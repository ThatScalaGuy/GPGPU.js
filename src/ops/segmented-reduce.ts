import type { DataType, TypedArray } from "../core/types";
import { DEFAULT_WORKGROUP_SIZE } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { GPUArray } from "../pipeline/gpu-array";
import { type OpInput, type OpOptions, resolveInput, inputDtype, finalize } from "../core/io";
import { computeWorkgroupGrid } from "../utils/workgroup";
import { dispatchOnly } from "../core/command";

/** The associative reduction applied per segment. */
export type SegmentedReduceOp = "sum" | "max" | "min" | "product";

/** Options for {@link gpuSegmentedReduce}: which segments exist and how to combine. */
export interface SegmentedReduceOpts {
  /** Number of output segments. Output length === numSegments. */
  numSegments: number;
  /** Reduction per segment (default `"sum"`). */
  op?: SegmentedReduceOp;
}

const WG = DEFAULT_WORKGROUP_SIZE;

// Identity per (op, dtype) — the value an empty segment reduces to, and the seed each
// accumulator slot is initialised with before any contribution lands. min/max reuse the
// same safe finite extremes the reduce family uses (toFixed renders the exact f32 max just
// above the representable max → shader compile error, so use 3.4e38, still > any real input).
const IDENTITY: Record<SegmentedReduceOp, Record<DataType, number>> = {
  sum: { f32: 0, i32: 0, u32: 0 },
  product: { f32: 1, i32: 1, u32: 1 },
  min: { f32: 3.4e38, i32: 2147483647, u32: 4294967295 },
  max: { f32: -3.4e38, i32: -2147483648, u32: 0 },
};

// Combine `acc` (the accumulator) with the incoming value `v`, as a WGSL expression over
// f32 (the `f32` shader bitcasts the slot to/from f32 around this). For the integer shaders
// the same expression is valid over i32/u32.
function combineExpr(op: SegmentedReduceOp): string {
  switch (op) {
    case "sum":
      return "acc + v";
    case "product":
      return "acc * v";
    case "min":
      return "min(acc, v)";
    case "max":
      return "max(acc, v)";
  }
}

// One thread per input element: atomically fold values[i] into out[segmentIds[i]].
//
// sum on i32/u32 is the only case with a native atomic (atomicAdd). Every other case —
// product/min/max on integers, and ALL f32 cases (WGSL has no atomic<f32>) — uses a
// compare-exchange loop over an atomic<u32> slot: read the current bits, compute the
// combined value, and try to swap the new bits in; retry on a lost race. f32 slots bitcast
// the bits to/from f32 around the combine; integer slots reinterpret the u32 bits as the
// signed/unsigned value. This is portable core WGSL (no extensions, no forward-progress
// assumptions beyond a standard CAS retry) and collision-safe for duplicate segment ids.
//
// An out-of-range segment id clamps to the last segment (the shader can't throw), matching
// the scatter/gather shaders.
function segmentedReduceShader(op: SegmentedReduceOp, elemType: DataType): string {
  const combine = combineExpr(op);

  // Native fast path: integer sum.
  if (op === "sum" && elemType !== "f32") {
    return `
@group(0) @binding(0) var<storage, read> values: array<${elemType}>;
@group(0) @binding(1) var<storage, read> segIds: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<atomic<${elemType}>>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(num_workgroups) nwg: vec3u) {
  let i = gid.y * (nwg.x * ${WG}u) + gid.x;
  if (i >= arrayLength(&values)) { return; }
  let s = min(segIds[i], arrayLength(&out) - 1u);
  atomicAdd(&out[s], values[i]);
}
`;
  }

  // f32: CAS loop over u32 bits, combining in f32.
  if (elemType === "f32") {
    return `
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> segIds: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<atomic<u32>>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(num_workgroups) nwg: vec3u) {
  let i = gid.y * (nwg.x * ${WG}u) + gid.x;
  if (i >= arrayLength(&values)) { return; }
  let s = min(segIds[i], arrayLength(&out) - 1u);
  let v = values[i];
  var old = atomicLoad(&out[s]);
  loop {
    let acc = bitcast<f32>(old);
    let res = atomicCompareExchangeWeak(&out[s], old, bitcast<u32>(${combine}));
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}
`;
  }

  // Integer min/max/product: CAS loop over the native integer slot.
  return `
@group(0) @binding(0) var<storage, read> values: array<${elemType}>;
@group(0) @binding(1) var<storage, read> segIds: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<atomic<${elemType}>>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(num_workgroups) nwg: vec3u) {
  let i = gid.y * (nwg.x * ${WG}u) + gid.x;
  if (i >= arrayLength(&values)) { return; }
  let s = min(segIds[i], arrayLength(&out) - 1u);
  let v = values[i];
  var acc = atomicLoad(&out[s]);
  loop {
    let res = atomicCompareExchangeWeak(&out[s], acc, ${combine});
    if (res.exchanged) { break; }
    acc = res.old_value;
  }
}
`;
}

export function gpuSegmentedReduce(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  values: OpInput,
  segmentIds: OpInput,
  opts: SegmentedReduceOpts & { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuSegmentedReduce(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  values: OpInput,
  segmentIds: OpInput,
  opts: SegmentedReduceOpts & OpOptions
): Promise<TypedArray>;
export async function gpuSegmentedReduce(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  values: OpInput,
  segmentIds: OpInput,
  opts: SegmentedReduceOpts & OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(values);
  const op = opts.op ?? "sum";
  const numSegments = opts.numSegments;
  const keepOnGpu = opts.keepOnGpu ?? false;

  if (numSegments <= 0) {
    throw new Error(`segmentedReduce: numSegments must be > 0, got ${numSegments}`);
  }

  const rvals = resolveInput(values, device, bufferPool, dtype);
  const rids = resolveInput(segmentIds, device, bufferPool, "u32");
  const n = rvals.length;
  const outBytes = numSegments * 4;

  // Accumulator buffer, seeded with the op's identity (so empty segments read back the
  // identity and the first real contribution combines correctly). Built on the CPU and
  // uploaded — small (numSegments) and avoids a second fill kernel.
  const seed = identityArray(op, dtype, numSegments);
  const out = bufferPool.acquire(
    device,
    outBytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  );
  device.queue.writeBuffer(out, 0, seed.buffer as ArrayBuffer, seed.byteOffset, seed.byteLength);

  if (n > 0) {
    const pipeline = await shaderCache.getOrCreate(
      device,
      segmentedReduceShader(op, dtype),
      `segreduce-${op}-${dtype}`
    );
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: rvals.buffer, size: n * 4 } },
        { binding: 1, resource: { buffer: rids.buffer, size: n * 4 } },
        { binding: 2, resource: { buffer: out, size: outBytes } },
      ],
    });
    dispatchOnly(device, pipeline, bindGroup, computeWorkgroupGrid(n));
  }

  rvals.release();
  rids.release();

  // Output dtype follows the values dtype.
  return finalize(new GPUArray(out, numSegments, dtype, device, bufferPool), keepOnGpu);
}

// Build a length-`numSegments` typed array filled with the op's identity for `dtype`.
function identityArray(op: SegmentedReduceOp, dtype: DataType, numSegments: number): TypedArray {
  const id = IDENTITY[op][dtype];
  if (dtype === "i32") return new Int32Array(numSegments).fill(id);
  if (dtype === "u32") return new Uint32Array(numSegments).fill(id);
  return new Float32Array(numSegments).fill(id);
}

/**
 * CPU reference for segmentedReduce: per-segment reduction of `values` keyed by
 * `segmentIds`. Out-of-range ids clamp to the last segment (matching the GPU shader).
 * Empty segments hold the op's identity. The output dtype follows `values`.
 */
export function cpuSegmentedReduce(
  values: TypedArray | number[],
  segmentIds: TypedArray | number[],
  opts: SegmentedReduceOpts
): TypedArray {
  const op = opts.op ?? "sum";
  const numSegments = opts.numSegments;
  if (numSegments <= 0) {
    throw new Error(`segmentedReduce: numSegments must be > 0, got ${numSegments}`);
  }

  const dtype: DataType =
    values instanceof Int32Array ? "i32" : values instanceof Uint32Array ? "u32" : "f32";

  const id = IDENTITY[op][dtype];
  const acc = new Array<number>(numSegments).fill(id);

  const combine = (a: number, b: number): number => {
    switch (op) {
      case "sum":
        return a + b;
      case "product":
        return a * b;
      case "min":
        return Math.min(a, b);
      case "max":
        return Math.max(a, b);
    }
  };

  for (let i = 0; i < values.length; i++) {
    let s = segmentIds[i] >>> 0; // read as u32
    if (s >= numSegments) s = numSegments - 1; // clamp out-of-range
    acc[s] = combine(acc[s], values[i]);
  }

  if (dtype === "i32") return Int32Array.from(acc);
  if (dtype === "u32") return Uint32Array.from(acc.map((x) => x >>> 0));
  return Float32Array.from(acc);
}
