import type { DataType, NumericArray, TypedArray } from "../core/types";
import { DEFAULT_WORKGROUP_SIZE, inferDataType } from "../core/types";
import { toTypedArray } from "../utils/data-conversion";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { createOutputBuffer, dispatchOnly, uploadBuffer } from "../core/command";
import { GPUArray } from "../pipeline/gpu-array";
import {
  type OpInput,
  type OpOptions,
  resolveInput,
  inputDtype,
  finalize,
} from "../core/io";
import { computeWorkgroupCount } from "../utils/workgroup";

/** Output-length convention, matching `numpy.convolve`'s `mode`. */
export type ConvolveMode = "full" | "same" | "valid";

/** Options for {@link gpuConvolve} / `gpu.convolve`. */
export interface ConvolveOpts extends OpOptions {
  /** Output length convention (default `"full"`). */
  mode?: ConvolveMode;
}

/**
 * Discrete 1-D convolution length for the given input/kernel sizes and mode
 * (the `numpy.convolve` convention). `n` is the signal length, `m` the kernel.
 */
export function convolveOutputLength(n: number, m: number, mode: ConvolveMode): number {
  if (mode === "valid") return Math.max(n, m) - Math.min(n, m) + 1;
  if (mode === "same") return Math.max(n, m);
  return n + m - 1; // full
}

// One output element per thread. The full convolution is out[t] = Σ_k a[t-k]*v[k]
// (the kernel v is reversed relative to a, i.e. true convolution, not correlation —
// matching numpy.convolve). `same`/`valid` are the `full` result shifted by a fixed
// offset so the centred / fully-overlapping window is returned; offset lives in the
// uniform. The kernel loop is sequential per thread; a out-of-range a-index contributes
// nothing (treated as zero padding), so no bounds-clamping read is needed.
function convolveShader(
  elemType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  const zero = elemType === "f32" ? "0.0" : "0";
  return `
struct Params { n: u32, m: u32, outLen: u32, offset: u32 }

@group(0) @binding(0) var<storage, read> a: array<${elemType}>;
@group(0) @binding(1) var<storage, read> v: array<${elemType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${elemType}>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= params.outLen) { return; }
  // Position in the full-mode output that this element maps to.
  let p = t + params.offset;
  var acc: ${elemType} = ${zero};
  // out[p] = sum over k of a[p-k] * v[k], for k where 0 <= p-k < n and 0 <= k < m.
  // k ranges over [max(0, p-(n-1)), min(p, m-1)].
  let kLo = select(0u, p - (params.n - 1u), p + 1u > params.n);
  var k = kLo;
  loop {
    if (k > p || k >= params.m) { break; }
    acc = acc + a[p - k] * v[k];
    k = k + 1u;
  }
  output[t] = acc;
}
`;
}

export function gpuConvolve(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  kernel: OpInput,
  opts: ConvolveOpts & { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuConvolve(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  kernel: OpInput,
  opts?: ConvolveOpts
): Promise<TypedArray>;
export async function gpuConvolve(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  kernel: OpInput,
  opts?: ConvolveOpts
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(input);
  const mode = opts?.mode ?? "full";
  const keepOnGpu = opts?.keepOnGpu ?? false;

  const ra = resolveInput(input, device, bufferPool, dtype);
  const rv = resolveInput(kernel, device, bufferPool, dtype);
  const n = ra.length;
  const m = rv.length;

  if (n === 0 || m === 0) {
    ra.release();
    rv.release();
    const empty = bufferPool.acquire(device, 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    return finalize(new GPUArray(empty, 0, dtype, device, bufferPool), keepOnGpu);
  }

  const outLen = convolveOutputLength(n, m, mode);
  // Offset into the `full` output (length n+m-1) at which this mode's window starts.
  const fullLen = n + m - 1;
  const offset = mode === "valid"
    ? Math.min(n, m) - 1
    : mode === "same"
      ? Math.floor((fullLen - outLen) / 2)
      : 0;

  const pipeline = await shaderCache.getOrCreate(
    device, convolveShader(dtype), `convolve-${dtype}`
  );

  const outByteSize = outLen * 4;
  const bufOut = createOutputBuffer(device, outByteSize, bufferPool);

  // Params uniform: 4 x u32, already 16-byte aligned.
  const params = new Uint32Array([n, m, outLen, offset]);
  const bufParams = uploadBuffer(device, params, GPUBufferUsage.UNIFORM, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ra.buffer, size: n * 4 } },
      { binding: 1, resource: { buffer: rv.buffer, size: m * 4 } },
      { binding: 2, resource: { buffer: bufOut, size: outByteSize } },
      { binding: 3, resource: { buffer: bufParams, size: 16 } },
    ],
  });

  dispatchOnly(device, pipeline, bindGroup, [computeWorkgroupCount(outLen)]);

  ra.release();
  rv.release();
  bufferPool.release(bufParams);

  return finalize(new GPUArray(bufOut, outLen, dtype, device, bufferPool), keepOnGpu);
}

/**
 * CPU reference for {@link gpuConvolve}: discrete 1-D convolution of `input` with
 * `kernel`, matching `numpy.convolve(input, kernel, mode)`. The kernel is reversed
 * relative to the signal (true convolution), and `mode` selects the output window.
 * Result dtype follows `input`.
 */
export function cpuConvolve(
  input: NumericArray,
  kernel: NumericArray,
  mode: ConvolveMode = "full"
): TypedArray {
  const dtype = inferDataType(input);
  const a = toTypedArray(input, dtype);
  const v = toTypedArray(kernel, dtype);
  const n = a.length;
  const m = v.length;

  const make = (len: number): TypedArray =>
    dtype === "i32" ? new Int32Array(len) : dtype === "u32" ? new Uint32Array(len) : new Float32Array(len);

  if (n === 0 || m === 0) return make(0);

  const outLen = convolveOutputLength(n, m, mode);
  const fullLen = n + m - 1;
  const offset =
    mode === "valid"
      ? Math.min(n, m) - 1
      : mode === "same"
        ? Math.floor((fullLen - outLen) / 2)
        : 0;

  const out = make(outLen);
  for (let t = 0; t < outLen; t++) {
    const p = t + offset;
    let acc = 0;
    const kLo = Math.max(0, p - (n - 1));
    const kHi = Math.min(p, m - 1);
    for (let k = kLo; k <= kHi; k++) {
      acc += a[p - k] * v[k];
    }
    out[t] = acc;
  }
  return out;
}
