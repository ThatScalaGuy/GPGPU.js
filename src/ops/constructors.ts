import type { DataType, TypedArray } from "../core/types";
import { DEFAULT_WORKGROUP_SIZE } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { createOutputBuffer, uploadBuffer, dispatchOnly } from "../core/command";
import { GPUArray } from "../pipeline/gpu-array";
import { type OpOptions, finalize } from "../core/io";
import { computeWorkgroupCount } from "../utils/workgroup";

/**
 * Options for the array constructors ({@link gpuZeros}, {@link gpuFull},
 * {@link gpuArange}): output `dtype` (default `"f32"`) plus the standard
 * `keepOnGpu` flag. {@link gpuLinspace} is f32-only and takes plain OpOptions.
 */
export interface ConstructorOpts extends OpOptions {
  dtype?: DataType;
}

const WG = DEFAULT_WORKGROUP_SIZE;

// --- Shader ----------------------------------------------------------------
// One parameterized sequence shader covers every constructor:
//   out[i] = start + T(i) * step        (fill: step = 0; arange: the step)
// with an optional exact last element (linspace pins out[n-1] to stop, the
// same trick NumPy uses to make the inclusive endpoint exact instead of
// accumulating rounding from the formula).
//
// All values arrive via the uniform, so no f32 literals appear in the WGSL.
// Pooled output buffers are NOT zero-initialized, so zeros MUST run this
// shader too (fill with 0) — every element i < n gets written.

function sequenceShader(dtype: DataType, workgroupSize = WG): string {
  const idx = dtype === "f32" ? "f32(i)" : dtype === "i32" ? "i32(i)" : "i";
  return `
struct Params { n: u32, use_last: u32, start: ${dtype}, step: ${dtype}, last: ${dtype} }

@group(0) @binding(0) var<storage, read_write> out: array<${dtype}>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  if (params.use_last == 1u && i + 1u == params.n) {
    out[i] = params.last;
    return;
  }
  let t = ${idx} * params.step;
  out[i] = params.start + t;
}
`;
}

// Uniform { n, use_last, start, step, last } — value fields encoded in the
// output dtype's bit pattern (DataView applies the same ToInt32/ToUint32/f32
// conversions the CPU references use, so both backends see identical bits).
function makeParams(
  dtype: DataType,
  n: number,
  useLast: boolean,
  start: number,
  step: number,
  last: number
): Uint32Array {
  const buf = new ArrayBuffer(32);
  const dv = new DataView(buf);
  dv.setUint32(0, n, true);
  dv.setUint32(4, useLast ? 1 : 0, true);
  if (dtype === "f32") {
    dv.setFloat32(8, start, true);
    dv.setFloat32(12, step, true);
    dv.setFloat32(16, last, true);
  } else if (dtype === "i32") {
    dv.setInt32(8, start, true);
    dv.setInt32(12, step, true);
    dv.setInt32(16, last, true);
  } else {
    dv.setUint32(8, start, true);
    dv.setUint32(12, step, true);
    dv.setUint32(16, last, true);
  }
  return new Uint32Array(buf);
}

// --- Shared helpers --------------------------------------------------------

function checkCount(name: string, param: string, n: number): void {
  if (!Number.isInteger(n) || n < 0) {
    throw new Error(`${name}: ${param} must be a nonnegative integer`);
  }
}

/** NumPy arange length: max(0, ceil((stop - start) / step)); step must be nonzero. */
function arangeLength(start: number, stop: number, step: number): number {
  if (step === 0) throw new Error("arange: step must be nonzero");
  return Math.max(0, Math.ceil((stop - start) / step));
}

/** The f32 step both backends use for linspace (num >= 2). */
function linspaceStep(start: number, stop: number, num: number): number {
  return Math.fround(Math.fround(Math.fround(stop) - Math.fround(start)) / (num - 1));
}

function emptyOf(dtype: DataType): TypedArray {
  if (dtype === "i32") return new Int32Array(0);
  if (dtype === "u32") return new Uint32Array(0);
  return new Float32Array(0);
}

// --- CPU references --------------------------------------------------------

export function cpuZeros(n: number, opts?: { dtype?: DataType }): TypedArray {
  return cpuFull(n, 0, opts);
}

export function cpuFull(n: number, value: number, opts?: { dtype?: DataType }): TypedArray {
  const dtype = opts?.dtype ?? "f32";
  if (dtype === "i32") return new Int32Array(n).fill(value | 0);
  if (dtype === "u32") return new Uint32Array(n).fill(value >>> 0);
  return new Float32Array(n).fill(Math.fround(value));
}

/**
 * CPU reference for {@link gpuArange}. Integer dtypes use wrapping 32-bit
 * arithmetic (Math.imul + typed-array coercion), f32 rounds each intermediate
 * with Math.fround — both matching what the GPU computes in the output dtype,
 * so exact comparisons hold (for f32, whenever i * step is exact, e.g. dyadic
 * steps; compare with a tolerance otherwise).
 */
export function cpuArange(
  start: number,
  stop: number,
  step: number,
  opts?: { dtype?: DataType }
): TypedArray {
  const dtype = opts?.dtype ?? "f32";
  const n = arangeLength(start, stop, step);
  if (dtype === "i32") {
    const out = new Int32Array(n);
    const s = start | 0;
    const st = step | 0;
    for (let i = 0; i < n; i++) out[i] = s + Math.imul(i, st);
    return out;
  }
  if (dtype === "u32") {
    const out = new Uint32Array(n);
    const s = start >>> 0;
    const st = step | 0;
    for (let i = 0; i < n; i++) out[i] = s + Math.imul(i, st);
    return out;
  }
  const out = new Float32Array(n);
  const s = Math.fround(start);
  const st = Math.fround(step);
  for (let i = 0; i < n; i++) out[i] = s + Math.fround(Math.fround(i) * st);
  return out;
}

/**
 * CPU reference for {@link gpuLinspace} (f32 only): `num` evenly spaced values
 * with both endpoints inclusive (the last element is pinned to `stop` exactly,
 * as on the GPU); `num === 1` yields `[start]`.
 */
export function cpuLinspace(start: number, stop: number, num: number): Float32Array {
  checkCount("linspace", "num", num);
  if (num === 0) return new Float32Array(0);
  const s = Math.fround(start);
  if (num === 1) return Float32Array.of(s);
  const out = new Float32Array(num);
  const st = linspaceStep(start, stop, num);
  for (let i = 0; i < num - 1; i++) out[i] = s + Math.fround(Math.fround(i) * st);
  out[num - 1] = Math.fround(stop);
  return out;
}

// --- GPU core --------------------------------------------------------------

async function runSequence(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  n: number,
  dtype: DataType,
  useLast: boolean,
  start: number,
  step: number,
  last: number,
  keepOnGpu: boolean
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();

  if (n === 0) {
    if (!keepOnGpu) return emptyOf(dtype);
    const empty = bufferPool.acquire(device, 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    return finalize(new GPUArray(empty, 0, dtype, device, bufferPool), true);
  }

  const pipeline = await shaderCache.getOrCreate(
    device,
    sequenceShader(dtype),
    `constructors-${dtype}`
  );

  const outBuf = createOutputBuffer(device, n * 4, bufferPool);

  const params = makeParams(dtype, n, useLast, start, step, last);
  const bufParams = uploadBuffer(device, params, GPUBufferUsage.UNIFORM, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: outBuf, size: n * 4 } },
      { binding: 1, resource: { buffer: bufParams, size: 20 } },
    ],
  });

  dispatchOnly(device, pipeline, bindGroup, [computeWorkgroupCount(n)]);

  bufferPool.release(bufParams);

  return finalize(new GPUArray(outBuf, n, dtype, device, bufferPool), keepOnGpu);
}

// --- GPU ops ---------------------------------------------------------------

/** `n` zeros in `dtype` (default f32). Actually writes every element — a pooled buffer holds junk. */
export function gpuZeros(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  n: number,
  opts: ConstructorOpts & { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuZeros(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  n: number,
  opts?: ConstructorOpts
): Promise<TypedArray>;
export async function gpuZeros(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  n: number,
  opts?: ConstructorOpts
): Promise<TypedArray | GPUArray> {
  return gpuFull(deviceManager, bufferPool, shaderCache, n, 0, opts);
}

/** `n` copies of `value` in `dtype` (default f32). */
export function gpuFull(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  n: number,
  value: number,
  opts: ConstructorOpts & { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuFull(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  n: number,
  value: number,
  opts?: ConstructorOpts
): Promise<TypedArray>;
export async function gpuFull(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  n: number,
  value: number,
  opts?: ConstructorOpts
): Promise<TypedArray | GPUArray> {
  checkCount("full", "n", n);
  const dtype = opts?.dtype ?? "f32";
  return runSequence(
    deviceManager,
    bufferPool,
    shaderCache,
    n,
    dtype,
    false,
    value,
    0,
    0,
    opts?.keepOnGpu ?? false
  );
}

/**
 * NumPy-style `arange(start, stop, step)`: length `max(0, ceil((stop - start) / step))`,
 * `value[i] = start + i * step` computed in the output dtype (f32 default; i32/u32
 * use wrapping integer arithmetic). Throws if `step` is zero.
 */
export function gpuArange(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  start: number,
  stop: number,
  step: number,
  opts: ConstructorOpts & { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuArange(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  start: number,
  stop: number,
  step: number,
  opts?: ConstructorOpts
): Promise<TypedArray>;
export async function gpuArange(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  start: number,
  stop: number,
  step: number,
  opts?: ConstructorOpts
): Promise<TypedArray | GPUArray> {
  const dtype = opts?.dtype ?? "f32";
  const n = arangeLength(start, stop, step);
  return runSequence(
    deviceManager,
    bufferPool,
    shaderCache,
    n,
    dtype,
    false,
    start,
    step,
    0,
    opts?.keepOnGpu ?? false
  );
}

/**
 * `num` evenly spaced f32 values from `start` to `stop`, endpoints inclusive
 * (the last element is pinned to `stop` exactly). `num === 1` yields `[start]`.
 * `num` must be a nonnegative integer.
 */
export function gpuLinspace(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  start: number,
  stop: number,
  num: number,
  opts: OpOptions & { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuLinspace(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  start: number,
  stop: number,
  num: number,
  opts?: OpOptions
): Promise<TypedArray>;
export async function gpuLinspace(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  start: number,
  stop: number,
  num: number,
  opts?: OpOptions
): Promise<TypedArray | GPUArray> {
  checkCount("linspace", "num", num);
  const keepOnGpu = opts?.keepOnGpu ?? false;
  const step = num >= 2 ? linspaceStep(start, stop, num) : 0;
  return runSequence(
    deviceManager,
    bufferPool,
    shaderCache,
    num,
    "f32",
    num >= 2,
    start,
    step,
    stop,
    keepOnGpu
  );
}
