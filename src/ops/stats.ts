import type { NumericArray, TypedArray } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { GPUArray } from "../pipeline/gpu-array";
import {
  type OpInput,
  type OpOptions,
  inputDtype,
  isGPUArray,
  finalize,
} from "../core/io";
import { gpuSum, gpuMax } from "./reduce";
import { gpuMap, gpuZip, gpuScalarBroadcast } from "./elementwise";
import { gpuCast } from "./cast";

// Statistics/ML convenience pack. Every op here is a COMPOSITION of existing gpu ops
// (reduce/map/zip/scalar-broadcast/cast) — no new WGSL. Intermediates are chained with
// { keepOnGpu: true } so multi-step ops round-trip through the CPU only for the final
// scalar readbacks, and every intermediate GPUArray created here is destroyed.

function inputLength(input: OpInput): number {
  return isGPUArray(input) ? input.length : input.length;
}

/** Arithmetic mean, sum/n. Empty input yields NaN (mean of nothing is undefined). */
export async function gpuMean(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput
): Promise<number> {
  const n = inputLength(input);
  if (n === 0) return NaN;
  const s = await gpuSum(deviceManager, bufferPool, shaderCache, input);
  return s / n;
}

// Integer inputs are cast to f32 before mean-centred arithmetic: the mean is generally
// fractional and deviations are negative, both of which i32/u32 elementwise math would
// truncate or wrap. Returns the working input plus whether we own (and must destroy) it.
async function asF32(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput
): Promise<{ x: OpInput; owned: GPUArray | null }> {
  if (inputDtype(input) === "f32") return { x: input, owned: null };
  const cast = await gpuCast(deviceManager, bufferPool, shaderCache, input, "f32", {
    keepOnGpu: true,
  });
  return { x: cast, owned: cast };
}

/**
 * Population variance (ddof=0, the NumPy default): mean of squared deviations.
 * Two-pass: compute the mean, subtract it via a scalar broadcast (the mean is passed
 * through a uniform, never embedded as a source literal), square, sum, divide by n.
 * Empty input yields NaN.
 */
export async function gpuVariance(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput
): Promise<number> {
  const n = inputLength(input);
  if (n === 0) return NaN;

  const { x, owned } = await asF32(deviceManager, bufferPool, shaderCache, input);
  const m = await gpuMean(deviceManager, bufferPool, shaderCache, x);
  const dev = await gpuScalarBroadcast(deviceManager, bufferPool, shaderCache, x, m, "-", {
    keepOnGpu: true,
  });
  if (owned) owned.destroy();
  const sq = await gpuZip(deviceManager, bufferPool, shaderCache, dev, dev, "a * b", {
    keepOnGpu: true,
  });
  dev.destroy();
  const s = await gpuSum(deviceManager, bufferPool, shaderCache, sq);
  sq.destroy();
  return s / n;
}

/** Population standard deviation, sqrt(variance). Empty input yields NaN. */
export async function gpuStd(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput
): Promise<number> {
  return Math.sqrt(await gpuVariance(deviceManager, bufferPool, shaderCache, input));
}

/** Dot product: elementwise multiply, then sum. Throws when the lengths differ. */
export async function gpuDot(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  a: OpInput,
  b: OpInput
): Promise<number> {
  const na = inputLength(a);
  const nb = inputLength(b);
  if (na !== nb) {
    throw new Error(`dot: inputs must have the same length (got ${na} and ${nb})`);
  }
  if (na === 0) return 0;
  const prod = await gpuZip(deviceManager, bufferPool, shaderCache, a, b, "a * b", {
    keepOnGpu: true,
  });
  const s = await gpuSum(deviceManager, bufferPool, shaderCache, prod);
  prod.destroy();
  return s;
}

/** L2 norm, sqrt(dot(x, x)). */
export async function gpuNorm(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput
): Promise<number> {
  return Math.sqrt(await gpuDot(deviceManager, bufferPool, shaderCache, input, input));
}

/**
 * Cosine similarity, dot(a, b) / (norm(a) * norm(b)). Throws when the lengths differ;
 * yields NaN when either norm is zero (direction of a zero vector is undefined).
 */
export async function gpuCosineSimilarity(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  a: OpInput,
  b: OpInput
): Promise<number> {
  // gpuDot validates the lengths before any GPU work.
  const d = await gpuDot(deviceManager, bufferPool, shaderCache, a, b);
  const na = await gpuNorm(deviceManager, bufferPool, shaderCache, a);
  const nb = await gpuNorm(deviceManager, bufferPool, shaderCache, b);
  if (na === 0 || nb === 0) return NaN;
  return d / (na * nb);
}

export function gpuSoftmax(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts: { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuSoftmax(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts?: OpOptions
): Promise<TypedArray>;
/**
 * Numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x))). Always f32
 * semantics — an i32/u32 input is cast first, so the result is the softmax of the
 * numeric values. keepOnGpu rides on the final divide step.
 */
export async function gpuSoftmax(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts?: OpOptions
): Promise<TypedArray | GPUArray> {
  const keepOnGpu = opts?.keepOnGpu ?? false;
  const n = inputLength(input);

  if (n === 0) {
    const device = await deviceManager.getDevice();
    const empty = bufferPool.acquire(
      device, 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    return finalize(new GPUArray(empty, 0, "f32", device, bufferPool), keepOnGpu);
  }

  const { x, owned } = await asF32(deviceManager, bufferPool, shaderCache, input);
  const m = await gpuMax(deviceManager, bufferPool, shaderCache, x);
  const shifted = await gpuScalarBroadcast(deviceManager, bufferPool, shaderCache, x, m, "-", {
    keepOnGpu: true,
  });
  if (owned) owned.destroy();
  const exps = await gpuMap(deviceManager, bufferPool, shaderCache, shifted, "Math.exp(x)", {
    keepOnGpu: true,
  });
  shifted.destroy();
  const s = await gpuSum(deviceManager, bufferPool, shaderCache, exps);
  const result = await gpuScalarBroadcast(
    deviceManager, bufferPool, shaderCache, exps, s, "/", opts
  );
  exps.destroy();
  return result;
}

// ---------------------------------------------------------------------------
// CPU reference implementations (plain sequential loops, f64 accumulation).
// ---------------------------------------------------------------------------

export function cpuMean(input: NumericArray): number {
  const n = input.length;
  if (n === 0) return NaN;
  let s = 0;
  for (let i = 0; i < n; i++) s += input[i];
  return s / n;
}

export function cpuVariance(input: NumericArray): number {
  const n = input.length;
  if (n === 0) return NaN;
  const m = cpuMean(input);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const d = input[i] - m;
    s += d * d;
  }
  return s / n;
}

export function cpuStd(input: NumericArray): number {
  return Math.sqrt(cpuVariance(input));
}

export function cpuDot(a: NumericArray, b: NumericArray): number {
  if (a.length !== b.length) {
    throw new Error(`dot: inputs must have the same length (got ${a.length} and ${b.length})`);
  }
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function cpuNorm(input: NumericArray): number {
  return Math.sqrt(cpuDot(input, input));
}

export function cpuCosineSimilarity(a: NumericArray, b: NumericArray): number {
  const d = cpuDot(a, b);
  const na = cpuNorm(a);
  const nb = cpuNorm(b);
  if (na === 0 || nb === 0) return NaN;
  return d / (na * nb);
}

export function cpuSoftmax(input: NumericArray): Float32Array {
  const n = input.length;
  const out = new Float32Array(n);
  if (n === 0) return out;
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (input[i] > m) m = input[i];
  const exps = new Float64Array(n);
  let s = 0;
  for (let i = 0; i < n; i++) {
    exps[i] = Math.exp(input[i] - m);
    s += exps[i];
  }
  for (let i = 0; i < n; i++) out[i] = exps[i] / s;
  return out;
}
