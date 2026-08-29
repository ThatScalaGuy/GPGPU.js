import type { DataType, NumericArray, TypedArray } from "../core/types";
import { inferDataType } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { GPUArray } from "../pipeline/gpu-array";
import {
  type OpInput,
  type OpOptions,
  inputDtype,
  isGPUArray,
} from "../core/io";
import { toTypedArray } from "../utils/data-conversion";
import { gpuArange } from "./constructors";
import { gpuSortByKey } from "./sort-by-key";
import { gpuSlice } from "./slice";

/** Options for {@link gpuTopK}: pick the largest (default) or smallest k, plus keepOnGpu. */
export interface TopKOptions extends OpOptions {
  /** Select the k largest values (default true); false selects the k smallest. */
  largest?: boolean;
}

/** Allocate a result array matching `dtype` (same helper shape as unique.ts). */
function resultArray(dtype: DataType, len: number): TypedArray {
  if (dtype === "i32") return new Int32Array(len);
  if (dtype === "u32") return new Uint32Array(len);
  return new Float32Array(len);
}

/** Clamp k to [0, n]; non-integers truncate (slice-style), k <= 0 selects nothing. */
function effectiveK(k: number, n: number): number {
  return Math.max(0, Math.min(Math.trunc(k), n));
}

/**
 * Top-k selection (torch.topk-like): `values` are the k largest elements sorted
 * descending (or the k smallest sorted ascending with `largest: false`), and
 * `indices` are their original positions (u32) with `input[indices[j]] === values[j]`.
 * `values` dtype follows the input. `k <= 0` yields an empty pair; `k >= n` yields
 * all n elements, sorted.
 *
 * Composition: a u32 iota (gpuArange) rides along as the value lane of a
 * key-value bitonic sort of the input (descending for `largest`), then both
 * sorted arrays are sliced to [0, k). The caller's input is never mutated
 * (gpuSortByKey copies both lanes before sorting).
 *
 * Ties: the bitonic sort is unstable, so the order among equal values — and which
 * of several equally-valued elements make the cut at the k boundary — is
 * unspecified. `input[indices[j]] === values[j]` always holds for tied values too.
 * Inherited pad caveat (from gpuSortByKey): a real key equal to the pad sentinel
 * (u32 0 with `largest: true`, or the type's max — u32 0xFFFFFFFF / i32 2147483647 /
 * f32 +Infinity — with `largest: false`) can have its paired index replaced by the
 * pad value 0 when n is not a power of two.
 */
export function gpuTopK(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  k: number,
  opts: TopKOptions & { keepOnGpu: true }
): Promise<{ values: GPUArray; indices: GPUArray }>;
export function gpuTopK(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  k: number,
  opts?: TopKOptions
): Promise<{ values: TypedArray; indices: TypedArray }>;
export async function gpuTopK(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  k: number,
  opts?: TopKOptions
): Promise<{ values: TypedArray | GPUArray; indices: TypedArray | GPUArray }> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(input);
  const keepOnGpu = opts?.keepOnGpu ?? false;
  const largest = opts?.largest ?? true;

  const n = isGPUArray(input) ? input.length : toTypedArray(input, dtype).length;
  const kEff = effectiveK(k, n);

  if (kEff === 0) {
    if (!keepOnGpu) {
      return { values: resultArray(dtype, 0), indices: new Uint32Array(0) };
    }
    // The pool cannot hand out 0-byte buffers; wrap minimum buckets in
    // length-0 GPUArrays (same pattern as gpuSlice / gpuUnique).
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    return {
      values: new GPUArray(bufferPool.acquire(device, 4, usage), 0, dtype, device, bufferPool),
      indices: new GPUArray(bufferPool.acquire(device, 4, usage), 0, "u32", device, bufferPool),
    };
  }

  // 1. iota 0..n-1 as u32 — each element's original position.
  const iota = await gpuArange(deviceManager, bufferPool, shaderCache, 0, n, 1, {
    dtype: "u32",
    keepOnGpu: true,
  });

  // 2. Sort (value, position) pairs by value; descending puts the largest first.
  // gpuSortByKey copies both lanes, so neither `input` nor `iota` is mutated.
  const [sortedVals, sortedIdx] = await gpuSortByKey(
    deviceManager,
    bufferPool,
    shaderCache,
    input,
    iota,
    { descending: largest, keepOnGpu: true }
  );
  iota.destroy();

  // 3. Take the first k of each lane. Passing keepOnGpu through gpuSlice performs
  // the readback (finalize) when the caller wants CPU arrays; the runtime flag
  // decides the shape, so widen the compile-time overload result accordingly.
  const values = (await gpuSlice(
    deviceManager, bufferPool, shaderCache, sortedVals, 0, kEff, { keepOnGpu }
  )) as TypedArray | GPUArray;
  const indices = (await gpuSlice(
    deviceManager, bufferPool, shaderCache, sortedIdx, 0, kEff, { keepOnGpu }
  )) as TypedArray | GPUArray;

  sortedVals.destroy();
  sortedIdx.destroy();

  return { values, indices };
}

/**
 * CPU reference for {@link gpuTopK}. Deterministic: ties break by ascending
 * original index (the GPU's tie order is unspecified instead, so compare
 * `values` exactly but verify `indices` via `input[indices[j]] === values[j]`).
 */
export function cpuTopK(
  input: NumericArray,
  k: number,
  largest = true
): { values: TypedArray; indices: Uint32Array } {
  const dtype = inferDataType(input);
  const arr = toTypedArray(input, dtype);
  const n = arr.length;
  const kEff = effectiveK(k, n);

  const order = Array.from({ length: n }, (_, i) => i);
  // Explicit comparisons (not subtraction): safe for u32 magnitudes and infinities.
  const cmp = (x: number, y: number): number => (x < y ? -1 : x > y ? 1 : 0);
  order.sort((a, b) => {
    const d = largest ? cmp(arr[b], arr[a]) : cmp(arr[a], arr[b]);
    return d !== 0 ? d : a - b;
  });

  const values = resultArray(dtype, kEff);
  const indices = new Uint32Array(kEff);
  for (let j = 0; j < kEff; j++) {
    values[j] = arr[order[j]];
    indices[j] = order[j];
  }
  return { values, indices };
}
