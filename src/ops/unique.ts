import type { DataType, NumericArray, TypedArray } from "../core/types";
import { DEFAULT_WORKGROUP_SIZE, inferDataType } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { dispatchOnly } from "../core/command";
import { GPUArray } from "../pipeline/gpu-array";
import {
  type OpInput,
  type OpOptions,
  inputDtype,
  finalize,
} from "../core/io";
import { toTypedArray } from "../utils/data-conversion";
import { computeWorkgroupCount } from "../utils/workgroup";
import { gpuSort } from "./sort";
import { gpuScan } from "./scan";

// On a SORTED array, flag the first element of each run of equal values: flags[i] = 1u when
// i == 0 or sorted[i] != sorted[i-1], else 0u. The inclusive scan of these flags then gives
// each unique value its 1-based output slot (the same flags->scan->compact shape as filter,
// but with a run-boundary flag instead of a user predicate). Equality is bit-exact for ints
// and for f32 (duplicates are bit-identical and sort adjacent); NaN is not handled, matching sort.
function boundaryFlagShader(
  elemType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  return `
@group(0) @binding(0) var<storage, read> sorted: array<${elemType}>;
@group(0) @binding(1) var<storage, read_write> flags: array<u32>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= arrayLength(&sorted)) { return; }
  flags[idx] = select(select(0u, 1u, sorted[idx] != sorted[idx - 1u]), 1u, idx == 0u);
}
`;
}

// Stream compaction: each run-boundary element (flags[idx]==1) writes its value to its
// compacted slot. `scanned` is the INCLUSIVE prefix sum of flags, so the 0-based output
// index of a kept element is scanned[idx] - 1. Identical in spirit to filter's compaction.
function compactShader(
  elemType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  return `
@group(0) @binding(0) var<storage, read> sorted: array<${elemType}>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> scanned: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<${elemType}>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= arrayLength(&sorted)) { return; }
  if (flags[idx] == 1u) {
    output[scanned[idx] - 1u] = sorted[idx];
  }
}
`;
}

export function gpuUnique(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts: { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuUnique(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts?: OpOptions
): Promise<TypedArray>;
export async function gpuUnique(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts?: OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(input);
  const keepOnGpu = opts?.keepOnGpu ?? false;

  const n = input instanceof GPUArray ? input.length : toTypedArray(input, dtype).length;

  if (n === 0) {
    const empty = bufferPool.acquire(device, 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    return finalize(new GPUArray(empty, 0, dtype, device, bufferPool), keepOnGpu);
  }

  // 1. Sort ascending so equal values become adjacent runs. gpuSort copies its input, so a
  // GPUArray input is never mutated. The result is a length-n GPUArray we own.
  const sorted = (await gpuSort(
    deviceManager, bufferPool, shaderCache, input, { keepOnGpu: true }
  )) as GPUArray;

  // 2. flags[i] = first-of-run boundary on the sorted data.
  const flagPipe = await shaderCache.getOrCreate(
    device, boundaryFlagShader(dtype), `unique-flags-${dtype}`
  );
  const flagsBuf = bufferPool.acquire(
    device, n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  );
  const flagGroup = device.createBindGroup({
    layout: flagPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sorted.buffer, size: n * 4 } },
      { binding: 1, resource: { buffer: flagsBuf, size: n * 4 } },
    ],
  });
  dispatchOnly(device, flagPipe, flagGroup, [computeWorkgroupCount(n)]);
  const flagsArr = new GPUArray(flagsBuf, n, "u32", device, bufferPool);

  // 3. inclusive scan of flags (gpuScan COPIES its input, so flagsArr survives for compaction).
  const scanned = (await gpuScan(
    deviceManager, bufferPool, shaderCache, flagsArr, (a, b) => a + b, 0, { keepOnGpu: true }
  )) as GPUArray;

  // 4. count = scanned[n-1] (single-u32 GPU->CPU readback, the only sync point).
  const staging = bufferPool.acquire(device, 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(scanned.buffer, (n - 1) * 4, staging, 0, 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const count = new Uint32Array(staging.getMappedRange().slice(0))[0];
  staging.unmap();
  bufferPool.release(staging);

  // 5. compact: each boundary element writes its value to output[scanned[i]-1]. Reads sorted,
  // flags and scanned, so it runs BEFORE they are destroyed. count >= 1 here (n >= 1).
  const outBuf = bufferPool.acquire(
    device, count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const compactPipe = await shaderCache.getOrCreate(
    device, compactShader(dtype), `unique-compact-${dtype}`
  );
  const compactGroup = device.createBindGroup({
    layout: compactPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sorted.buffer, size: n * 4 } },
      { binding: 1, resource: { buffer: flagsArr.buffer, size: n * 4 } },
      { binding: 2, resource: { buffer: scanned.buffer, size: n * 4 } },
      { binding: 3, resource: { buffer: outBuf, size: count * 4 } },
    ],
  });
  dispatchOnly(device, compactPipe, compactGroup, [computeWorkgroupCount(n)]);

  sorted.destroy();
  flagsArr.destroy();
  scanned.destroy();

  return finalize(new GPUArray(outBuf, count, dtype, device, bufferPool), keepOnGpu);
}

/** Allocate a zero-filled result array matching `dtype` (mirrors the cpu-ops helper). */
function resultArray(dtype: DataType, len: number): TypedArray {
  if (dtype === "i32") return new Int32Array(len);
  if (dtype === "u32") return new Uint32Array(len);
  return new Float32Array(len);
}

// Sorted distinct values of the input (NumPy np.unique semantics). Output dtype follows the
// input; values are copied verbatim, so the GPU and CPU paths agree exactly for every dtype
// (no floating-point reassociation). NaN is not handled, matching sort.
export function cpuUnique(input: NumericArray): TypedArray {
  const dtype = inferDataType(input);
  const arr = toTypedArray(input, dtype).slice();
  arr.sort((a, b) => a - b);
  const out: number[] = [];
  for (let i = 0; i < arr.length; i++) {
    if (i === 0 || arr[i] !== arr[i - 1]) out.push(arr[i]);
  }
  const result = resultArray(dtype, out.length);
  result.set(out);
  return result;
}
