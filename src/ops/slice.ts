import type { NumericArray, TypedArray } from "../core/types";
import { inferDataType } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { GPUArray } from "../pipeline/gpu-array";
import {
  type OpInput,
  type OpOptions,
  resolveInput,
  inputDtype,
  finalize,
} from "../core/io";
import { toTypedArray } from "../utils/data-conversion";

// JavaScript Array.prototype.slice index normalization: indices are truncated to
// integers, negative values count from the end, and everything clamps into
// [0, length]. Returns the clamped start plus the (possibly zero) result length.
function normalizeRange(
  length: number,
  begin?: number,
  end?: number
): { start: number; len: number } {
  let b = begin === undefined ? 0 : Math.trunc(begin);
  let e = end === undefined ? length : Math.trunc(end);
  b = b < 0 ? Math.max(length + b, 0) : Math.min(b, length);
  e = e < 0 ? Math.max(length + e, 0) : Math.min(e, length);
  return { start: b, len: Math.max(e - b, 0) };
}

/**
 * CPU reference for {@link gpuSlice}: JavaScript `Array.prototype.slice`
 * semantics (negative indices count from the end, out-of-range clamps).
 * Always returns a fresh typed array matching the input dtype.
 */
export function cpuSlice(input: NumericArray, begin?: number, end?: number): TypedArray {
  const dtype = inferDataType(input);
  // TypedArray.prototype.slice implements exactly the semantics we want and
  // always copies, so an already-typed input is never aliased.
  return toTypedArray(input, dtype).slice(begin, end);
}

/**
 * Sub-range extraction with `Array.prototype.slice` semantics. The result dtype
 * follows the input; the result is always a copy (a GPUArray input is never
 * mutated or aliased). No compute pass — a single buffer-to-buffer copy.
 */
export function gpuSlice(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  begin: number | undefined,
  end: number | undefined,
  opts: { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuSlice(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  begin?: number,
  end?: number,
  opts?: OpOptions
): Promise<TypedArray>;
export async function gpuSlice(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  // Unused (no shader), kept for the uniform op signature.
  _shaderCache: ShaderCache,
  input: OpInput,
  begin?: number,
  end?: number,
  opts?: OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(input);
  const keepOnGpu = opts?.keepOnGpu ?? false;

  // COPY_SRC so an uploaded CPU array can serve as the copy source (a GPUArray
  // input is reused in place; every op-produced buffer already carries COPY_SRC).
  const rsrc = resolveInput(
    input, device, bufferPool, dtype,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const { start, len } = normalizeRange(rsrc.length, begin, end);

  if (len === 0) {
    rsrc.release();
    // The pool cannot hand out a 0-byte buffer; acquire the minimum bucket and
    // carry length 0 (toArray copies 0 bytes, destroy releases the bucket).
    const empty = bufferPool.acquire(device, 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    return finalize(new GPUArray(empty, 0, dtype, device, bufferPool), keepOnGpu);
  }

  const outBuf = bufferPool.acquire(
    device,
    len * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  );

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(rsrc.buffer, start * 4, outBuf, 0, len * 4);
  device.queue.submit([encoder.finish()]);

  rsrc.release();

  return finalize(new GPUArray(outBuf, len, dtype, device, bufferPool), keepOnGpu);
}
