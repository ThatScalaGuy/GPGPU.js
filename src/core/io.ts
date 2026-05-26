import type { DataType, NumericArray, TypedArray } from "./types";
import { inferDataType } from "./types";
import { toTypedArray } from "../utils/data-conversion";
import { BufferPool } from "./buffer-pool";
import { uploadBuffer } from "./command";
import { GPUArray } from "../pipeline/gpu-array";

/** Anything an op can take: a CPU array or a GPU-resident handle. */
export type OpInput = NumericArray | GPUArray;

/** Per-op flag controlling whether the result stays on the GPU. */
export interface OpOptions {
  keepOnGpu?: boolean;
}

export function isGPUArray(x: unknown): x is GPUArray {
  return x instanceof GPUArray;
}

/** Data type of an input — from the GPUArray itself, or inferred from the CPU array. */
export function inputDtype(input: OpInput): DataType {
  return input instanceof GPUArray ? input.dtype : inferDataType(input);
}

export interface ResolvedInput {
  buffer: GPUBuffer;
  length: number;
  /** True when the buffer is the caller's GPUArray (reused in place, not owned by the op). */
  isGpu: boolean;
  /** Release any buffer the op allocated for this input (no-op for a GPUArray input). */
  release(): void;
}

/**
 * Turn an op input into a GPU buffer. A GPUArray is reused in place (no upload, no
 * release); a CPU array is uploaded to a pooled buffer that `release()` returns.
 */
export function resolveInput(
  input: OpInput,
  device: GPUDevice,
  pool: BufferPool,
  dtype: DataType,
  usage: number = GPUBufferUsage.STORAGE
): ResolvedInput {
  if (input instanceof GPUArray) {
    if (input.isDestroyed) throw new Error("GPUArray has been destroyed");
    return { buffer: input.buffer, length: input.length, isGpu: true, release() {} };
  }
  const arr = toTypedArray(input, dtype);
  const buffer = uploadBuffer(device, arr, usage, pool);
  return { buffer, length: arr.length, isGpu: false, release: () => pool.release(buffer) };
}

/**
 * Final step of an array-returning op: either hand back the GPU-resident handle
 * (`keepOnGpu`) or read it to the CPU and free the buffer.
 */
export async function finalize(
  out: GPUArray,
  keepOnGpu: boolean
): Promise<TypedArray | GPUArray> {
  if (keepOnGpu) return out;
  const arr = await out.toArray();
  out.destroy();
  return arr;
}
