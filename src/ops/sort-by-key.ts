import type { DataType, TypedArray } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { uploadBuffer, viewFor } from "../core/command";
import { GPUArray } from "../pipeline/gpu-array";
import {
  type OpInput,
  type OpOptions,
  inputDtype,
  isGPUArray,
  finalize,
} from "../core/io";
import { toTypedArray } from "../utils/data-conversion";
import { computeWorkgroupCount } from "../utils/workgroup";
import { bitonicSortByKeyShader } from "../codegen/templates";

function nextPowerOf2(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

// Sentinel for padding key lanes in an ascending sort: the largest value of the type so
// padding always sorts to the end and is trimmed off on readback.
const SORT_PAD: Record<DataType, number> = {
  f32: Infinity,
  i32: 2147483647,
  u32: 4294967295,
};

// Upload one data array into `bufData` padded to `paddedSize`. Keys pad with the type-max
// sentinel; values pad with 0 (padded values ride along with sentinel keys to the tail and
// are trimmed). Handles a GPUArray input (copy the real region, then fill the pad tail) and a
// CPU input (pad + single writeBuffer) — mirrors gpuSort for each buffer.
function uploadPadded(
  device: GPUDevice,
  bufData: GPUBuffer,
  input: OpInput,
  dtype: DataType,
  originalSize: number,
  paddedSize: number,
  pad: number
): void {
  const byteSize = paddedSize * 4;
  if (isGPUArray(input)) {
    if (input.isDestroyed) throw new Error("GPUArray has been destroyed");
    // Copy the input (don't mutate it), then fill the pad lanes with the sentinel.
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(input.buffer, 0, bufData, 0, originalSize * 4);
    device.queue.submit([copyEncoder.finish()]);
    if (paddedSize > originalSize) {
      const tail = viewFor(dtype, new ArrayBuffer((paddedSize - originalSize) * 4));
      tail.fill(pad);
      device.queue.writeBuffer(bufData, originalSize * 4, tail.buffer as ArrayBuffer, tail.byteOffset, tail.byteLength);
    }
  } else {
    // Pad to next power of 2, then upload in one shot.
    const arr = toTypedArray(input, dtype);
    const padded = viewFor(dtype, new ArrayBuffer(byteSize));
    padded.set(arr);
    for (let i = originalSize; i < paddedSize; i++) {
      padded[i] = pad;
    }
    device.queue.writeBuffer(bufData, 0, padded.buffer as ArrayBuffer, padded.byteOffset, padded.byteLength);
  }
}

export function gpuSortByKey(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  keys: OpInput,
  values: OpInput,
  opts: { keepOnGpu: true }
): Promise<[GPUArray, GPUArray]>;
export function gpuSortByKey(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  keys: OpInput,
  values: OpInput,
  opts?: OpOptions
): Promise<[TypedArray, TypedArray]>;
export async function gpuSortByKey(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  keys: OpInput,
  values: OpInput,
  opts?: OpOptions
): Promise<[TypedArray, TypedArray] | [GPUArray, GPUArray]> {
  const device = await deviceManager.getDevice();
  const keyDtype = inputDtype(keys);
  const valDtype = inputDtype(values);
  const keepOnGpu = opts?.keepOnGpu ?? false;

  const originalSize = isGPUArray(keys) ? keys.length : toTypedArray(keys, keyDtype).length;
  const paddedSize = nextPowerOf2(originalSize);
  const byteSize = paddedSize * 4;

  const shader = bitonicSortByKeyShader(keyDtype, valDtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `bitonic-sort-by-key-${keyDtype}-${valDtype}`);

  // Both buffers need read_write storage + copy (sorted/permuted in place).
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const keysBuf = bufferPool.acquire(device, byteSize, usage);
  const valsBuf = bufferPool.acquire(device, byteSize, usage);

  // Keys pad with the type-max sentinel so padding sorts to the tail; values pad with 0.
  uploadPadded(device, keysBuf, keys, keyDtype, originalSize, paddedSize, SORT_PAD[keyDtype]);
  uploadPadded(device, valsBuf, values, valDtype, originalSize, paddedSize, 0);

  const numPairs = paddedSize / 2;
  const workgroupCount = computeWorkgroupCount(numPairs);

  // Bitonic sort stages
  for (let blockSize = 2; blockSize <= paddedSize; blockSize *= 2) {
    for (let subBlockSize = blockSize / 2; subBlockSize >= 1; subBlockSize /= 2) {
      const params = new Uint32Array([blockSize, subBlockSize, paddedSize, 0]);
      const bufParams = uploadBuffer(device, params, GPUBufferUsage.UNIFORM, bufferPool);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: keysBuf, size: byteSize } },
          { binding: 1, resource: { buffer: valsBuf, size: byteSize } },
          { binding: 2, resource: { buffer: bufParams, size: params.byteLength } },
        ],
      });

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroupCount);
      pass.end();
      device.queue.submit([encoder.finish()]);

      bufferPool.release(bufParams);
    }
  }

  // Logical length is the original size; the pad tail is hidden by toArray's slice and
  // by length-based binding when these GPUArrays are fed into another op.
  const outKeys = await finalize(new GPUArray(keysBuf, originalSize, keyDtype, device, bufferPool), keepOnGpu);
  const outVals = await finalize(new GPUArray(valsBuf, originalSize, valDtype, device, bufferPool), keepOnGpu);
  return [outKeys, outVals] as [TypedArray, TypedArray] | [GPUArray, GPUArray];
}
