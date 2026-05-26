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
import { bitonicSortShader } from "../codegen/templates";

function nextPowerOf2(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

// Sentinel for padding lanes in an ascending sort: the largest value of the type so
// padding always sorts to the end and is trimmed off on readback.
const SORT_PAD: Record<DataType, number> = {
  f32: Infinity,
  i32: 2147483647,
  u32: 4294967295,
};

export function gpuSort(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts: { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuSort(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts?: OpOptions
): Promise<TypedArray>;
export async function gpuSort(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts?: OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(input);
  const keepOnGpu = opts?.keepOnGpu ?? false;

  const originalSize = isGPUArray(input) ? input.length : toTypedArray(input, dtype).length;
  const paddedSize = nextPowerOf2(originalSize);
  const byteSize = paddedSize * 4;

  const shader = bitonicSortShader(dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `bitonic-sort-${dtype}`);

  // Data buffer needs read_write storage + copy (sorted in place).
  const bufData = bufferPool.acquire(
    device,
    byteSize,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  );

  if (isGPUArray(input)) {
    if (input.isDestroyed) throw new Error("GPUArray has been destroyed");
    // Copy the input (don't mutate it), then fill the pad lanes with the sentinel.
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(input.buffer, 0, bufData, 0, originalSize * 4);
    device.queue.submit([copyEncoder.finish()]);
    if (paddedSize > originalSize) {
      const pad = viewFor(dtype, new ArrayBuffer((paddedSize - originalSize) * 4));
      pad.fill(SORT_PAD[dtype]);
      device.queue.writeBuffer(bufData, originalSize * 4, pad.buffer as ArrayBuffer, pad.byteOffset, pad.byteLength);
    }
  } else {
    // Pad to next power of 2 with the type's max value, then upload in one shot.
    const arr = toTypedArray(input, dtype);
    const padded = viewFor(dtype, new ArrayBuffer(byteSize));
    padded.set(arr);
    for (let i = originalSize; i < paddedSize; i++) {
      padded[i] = SORT_PAD[dtype];
    }
    device.queue.writeBuffer(bufData, 0, padded.buffer as ArrayBuffer, padded.byteOffset, padded.byteLength);
  }

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
          { binding: 0, resource: { buffer: bufData, size: byteSize } },
          { binding: 1, resource: { buffer: bufParams, size: params.byteLength } },
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
  // by length-based binding when this GPUArray is fed into another op.
  return finalize(new GPUArray(bufData, originalSize, dtype, device, bufferPool), keepOnGpu);
}
