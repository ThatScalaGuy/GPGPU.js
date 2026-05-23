import type { NumericArray } from "../core/types";
import { DEFAULT_WORKGROUP_SIZE, toFloat32Array } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { dispatchAndRead, uploadBuffer } from "../core/command";
import { computeWorkgroupCount } from "../utils/workgroup";
import { bitonicSortShader } from "../codegen/templates";

function nextPowerOf2(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

export async function gpuSort(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: NumericArray
): Promise<Float32Array> {
  const device = await deviceManager.getDevice();
  const arr = toFloat32Array(input);
  const originalSize = arr.length;

  // Pad to next power of 2 with Infinity
  const paddedSize = nextPowerOf2(originalSize);
  const padded = new Float32Array(paddedSize);
  padded.set(arr);
  for (let i = originalSize; i < paddedSize; i++) {
    padded[i] = Infinity;
  }

  const byteSize = paddedSize * 4;

  const shader = bitonicSortShader();
  const pipeline = await shaderCache.getOrCreate(device, shader, "bitonic-sort");

  // Data buffer needs read_write storage + copy
  const bufData = bufferPool.acquire(
    device,
    byteSize,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  );
  device.queue.writeBuffer(bufData, 0, padded);

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

  // Read back result
  const staging = bufferPool.acquire(
    device,
    byteSize,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  );

  const copyEncoder = device.createCommandEncoder();
  copyEncoder.copyBufferToBuffer(bufData, 0, staging, 0, byteSize);
  device.queue.submit([copyEncoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();

  bufferPool.release(staging);
  bufferPool.release(bufData);

  return result.slice(0, originalSize);
}
