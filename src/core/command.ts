import type { DataType } from "./types";
import { BufferPool } from "./buffer-pool";

/** Wrap a copied ArrayBuffer in the typed-array view that matches `dtype`. */
export function viewFor(
  dtype: DataType,
  buffer: ArrayBuffer
): Float32Array | Int32Array | Uint32Array {
  if (dtype === "i32") return new Int32Array(buffer);
  if (dtype === "u32") return new Uint32Array(buffer);
  return new Float32Array(buffer);
}

export async function dispatchAndRead(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroupCount: [number, number?, number?],
  outputBuffer: GPUBuffer,
  outputSize: number,
  bufferPool: BufferPool,
  dtype: DataType = "f32"
): Promise<Float32Array | Int32Array | Uint32Array> {
  const staging = bufferPool.acquire(
    device,
    outputSize,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  );

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(...workgroupCount);
  pass.end();

  encoder.copyBufferToBuffer(outputBuffer, 0, staging, 0, outputSize);
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const result = viewFor(dtype, staging.getMappedRange().slice(0));
  staging.unmap();
  bufferPool.release(staging);

  return result;
}

export function dispatchOnly(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroupCount: [number, number?, number?]
): void {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(...workgroupCount);
  pass.end();
  device.queue.submit([encoder.finish()]);
}

export function uploadBuffer(
  device: GPUDevice,
  data: ArrayBufferView,
  usage: number,
  bufferPool: BufferPool
): GPUBuffer {
  const buffer = bufferPool.acquire(
    device,
    data.byteLength,
    usage | GPUBufferUsage.COPY_DST
  );
  device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  return buffer;
}

export function createOutputBuffer(
  device: GPUDevice,
  size: number,
  bufferPool: BufferPool
): GPUBuffer {
  return bufferPool.acquire(
    device,
    size,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
}
