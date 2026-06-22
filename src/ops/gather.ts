import type { TypedArray } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { createOutputBuffer, dispatchOnly } from "../core/command";
import { GPUArray } from "../pipeline/gpu-array";
import {
  type OpInput,
  type OpOptions,
  resolveInput,
  inputDtype,
  finalize,
} from "../core/io";
import { computeWorkgroupCount } from "../utils/workgroup";
import { gatherShader } from "../codegen/templates";

export function gpuGather(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  src: OpInput,
  idx: OpInput,
  opts: { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuGather(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  src: OpInput,
  idx: OpInput,
  opts?: OpOptions
): Promise<TypedArray>;
export async function gpuGather(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  src: OpInput,
  idx: OpInput,
  opts?: OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  // Output mirrors src's dtype; idx is coerced to u32 (resolveInput's dtype param uploads
  // a plain array as u32 and validates a GPUArray index already lives in u32).
  const dtype = inputDtype(src);

  const rsrc = resolveInput(src, device, bufferPool, dtype);
  const ridx = resolveInput(idx, device, bufferPool, "u32");
  const size = ridx.length;
  const byteSize = size * 4;

  const shader = gatherShader(dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `gather-${dtype}`);

  const bufOut = createOutputBuffer(device, byteSize, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: rsrc.buffer, size: rsrc.length * 4 } },
      { binding: 1, resource: { buffer: ridx.buffer, size: byteSize } },
      { binding: 2, resource: { buffer: bufOut, size: byteSize } },
    ],
  });

  dispatchOnly(device, pipeline, bindGroup, [computeWorkgroupCount(size)]);

  rsrc.release();
  ridx.release();

  return finalize(new GPUArray(bufOut, size, dtype, device, bufferPool), opts?.keepOnGpu ?? false);
}
