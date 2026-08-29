import type { DataType, TypedArray } from "../core/types";
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
import { computeWorkgroupGrid } from "../utils/workgroup";
import { castShader } from "../codegen/templates";

export function gpuCast(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  toDtype: DataType,
  opts: { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuCast(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  toDtype: DataType,
  opts?: OpOptions
): Promise<TypedArray>;
export async function gpuCast(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  toDtype: DataType,
  opts?: OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const fromDtype = inputDtype(input);

  const rin = resolveInput(input, device, bufferPool, fromDtype);
  const length = rin.length;
  const byteSize = length * 4;

  const shader = castShader(fromDtype, toDtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `cast-${fromDtype}-${toDtype}`);

  const bufOut = createOutputBuffer(device, byteSize, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: rin.buffer, size: byteSize } },
      { binding: 1, resource: { buffer: bufOut, size: byteSize } },
    ],
  });

  dispatchOnly(device, pipeline, bindGroup, computeWorkgroupGrid(length));

  rin.release();

  return finalize(new GPUArray(bufOut, length, toDtype, device, bufferPool), opts?.keepOnGpu ?? false);
}
