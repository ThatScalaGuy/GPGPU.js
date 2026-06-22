import type { TypedArray, SearchSortedOpts } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { createOutputBuffer, dispatchOnly } from "../core/command";
import { GPUArray } from "../pipeline/gpu-array";
import {
  type OpInput,
  resolveInput,
  inputDtype,
  finalize,
} from "../core/io";
import { computeWorkgroupCount } from "../utils/workgroup";
import { searchsortedShader } from "../codegen/templates";

export function gpuSearchsorted(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  sorted: OpInput,
  queries: OpInput,
  opts: SearchSortedOpts & { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuSearchsorted(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  sorted: OpInput,
  queries: OpInput,
  opts?: SearchSortedOpts & { keepOnGpu?: boolean }
): Promise<TypedArray>;
export async function gpuSearchsorted(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  sorted: OpInput,
  queries: OpInput,
  opts?: SearchSortedOpts & { keepOnGpu?: boolean }
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  // sorted and queries share the input dtype; the output is always u32 insertion indices.
  const dtype = inputDtype(sorted);
  const side = opts?.side ?? "left";

  const rsorted = resolveInput(sorted, device, bufferPool, dtype);
  const rqueries = resolveInput(queries, device, bufferPool, dtype);
  const size = rqueries.length;
  const byteSize = size * 4;

  const shader = searchsortedShader(dtype, side);
  const pipeline = await shaderCache.getOrCreate(device, shader, `searchsorted-${side}-${dtype}`);

  const bufOut = createOutputBuffer(device, byteSize, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: rsorted.buffer, size: rsorted.length * 4 } },
      { binding: 1, resource: { buffer: rqueries.buffer, size: byteSize } },
      { binding: 2, resource: { buffer: bufOut, size: byteSize } },
    ],
  });

  dispatchOnly(device, pipeline, bindGroup, [computeWorkgroupCount(size)]);

  rsorted.release();
  rqueries.release();

  return finalize(new GPUArray(bufOut, size, "u32", device, bufferPool), opts?.keepOnGpu ?? false);
}
