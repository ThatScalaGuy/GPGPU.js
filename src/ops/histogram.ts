import type { TypedArray } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { GPUArray } from "../pipeline/gpu-array";
import { type OpInput, type OpOptions, resolveInput, inputDtype, finalize } from "../core/io";
import type { HistogramOpts } from "../core/types";
import { uploadBuffer, dispatchOnly } from "../core/command";
import { computeWorkgroupGrid } from "../utils/workgroup";
import { histogramShader } from "../codegen/templates";

export function gpuHistogram(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts: HistogramOpts & { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuHistogram(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts: HistogramOpts & OpOptions
): Promise<TypedArray>;
export async function gpuHistogram(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts: HistogramOpts & OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(input);
  const { bins, min, max } = opts;

  const rin = resolveInput(input, device, bufferPool, dtype);
  const histBytes = bins * 4;

  // Zeroed atomic counter buffer.
  const hist = bufferPool.acquire(
    device,
    histBytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  );
  device.queue.writeBuffer(hist, 0, new Uint32Array(bins));

  // Mixed-type uniform: { bins: u32, minVal: f32, maxVal: f32 } in a 16-byte buffer.
  const paramsAB = new ArrayBuffer(16);
  new Uint32Array(paramsAB, 0, 1)[0] = bins;
  new Float32Array(paramsAB, 4, 2).set([min, max]);
  const bufParams = uploadBuffer(device, new Uint8Array(paramsAB), GPUBufferUsage.UNIFORM, bufferPool);

  const pipeline = await shaderCache.getOrCreate(device, histogramShader(dtype), `histogram-${dtype}`);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: rin.buffer, size: rin.length * 4 } },
      { binding: 1, resource: { buffer: hist, size: histBytes } },
      { binding: 2, resource: { buffer: bufParams, size: 16 } },
    ],
  });

  dispatchOnly(device, pipeline, bindGroup, computeWorkgroupGrid(rin.length));

  rin.release();
  bufferPool.release(bufParams);

  // Counts are u32 regardless of input dtype.
  return finalize(new GPUArray(hist, bins, "u32", device, bufferPool), opts.keepOnGpu ?? false);
}
