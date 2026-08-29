import type { TypedArray } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { GPUArray } from "../pipeline/gpu-array";
import { type OpInput, resolveInput, inputDtype, finalize } from "../core/io";
import type { ScatterOpts } from "../core/types";
import type { OpOptions } from "../core/io";
import { computeWorkgroupGrid } from "../utils/workgroup";
import { dispatchOnly } from "../core/command";
import { scatterSetShader, scatterAddShader } from "../codegen/templates";

export function gpuScatter(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  dst: OpInput,
  idx: OpInput,
  vals: OpInput,
  opts: ScatterOpts & { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuScatter(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  dst: OpInput,
  idx: OpInput,
  vals: OpInput,
  opts?: ScatterOpts & OpOptions
): Promise<TypedArray>;
export async function gpuScatter(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  dst: OpInput,
  idx: OpInput,
  vals: OpInput,
  opts?: ScatterOpts & OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(dst);
  const mode = opts?.mode ?? "set";

  // dst must be readable as a copy source. A GPUArray's buffer already has COPY_SRC;
  // a CPU array is uploaded with the usage we pass here.
  const rdst = resolveInput(dst, device, bufferPool, dtype, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const ridx = resolveInput(idx, device, bufferPool, "u32");
  const rvals = resolveInput(vals, device, bufferPool, dtype);

  const n = rdst.length;          // output length
  const m = ridx.length;          // number of scatters
  const byteSize = n * 4;

  // out starts as a copy of dst, then is modified in place.
  const out = bufferPool.acquire(
    device,
    byteSize,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  );
  const copyEnc = device.createCommandEncoder();
  copyEnc.copyBufferToBuffer(rdst.buffer, 0, out, 0, byteSize);
  device.queue.submit([copyEnc.finish()]);

  const shader = mode === "add" ? scatterAddShader(dtype) : scatterSetShader(dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `scatter-${mode}-${dtype}`);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: out, size: byteSize } },
      { binding: 1, resource: { buffer: ridx.buffer, size: m * 4 } },
      { binding: 2, resource: { buffer: rvals.buffer, size: m * 4 } },
    ],
  });

  dispatchOnly(device, pipeline, bindGroup, computeWorkgroupGrid(m));

  rdst.release();
  ridx.release();
  rvals.release();

  return finalize(new GPUArray(out, n, dtype, device, bufferPool), opts?.keepOnGpu ?? false);
}
