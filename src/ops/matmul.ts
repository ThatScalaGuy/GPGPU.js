import type { MatMulOpts, TypedArray } from "../core/types";
import { MATMUL_TILE_SIZE } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { createOutputBuffer, dispatchOnly, uploadBuffer } from "../core/command";
import { GPUArray } from "../pipeline/gpu-array";
import {
  type OpInput,
  type OpOptions,
  resolveInput,
  inputDtype,
  finalize,
} from "../core/io";
import { matmulShader } from "../codegen/templates";

export function gpuMatmul(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  a: OpInput,
  b: OpInput,
  opts: MatMulOpts & { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuMatmul(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  a: OpInput,
  b: OpInput,
  opts: MatMulOpts & OpOptions
): Promise<TypedArray>;
export async function gpuMatmul(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  a: OpInput,
  b: OpInput,
  opts: MatMulOpts & OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const { rowsA, colsA, colsB } = opts;
  const dtype = inputDtype(a);

  const ra = resolveInput(a, device, bufferPool, dtype);
  const rb = resolveInput(b, device, bufferPool, dtype);
  const outputSize = rowsA * colsB;
  const outputByteSize = outputSize * 4;

  const shader = matmulShader(dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `matmul-${dtype}`);

  const bufOut = createOutputBuffer(device, outputByteSize, bufferPool);

  // Uniform buffer for dimensions (3 x u32 = 12 bytes, padded to 16)
  const dims = new Uint32Array([rowsA, colsA, colsB, 0]);
  const bufDims = uploadBuffer(device, dims, GPUBufferUsage.UNIFORM, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ra.buffer, size: ra.length * 4 } },
      { binding: 1, resource: { buffer: rb.buffer, size: rb.length * 4 } },
      { binding: 2, resource: { buffer: bufOut, size: outputByteSize } },
      { binding: 3, resource: { buffer: bufDims, size: dims.byteLength } },
    ],
  });

  const workgroupsX = Math.ceil(colsB / MATMUL_TILE_SIZE);
  const workgroupsY = Math.ceil(rowsA / MATMUL_TILE_SIZE);

  dispatchOnly(device, pipeline, bindGroup, [workgroupsX, workgroupsY]);

  ra.release();
  rb.release();
  bufferPool.release(bufDims);

  return finalize(
    new GPUArray(bufOut, outputSize, dtype, device, bufferPool),
    opts.keepOnGpu ?? false
  );
}
