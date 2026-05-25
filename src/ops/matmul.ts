import type { NumericArray, MatMulOpts } from "../core/types";
import { MATMUL_TILE_SIZE, inferDataType } from "../core/types";
import { toTypedArray } from "../utils/data-conversion";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { dispatchAndRead, uploadBuffer, createOutputBuffer } from "../core/command";
import { matmulShader } from "../codegen/templates";

export async function gpuMatmul(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  a: NumericArray,
  b: NumericArray,
  opts: MatMulOpts
): Promise<Float32Array | Int32Array | Uint32Array> {
  const device = await deviceManager.getDevice();
  const { rowsA, colsA, colsB } = opts;
  const dtype = inferDataType(a);

  const arrA = toTypedArray(a, dtype);
  const arrB = toTypedArray(b, dtype);
  const outputSize = rowsA * colsB;
  const outputByteSize = outputSize * 4;

  const shader = matmulShader(dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `matmul-${dtype}`);

  const bufA = uploadBuffer(device, arrA, GPUBufferUsage.STORAGE, bufferPool);
  const bufB = uploadBuffer(device, arrB, GPUBufferUsage.STORAGE, bufferPool);
  const bufOut = createOutputBuffer(device, outputByteSize, bufferPool);

  // Uniform buffer for dimensions (3 x u32 = 12 bytes, padded to 16)
  const dims = new Uint32Array([rowsA, colsA, colsB, 0]);
  const bufDims = uploadBuffer(device, dims, GPUBufferUsage.UNIFORM, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufA, size: arrA.byteLength } },
      { binding: 1, resource: { buffer: bufB, size: arrB.byteLength } },
      { binding: 2, resource: { buffer: bufOut, size: outputByteSize } },
      { binding: 3, resource: { buffer: bufDims, size: dims.byteLength } },
    ],
  });

  const workgroupsX = Math.ceil(colsB / MATMUL_TILE_SIZE);
  const workgroupsY = Math.ceil(rowsA / MATMUL_TILE_SIZE);

  const result = await dispatchAndRead(
    device, pipeline, bindGroup,
    [workgroupsX, workgroupsY], bufOut, outputByteSize, bufferPool, dtype
  );

  bufferPool.release(bufA);
  bufferPool.release(bufB);
  bufferPool.release(bufOut);
  bufferPool.release(bufDims);

  return result.slice(0, outputSize);
}
