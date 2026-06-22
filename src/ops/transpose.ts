import type { TransposeOpts, TypedArray } from "../core/types";
import { TRANSPOSE_TILE_SIZE } from "../core/types";
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
  isGPUArray,
  finalize,
} from "../core/io";
import { transposeShader } from "../codegen/templates";

export function gpuTranspose(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts: TransposeOpts & { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuTranspose(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts?: TransposeOpts & OpOptions
): Promise<TypedArray>;
export async function gpuTranspose(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts?: TransposeOpts & OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(input);

  // Resolve the input dims (which describe the row-major R×C input). Explicit { rows, cols }
  // wins; otherwise infer from a reshaped 2D GPUArray's shape.
  let rows: number;
  let cols: number;
  if (opts?.rows != null && opts?.cols != null) {
    rows = opts.rows;
    cols = opts.cols;
  } else if (isGPUArray(input) && input.shape?.length === 2) {
    [rows, cols] = input.shape;
  } else {
    throw new Error("transpose: provide { rows, cols } or pass a reshaped 2D GPUArray");
  }

  const rin = resolveInput(input, device, bufferPool, dtype);
  if (rows * cols !== rin.length) {
    throw new Error(`transpose: rows*cols = ${rows * cols} but array has ${rin.length}`);
  }

  const outputSize = rows * cols;
  const outputByteSize = outputSize * 4;

  const shader = transposeShader(dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `transpose-${dtype}`);

  const bufOut = createOutputBuffer(device, outputByteSize, bufferPool);

  // Dims uniform: 16-byte buffer (2 x u32 padded to 16).
  const dims = new Uint32Array([rows, cols, 0, 0]);
  const bufDims = uploadBuffer(device, dims, GPUBufferUsage.UNIFORM, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: rin.buffer, size: rin.length * 4 } },
      { binding: 1, resource: { buffer: bufOut, size: outputByteSize } },
      { binding: 2, resource: { buffer: bufDims, size: 16 } },
    ],
  });

  // wid.x over columns, wid.y over rows.
  const workgroupsX = Math.ceil(cols / TRANSPOSE_TILE_SIZE);
  const workgroupsY = Math.ceil(rows / TRANSPOSE_TILE_SIZE);

  dispatchOnly(device, pipeline, bindGroup, [workgroupsX, workgroupsY]);

  rin.release();
  bufferPool.release(bufDims);

  // The result is logically cols × rows.
  return finalize(
    new GPUArray(bufOut, outputSize, dtype, device, bufferPool, { shape: [cols, rows] }),
    opts?.keepOnGpu ?? false
  );
}
