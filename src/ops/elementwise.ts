import type { NumericArray } from "../core/types";
import { inferDataType } from "../core/types";
import { toTypedArray } from "../utils/data-conversion";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { dispatchAndRead, uploadBuffer, createOutputBuffer } from "../core/command";
import { computeWorkgroupCount } from "../utils/workgroup";
import { parseExpression } from "../codegen/expression-parser";
import { emitWGSL } from "../codegen/wgsl-emitter";
import {
  mapShader,
  elementwiseBinaryShader,
  scalarBroadcastShader,
} from "../codegen/templates";

export async function gpuElementwiseBinary(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  a: NumericArray,
  b: NumericArray,
  op: string
): Promise<Float32Array | Int32Array | Uint32Array> {
  const device = await deviceManager.getDevice();
  const dtype = inferDataType(a);
  const arrA = toTypedArray(a, dtype);
  const arrB = toTypedArray(b, dtype);
  const size = arrA.length;
  const byteSize = size * 4;

  const shader = elementwiseBinaryShader(op, dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `elementwise-${op}-${dtype}`);

  const bufA = uploadBuffer(device, arrA, GPUBufferUsage.STORAGE, bufferPool);
  const bufB = uploadBuffer(device, arrB, GPUBufferUsage.STORAGE, bufferPool);
  const bufOut = createOutputBuffer(device, byteSize, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufA, size: byteSize } },
      { binding: 1, resource: { buffer: bufB, size: byteSize } },
      { binding: 2, resource: { buffer: bufOut, size: byteSize } },
    ],
  });

  const workgroupCount = computeWorkgroupCount(size);
  const result = await dispatchAndRead(
    device, pipeline, bindGroup,
    [workgroupCount], bufOut, byteSize, bufferPool, dtype
  );

  bufferPool.release(bufA);
  bufferPool.release(bufB);
  bufferPool.release(bufOut);

  return result.slice(0, size);
}

export async function gpuScalarBroadcast(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: NumericArray,
  scalar: number,
  op: string
): Promise<Float32Array | Int32Array | Uint32Array> {
  const device = await deviceManager.getDevice();
  const dtype = inferDataType(input);
  const arr = toTypedArray(input, dtype);
  const size = arr.length;
  const byteSize = size * 4;

  const shader = scalarBroadcastShader(op, dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `scalar-${op}-${dtype}`);

  const bufInput = uploadBuffer(device, arr, GPUBufferUsage.STORAGE, bufferPool);
  const bufOut = createOutputBuffer(device, byteSize, bufferPool);

  // Uniform buffer for the scalar param, in the same dtype as the data.
  const uniformData = toTypedArray([scalar], dtype);
  const bufUniform = uploadBuffer(device, uniformData, GPUBufferUsage.UNIFORM, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufInput, size: byteSize } },
      { binding: 1, resource: { buffer: bufOut, size: byteSize } },
      { binding: 2, resource: { buffer: bufUniform, size: bufUniform.size } },
    ],
  });

  const workgroupCount = computeWorkgroupCount(size);
  const result = await dispatchAndRead(
    device, pipeline, bindGroup,
    [workgroupCount], bufOut, byteSize, bufferPool, dtype
  );

  bufferPool.release(bufInput);
  bufferPool.release(bufOut);
  bufferPool.release(bufUniform);

  return result.slice(0, size);
}

export async function gpuMap(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: NumericArray,
  fn: ((x: number) => number) | string
): Promise<Float32Array | Int32Array | Uint32Array> {
  const device = await deviceManager.getDevice();
  const dtype = inferDataType(input);
  const arr = toTypedArray(input, dtype);
  const size = arr.length;
  const byteSize = size * 4;

  const ir = parseExpression(fn, ["x"]);
  const expression = emitWGSL(ir, dtype);
  const shader = mapShader(expression, dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `map-${dtype}`);

  const bufInput = uploadBuffer(device, arr, GPUBufferUsage.STORAGE, bufferPool);
  const bufOut = createOutputBuffer(device, byteSize, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufInput, size: byteSize } },
      { binding: 1, resource: { buffer: bufOut, size: byteSize } },
    ],
  });

  const workgroupCount = computeWorkgroupCount(size);
  const result = await dispatchAndRead(
    device, pipeline, bindGroup,
    [workgroupCount], bufOut, byteSize, bufferPool, dtype
  );

  bufferPool.release(bufInput);
  bufferPool.release(bufOut);

  return result.slice(0, size);
}
