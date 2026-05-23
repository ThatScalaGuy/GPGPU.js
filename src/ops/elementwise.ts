import type { NumericArray } from "../core/types";
import { DEFAULT_WORKGROUP_SIZE, toFloat32Array } from "../core/types";
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
): Promise<Float32Array> {
  const device = await deviceManager.getDevice();
  const arrA = toFloat32Array(a);
  const arrB = toFloat32Array(b);
  const size = arrA.length;
  const byteSize = size * 4;

  const shader = elementwiseBinaryShader(op);
  const pipeline = await shaderCache.getOrCreate(device, shader, `elementwise-${op}`);

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
    [workgroupCount], bufOut, byteSize, bufferPool
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
): Promise<Float32Array> {
  const device = await deviceManager.getDevice();
  const arr = toFloat32Array(input);
  const size = arr.length;
  const byteSize = size * 4;

  const shader = scalarBroadcastShader(op);
  const pipeline = await shaderCache.getOrCreate(device, shader, `scalar-${op}`);

  const bufInput = uploadBuffer(device, arr, GPUBufferUsage.STORAGE, bufferPool);
  const bufOut = createOutputBuffer(device, byteSize, bufferPool);

  // Uniform buffer for scalar param (16 bytes min for alignment)
  const uniformData = new Float32Array([scalar]);
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
    [workgroupCount], bufOut, byteSize, bufferPool
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
): Promise<Float32Array> {
  const device = await deviceManager.getDevice();
  const arr = toFloat32Array(input);
  const size = arr.length;
  const byteSize = size * 4;

  const ir = parseExpression(fn, ["x"]);
  const expression = emitWGSL(ir);
  const shader = mapShader(expression);
  const pipeline = await shaderCache.getOrCreate(device, shader, "map");

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
    [workgroupCount], bufOut, byteSize, bufferPool
  );

  bufferPool.release(bufInput);
  bufferPool.release(bufOut);

  return result.slice(0, size);
}
