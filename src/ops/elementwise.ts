import type { TypedArray } from "../core/types";
import { toTypedArray } from "../utils/data-conversion";
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
import { computeWorkgroupCount } from "../utils/workgroup";
import { parseExpression } from "../codegen/expression-parser";
import { emitWGSL } from "../codegen/wgsl-emitter";
import {
  mapShader,
  elementwiseBinaryShader,
  scalarBroadcastShader,
} from "../codegen/templates";

export function gpuElementwiseBinary(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  a: OpInput,
  b: OpInput,
  op: string,
  opts: { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuElementwiseBinary(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  a: OpInput,
  b: OpInput,
  op: string,
  opts?: OpOptions
): Promise<TypedArray>;
export async function gpuElementwiseBinary(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  a: OpInput,
  b: OpInput,
  op: string,
  opts?: OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(a);

  if (isGPUArray(a) && isGPUArray(b)) {
    if (a.dtype !== b.dtype) throw new Error("Elementwise inputs must share a data type");
    if (a.length !== b.length) throw new Error("Elementwise inputs must have the same length");
  }

  const ra = resolveInput(a, device, bufferPool, dtype);
  const rb = resolveInput(b, device, bufferPool, dtype);
  const size = ra.length;
  const byteSize = size * 4;

  const shader = elementwiseBinaryShader(op, dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `elementwise-${op}-${dtype}`);

  const bufOut = createOutputBuffer(device, byteSize, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ra.buffer, size: byteSize } },
      { binding: 1, resource: { buffer: rb.buffer, size: byteSize } },
      { binding: 2, resource: { buffer: bufOut, size: byteSize } },
    ],
  });

  dispatchOnly(device, pipeline, bindGroup, [computeWorkgroupCount(size)]);

  ra.release();
  rb.release();

  return finalize(new GPUArray(bufOut, size, dtype, device, bufferPool), opts?.keepOnGpu ?? false);
}

export function gpuScalarBroadcast(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  scalar: number,
  op: string,
  opts: { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuScalarBroadcast(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  scalar: number,
  op: string,
  opts?: OpOptions
): Promise<TypedArray>;
export async function gpuScalarBroadcast(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  scalar: number,
  op: string,
  opts?: OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(input);

  const rin = resolveInput(input, device, bufferPool, dtype);
  const size = rin.length;
  const byteSize = size * 4;

  const shader = scalarBroadcastShader(op, dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `scalar-${op}-${dtype}`);

  const bufOut = createOutputBuffer(device, byteSize, bufferPool);

  // Uniform buffer for the scalar param, in the same dtype as the data.
  const uniformData = toTypedArray([scalar], dtype);
  const bufUniform = uploadBuffer(device, uniformData, GPUBufferUsage.UNIFORM, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: rin.buffer, size: byteSize } },
      { binding: 1, resource: { buffer: bufOut, size: byteSize } },
      { binding: 2, resource: { buffer: bufUniform, size: bufUniform.size } },
    ],
  });

  dispatchOnly(device, pipeline, bindGroup, [computeWorkgroupCount(size)]);

  rin.release();
  bufferPool.release(bufUniform);

  return finalize(new GPUArray(bufOut, size, dtype, device, bufferPool), opts?.keepOnGpu ?? false);
}

export function gpuMap(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  fn: ((x: number) => number) | string,
  opts: { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuMap(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  fn: ((x: number) => number) | string,
  opts?: OpOptions
): Promise<TypedArray>;
export async function gpuMap(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  fn: ((x: number) => number) | string,
  opts?: OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(input);

  const rin = resolveInput(input, device, bufferPool, dtype);
  const size = rin.length;
  const byteSize = size * 4;

  const ir = parseExpression(fn, ["x"]);
  const expression = emitWGSL(ir, dtype);
  const shader = mapShader(expression, dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `map-${dtype}`);

  const bufOut = createOutputBuffer(device, byteSize, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: rin.buffer, size: byteSize } },
      { binding: 1, resource: { buffer: bufOut, size: byteSize } },
    ],
  });

  dispatchOnly(device, pipeline, bindGroup, [computeWorkgroupCount(size)]);

  rin.release();

  return finalize(new GPUArray(bufOut, size, dtype, device, bufferPool), opts?.keepOnGpu ?? false);
}
