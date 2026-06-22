import type { DataType, TypedArray } from "../core/types";
import { toTypedArray } from "../utils/data-conversion";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { createOutputBuffer, dispatchOnly, uploadBuffer } from "../core/command";
import { GPUArray } from "../pipeline/gpu-array";
import {
  type OpInput,
  type OpOptions,
  type MapOptions,
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
  zipShader,
  elementwiseBinaryShader,
  scalarBroadcastShader,
  broadcastBinaryShader,
} from "../codegen/templates";
import {
  MAX_BROADCAST_RANK,
  broadcastShapes,
  broadcastStrides,
  paddedOutShape,
} from "../codegen/broadcast";

// The logical shape of an operand: a reshaped GPUArray carries one; anything else is read as a
// flat 1-D vector of its length (matching how NumPy treats a bare vector).
function shapeOf(input: OpInput, length: number): readonly number[] {
  return isGPUArray(input) && input.shape ? input.shape : [length];
}

// Run a broadcasting elementwise dispatch: one thread per OUTPUT element, reading a/b through
// per-operand broadcast strides. `combine` is the body of broadcastBinaryShader ("$a OP $b" or
// an emitWGSL expression over a/b). Shared by gpuElementwiseBinary and gpuZip.
async function dispatchBroadcast(
  device: GPUDevice,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  aBuf: GPUBuffer,
  bBuf: GPUBuffer,
  aLen: number,
  bLen: number,
  shapeA: readonly number[],
  shapeB: readonly number[],
  dtype: DataType,
  combine: string,
  cacheKey: string,
  source?: string
): Promise<{ buffer: GPUBuffer; length: number; outShape: number[] }> {
  const outShape = broadcastShapes(shapeA, shapeB);
  const total = outShape.reduce((a, b) => a * b, 1);

  const shader = broadcastBinaryShader(combine, dtype, MAX_BROADCAST_RANK);
  const pipeline = await shaderCache.getOrCreate(device, shader, cacheKey, source);

  const outBuf = createOutputBuffer(device, total * 4, bufferPool);

  // Params uniform (64 bytes): outShape[0..3], strideA[4..7], strideB[8..11], total[12].
  const padShape = paddedOutShape(outShape);
  const strideA = broadcastStrides(shapeA, outShape);
  const strideB = broadcastStrides(shapeB, outShape);
  const params = new Uint32Array(16);
  params.set(padShape, 0);
  params.set(strideA, 4);
  params.set(strideB, 8);
  params[12] = total;
  const bufParams = uploadBuffer(device, params, GPUBufferUsage.UNIFORM, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuf, size: aLen * 4 } },
      { binding: 1, resource: { buffer: bBuf, size: bLen * 4 } },
      { binding: 2, resource: { buffer: outBuf, size: total * 4 } },
      { binding: 3, resource: { buffer: bufParams, size: bufParams.size } },
    ],
  });

  dispatchOnly(device, pipeline, bindGroup, [computeWorkgroupCount(total)]);
  bufferPool.release(bufParams);

  return { buffer: outBuf, length: total, outShape };
}

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

  if (isGPUArray(a) && isGPUArray(b) && a.dtype !== b.dtype) {
    throw new Error("Elementwise inputs must share a data type");
  }

  const ra = resolveInput(a, device, bufferPool, dtype);
  const rb = resolveInput(b, device, bufferPool, dtype);

  // Different lengths → NumPy broadcasting (throws if the shapes aren't compatible). Equal
  // lengths take the untouched one-thread-per-element fast path below.
  if (ra.length !== rb.length) {
    const out = await dispatchBroadcast(
      device, bufferPool, shaderCache,
      ra.buffer, rb.buffer, ra.length, rb.length,
      shapeOf(a, ra.length), shapeOf(b, rb.length),
      dtype, `a ${op} b`, `broadcast-${op}-${dtype}`
    );
    ra.release();
    rb.release();
    return finalize(
      new GPUArray(out.buffer, out.length, dtype, device, bufferPool, { shape: out.outShape }),
      opts?.keepOnGpu ?? false
    );
  }

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

export function gpuZip(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  a: OpInput,
  b: OpInput,
  fn: ((a: number, b: number) => number) | string,
  opts: { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuZip(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  a: OpInput,
  b: OpInput,
  fn: ((a: number, b: number) => number) | string,
  opts?: OpOptions
): Promise<TypedArray>;
export async function gpuZip(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  a: OpInput,
  b: OpInput,
  fn: ((a: number, b: number) => number) | string,
  opts?: OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(a);

  if (isGPUArray(a) && isGPUArray(b) && a.dtype !== b.dtype) {
    throw new Error("Elementwise inputs must share a data type");
  }

  const ra = resolveInput(a, device, bufferPool, dtype);
  const rb = resolveInput(b, device, bufferPool, dtype);

  const ir = parseExpression(fn, ["a", "b"]);
  const expression = emitWGSL(ir, dtype);
  const source = typeof fn === "string" ? fn : fn.toString();

  // Different lengths → NumPy broadcasting (the emitted expression already reads locals a/b).
  // Equal lengths take the untouched fast path below.
  if (ra.length !== rb.length) {
    const out = await dispatchBroadcast(
      device, bufferPool, shaderCache,
      ra.buffer, rb.buffer, ra.length, rb.length,
      shapeOf(a, ra.length), shapeOf(b, rb.length),
      dtype, expression, `broadcast-zip-${dtype}`, source
    );
    ra.release();
    rb.release();
    return finalize(
      new GPUArray(out.buffer, out.length, dtype, device, bufferPool, { shape: out.outShape }),
      opts?.keepOnGpu ?? false
    );
  }

  const size = ra.length;
  const byteSize = size * 4;

  const shader = zipShader(expression, dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `zip-${dtype}`, source);

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
  fn: ((x: number, i: number, len: number) => number) | string,
  opts: MapOptions & { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuMap(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  fn: ((x: number, i: number, len: number) => number) | string,
  opts?: MapOptions
): Promise<TypedArray>;
export async function gpuMap(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  fn: ((x: number, i: number, len: number) => number) | string,
  opts?: MapOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(input);

  const rin = resolveInput(input, device, bufferPool, dtype);
  const size = rin.length;
  const byteSize = size * 4;

  // Captured const arrays, bound read-only after input/output. A GPUArray binds in place
  // (no re-upload); a plain array uploads to a pooled buffer for this call.
  const constNames = opts?.consts ? Object.keys(opts.consts) : [];
  const resolvedConsts = constNames.map((name) => {
    const value = opts!.consts![name];
    return { name, dtype: inputDtype(value), resolved: resolveInput(value, device, bufferPool, inputDtype(value)) };
  });

  const ir = parseExpression(fn, ["x", "i", "len"], constNames);
  const expression = emitWGSL(ir, dtype);
  const shader = mapShader(expression, dtype, resolvedConsts.map((c) => ({ name: c.name, dtype: c.dtype })));
  const source = typeof fn === "string" ? fn : fn.toString();
  // The const names go in the cache key so kernels with different captures don't collide.
  const pipeline = await shaderCache.getOrCreate(device, shader, `map-${dtype}-${constNames.join(",")}`, source);

  const bufOut = createOutputBuffer(device, byteSize, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: rin.buffer, size: byteSize } },
      { binding: 1, resource: { buffer: bufOut, size: byteSize } },
      ...resolvedConsts.map((c, i) => ({
        binding: i + 2,
        resource: { buffer: c.resolved.buffer, size: c.resolved.length * 4 },
      })),
    ],
  });

  dispatchOnly(device, pipeline, bindGroup, [computeWorkgroupCount(size)]);

  rin.release();
  for (const c of resolvedConsts) c.resolved.release();

  return finalize(new GPUArray(bufOut, size, dtype, device, bufferPool), opts?.keepOnGpu ?? false);
}
