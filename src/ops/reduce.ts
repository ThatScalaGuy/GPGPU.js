import type { DataType } from "../core/types";
import { REDUCE_WORKGROUP_SIZE } from "../core/types";
import { toTypedArray } from "../utils/data-conversion";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { uploadBuffer, viewFor } from "../core/command";
import { type OpInput, inputDtype, isGPUArray } from "../core/io";
import { parseExpression } from "../codegen/expression-parser";
import { emitWGSL, formatLiteral } from "../codegen/wgsl-emitter";
import { argReduceShader, reduceShader } from "../codegen/templates";

export async function gpuReduce(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  fn: ((a: number, b: number) => number) | string,
  identity: number
): Promise<number> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(input);

  const ir = parseExpression(fn, ["a", "b"]);
  const reduceExpr = emitWGSL(ir, dtype);
  const identityStr = formatLiteral(identity, dtype);

  const shader = reduceShader(reduceExpr, identityStr, dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `reduce-${dtype}`);

  // Ping-pong between two on-device buffers so the multi-pass reduction never
  // round-trips through the CPU. The first pass reads `src` and writes a separate
  // `dst`, so a GPUArray input can serve as the initial `src` directly — no upload,
  // and we must not release it.
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  let initialSize: number;
  let bufA: GPUBuffer;
  let ownsA: boolean;

  if (isGPUArray(input)) {
    if (input.isDestroyed) throw new Error("GPUArray has been destroyed");
    initialSize = input.length;
    if (initialSize === 0) return identity;
    bufA = input.buffer;
    ownsA = false;
  } else {
    const inputData = toTypedArray(input, dtype);
    if (inputData.length === 0) return identity;
    if (inputData.length === 1) return inputData[0];
    initialSize = inputData.length;
    bufA = bufferPool.acquire(device, inputData.byteLength, usage);
    device.queue.writeBuffer(
      bufA, 0, inputData.buffer as ArrayBuffer, inputData.byteOffset, inputData.byteLength
    );
    ownsA = true;
  }

  const firstOutCount = Math.ceil(initialSize / REDUCE_WORKGROUP_SIZE);
  const bufB = bufferPool.acquire(device, firstOutCount * 4, usage);
  // From the second pass on, bufB's ping-pong partner must never be the caller's
  // GPUArray buffer — writing partials there would corrupt the resident data. A CPU
  // input's upload buffer is ours to recycle; a GPUArray input gets a scratch buffer
  // sized for the second pass's output.
  const scratch = ownsA
    ? null
    : bufferPool.acquire(device, Math.ceil(firstOutCount / REDUCE_WORKGROUP_SIZE) * 4, usage);
  const pong = ownsA ? bufA : scratch!;

  const encoder = device.createCommandEncoder();

  let src = bufA;
  let dst = bufB;
  let size = initialSize;

  while (size > 1) {
    const workgroupCount = Math.ceil(size / REDUCE_WORKGROUP_SIZE);

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src, size: size * 4 } },
        { binding: 1, resource: { buffer: dst, size: workgroupCount * 4 } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();

    size = workgroupCount;
    src = dst;
    dst = dst === bufB ? pong : bufB;
  }

  // After the final pass, `src` holds the single reduced value.
  const staging = bufferPool.acquire(
    device, 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  );
  encoder.copyBufferToBuffer(src, 0, staging, 0, 4);
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const result = viewFor(dtype, staging.getMappedRange().slice(0))[0];
  staging.unmap();

  bufferPool.release(staging);
  bufferPool.release(bufB);
  if (scratch) bufferPool.release(scratch);
  if (ownsA) bufferPool.release(bufA);

  return result;
}

export async function gpuSum(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput
): Promise<number> {
  return gpuReduce(deviceManager, bufferPool, shaderCache, input, (a, b) => a + b, 0);
}

// Identity for `min` is the largest value of the type (so padding lanes never win).
// f32 uses +3.4e38 (not the exact max 3.4028235e+38, which `toFixed` renders just above
// the representable max → shader compile error); still larger than any real input.
const MIN_IDENTITY: Record<DataType, number> = {
  f32: 3.4e38,
  i32: 2147483647,
  u32: 4294967295,
};

// Identity for `max` is the smallest value of the type.
const MAX_IDENTITY: Record<DataType, number> = {
  f32: -3.4e38,
  i32: -2147483648,
  u32: 0,
};

export async function gpuMin(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput
): Promise<number> {
  return gpuReduce(
    deviceManager, bufferPool, shaderCache, input,
    "Math.min(a, b)", MIN_IDENTITY[inputDtype(input)]
  );
}

export async function gpuMax(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput
): Promise<number> {
  return gpuReduce(
    deviceManager, bufferPool, shaderCache, input,
    "Math.max(a, b)", MAX_IDENTITY[inputDtype(input)]
  );
}

export async function gpuProduct(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput
): Promise<number> {
  return gpuReduce(deviceManager, bufferPool, shaderCache, input, (a, b) => a * b, 1);
}

// argmin/argmax: the multi-block ping-pong reduction of gpuReduce, but carrying
// (value, index) pairs in twin buffers so the surviving extremum's index can be read back.
// `mode` picks the value identity (the same safe ±3.4e38 constants min/max use, so padding
// lanes never win) and the WGSL comparison; first-occurrence tie-break lives in the shader.
async function gpuArgReduce(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  mode: "min" | "max"
): Promise<number> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(input);

  const identity = mode === "min" ? MIN_IDENTITY[dtype] : MAX_IDENTITY[dtype];
  const identityStr = formatLiteral(identity, dtype);
  const better = mode === "min" ? "<" : ">";

  const shader = argReduceShader(identityStr, better, dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `argreduce-${mode}-${dtype}`);

  // Value buffers ping-pong exactly like gpuReduce; index buffers shadow them. The first
  // pass reads `valA` and seeds indices from the global index (firstPass uniform), so a
  // GPUArray input serves as the initial value `src` directly — no upload, no release.
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  let initialSize: number;
  let valA: GPUBuffer;
  let ownsValA: boolean;

  if (isGPUArray(input)) {
    if (input.isDestroyed) throw new Error("GPUArray has been destroyed");
    initialSize = input.length;
    if (initialSize === 0) return -1;
    if (initialSize === 1) return 0;
    valA = input.buffer;
    ownsValA = false;
  } else {
    const inputData = toTypedArray(input, dtype);
    if (inputData.length === 0) return -1;
    if (inputData.length === 1) return 0;
    initialSize = inputData.length;
    valA = bufferPool.acquire(device, inputData.byteLength, usage);
    device.queue.writeBuffer(
      valA, 0, inputData.buffer as ArrayBuffer, inputData.byteOffset, inputData.byteLength
    );
    ownsValA = true;
  }

  const firstOutCount = Math.ceil(initialSize / REDUCE_WORKGROUP_SIZE);
  const valB = bufferPool.acquire(device, firstOutCount * 4, usage);
  // Like gpuReduce: valB's ping-pong partner from the second pass on must never be
  // the caller's GPUArray buffer (partials would corrupt the resident data).
  const valScratch = ownsValA
    ? null
    : bufferPool.acquire(device, Math.ceil(firstOutCount / REDUCE_WORKGROUP_SIZE) * 4, usage);
  const valPong = ownsValA ? valA : valScratch!;
  // Index buffers mirror the value buffers' shapes. idxA is only the carried-index source
  // from the second pass on; on the first pass it is bound but ignored (firstPass uniform).
  const idxA = bufferPool.acquire(device, initialSize * 4, usage);
  const idxB = bufferPool.acquire(device, firstOutCount * 4, usage);

  // One uniform per firstPass value: both are written before the single submit, so reusing
  // a single buffer across passes would clobber the flag. firstPass=1 runs once.
  const uFirst = uploadBuffer(device, new Uint32Array([1]), GPUBufferUsage.UNIFORM, bufferPool);
  const uRest = uploadBuffer(device, new Uint32Array([0]), GPUBufferUsage.UNIFORM, bufferPool);

  const encoder = device.createCommandEncoder();

  let srcVal = valA;
  let dstVal = valB;
  let srcIdx = idxA;
  let dstIdx = idxB;
  let size = initialSize;
  let firstPass = true;

  while (size > 1) {
    const workgroupCount = Math.ceil(size / REDUCE_WORKGROUP_SIZE);

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: srcVal, size: size * 4 } },
        { binding: 1, resource: { buffer: srcIdx, size: size * 4 } },
        { binding: 2, resource: { buffer: dstVal, size: workgroupCount * 4 } },
        { binding: 3, resource: { buffer: dstIdx, size: workgroupCount * 4 } },
        { binding: 4, resource: { buffer: firstPass ? uFirst : uRest } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();

    size = workgroupCount;
    srcVal = dstVal;
    dstVal = dstVal === valB ? valPong : valB;
    [srcIdx, dstIdx] = [dstIdx, srcIdx];
    firstPass = false;
  }

  // After the final swap, `srcIdx` holds the winning element's original index.
  const staging = bufferPool.acquire(
    device, 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  );
  encoder.copyBufferToBuffer(srcIdx, 0, staging, 0, 4);
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const result = new Uint32Array(staging.getMappedRange().slice(0))[0];
  staging.unmap();

  bufferPool.release(staging);
  bufferPool.release(valB);
  if (valScratch) bufferPool.release(valScratch);
  bufferPool.release(idxA);
  bufferPool.release(idxB);
  bufferPool.release(uFirst);
  bufferPool.release(uRest);
  if (ownsValA) bufferPool.release(valA);

  return result;
}

export async function gpuArgmin(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput
): Promise<number> {
  return gpuArgReduce(deviceManager, bufferPool, shaderCache, input, "min");
}

export async function gpuArgmax(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput
): Promise<number> {
  return gpuArgReduce(deviceManager, bufferPool, shaderCache, input, "max");
}
