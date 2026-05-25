import type { DataType, NumericArray } from "../core/types";
import { REDUCE_WORKGROUP_SIZE, inferDataType } from "../core/types";
import { toTypedArray } from "../utils/data-conversion";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { viewFor } from "../core/command";
import { parseExpression } from "../codegen/expression-parser";
import { emitWGSL, formatLiteral } from "../codegen/wgsl-emitter";
import { reduceShader } from "../codegen/templates";

export async function gpuReduce(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: NumericArray,
  fn: ((a: number, b: number) => number) | string,
  identity: number
): Promise<number> {
  const device = await deviceManager.getDevice();
  const dtype = inferDataType(input);

  const ir = parseExpression(fn, ["a", "b"]);
  const reduceExpr = emitWGSL(ir, dtype);
  const identityStr = formatLiteral(identity, dtype);

  const shader = reduceShader(reduceExpr, identityStr, dtype);
  const pipeline = await shaderCache.getOrCreate(device, shader, `reduce-${dtype}`);

  const inputData = toTypedArray(input, dtype);

  if (inputData.length === 0) return identity;
  if (inputData.length === 1) return inputData[0];

  // Ping-pong between two on-device buffers so the multi-pass reduction never
  // round-trips through the CPU. Upload once, dispatch every pass into a single
  // command encoder, read back only the final value.
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const firstOutCount = Math.ceil(inputData.length / REDUCE_WORKGROUP_SIZE);

  const bufA = bufferPool.acquire(device, inputData.byteLength, usage);
  const bufB = bufferPool.acquire(device, firstOutCount * 4, usage);
  device.queue.writeBuffer(
    bufA, 0, inputData.buffer as ArrayBuffer, inputData.byteOffset, inputData.byteLength
  );

  const encoder = device.createCommandEncoder();

  let src = bufA;
  let dst = bufB;
  let size = inputData.length;

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
    [src, dst] = [dst, src];
  }

  // After the final swap, `src` holds the single reduced value.
  const staging = bufferPool.acquire(
    device, 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  );
  encoder.copyBufferToBuffer(src, 0, staging, 0, 4);
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const result = viewFor(dtype, staging.getMappedRange().slice(0))[0];
  staging.unmap();

  bufferPool.release(staging);
  bufferPool.release(bufA);
  bufferPool.release(bufB);

  return result;
}

export async function gpuSum(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: NumericArray
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
  input: NumericArray
): Promise<number> {
  return gpuReduce(
    deviceManager, bufferPool, shaderCache, input,
    "Math.min(a, b)", MIN_IDENTITY[inferDataType(input)]
  );
}

export async function gpuMax(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: NumericArray
): Promise<number> {
  return gpuReduce(
    deviceManager, bufferPool, shaderCache, input,
    "Math.max(a, b)", MAX_IDENTITY[inferDataType(input)]
  );
}

export async function gpuProduct(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: NumericArray
): Promise<number> {
  return gpuReduce(deviceManager, bufferPool, shaderCache, input, (a, b) => a * b, 1);
}
