import type { NumericArray } from "../core/types";
import { REDUCE_WORKGROUP_SIZE, toFloat32Array } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { parseExpression } from "../codegen/expression-parser";
import { emitWGSL } from "../codegen/wgsl-emitter";
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

  const ir = parseExpression(fn, ["a", "b"]);
  const reduceExpr = emitWGSL(ir);
  const identityStr = Number.isInteger(identity) ? identity.toFixed(1) : String(identity);

  const shader = reduceShader(reduceExpr, identityStr);
  const pipeline = await shaderCache.getOrCreate(device, shader, "reduce");

  const inputData = toFloat32Array(input);

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
  const result = new Float32Array(staging.getMappedRange().slice(0))[0];
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

export async function gpuMin(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: NumericArray
): Promise<number> {
  return gpuReduce(
    deviceManager, bufferPool, shaderCache, input,
    "min(a, b)", 3.4028235e+38
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
    "max(a, b)", -3.4028235e+38
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
