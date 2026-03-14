import type { NumericArray } from "../core/types";
import { REDUCE_WORKGROUP_SIZE, toFloat32Array } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { dispatchAndRead, uploadBuffer, createOutputBuffer } from "../core/command";
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

  let data = toFloat32Array(input);

  while (data.length > 1) {
    const size = data.length;
    const workgroupCount = Math.ceil(size / REDUCE_WORKGROUP_SIZE);
    const inputByteSize = size * 4;
    const outputByteSize = workgroupCount * 4;

    const bufInput = uploadBuffer(device, data, GPUBufferUsage.STORAGE, bufferPool);
    const bufOut = createOutputBuffer(device, outputByteSize, bufferPool);

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufInput, size: inputByteSize } },
        { binding: 1, resource: { buffer: bufOut, size: outputByteSize } },
      ],
    });

    data = await dispatchAndRead(
      device, pipeline, bindGroup,
      [workgroupCount], bufOut, outputByteSize, bufferPool
    );

    bufferPool.release(bufInput);
    bufferPool.release(bufOut);

    data = data.slice(0, workgroupCount);
  }

  return data[0] ?? identity;
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
