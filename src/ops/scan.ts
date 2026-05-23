import type { NumericArray } from "../core/types";
import { DEFAULT_WORKGROUP_SIZE, toFloat32Array } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { dispatchAndRead, uploadBuffer, createOutputBuffer } from "../core/command";
import { computeWorkgroupCount } from "../utils/workgroup";
import { parseExpression } from "../codegen/expression-parser";
import { emitWGSL } from "../codegen/wgsl-emitter";
import { scanShader } from "../codegen/templates";

export async function gpuScan(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: NumericArray,
  fn: ((a: number, b: number) => number) | string = (a, b) => a + b,
  identity: number = 0
): Promise<Float32Array> {
  const device = await deviceManager.getDevice();
  const arr = toFloat32Array(input);
  const size = arr.length;
  const byteSize = size * 4;

  const ir = parseExpression(fn, ["a", "b"]);
  const scanExpr = emitWGSL(ir);
  const identityStr = Number.isInteger(identity) ? identity.toFixed(1) : String(identity);

  const shader = scanShader(scanExpr, identityStr);
  const pipeline = await shaderCache.getOrCreate(device, shader, "scan");

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
