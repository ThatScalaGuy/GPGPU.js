import type { NumericArray } from "../core/types";
import { DEFAULT_WORKGROUP_SIZE, toFloat32Array } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { parseExpression } from "../codegen/expression-parser";
import { emitWGSL } from "../codegen/wgsl-emitter";
import { blockScanShader, scanAddOffsetsShader } from "../codegen/templates";

const WG = DEFAULT_WORKGROUP_SIZE;

export async function gpuScan(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: NumericArray,
  fn: ((a: number, b: number) => number) | string = (a, b) => a + b,
  identity: number = 0
): Promise<Float32Array> {
  const arr = toFloat32Array(input);
  const size = arr.length;
  if (size === 0) return new Float32Array(0);

  const device = await deviceManager.getDevice();

  const ir = parseExpression(fn, ["a", "b"]);
  const scanExpr = emitWGSL(ir);
  const identityStr = Number.isInteger(identity) ? identity.toFixed(1) : String(identity);

  const blockScanPipe = await shaderCache.getOrCreate(
    device, blockScanShader(scanExpr, identityStr), "scan-block"
  );
  const addOffsetsPipe = await shaderCache.getOrCreate(
    device, scanAddOffsetsShader(scanExpr), "scan-add"
  );

  // Single command encoder for every level so the multi-block scan never round-trips
  // through the CPU between passes (mirrors the gpuReduce design).
  const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const dataBuf = bufferPool.acquire(device, size * 4, storageUsage);
  device.queue.writeBuffer(dataBuf, 0, arr.buffer as ArrayBuffer, arr.byteOffset, arr.byteLength);

  const encoder = device.createCommandEncoder();
  const cleanup: GPUBuffer[] = [dataBuf];

  const makeParams = (n: number): GPUBuffer => {
    const buf = bufferPool.acquire(device, 4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(buf, 0, new Uint32Array([n]));
    cleanup.push(buf);
    return buf;
  };

  // Scan `buf` (n elements) in place: per-block scan, recurse on the block sums, then
  // fold each block's offset back in.
  const scanRec = (buf: GPUBuffer, n: number): void => {
    const blockCount = Math.ceil(n / WG);
    const blockSums = bufferPool.acquire(device, blockCount * 4, storageUsage);
    cleanup.push(blockSums);
    const params = makeParams(n);

    const blockGroup = device.createBindGroup({
      layout: blockScanPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buf, size: n * 4 } },
        { binding: 1, resource: { buffer: blockSums, size: blockCount * 4 } },
        { binding: 2, resource: { buffer: params } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(blockScanPipe);
    pass.setBindGroup(0, blockGroup);
    pass.dispatchWorkgroups(blockCount);
    pass.end();

    if (blockCount > 1) {
      scanRec(blockSums, blockCount);

      const addGroup = device.createBindGroup({
        layout: addOffsetsPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buf, size: n * 4 } },
          { binding: 1, resource: { buffer: blockSums, size: blockCount * 4 } },
          { binding: 2, resource: { buffer: params } },
        ],
      });

      const addPass = encoder.beginComputePass();
      addPass.setPipeline(addOffsetsPipe);
      addPass.setBindGroup(0, addGroup);
      addPass.dispatchWorkgroups(blockCount);
      addPass.end();
    }
  };

  scanRec(dataBuf, size);

  const byteSize = size * 4;
  const staging = bufferPool.acquire(
    device, byteSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  );
  encoder.copyBufferToBuffer(dataBuf, 0, staging, 0, byteSize);
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();

  bufferPool.release(staging);
  for (const buf of cleanup) bufferPool.release(buf);

  return result.slice(0, size);
}
