import type { TypedArray } from "../core/types";
import { DEFAULT_WORKGROUP_SIZE } from "../core/types";
import { toTypedArray } from "../utils/data-conversion";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { GPUArray } from "../pipeline/gpu-array";
import {
  type OpInput,
  type OpOptions,
  inputDtype,
  isGPUArray,
  finalize,
} from "../core/io";
import { parseExpression } from "../codegen/expression-parser";
import { emitWGSL, formatLiteral } from "../codegen/wgsl-emitter";
import { blockScanShader, scanAddOffsetsShader } from "../codegen/templates";

const WG = DEFAULT_WORKGROUP_SIZE;

type ScanFn = ((a: number, b: number) => number) | string;

export function gpuScan(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  fn: ScanFn | undefined,
  identity: number | undefined,
  opts: { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuScan(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  fn?: ScanFn,
  identity?: number,
  opts?: OpOptions
): Promise<TypedArray>;
export async function gpuScan(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  fn: ScanFn = (a, b) => a + b,
  identity: number = 0,
  opts?: OpOptions
): Promise<TypedArray | GPUArray> {
  const dtype = inputDtype(input);
  const device = await deviceManager.getDevice();
  const keepOnGpu = opts?.keepOnGpu ?? false;

  const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  // Resolve length + how to seed the working buffer without mutating a GPUArray input.
  let size: number;
  let srcArr: TypedArray | null = null;
  if (isGPUArray(input)) {
    if (input.isDestroyed) throw new Error("GPUArray has been destroyed");
    size = input.length;
  } else {
    srcArr = toTypedArray(input, dtype);
    size = srcArr.length;
  }

  if (size === 0) {
    return finalize(
      new GPUArray(bufferPool.acquire(device, 4, storageUsage), 0, dtype, device, bufferPool),
      keepOnGpu
    );
  }

  const ir = parseExpression(fn, ["a", "b"]);
  const scanExpr = emitWGSL(ir, dtype);
  const identityStr = formatLiteral(identity, dtype);

  const blockScanPipe = await shaderCache.getOrCreate(
    device, blockScanShader(scanExpr, identityStr, dtype), `scan-block-${dtype}`
  );
  const addOffsetsPipe = await shaderCache.getOrCreate(
    device, scanAddOffsetsShader(scanExpr, dtype), `scan-add-${dtype}`
  );

  // Single command encoder for every level so the multi-block scan never round-trips
  // through the CPU between passes (mirrors the gpuReduce design).
  const dataBuf = bufferPool.acquire(device, size * 4, storageUsage);

  const encoder = device.createCommandEncoder();

  // Seed the (in-place) working buffer. A GPUArray input is copied so we never mutate it.
  if (srcArr) {
    device.queue.writeBuffer(dataBuf, 0, srcArr.buffer as ArrayBuffer, srcArr.byteOffset, srcArr.byteLength);
  } else {
    encoder.copyBufferToBuffer((input as GPUArray).buffer, 0, dataBuf, 0, size * 4);
  }

  // Temp buffers (block sums + params); dataBuf is kept for the result.
  const cleanup: GPUBuffer[] = [];

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
  device.queue.submit([encoder.finish()]);

  for (const buf of cleanup) bufferPool.release(buf);

  return finalize(new GPUArray(dataBuf, size, dtype, device, bufferPool), keepOnGpu);
}
