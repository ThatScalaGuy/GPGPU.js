import type { TypedArray } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { dispatchOnly } from "../core/command";
import { GPUArray } from "../pipeline/gpu-array";
import {
  type OpInput,
  type OpOptions,
  resolveInput,
  inputDtype,
  finalize,
} from "../core/io";
import { computeWorkgroupGrid } from "../utils/workgroup";
import { parseExpression } from "../codegen/expression-parser";
import { emitWGSL } from "../codegen/wgsl-emitter";
import { gpuScan } from "./scan";
import { predicateFlagShader, compactShader } from "../codegen/templates";

export function gpuFilter(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  predicate: ((x: number, i: number, len: number) => boolean) | string,
  opts: { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuFilter(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  predicate: ((x: number, i: number, len: number) => boolean) | string,
  opts?: OpOptions
): Promise<TypedArray>;
export async function gpuFilter(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  predicate: ((x: number, i: number, len: number) => boolean) | string,
  opts?: OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const dtype = inputDtype(input);
  const keepOnGpu = opts?.keepOnGpu ?? false;

  const rin = resolveInput(input, device, bufferPool, dtype);
  const n = rin.length;

  if (n === 0) {
    rin.release();
    const empty = bufferPool.acquire(device, 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    return finalize(new GPUArray(empty, 0, dtype, device, bufferPool), keepOnGpu);
  }

  // 1. flags = predicate(x, i, len) ? 1 : 0. The predicate is a BOOLEAN expression wrapped in
  // select(0u, 1u, (expr)) by predicateFlagShader.
  // parseExpression only stringifies the fn; cast the boolean-returning predicate to its
  // number-returning signature so codegen accepts it (the WGSL is wrapped in select(0u,1u,...)).
  const ir = parseExpression(
    predicate as ((...args: number[]) => number) | string,
    ["x", "i", "len"]
  );
  const expr = emitWGSL(ir, dtype);
  const source = typeof predicate === "string" ? predicate : predicate.toString();
  const flagPipe = await shaderCache.getOrCreate(
    device, predicateFlagShader(expr, dtype), `filter-flags-${dtype}`, source
  );
  const flagsBuf = bufferPool.acquire(
    device, n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  );
  const flagGroup = device.createBindGroup({
    layout: flagPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: rin.buffer, size: n * 4 } },
      { binding: 1, resource: { buffer: flagsBuf, size: n * 4 } },
    ],
  });
  dispatchOnly(device, flagPipe, flagGroup, computeWorkgroupGrid(n));
  const flagsArr = new GPUArray(flagsBuf, n, "u32", device, bufferPool);

  // 2. inclusive scan of flags (gpuScan COPIES its input, so flagsArr survives for the compact pass).
  const scanned = (await gpuScan(
    deviceManager, bufferPool, shaderCache, flagsArr, (a, b) => a + b, 0, { keepOnGpu: true }
  )) as GPUArray;

  // 3. count = scanned[n-1]  (single-u32 GPU->CPU readback)
  const staging = bufferPool.acquire(device, 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(scanned.buffer, (n - 1) * 4, staging, 0, 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const count = new Uint32Array(staging.getMappedRange().slice(0))[0];
  staging.unmap();
  bufferPool.release(staging);

  // 4. compact: each kept element i writes input[i] to output[scanned[i]-1]. Runs BEFORE
  // destroying flagsArr/scanned (it reads both). Skip the dispatch when nothing is kept.
  const outBuf = bufferPool.acquire(
    device, Math.max(count, 1) * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  if (count > 0) {
    const compactPipe = await shaderCache.getOrCreate(
      device, compactShader(dtype), `filter-compact-${dtype}`
    );
    const compactGroup = device.createBindGroup({
      layout: compactPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: rin.buffer, size: n * 4 } },
        { binding: 1, resource: { buffer: flagsArr.buffer, size: n * 4 } },
        { binding: 2, resource: { buffer: scanned.buffer, size: n * 4 } },
        { binding: 3, resource: { buffer: outBuf, size: count * 4 } },
      ],
    });
    dispatchOnly(device, compactPipe, compactGroup, computeWorkgroupGrid(n));
  }

  rin.release();
  flagsArr.destroy();
  scanned.destroy();

  return finalize(new GPUArray(outBuf, count, dtype, device, bufferPool), keepOnGpu);
}
