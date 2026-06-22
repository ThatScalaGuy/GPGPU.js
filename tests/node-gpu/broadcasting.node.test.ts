import { describe, it, expect } from "vitest";
import { GPU } from "../../src/gpu";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { gpuElementwiseBinary, gpuZip } from "../../src/ops/elementwise";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";
import { broadcastShapes } from "../../src/codegen/broadcast";
import { expectClose } from "../shared/tolerance";

// Broadcasting needs the logical shape a reshape attaches to a GPUArray, so these go through the
// public API (upload → reshape → op). Integer cases assert exact equality; the f32 case uses the
// reduction tolerance helper for parity with the rest of the suite (the values are exact here, but
// keep the discipline). The shapes exercised: [n]vs[1], [m,n]vs[n], [m,n]vs[m,1], + incompatible.
const gpu = new GPU();

describe("broadcasting (real GPU)", () => {
  it("[n] vs [1]: a length-1 array broadcasts against a vector", async () => {
    const a = await gpu.reshape(new Int32Array([1, 2, 3, 4]), [4]);
    const b = await gpu.reshape(new Int32Array([10]), [1]);
    const out = (await gpu.add(a, b, { keepOnGpu: true })) as GPUArray;
    expect(out.shape).toEqual([4]);
    expect(Array.from(await out.toArray())).toEqual([11, 12, 13, 14]);
    a.destroy();
    b.destroy();
    out.destroy();
  });

  it("[m,n] vs [n]: a row vector broadcasts across every row", async () => {
    const mat = await gpu.reshape(new Int32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const row = await gpu.reshape(new Int32Array([10, 20, 30]), [3]);
    const out = (await gpu.add(mat, row, { keepOnGpu: true })) as GPUArray;
    expect(out.shape).toEqual([2, 3]);
    expect(Array.from(await out.toArray())).toEqual([11, 22, 33, 14, 25, 36]);
    mat.destroy();
    row.destroy();
    out.destroy();
  });

  it("[m,n] vs [m,1]: a column vector broadcasts across every column", async () => {
    const mat = await gpu.reshape(new Int32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const col = await gpu.reshape(new Int32Array([100, 200]), [2, 1]);
    const out = (await gpu.add(mat, col, { keepOnGpu: true })) as GPUArray;
    expect(out.shape).toEqual([2, 3]);
    expect(Array.from(await out.toArray())).toEqual([101, 102, 103, 204, 205, 206]);
    mat.destroy();
    col.destroy();
    out.destroy();
  });

  it("broadcasting is symmetric: [n] op [m,n] yields the [m,n] result shape", async () => {
    const row = await gpu.reshape(new Int32Array([10, 20, 30]), [3]);
    const mat = await gpu.reshape(new Int32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const out = (await gpu.add(row, mat, { keepOnGpu: true })) as GPUArray;
    expect(out.shape).toEqual([2, 3]);
    expect(Array.from(await out.toArray())).toEqual([11, 22, 33, 14, 25, 36]);
    row.destroy();
    mat.destroy();
    out.destroy();
  });

  it("subtract / multiply / divide broadcast a row vector across a matrix", async () => {
    const mat = await gpu.reshape([2, 4, 6, 8, 10, 12], [2, 3]);
    const row = await gpu.reshape([2, 4, 6], [3]);

    const sub = (await gpu.subtract(mat, row, { keepOnGpu: true })) as GPUArray;
    const mul = (await gpu.multiply(mat, row, { keepOnGpu: true })) as GPUArray;
    const div = (await gpu.divide(mat, row, { keepOnGpu: true })) as GPUArray;

    const subArr = await sub.toArray();
    const mulArr = await mul.toArray();
    const divArr = await div.toArray();
    const expectedSub = [0, 0, 0, 6, 6, 6];
    const expectedMul = [4, 16, 36, 16, 40, 72];
    const expectedDiv = [1, 1, 1, 4, 2.5, 2];
    for (let i = 0; i < 6; i++) {
      expectClose(subArr[i], expectedSub[i]);
      expectClose(mulArr[i], expectedMul[i]);
      expectClose(divArr[i], expectedDiv[i]);
    }
    mat.destroy();
    row.destroy();
    sub.destroy();
    mul.destroy();
    div.destroy();
  });

  it("zip broadcasts a custom two-arg fn over mismatched shapes", async () => {
    const mat = await gpu.reshape(new Int32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const col = await gpu.reshape(new Int32Array([10, 20]), [2, 1]);
    const out = (await gpu.zip(mat, col, (a, b) => a * b, { keepOnGpu: true })) as GPUArray;
    expect(out.shape).toEqual([2, 3]);
    // row0 * 10 => 10,20,30 ; row1 * 20 => 80,100,120
    expect(Array.from(await out.toArray())).toEqual([10, 20, 30, 80, 100, 120]);
    mat.destroy();
    col.destroy();
    out.destroy();
  });

  it("a bare (unshaped) GPUArray is treated as 1-D for broadcasting", async () => {
    // `upload` attaches no shape, so the scalar-like length-1 operand reads as [1].
    const mat = await gpu.reshape(new Int32Array([1, 2, 3, 4]), [2, 2]);
    const scalar = await gpu.upload(new Int32Array([5]));
    const out = (await gpu.multiply(mat, scalar, { keepOnGpu: true })) as GPUArray;
    expect(Array.from(await out.toArray())).toEqual([5, 10, 15, 20]);
    mat.destroy();
    scalar.destroy();
    out.destroy();
  });

  it("throws on incompatible shapes ([m,n] vs [k] with k != n)", async () => {
    const mat = await gpu.reshape(new Int32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const bad = await gpu.reshape(new Int32Array([1, 2]), [2]);
    await expect(gpu.add(mat, bad, { keepOnGpu: true })).rejects.toThrow(/[Cc]annot broadcast/);
    mat.destroy();
    bad.destroy();
  });

  it("equal-length inputs keep the elementwise fast path (no broadcast shape attached)", async () => {
    const deviceManager = new DeviceManager();
    const bufferPool = new BufferPool();
    const shaderCache = new ShaderCache();
    // Drive gpuElementwiseBinary directly with two same-length CPU arrays: the result must be a
    // plain elementwise sum, with no shape (the fast path is untouched).
    const out = (await gpuElementwiseBinary(
      deviceManager, bufferPool, shaderCache,
      new Int32Array([1, 2, 3]), new Int32Array([10, 20, 30]), "+", { keepOnGpu: true }
    )) as GPUArray;
    expect(out.shape).toBeUndefined();
    expect(Array.from(await out.toArray())).toEqual([11, 22, 33]);
    out.destroy();
  });

  it("gpuZip broadcasts directly with shaped GPUArray inputs", async () => {
    const deviceManager = new DeviceManager();
    const bufferPool = new BufferPool();
    const shaderCache = new ShaderCache();
    const device = await deviceManager.getDevice();
    const { uploadBuffer } = await import("../../src/core/command");
    const matBuf = uploadBuffer(device, new Int32Array([1, 2, 3, 4, 5, 6]), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, bufferPool);
    const rowBuf = uploadBuffer(device, new Int32Array([10, 20, 30]), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, bufferPool);
    const mat = new GPUArray(matBuf, 6, "i32", device, bufferPool, { shape: [2, 3] });
    const row = new GPUArray(rowBuf, 3, "i32", device, bufferPool, { shape: [3] });
    const out = (await gpuZip(deviceManager, bufferPool, shaderCache, mat, row, (a, b) => a + b, { keepOnGpu: true })) as GPUArray;
    expect(out.shape).toEqual([2, 3]);
    expect(Array.from(await out.toArray())).toEqual([11, 22, 33, 14, 25, 36]);
    mat.destroy();
    row.destroy();
    out.destroy();
  });
});

// Pure shape-algebra unit checks — no GPU — pin the NumPy rules independent of any kernel.
describe("broadcastShapes (unit)", () => {
  it("computes the elementwise-max result shape", () => {
    expect(broadcastShapes([4], [1])).toEqual([4]);
    expect(broadcastShapes([2, 3], [3])).toEqual([2, 3]);
    expect(broadcastShapes([2, 3], [2, 1])).toEqual([2, 3]);
    expect(broadcastShapes([2, 1, 3], [4, 1])).toEqual([2, 4, 3]);
  });

  it("throws when a non-1 dimension pair disagrees", () => {
    expect(() => broadcastShapes([2, 3], [2])).toThrow(/[Cc]annot broadcast/);
    expect(() => broadcastShapes([3], [4])).toThrow(/[Cc]annot broadcast/);
  });
});
