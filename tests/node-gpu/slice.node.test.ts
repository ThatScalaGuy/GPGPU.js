import { describe, it, expect } from "vitest";
import { gpuSlice, cpuSlice } from "../../src/ops/slice";
import { gpuSum } from "../../src/ops/reduce";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";
import { expectClose } from "../shared/tolerance";

// Drive the GPU slice implementation directly (no CPU fallback wrapper), so a broken
// copy path fails the test loudly instead of silently passing through the fallback.
// Values are copied verbatim (no arithmetic), so GPU and CPU agree exactly for every dtype.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

describe("slice (real GPU)", () => {
  const data = new Float32Array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);

  it("middle slice matches cpuSlice exactly", async () => {
    const out = (await gpuSlice(...args, data, 2, 7)) as Float32Array;
    expect(out).toBeInstanceOf(Float32Array);
    expect(Array.from(out)).toEqual(Array.from(cpuSlice(data, 2, 7)));
    expect(Array.from(out)).toEqual([12, 13, 14, 15, 16]);
  });

  it("prefix slice (explicit end only)", async () => {
    const out = (await gpuSlice(...args, data, 0, 3)) as Float32Array;
    expect(Array.from(out)).toEqual([10, 11, 12]);
  });

  it("suffix slice (begin only, end defaults to length)", async () => {
    const out = (await gpuSlice(...args, data, 7)) as Float32Array;
    expect(Array.from(out)).toEqual([17, 18, 19]);
  });

  it("no arguments copies the whole array", async () => {
    const out = (await gpuSlice(...args, data)) as Float32Array;
    expect(Array.from(out)).toEqual(Array.from(data));
  });

  it("negative begin counts from the end", async () => {
    const out = (await gpuSlice(...args, data, -3)) as Float32Array;
    expect(Array.from(out)).toEqual([17, 18, 19]);
  });

  it("negative end counts from the end", async () => {
    const out = (await gpuSlice(...args, data, 1, -6)) as Float32Array;
    expect(Array.from(out)).toEqual([11, 12, 13]);
  });

  it("out-of-range begin/end clamp (full copy)", async () => {
    const out = (await gpuSlice(...args, data, -100, 100)) as Float32Array;
    expect(Array.from(out)).toEqual(Array.from(data));
  });

  it("begin >= end yields an empty result", async () => {
    const out = (await gpuSlice(...args, data, 7, 2)) as Float32Array;
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(0);
  });

  it("begin === length yields an empty result", async () => {
    const out = (await gpuSlice(...args, data, data.length)) as Float32Array;
    expect(out.length).toBe(0);
  });

  it("empty input yields an empty result", async () => {
    const out = (await gpuSlice(...args, new Float32Array(0))) as Float32Array;
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(0);
  });

  it("length-1 input: full and empty slices", async () => {
    const one = new Float32Array([42]);
    expect(Array.from((await gpuSlice(...args, one)) as Float32Array)).toEqual([42]);
    expect(((await gpuSlice(...args, one, 1)) as Float32Array).length).toBe(0);
  });

  // Sweep (begin, end) combinations, including undefined, negatives, clamping and
  // non-integer indices, against the cpuSlice reference.
  const combos: [number | undefined, number | undefined][] = [
    [undefined, undefined],
    [3, undefined],
    [undefined, 4],
    [2, 8],
    [-4, -1],
    [-100, 100],
    [8, 3],
    [1.9, 7.2],
    [-2.5, undefined],
  ];
  for (const [b, e] of combos) {
    it(`matches cpuSlice for (${b}, ${e})`, async () => {
      const out = (await gpuSlice(...args, data, b, e)) as Float32Array;
      expect(Array.from(out)).toEqual(Array.from(cpuSlice(data, b, e)));
    });
  }

  it("i32 input: dtype and values preserved", async () => {
    const input = new Int32Array([-5, -4, -3, -2, -1, 0, 1]);
    const out = (await gpuSlice(...args, input, 1, -1)) as Int32Array;
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual(Array.from(cpuSlice(input, 1, -1)));
    expect(Array.from(out)).toEqual([-4, -3, -2, -1, 0]);
  });

  it("u32 input: dtype and values preserved (large values)", async () => {
    const input = new Uint32Array([4294967295, 0, 2147483648, 7]);
    const out = (await gpuSlice(...args, input, 0, 3)) as Uint32Array;
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([4294967295, 0, 2147483648]);
  });

  it("number[] input yields a Float32Array", async () => {
    const out = (await gpuSlice(...args, [1, 2, 3, 4], 1, 3)) as Float32Array;
    expect(out).toBeInstanceOf(Float32Array);
    expect(Array.from(out)).toEqual([2, 3]);
  });

  it("accepts a GPUArray input and does not mutate it (copy, not view)", async () => {
    const device = await deviceManager.getDevice();
    const src = new Float32Array([1, 2, 3, 4, 5, 6]);
    const buf = bufferPool.acquire(
      device,
      src.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    device.queue.writeBuffer(buf, 0, src.buffer);
    const gpuInput = new GPUArray(buf, src.length, "f32", device, bufferPool);

    const out = await gpuSlice(...args, gpuInput, 2, 5, { keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    // A copy, never a view of the input's buffer.
    expect(out.buffer).not.toBe(gpuInput.buffer);
    expect(Array.from(await out.toArray())).toEqual([3, 4, 5]);
    out.destroy();

    // The input GPUArray must be untouched and still usable.
    expect(Array.from(await gpuInput.toArray())).toEqual([1, 2, 3, 4, 5, 6]);
    gpuInput.destroy();
  });

  it("keepOnGpu result feeds another op (gpuSum)", async () => {
    const input = new Float32Array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]);
    const out = await gpuSlice(...args, input, 1, 4, { keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.length).toBe(3);
    expect(out.dtype).toBe("f32");
    const sum = await gpuSum(deviceManager, bufferPool, shaderCache, out);
    expectClose(sum, 1.5 + 2.5 + 3.5);
    out.destroy();
  });

  it("keepOnGpu empty result is a safe zero-length GPUArray", async () => {
    const out = await gpuSlice(...args, data, 5, 5, { keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.length).toBe(0);
    expect((await out.toArray()).length).toBe(0);
    out.destroy();
  });
});

describe("cpuSlice", () => {
  it("does not alias an already-typed input", () => {
    const input = new Float32Array([1, 2, 3]);
    const out = cpuSlice(input);
    out[0] = 99;
    expect(input[0]).toBe(1);
  });

  it("follows JS slice semantics for negatives and clamping", () => {
    const input = new Int32Array([0, 1, 2, 3, 4]);
    expect(Array.from(cpuSlice(input, -2))).toEqual([3, 4]);
    expect(Array.from(cpuSlice(input, 1, -1))).toEqual([1, 2, 3]);
    expect(Array.from(cpuSlice(input, 10))).toEqual([]);
    expect(Array.from(cpuSlice(input, -99, 99))).toEqual([0, 1, 2, 3, 4]);
  });
});
