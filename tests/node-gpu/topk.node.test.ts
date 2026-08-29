import { describe, it, expect } from "vitest";
import { gpuTopK, cpuTopK } from "../../src/ops/topk";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";

// Drive the GPU top-k implementation directly (no CPU fallback wrapper), so a broken
// composition fails the test loudly instead of silently passing through the fallback.
// Values are moved verbatim (sort + slice, no arithmetic), so GPU-vs-CPU value
// comparisons are exact. Indices are NOT compared to the CPU reference — the bitonic
// sort is unstable, so tied values may report different (equally valid) positions;
// instead every index is verified against the ORIGINAL input: input[indices[j]] === values[j].
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

// Deterministic pseudo-random f32 values (LCG) so failures reproduce.
function lcgFloats(n: number, seed = 12345): Float32Array {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = Math.fround((s / 4294967296) * 2000 - 1000);
  }
  return out;
}

function expectPairing(
  input: ArrayLike<number>,
  values: ArrayLike<number>,
  indices: ArrayLike<number>
): void {
  expect(indices.length).toBe(values.length);
  for (let j = 0; j < indices.length; j++) {
    expect(input[indices[j]]).toBe(values[j]);
  }
}

describe("topk (real GPU)", () => {
  it("k=10 of 100k values: values match cpuTopK exactly, indices pair with the input", async () => {
    const input = lcgFloats(100000);
    const { values, indices } = await gpuTopK(...args, input, 10);
    expect(values).toBeInstanceOf(Float32Array);
    expect(indices).toBeInstanceOf(Uint32Array);
    expect(values.length).toBe(10);

    const ref = cpuTopK(input, 10);
    expect(Array.from(values)).toEqual(Array.from(ref.values));
    expectPairing(input, values, indices);
  });

  it("largest: false returns the k smallest, ascending", async () => {
    const input = lcgFloats(1000, 999);
    const { values, indices } = await gpuTopK(...args, input, 7, { largest: false });
    const ref = cpuTopK(input, 7, false);
    expect(Array.from(values)).toEqual(Array.from(ref.values));
    expectPairing(input, values, indices);
  });

  it("k = 0 yields an empty pair", async () => {
    const { values, indices } = await gpuTopK(...args, new Float32Array([3, 1, 2]), 0);
    expect(values).toBeInstanceOf(Float32Array);
    expect(indices).toBeInstanceOf(Uint32Array);
    expect(values.length).toBe(0);
    expect(indices.length).toBe(0);
  });

  it("negative k yields an empty pair", async () => {
    const { values, indices } = await gpuTopK(...args, new Float32Array([3, 1, 2]), -5);
    expect(values.length).toBe(0);
    expect(indices.length).toBe(0);
  });

  it("k > n returns all n, sorted (indices are a full permutation)", async () => {
    const input = new Float32Array([3, 1, 4, 1, 5]);
    const { values, indices } = await gpuTopK(...args, input, 100);
    expect(Array.from(values)).toEqual([5, 4, 3, 1, 1]);
    expectPairing(input, values, indices);
    // All n positions appear exactly once.
    expect(Array.from(indices).sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4]);
  });

  it("k = n returns all n, sorted", async () => {
    const input = new Float32Array([-2, 7, 0, 7]);
    const { values, indices } = await gpuTopK(...args, input, 4);
    expect(Array.from(values)).toEqual([7, 7, 0, -2]);
    expectPairing(input, values, indices);
    expect(Array.from(indices).sort((a, b) => a - b)).toEqual([0, 1, 2, 3]);
  });

  it("i32 input: values dtype follows input, indices are u32", async () => {
    const input = new Int32Array([-5, 3, -7, 12, 0, 3]);
    const { values, indices } = await gpuTopK(...args, input, 3);
    expect(values).toBeInstanceOf(Int32Array);
    expect(indices).toBeInstanceOf(Uint32Array);
    const ref = cpuTopK(input, 3);
    expect(Array.from(values)).toEqual(Array.from(ref.values));
    expect(Array.from(values)).toEqual([12, 3, 3]);
    expectPairing(input, values, indices);
  });

  it("i32 input, largest: false (negative values)", async () => {
    const input = new Int32Array([-5, 3, -7, 12, 0]);
    const { values, indices } = await gpuTopK(...args, input, 2, { largest: false });
    expect(Array.from(values)).toEqual([-7, -5]);
    expectPairing(input, values, indices);
  });

  it("u32 input: large values kept exact", async () => {
    const input = new Uint32Array([7, 4294967294, 0, 2147483648, 1]);
    const { values, indices } = await gpuTopK(...args, input, 2);
    expect(values).toBeInstanceOf(Uint32Array);
    expect(Array.from(values)).toEqual([4294967294, 2147483648]);
    expectPairing(input, values, indices);
  });

  it("empty input yields an empty pair regardless of k", async () => {
    const { values, indices } = await gpuTopK(...args, new Float32Array(0), 5);
    expect(values).toBeInstanceOf(Float32Array);
    expect(values.length).toBe(0);
    expect(indices.length).toBe(0);
  });

  it("length-1 input", async () => {
    const { values, indices } = await gpuTopK(...args, new Float32Array([42]), 1);
    expect(Array.from(values)).toEqual([42]);
    expect(Array.from(indices)).toEqual([0]);
  });

  it("accepts a resident GPUArray input and does not mutate it", async () => {
    const device = await deviceManager.getDevice();
    const src = new Float32Array([6, 2, 9, 4, 8, 1]);
    const buf = bufferPool.acquire(
      device,
      src.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    device.queue.writeBuffer(buf, 0, src.buffer);
    const gpuInput = new GPUArray(buf, src.length, "f32", device, bufferPool);

    const { values, indices } = await gpuTopK(...args, gpuInput, 3);
    expect(Array.from(values)).toEqual([9, 8, 6]);
    expectPairing(src, values, indices);

    // The input GPUArray must be untouched and still usable.
    expect(Array.from(await gpuInput.toArray())).toEqual([6, 2, 9, 4, 8, 1]);
    gpuInput.destroy();
  });

  it("keepOnGpu returns a usable GPUArray pair", async () => {
    const input = new Float32Array([0.5, -1.5, 3.25, 2, -7, 3.25, 0]);
    const out = await gpuTopK(...args, input, 4, { keepOnGpu: true });
    expect(out.values).toBeInstanceOf(GPUArray);
    expect(out.indices).toBeInstanceOf(GPUArray);
    expect(out.values.length).toBe(4);
    expect(out.indices.length).toBe(4);
    expect(out.values.dtype).toBe("f32");
    expect(out.indices.dtype).toBe("u32");

    const values = await out.values.toArray();
    const indices = await out.indices.toArray();
    expect(Array.from(values)).toEqual(Array.from(cpuTopK(input, 4).values));
    expectPairing(input, values, indices);
    out.values.destroy();
    out.indices.destroy();
  });

  it("keepOnGpu with largest: false", async () => {
    const input = new Int32Array([10, -3, 7, -3, 5]);
    const out = await gpuTopK(...args, input, 2, { keepOnGpu: true, largest: false });
    const values = await out.values.toArray();
    const indices = await out.indices.toArray();
    expect(Array.from(values)).toEqual([-3, -3]);
    expectPairing(input, values, indices);
    out.values.destroy();
    out.indices.destroy();
  });

  it("keepOnGpu empty pair is safe (toArray both)", async () => {
    const out = await gpuTopK(...args, new Float32Array([1, 2]), 0, { keepOnGpu: true });
    expect(out.values).toBeInstanceOf(GPUArray);
    expect(out.indices).toBeInstanceOf(GPUArray);
    expect((await out.values.toArray()).length).toBe(0);
    expect((await out.indices.toArray()).length).toBe(0);
    out.values.destroy();
    out.indices.destroy();
  });
});

describe("cpuTopK", () => {
  it("selects the k largest, descending, ties broken by index", () => {
    const { values, indices } = cpuTopK([3, 1, 4, 1, 5], 3);
    expect(Array.from(values)).toEqual([5, 4, 3]);
    expect(Array.from(indices)).toEqual([4, 2, 0]);
  });

  it("largest: false selects the k smallest, ascending", () => {
    const { values, indices } = cpuTopK([3, 1, 4, 1, 5], 2, false);
    expect(Array.from(values)).toEqual([1, 1]);
    expect(Array.from(indices)).toEqual([1, 3]);
  });

  it("clamps k to [0, n]", () => {
    expect(cpuTopK([1, 2], 100).values.length).toBe(2);
    expect(cpuTopK([1, 2], -1).values.length).toBe(0);
    expect(cpuTopK([], 3).values.length).toBe(0);
  });
});
