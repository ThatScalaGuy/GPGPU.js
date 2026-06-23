import { describe, it, expect } from "vitest";
import { Pipeline } from "../../src/pipeline/pipeline";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";

// Drive Pipeline directly against the real GPU (no CPU fallback), so a broken
// reduce/map shader fails the test loudly instead of silently passing.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();

const pipeline = () => new Pipeline(deviceManager, bufferPool, shaderCache);

describe("pipeline (real GPU)", () => {
  it("runs a map-only chain", async () => {
    const result = await pipeline()
      .map((x) => x * 2)
      .map((x) => x + 1)
      .run([1, 2, 3, 4, 5]);
    expect(Array.from(result as Float32Array)).toEqual([3, 5, 7, 9, 11]);
  });

  // Regression: the reduce shader used `shared` (a reserved WGSL keyword), so
  // `pipeline-reduce` failed createShaderModule and run() threw instead of returning a value.
  it("runs map -> reduce (sum)", async () => {
    const result = await pipeline()
      .map((x) => x * 2)
      .map((x) => x + 1)
      .reduce((a, b) => a + b, 0)
      .run([1, 2, 3, 4, 5]);
    expect(result).toBe(35); // [3,5,7,9,11] -> 35
  });

  it("reduces across many workgroups", async () => {
    // n chosen so the sum (12,502,500) stays within f32's exact-integer range
    // (< 2^24) while still spanning many workgroups + a multi-pass reduction.
    const n = 5000;
    const input = Array.from({ length: n }, (_, i) => i + 1);
    const result = await pipeline().reduce((a, b) => a + b, 0).run(input);
    expect(result).toBe((n * (n + 1)) / 2);
  });

  it("reduces with a max expression", async () => {
    const result = await pipeline()
      .reduce((a, b) => Math.max(a, b), -3.4e38)
      .run([3, 1, 4, 1, 5, 9, 2, 6]);
    expect(result).toBe(9);
  });

  it("chains map -> filter -> scan -> reduce", async () => {
    // [1..6] *2 -> [2,4,6,8,10,12]; keep >5 -> [6,8,10,12];
    // inclusive prefix sum -> [6,14,24,36]; sum -> 80.
    const result = await pipeline()
      .map((x) => x * 2)
      .filter((x) => x > 5)
      .scan()
      .reduce((a, b) => a + b, 0)
      .run([1, 2, 3, 4, 5, 6]);
    expect(result).toBe(80);
  });

  it("runs a scan (inclusive prefix sum)", async () => {
    const result = await pipeline().scan().run([1, 2, 3, 4]);
    expect(Array.from(result as Float32Array)).toEqual([1, 3, 6, 10]);
  });

  it("filters with a string predicate", async () => {
    const result = await pipeline().filter("x > 3").run([1, 2, 3, 4, 5, 6]);
    expect(Array.from(result as Float32Array)).toEqual([4, 5, 6]);
  });

  it("sorts the stream", async () => {
    const result = await pipeline().sort().run([5, 3, 1, 4, 2]);
    expect(Array.from(result as Float32Array)).toEqual([1, 2, 3, 4, 5]);
  });

  it("returns sorted distinct values (sort -> unique)", async () => {
    const result = await pipeline().sort().unique().run([3, 1, 2, 3, 1, 2, 5, 4]);
    expect(Array.from(result as Float32Array)).toEqual([1, 2, 3, 4, 5]);
  });

  it("casts mid-chain, then maps in the new dtype", async () => {
    // f32 input truncates to i32, then *2 runs as integer math.
    const result = await pipeline()
      .cast("i32")
      .map((x) => x * 2)
      .run([1.2, 2.8, 3.5]);
    expect(result).toBeInstanceOf(Int32Array);
    expect(Array.from(result as Int32Array)).toEqual([2, 4, 6]);
  });

  it("returns an empty array when a filter keeps nothing", async () => {
    const result = await pipeline().filter((x) => x > 100).run([1, 2, 3]);
    expect((result as Float32Array).length).toBe(0);
  });

  it("returns the identity when reducing an empty (filtered-out) stream", async () => {
    const result = await pipeline()
      .filter((x) => x > 100)
      .reduce((a, b) => a + b, 0)
      .run([1, 2, 3]);
    expect(result).toBe(0);
  });

  it("fuses consecutive maps without changing results", async () => {
    // Three maps fuse into one dispatch; the output must match the math run on the CPU.
    const input = [1.5, 2.5, 3.5, 4.5];
    const result = (await pipeline()
      .map((x) => x * 2)
      .map((x) => x + 1)
      .map((x) => Math.sqrt(x))
      .run(input)) as Float32Array;
    const expected = input.map((x) => Math.sqrt(x * 2 + 1));
    expect(result.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 4);
    }
  });

  it("chains a GPU-resident array between pipelines without touching the input", async () => {
    const g = (await pipeline()
      .map((x) => x + 1)
      .run([1, 2, 3], { keepOnGpu: true })) as GPUArray;
    const out = (await pipeline()
      .map((x) => x * 10)
      .run(g, { keepOnGpu: true })) as GPUArray;

    expect(Array.from(await out.toArray())).toEqual([20, 30, 40]);
    expect(Array.from(await g.toArray())).toEqual([2, 3, 4]); // input untouched
    g.destroy();
    out.destroy();
  });

  it("rejects a step after a terminal reduce", () => {
    expect(() => pipeline().reduce((a, b) => a + b, 0).map((x) => x)).toThrow(/terminal/);
  });

  it("computes a histogram (u32 counts of length `bins`)", async () => {
    const result = await pipeline()
      .histogram({ bins: 5, min: 0, max: 10 })
      .run([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(result).toBeInstanceOf(Uint32Array);
    expect(Array.from(result as Uint32Array)).toEqual([2, 2, 2, 2, 2]);
  });

  it("chains histogram -> scan into a CDF (dtype flips to u32)", async () => {
    const result = await pipeline()
      .histogram({ bins: 5, min: 0, max: 10 })
      .scan()
      .run([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(result).toBeInstanceOf(Uint32Array);
    expect(Array.from(result as Uint32Array)).toEqual([2, 4, 6, 8, 10]);
  });

  it("keeps a histogram on the GPU as a u32 array", async () => {
    const g = (await pipeline()
      .histogram({ bins: 5, min: 0, max: 10 })
      .run([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], { keepOnGpu: true })) as GPUArray;
    expect(g.dtype).toBe("u32");
    expect(g.length).toBe(5);
    expect(Array.from(await g.toArray())).toEqual([2, 2, 2, 2, 2]);
    g.destroy();
  });

  it("convolves with a fixed kernel (valid mode)", async () => {
    // numpy.convolve([1,2,3,4,5], [1,1,1], "valid") -> [6, 9, 12]
    const result = await pipeline()
      .convolve([1, 1, 1], { mode: "valid" })
      .run([1, 2, 3, 4, 5]);
    expect(Array.from(result as Float32Array)).toEqual([6, 9, 12]);
  });

  it("gathers by a fixed index array (reverse)", async () => {
    const result = await pipeline()
      .gather([4, 3, 2, 1, 0])
      .run([10, 20, 30, 40, 50]);
    expect(Array.from(result as Float32Array)).toEqual([50, 40, 30, 20, 10]);
  });

  it("composes map -> gather", async () => {
    const result = await pipeline()
      .map((x) => x * 2)
      .gather([0, 0, 2])
      .run([1, 2, 3]);
    expect(Array.from(result as Float32Array)).toEqual([2, 2, 6]);
  });
});
