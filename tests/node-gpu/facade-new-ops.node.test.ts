import { describe, it, expect } from "vitest";
import { GPU } from "../../src/gpu";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { expectClose } from "../shared/tolerance";

// End-to-end smoke tests for the public gpu.* wiring of the new ops (the op
// suites drive the gpu* functions directly; this covers the facade overloads,
// fallback routing, and cross-op flows a user would actually write).
const gpu = new GPU({ fallback: "throw" });

describe("stats facade (real GPU)", () => {
  it("mean / variance / std / dot / norm", async () => {
    expect(await gpu.mean([1, 2, 3, 4])).toBe(2.5);
    expectClose(await gpu.variance([1, 2, 3, 4]), 1.25);
    expectClose(await gpu.std([1, 2, 3, 4]), Math.sqrt(1.25));
    expect(await gpu.dot([1, 2, 3], [4, 5, 6])).toBe(32);
    expectClose(await gpu.norm([3, 4]), 5);
  });

  it("cosineSimilarity: identical ≈ 1, orthogonal ≈ 0", async () => {
    expectClose(await gpu.cosineSimilarity([1, 2, 3], [1, 2, 3]), 1, { eps: 1e-5 });
    expectClose(await gpu.cosineSimilarity([1, 0], [0, 1]), 0, { eps: 1e-6 });
  });

  it("softmax sums to 1 and is shift-invariant", async () => {
    const s = (await gpu.softmax([1, 2, 3])) as Float32Array;
    expectClose(s[0] + s[1] + s[2], 1, { eps: 1e-5 });
    const shifted = (await gpu.softmax([101, 102, 103])) as Float32Array;
    for (let i = 0; i < 3; i++) expectClose(s[i], shifted[i], { eps: 1e-5 });
  });
});

describe("constructors facade (real GPU)", () => {
  it("zeros / full / arange / linspace", async () => {
    expect(Array.from((await gpu.zeros(4)) as Float32Array)).toEqual([0, 0, 0, 0]);
    expect(Array.from((await gpu.full(3, 7.5)) as Float32Array)).toEqual([7.5, 7.5, 7.5]);
    expect(Array.from((await gpu.arange(0, 5)) as Float32Array)).toEqual([0, 1, 2, 3, 4]);
    expect(Array.from((await gpu.arange(10, 0, -2, { dtype: "i32" })) as Int32Array)).toEqual([10, 8, 6, 4, 2]);
    expect(Array.from((await gpu.linspace(0, 1, 5)) as Float32Array)).toEqual([0, 0.25, 0.5, 0.75, 1]);
  });

  it("arange keepOnGpu feeds gather as u32 indices", async () => {
    const idx = (await gpu.arange(0, 3, 1, { dtype: "u32", keepOnGpu: true })) as GPUArray;
    const r = (await gpu.gather([9, 8, 7, 6], idx, { keepOnGpu: false })) as Float32Array;
    expect(Array.from(r)).toEqual([9, 8, 7]);
    idx.destroy();
  });
});

describe("sort options + topK + slice facade (real GPU)", () => {
  it("sort descending", async () => {
    const r = (await gpu.sort([3, -1, 5, 0], { descending: true })) as Float32Array;
    expect(Array.from(r)).toEqual([5, 3, 0, -1]);
  });

  it("sortByKey descending", async () => {
    const [k, v] = (await gpu.sortByKey([1, 3, 2], [10, 30, 20], { descending: true })) as [Float32Array, Float32Array];
    expect(Array.from(k)).toEqual([3, 2, 1]);
    expect(Array.from(v)).toEqual([30, 20, 10]);
  });

  it("topK finds the k largest with valid indices (vector-search shape)", async () => {
    const n = 50_000;
    const scores = new Float32Array(n);
    let s = 123 >>> 0;
    for (let i = 0; i < n; i++) { s = (s * 1664525 + 1013904223) >>> 0; scores[i] = (s >>> 8) / 16777216; }
    scores[41_234] = 2; scores[7] = 3; scores[49_999] = 4;

    const { values, indices } = await gpu.topK(scores, 3);
    expect(Array.from(values as Float32Array)).toEqual([4, 3, 2]);
    expect(Array.from(indices as Uint32Array)).toEqual([49_999, 7, 41_234]);
  });

  it("slice with negative indices", async () => {
    const r = (await gpu.slice([1, 2, 3, 4, 5], -3, -1)) as Float32Array;
    expect(Array.from(r)).toEqual([3, 4]);
  });
});

describe("ifft facade (real GPU)", () => {
  it("fft → ifft round-trips a real signal", async () => {
    const signal = [1, 2, 3, 4, 3, 2, 1, 0];
    const spec = (await gpu.fft(signal)) as Float32Array;
    const back = (await gpu.ifft(spec)) as Float32Array;
    for (let i = 0; i < signal.length; i++) {
      expectClose(back[2 * i], signal[i], { eps: 1e-4 });     // re
      expectClose(back[2 * i + 1], 0, { eps: 1e-4 });          // im
    }
  });
});

describe("new ops fall back to CPU without WebGPU", () => {
  const noGpu = new GPU();
  (noGpu as unknown as { deviceManager: { isAvailable(): boolean } }).deviceManager = {
    isAvailable: () => false,
  } as never;

  it("mean / topK / arange / sort descending run on CPU", async () => {
    expect(await noGpu.mean([1, 2, 3])).toBe(2);
    const { values } = await noGpu.topK([5, 1, 9, 3], 2);
    expect(Array.from(values as Float32Array)).toEqual([9, 5]);
    expect(Array.from((await noGpu.arange(0, 3)) as Float32Array)).toEqual([0, 1, 2]);
    const r = (await noGpu.sort([1, 3, 2], { descending: true })) as Float32Array;
    expect(Array.from(r)).toEqual([3, 2, 1]);
  });
});
