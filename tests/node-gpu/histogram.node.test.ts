import { describe, it, expect } from "vitest";
import { gpuHistogram } from "../../src/ops/histogram";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";

// Drive the GPU histogram implementation directly (no CPU fallback wrapper), so a broken
// histogram shader / atomics path fails the test loudly instead of silently passing through
// the fallback. Counts are integers, so every assertion uses exact toEqual.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

describe("histogram (real GPU)", () => {
  it("even distribution: 0..9 into 5 bins over [0,10]", async () => {
    const input = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const out = (await gpuHistogram(...args, input, { bins: 5, min: 0, max: 10 })) as Uint32Array;
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([2, 2, 2, 2, 2]);
  });

  it("value == max clamps to the last bin", async () => {
    const input = new Float32Array([10]);
    const out = (await gpuHistogram(...args, input, { bins: 5, min: 0, max: 10 })) as Uint32Array;
    expect(Array.from(out)).toEqual([0, 0, 0, 0, 1]);
  });

  it("out-of-range values clamp to the edge bins", async () => {
    const input = new Float32Array([-5, 15]);
    const out = (await gpuHistogram(...args, input, { bins: 5, min: 0, max: 10 })) as Uint32Array;
    expect(Array.from(out)).toEqual([1, 0, 0, 0, 1]);
  });

  it("bins=1: every element lands in the single bin", async () => {
    const input = new Float32Array([-3, 0, 4, 100, 7]);
    const out = (await gpuHistogram(...args, input, { bins: 1, min: 0, max: 10 })) as Uint32Array;
    expect(Array.from(out)).toEqual([input.length]);
  });

  it("max == min guard: everything falls in bin 0", async () => {
    const input = new Float32Array([3, 3, 3]);
    const out = (await gpuHistogram(...args, input, { bins: 4, min: 3, max: 3 })) as Uint32Array;
    expect(Array.from(out)).toEqual([3, 0, 0, 0]);
  });

  it("i32 input buckets correctly", async () => {
    const input = new Int32Array([-2, 0, 2, 4, 6, 8, 9]);
    const out = (await gpuHistogram(...args, input, { bins: 5, min: 0, max: 10 })) as Uint32Array;
    expect(out).toBeInstanceOf(Uint32Array);
    // -2 -> bin 0; 0 -> 0; 2 -> 1; 4 -> 2; 6 -> 3; 8 -> 4; 9 -> 4
    expect(Array.from(out)).toEqual([2, 1, 1, 1, 2]);
  });

  it("u32 input buckets correctly", async () => {
    const input = new Uint32Array([0, 1, 2, 5, 9, 9]);
    const out = (await gpuHistogram(...args, input, { bins: 5, min: 0, max: 10 })) as Uint32Array;
    expect(out).toBeInstanceOf(Uint32Array);
    // 0 -> 0; 1 -> 0; 2 -> 1; 5 -> 2; 9 -> 4; 9 -> 4
    expect(Array.from(out)).toEqual([2, 1, 1, 0, 2]);
  });

  it("keepOnGpu returns a GPUArray of u32 counts with length === bins", async () => {
    const input = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const out = await gpuHistogram(...args, input, { bins: 5, min: 0, max: 10, keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.dtype).toBe("u32");
    expect(out.length).toBe(5);
    const arr = await out.toArray();
    expect(Array.from(arr)).toEqual([2, 2, 2, 2, 2]);
    out.destroy();
  });

  // Large input over many workgroups with heavy atomic contention (each value maps to one
  // of 7 bins). Assert the exact per-bin counts and that they sum to the input length.
  it("multi-block: 10000 elements with atomic contention sum exactly", async () => {
    const n = 10000;
    const bins = 7;
    const input = new Uint32Array(n);
    for (let i = 0; i < n; i++) input[i] = i % 7;
    const out = (await gpuHistogram(...args, input, { bins, min: 0, max: 7 })) as Uint32Array;

    const expected = new Array(bins).fill(0);
    for (let i = 0; i < n; i++) expected[i % 7]++;

    expect(Array.from(out)).toEqual(expected);
    expect(Array.from(out).reduce((a, b) => a + b, 0)).toEqual(n);
  });

  it("non-zero min: offset binning over [2,8]", async () => {
    const input = new Float32Array([2, 3, 5, 8]);
    const out = (await gpuHistogram(...args, input, { bins: 3, min: 2, max: 8 })) as Uint32Array;
    // (x-2)/6*3: 2 -> 0, 3 -> 0, 5 -> 1, 8 (== max) clamps to 2
    expect(Array.from(out)).toEqual([2, 1, 1]);
  });

  // Re-running the same histogram must not accumulate: the first call reads back and releases
  // its counter buffer to the pool, the second reuses that same bucket. A correct second result
  // proves the atomic buffer is re-zeroed on every call (guards the stale-bytes pooling risk).
  it("re-zeroes the counter buffer across pooled reuse", async () => {
    const input = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const first = (await gpuHistogram(...args, input, { bins: 5, min: 0, max: 10 })) as Uint32Array;
    const second = (await gpuHistogram(...args, input, { bins: 5, min: 0, max: 10 })) as Uint32Array;
    expect(Array.from(first)).toEqual([2, 2, 2, 2, 2]);
    expect(Array.from(second)).toEqual(Array.from(first));
  });
});
