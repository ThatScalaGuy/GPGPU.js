import { describe, it, expect } from "vitest";
import { GPU } from "../../src/gpu";
import { GPUArray } from "../../src/pipeline/gpu-array";

// Regression suite for the resident-input reduce corruption: gpuReduce/gpuArgReduce
// ping-ponged partial results into the caller's GPUArray buffer from the second pass
// on, so any reduction over a resident array longer than one workgroup (256) silently
// destroyed the caller's data. The old tests never saw it — their resident inputs
// were a handful of elements (single pass).
const gpu = new GPU({ fallback: "throw" });

describe("reduce family leaves a resident GPUArray input intact (real GPU)", () => {
  it("sum over 100k resident elements is correct and does not mutate the input", async () => {
    const n = 100_000;
    const ones = new Float32Array(n).fill(1);
    const g = await gpu.upload(ones);

    expect(await gpu.sum(g)).toBe(n);

    const after = await g.toArray();
    expect(after.length).toBe(n);
    // Corruption wrote per-workgroup partials (e.g. 256) into the head of the buffer.
    expect(after[0]).toBe(1);
    expect(after[1]).toBe(1);
    expect(after.every((x) => x === 1)).toBe(true);
    g.destroy();
  });

  it("sum on the same resident array twice returns the same value", async () => {
    const n = 10_000;
    const data = Float32Array.from({ length: n }, (_, i) => (i % 7) + 1);
    const expected = data.reduce((a, b) => a + b, 0);
    const g = await gpu.upload(data);

    expect(await gpu.sum(g)).toBe(expected);
    expect(await gpu.sum(g)).toBe(expected); // second call sees uncorrupted input
    g.destroy();
  });

  it("min/max/product over multi-pass resident input do not mutate it", async () => {
    const n = 1_000;
    const data = Float32Array.from({ length: n }, (_, i) => i + 1);
    const g = await gpu.upload(data);

    expect(await gpu.min(g)).toBe(1);
    expect(await gpu.max(g)).toBe(n);

    const after = await g.toArray();
    expect(Array.from(after.slice(0, 4))).toEqual([1, 2, 3, 4]);
    expect(after[n - 1]).toBe(n);
    g.destroy();
  });

  it("argmax over multi-pass resident input is correct and does not mutate it", async () => {
    const n = 1_000;
    const data = Float32Array.from({ length: n }, (_, i) => i % 100);
    data[637] = 999; // unique maximum
    const g = await gpu.upload(data);

    expect(await gpu.argmax(g)).toBe(637);

    const after = await g.toArray();
    expect(after[0]).toBe(0);
    expect(after[637]).toBe(999);
    g.destroy();
  });
});

describe("GPUArray dtype guard (real GPU)", () => {
  it("rejects an f32 GPUArray where u32 indices are expected", async () => {
    const src = [10, 20, 30];
    const idx = await gpu.upload([0, 2]); // f32 — would be bit-reinterpreted as u32
    await expect(gpu.gather(src, idx)).rejects.toThrow(/dtype mismatch/);
    idx.destroy();
  });

  it("rejects a mixed CPU-f32 / GPUArray-i32 elementwise pair", async () => {
    const gi = (await gpu.cast([1, 2, 3], "i32", { keepOnGpu: true })) as GPUArray;
    await expect(gpu.add([1, 2, 3], gi)).rejects.toThrow(/dtype mismatch/);
    gi.destroy();
  });
});
