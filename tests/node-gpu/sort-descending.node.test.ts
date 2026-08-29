import { describe, it, expect } from "vitest";
import { gpuSort } from "../../src/ops/sort";
import { gpuSortByKey } from "../../src/ops/sort-by-key";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";

// Drive the GPU sorts directly (no CPU fallback wrapper), so a broken descending shader or
// pad sentinel fails the test loudly. Reference for descending output: an ascending CPU sort
// reversed — sorting is exact (no float reassociation), so element-wise exact equality holds.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

/** Ascending CPU sort, reversed — the exact reference for a descending sort. */
function descRef(data: ArrayLike<number>): number[] {
  return Array.from(data)
    .sort((a, b) => a - b)
    .reverse();
}

/** Deterministic PRNG so the big cases are reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function expectExactlyEqual(got: ArrayLike<number>, want: number[]): void {
  expect(got.length).toBe(want.length);
  let mismatches = 0;
  for (let i = 0; i < want.length; i++) {
    if (got[i] !== want[i]) mismatches++;
  }
  expect(mismatches).toBe(0);
}

describe("sort descending (real GPU)", () => {
  it("f32 with duplicates and negatives, length 1000 (non-power-of-2, multi-block)", async () => {
    const rand = mulberry32(42);
    const n = 1000;
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      // duplicates (coarse rounding) and negatives; fround-exact f32 values
      data[i] = Math.fround(Math.round((rand() - 0.5) * 200) / 4);
    }
    const out = (await gpuSort(...args, data, { descending: true })) as Float32Array;
    expectExactlyEqual(out, descRef(data));
  });

  it("f32 with duplicates and negatives, length 50000", async () => {
    const rand = mulberry32(1337);
    const n = 50000;
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      data[i] = Math.fround((rand() - 0.5) * 1e6);
    }
    const out = (await gpuSort(...args, data, { descending: true })) as Float32Array;
    expectExactlyEqual(out, descRef(data));
  });

  it("i32 descending (incl. negatives and INT32_MIN/MAX)", async () => {
    const data = new Int32Array([5, -3, 2147483647, 0, -2147483648, 7, -3, 100]);
    const out = (await gpuSort(...args, data, { descending: true })) as Int32Array;
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual(descRef(data));
  });

  it("u32 descending (incl. 0 and UINT32_MAX; pad sentinel 0 must not leak)", async () => {
    // length 6 pads to 8 with sentinel 0 in descending mode: the two pad lanes must be
    // trimmed, leaving exactly one real 0 at the end.
    const data = new Uint32Array([4294967295, 0, 17, 1, 4294967295, 3]);
    const out = (await gpuSort(...args, data, { descending: true })) as Uint32Array;
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([4294967295, 4294967295, 17, 3, 1, 0]);
  });

  it("non-power-of-2 length 5: pad lanes leak into neither end", async () => {
    const data = new Float32Array([-1.5, 3, -1.5, 0.25, 2]);
    const out = (await gpuSort(...args, data, { descending: true })) as Float32Array;
    expect(Array.from(out)).toEqual([3, 2, 0.25, -1.5, -1.5]);
  });

  it("descending and ascending pipelines do not collide in the shader cache", async () => {
    const data = new Float32Array([2, -7, 5, 0]);
    // ascending -> descending -> ascending on the same ShaderCache instance
    const asc1 = (await gpuSort(...args, data)) as Float32Array;
    expect(Array.from(asc1)).toEqual([-7, 0, 2, 5]);
    const desc = (await gpuSort(...args, data, { descending: true })) as Float32Array;
    expect(Array.from(desc)).toEqual([5, 2, 0, -7]);
    const asc2 = (await gpuSort(...args, data, { descending: false })) as Float32Array;
    expect(Array.from(asc2)).toEqual([-7, 0, 2, 5]);
  });

  it("length 1 and empty input", async () => {
    const one = (await gpuSort(...args, new Float32Array([42]), { descending: true })) as Float32Array;
    expect(Array.from(one)).toEqual([42]);
    const none = (await gpuSort(...args, new Float32Array([]), { descending: true })) as Float32Array;
    expect(none.length).toBe(0);
  });

  it("keepOnGpu returns a GPUArray that reads back descending", async () => {
    const data = new Float32Array([1, 4, -2, 9, 0]);
    const out = await gpuSort(...args, data, { descending: true, keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.length).toBe(5);
    expect(out.dtype).toBe("f32");
    expect(Array.from(await out.toArray())).toEqual([9, 4, 1, 0, -2]);
    out.destroy();
  });

  it("accepts a GPUArray input and does not mutate it", async () => {
    const device = await deviceManager.getDevice();
    const { uploadBuffer } = await import("../../src/core/command");
    const src = new Float32Array([3, -1, 2, 8, -5]);
    const input = new GPUArray(
      uploadBuffer(device, src, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, bufferPool),
      src.length,
      "f32",
      device,
      bufferPool
    );
    const out = (await gpuSort(...args, input, { descending: true })) as Float32Array;
    expect(Array.from(out)).toEqual([8, 3, 2, -1, -5]);
    // input untouched
    expect(Array.from(await input.toArray())).toEqual([3, -1, 2, 8, -5]);
    input.destroy();
  });
});

describe("sortByKey descending (real GPU)", () => {
  // DISTINCT keys everywhere we assert value placement — bitonic sort is not stable.
  it("keys sort descending and each key keeps its original value (position map)", async () => {
    const rand = mulberry32(7);
    const n = 500; // non-power-of-2, multi-block
    const keys = new Float32Array(n);
    const values = new Uint32Array(n);
    const perm: number[] = [];
    for (let i = 0; i < n; i++) perm.push(i);
    // distinct keys via a shuffled range
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [perm[i], perm[j]] = [perm[j], perm[i]];
    }
    const pair = new Map<number, number>();
    for (let i = 0; i < n; i++) {
      keys[i] = Math.fround(perm[i] * 0.5 - 100);
      values[i] = perm[i] * 3 + 1;
      pair.set(keys[i], values[i]);
    }
    const [k, v] = (await gpuSortByKey(...args, keys, values, { descending: true })) as [
      Float32Array,
      Uint32Array,
    ];
    expectExactlyEqual(k, descRef(keys));
    for (let i = 0; i < n; i++) {
      expect(v[i]).toBe(pair.get(k[i]));
    }
  });

  it("i32 keys descending with negatives, values follow", async () => {
    const keys = new Int32Array([3, -5, 0, -1, 2]);
    const values = new Float32Array([30, -50, 0, -10, 20]); // value = key * 10
    const [k, v] = (await gpuSortByKey(...args, keys, values, { descending: true })) as [
      Int32Array,
      Float32Array,
    ];
    expect(Array.from(k)).toEqual([3, 2, 0, -1, -5]);
    expect(Array.from(v)).toEqual([30, 20, 0, -10, -50]);
  });

  // Keys deliberately avoid 0: in descending mode the pad sentinel IS 0, so a leaked pad
  // lane would surface as a key 0 in the output. (A real key equal to the sentinel behaves
  // like any duplicate key — same as u32 MAX in ascending mode — so we keep keys distinct
  // from it here.)
  it("u32 keys descending, non-power-of-2: pad key 0 leaks into neither end", async () => {
    const keys = new Uint32Array([10, 3, 30, 20, 5]);
    const values = new Uint32Array([100, 33, 300, 200, 50]);
    const [k, v] = (await gpuSortByKey(...args, keys, values, { descending: true })) as [
      Uint32Array,
      Uint32Array,
    ];
    expect(Array.from(k)).toEqual([30, 20, 10, 5, 3]);
    expect(Array.from(v)).toEqual([300, 200, 100, 50, 33]);
  });

  it("keepOnGpu round-trip and GPUArray inputs not mutated", async () => {
    const device = await deviceManager.getDevice();
    const { uploadBuffer } = await import("../../src/core/command");
    const mk = (data: Float32Array | Uint32Array, dtype: "f32" | "u32") =>
      new GPUArray(
        uploadBuffer(device, data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, bufferPool),
        data.length,
        dtype,
        device,
        bufferPool
      );
    const keys = mk(new Float32Array([3, 1, 2]), "f32");
    const values = mk(new Uint32Array([30, 10, 20]), "u32");

    const [k, v] = await gpuSortByKey(...args, keys, values, {
      descending: true,
      keepOnGpu: true,
    });
    expect(k).toBeInstanceOf(GPUArray);
    expect(v).toBeInstanceOf(GPUArray);
    expect(Array.from(await k.toArray())).toEqual([3, 2, 1]);
    expect(Array.from(await v.toArray())).toEqual([30, 20, 10]);
    // inputs untouched
    expect(Array.from(await keys.toArray())).toEqual([3, 1, 2]);
    expect(Array.from(await values.toArray())).toEqual([30, 10, 20]);
    keys.destroy();
    values.destroy();
    k.destroy();
    v.destroy();
  });

  it("default stays ascending after a descending call (cache key separation)", async () => {
    const keys = new Float32Array([3, 1, 2]);
    const values = new Float32Array([30, 10, 20]);
    await gpuSortByKey(...args, keys, values, { descending: true });
    const [k, v] = (await gpuSortByKey(...args, keys, values)) as [Float32Array, Float32Array];
    expect(Array.from(k)).toEqual([1, 2, 3]);
    expect(Array.from(v)).toEqual([10, 20, 30]);
  });

  it("length 1", async () => {
    const [k, v] = (await gpuSortByKey(...args, new Float32Array([5]), new Float32Array([50]), {
      descending: true,
    })) as [Float32Array, Float32Array];
    expect(Array.from(k)).toEqual([5]);
    expect(Array.from(v)).toEqual([50]);
  });
});
