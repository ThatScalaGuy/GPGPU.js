import { describe, it, expect } from "vitest";
import { gpuSortByKey } from "../../src/ops/sort-by-key";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";

// Drive the GPU sort-by-key directly (no CPU fallback wrapper), so a broken bitonic-by-key
// shader / padding path fails the test loudly instead of silently passing through the
// fallback. Keys for any test that asserts an exact value order are DISTINCT — bitonic sort
// is not stable, so equal keys leave their values in an unspecified relative order.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

describe("sortByKey (real GPU)", () => {
  it("basic: sorts keys ascending and permutes values to follow", async () => {
    const keys = new Float32Array([3, 1, 2]);
    const values = new Float32Array([30, 10, 20]);
    const [k, v] = (await gpuSortByKey(...args, keys, values)) as [Float32Array, Float32Array];
    expect(Array.from(k)).toEqual([1, 2, 3]);
    expect(Array.from(v)).toEqual([10, 20, 30]);
  });

  it("reverse permutation: fully reversed input sorts correctly", async () => {
    const keys = new Float32Array([5, 4, 3, 2, 1]);
    const values = new Float32Array([50, 40, 30, 20, 10]);
    const [k, v] = (await gpuSortByKey(...args, keys, values)) as [Float32Array, Float32Array];
    expect(Array.from(k)).toEqual([1, 2, 3, 4, 5]);
    expect(Array.from(v)).toEqual([10, 20, 30, 40, 50]);
  });

  it("independent value dtype: f32 keys carry u32 payload ids", async () => {
    const keys = new Float32Array([2.5, 0.5, 1.5]);
    const values = new Uint32Array([200, 0, 100]);
    const [k, v] = (await gpuSortByKey(...args, keys, values)) as [Float32Array, Uint32Array];
    expect(k).toBeInstanceOf(Float32Array);
    expect(v).toBeInstanceOf(Uint32Array);
    expect(Array.from(k)).toEqual([0.5, 1.5, 2.5]);
    // payloads follow their keys: 0 -> key 0.5, 100 -> key 1.5, 200 -> key 2.5
    expect(Array.from(v)).toEqual([0, 100, 200]);
  });

  it("i32 keys (incl. negatives) sort correctly with values following", async () => {
    const keys = new Int32Array([3, -5, 0, -1, 2]);
    const values = new Float32Array([3, -5, 0, -1, 2]); // value == key for an easy check
    const [k, v] = (await gpuSortByKey(...args, keys, values)) as [Int32Array, Float32Array];
    expect(k).toBeInstanceOf(Int32Array);
    expect(Array.from(k)).toEqual([-5, -1, 0, 2, 3]);
    expect(Array.from(v)).toEqual([-5, -1, 0, 2, 3]);
  });

  it("non-power-of-2 length 5: padding does not disturb the result", async () => {
    const keys = new Float32Array([40, 10, 50, 20, 30]);
    const values = new Float32Array([4, 1, 5, 2, 3]);
    const [k, v] = (await gpuSortByKey(...args, keys, values)) as [Float32Array, Float32Array];
    expect(Array.from(k)).toEqual([10, 20, 30, 40, 50]);
    expect(Array.from(v)).toEqual([1, 2, 3, 4, 5]);
  });

  it("non-power-of-2 length 6: padding does not disturb the result", async () => {
    const keys = new Float32Array([6, 5, 4, 3, 2, 1]);
    const values = new Float32Array([60, 50, 40, 30, 20, 10]);
    const [k, v] = (await gpuSortByKey(...args, keys, values)) as [Float32Array, Float32Array];
    expect(Array.from(k)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(Array.from(v)).toEqual([10, 20, 30, 40, 50, 60]);
  });

  it("keepOnGpu returns [GPUArray, GPUArray] of length N", async () => {
    const keys = new Float32Array([3, 1, 2]);
    const values = new Uint32Array([30, 10, 20]);
    const [k, v] = await gpuSortByKey(...args, keys, values, { keepOnGpu: true });
    expect(k).toBeInstanceOf(GPUArray);
    expect(v).toBeInstanceOf(GPUArray);
    expect(k.length).toBe(3);
    expect(v.length).toBe(3);
    expect(k.dtype).toBe("f32");
    expect(v.dtype).toBe("u32");
    expect(Array.from(await k.toArray())).toEqual([1, 2, 3]);
    expect(Array.from(await v.toArray())).toEqual([10, 20, 30]);
    k.destroy();
    v.destroy();
  });

  // Length well over the 64-wide workgroup and not on a single bitonic block, with DISTINCT
  // keys so the value placement is deterministic: values[i] must equal keys[i]*10 after sort.
  it("multi-block, distinct keys (length 1000)", async () => {
    const n = 1000;
    const keys = new Float32Array(n);
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const key = n - 1 - i; // reversed range 999..0, all distinct
      keys[i] = key;
      values[i] = key * 10;
    }
    const [k, v] = (await gpuSortByKey(...args, keys, values)) as [Float32Array, Float32Array];
    for (let i = 0; i < n; i++) {
      expect(k[i]).toBe(i);
      expect(v[i]).toBe(i * 10);
    }
  });

  it("accepts GPUArray inputs for keys and values", async () => {
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

    const [k, v] = await gpuSortByKey(...args, keys, values, { keepOnGpu: true });
    expect(k).toBeInstanceOf(GPUArray);
    expect(v).toBeInstanceOf(GPUArray);
    expect(Array.from(await k.toArray())).toEqual([1, 2, 3]);
    expect(Array.from(await v.toArray())).toEqual([10, 20, 30]);
    // inputs are reused in place, not consumed
    expect(Array.from(await keys.toArray())).toEqual([3, 1, 2]);
    expect(Array.from(await values.toArray())).toEqual([30, 10, 20]);
    keys.destroy();
    values.destroy();
    k.destroy();
    v.destroy();
  });

  // NOT stable: with duplicate keys the per-duplicate value order is unspecified. Assert ONLY
  // (a) keys are sorted ascending and (b) the multiset of values is preserved — never a
  // specific value at a specific position among equal keys.
  it("duplicate keys: keys sorted and value multiset preserved (order unspecified)", async () => {
    const keys = new Float32Array([1, 2, 1, 2, 1, 2]);
    const values = new Float32Array([10, 20, 11, 21, 12, 22]);
    const [k, v] = (await gpuSortByKey(...args, keys, values)) as [Float32Array, Float32Array];
    // keys ascending
    for (let i = 1; i < k.length; i++) expect(k[i]).toBeGreaterThanOrEqual(k[i - 1]);
    expect(Array.from(k)).toEqual([1, 1, 1, 2, 2, 2]);
    // value multiset unchanged (sorted copies match) — does NOT assume any per-key order
    expect(Array.from(v).sort((a, b) => a - b)).toEqual([10, 11, 12, 20, 21, 22]);
  });
});
