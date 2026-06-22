import { describe, it, expect } from "vitest";
import { gpuUnique, cpuUnique } from "../../src/ops/unique";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";
import { expectClose } from "../shared/tolerance";

// Drive the GPU unique (sort -> boundary flags -> scan -> compaction) directly, with no CPU
// fallback wrapper, so a broken pass fails loudly instead of silently passing through the
// fallback. unique returns SORTED distinct values; values are copied verbatim, so the GPU and
// CPU paths agree exactly for ints AND f32 (no floating-point reassociation).
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

describe("unique (real GPU)", () => {
  it("returns sorted distinct values of an unsorted input", async () => {
    const input = new Float32Array([3, 1, 2, 3, 1, 2, 3]);
    const out = (await gpuUnique(...args, input)) as Float32Array;
    expect(Array.from(out)).toEqual([1, 2, 3]);
  });

  it("all-distinct input returns the sorted input (same length)", async () => {
    const input = new Float32Array([5, 4, 3, 2, 1]);
    const out = (await gpuUnique(...args, input)) as Float32Array;
    expect(out.length).toBe(5);
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5]);
  });

  it("all-equal input collapses to a single element", async () => {
    const input = new Float32Array([7, 7, 7, 7, 7, 7]);
    const out = (await gpuUnique(...args, input)) as Float32Array;
    expect(Array.from(out)).toEqual([7]);
  });

  it("single element", async () => {
    const out = (await gpuUnique(...args, new Float32Array([42]))) as Float32Array;
    expect(Array.from(out)).toEqual([42]);
  });

  it("empty input returns a length-0 result", async () => {
    const out = (await gpuUnique(...args, new Float32Array([]))) as Float32Array;
    expect(out.length).toBe(0);
  });

  it("preserves i32 dtype (incl. negatives and duplicates)", async () => {
    const input = new Int32Array([3, -5, 0, -1, 2, -5, 0, 3]);
    const out = (await gpuUnique(...args, input)) as Int32Array;
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([-5, -1, 0, 2, 3]);
  });

  it("preserves u32 dtype", async () => {
    const input = new Uint32Array([10, 5, 10, 5, 20, 5]);
    const out = (await gpuUnique(...args, input)) as Uint32Array;
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([5, 10, 20]);
  });

  it("dedupes f32 values exactly (bit-identical duplicates)", async () => {
    const input = new Float32Array([1.5, 0.25, 1.5, 0.25, 2.75]);
    const out = (await gpuUnique(...args, input)) as Float32Array;
    expect(Array.from(out)).toEqual([0.25, 1.5, 2.75]);
  });

  it("keepOnGpu returns a GPUArray of length === unique count", async () => {
    const input = new Float32Array([3, 1, 2, 3, 1, 2]);
    const out = await gpuUnique(...args, input, { keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.length).toBe(3);
    expect(out.dtype).toBe("f32");
    expect(Array.from(await out.toArray())).toEqual([1, 2, 3]);
    out.destroy();
  });

  it("does not mutate a GPUArray input", async () => {
    const device = await deviceManager.getDevice();
    const data = new Float32Array([3, 1, 2, 3, 1, 2]);
    const buf = bufferPool.acquire(
      device,
      data.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    device.queue.writeBuffer(buf, 0, data.buffer);
    const gpuInput = new GPUArray(buf, data.length, "f32", device, bufferPool);

    const out = (await gpuUnique(...args, gpuInput)) as Float32Array;
    expect(Array.from(out)).toEqual([1, 2, 3]);

    // The input GPUArray must be untouched (sort copies it, flags write their own buffer).
    expect(Array.from(await gpuInput.toArray())).toEqual([3, 1, 2, 3, 1, 2]);
    gpuInput.destroy();
  });

  // Size sweep across block boundaries (workgroup 64, scan multi-block, bitonic power-of-2
  // padding). Each size mixes ~half duplicates so the count is data-dependent and crosses
  // block edges. Exact equality for ints; floats are copied verbatim so they match exactly too,
  // but assert with expectClose per the float-comparison convention.
  const sizes = [1, 63, 64, 65, 127, 128, 129, 1000, 4096, 65536, 1_000_000];
  for (const size of sizes) {
    it(`matches cpuUnique for size ${size} (i32)`, async () => {
      const input = new Int32Array(size);
      // Values in [0, size/2) so roughly half are duplicates; unsorted via a stride pattern.
      const mod = Math.max(1, Math.floor(size / 2));
      for (let i = 0; i < size; i++) input[i] = ((i * 2654435761) >>> 0) % mod;
      const expected = cpuUnique(input);
      const out = (await gpuUnique(...args, input)) as Int32Array;
      expect(out).toBeInstanceOf(Int32Array);
      expect(out.length).toBe(expected.length);
      expect(Array.from(out)).toEqual(Array.from(expected));
    });

    it(`matches cpuUnique for size ${size} (f32)`, async () => {
      const input = new Float32Array(size);
      const mod = Math.max(1, Math.floor(size / 2));
      for (let i = 0; i < size; i++) input[i] = (((i * 40503) >>> 0) % mod) * 0.5;
      const expected = cpuUnique(input);
      const out = (await gpuUnique(...args, input)) as Float32Array;
      expect(out.length).toBe(expected.length);
      for (let i = 0; i < expected.length; i++) expectClose(out[i], expected[i]);
    });
  }
});
