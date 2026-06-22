import { describe, it, expect } from "vitest";
import { gpuFilter } from "../../src/ops/filter";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";

// Drive the GPU filter (stream compaction) directly — no CPU fallback wrapper — so a broken
// flags/scan/compaction pass fails loudly instead of silently passing through the fallback.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

describe("filter (real GPU)", () => {
  it("keeps elements matching an arrow predicate", async () => {
    const input = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = (await gpuFilter(...args, input, (x) => x > 3)) as Float32Array;
    expect(Array.from(out)).toEqual([4, 5, 6]);
  });

  it("accepts a string predicate", async () => {
    const input = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = (await gpuFilter(...args, input, "x % 2 == 0")) as Float32Array;
    expect(Array.from(out)).toEqual([2, 4, 6]);
  });

  it("predicate can use the element index i", async () => {
    const input = new Float32Array([10, 20, 30, 40]);
    const out = (await gpuFilter(...args, input, (x, i) => i % 2 == 0)) as Float32Array;
    expect(Array.from(out)).toEqual([10, 30]);
  });

  it("predicate can use the length len", async () => {
    const input = new Float32Array([0, 1, 2, 3, 4, 5]);
    const out = (await gpuFilter(...args, input, (x, i, len) => i < len / 2)) as Float32Array;
    expect(Array.from(out)).toEqual([0, 1, 2]);
  });

  it("all kept → result equals input (same length)", async () => {
    const input = new Float32Array([1, 2, 3, 4, 5]);
    const out = (await gpuFilter(...args, input, (x) => x > 0)) as Float32Array;
    expect(out.length).toBe(input.length);
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5]);
  });

  it("none kept → length-0 result", async () => {
    const input = new Float32Array([1, 2, 3, 4, 5]);
    const out = (await gpuFilter(...args, input, (x) => x > 100)) as Float32Array;
    expect(out.length).toBe(0);
  });

  it("filters an i32 input", async () => {
    const input = new Int32Array([-3, -2, -1, 0, 1, 2, 3]);
    const out = (await gpuFilter(...args, input, (x) => x >= 0)) as Int32Array;
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([0, 1, 2, 3]);
  });

  it("filters a u32 input", async () => {
    const input = new Uint32Array([1, 2, 3, 4, 5, 6]);
    const out = (await gpuFilter(...args, input, "x % 3 == 0")) as Uint32Array;
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([3, 6]);
  });

  it("keepOnGpu returns a GPUArray of length === count", async () => {
    const input = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = await gpuFilter(...args, input, (x) => x > 3, { keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.length).toBe(3);
    expect(out.dtype).toBe("f32");
    const arr = await out.toArray();
    expect(Array.from(arr)).toEqual([4, 5, 6]);
    out.destroy();
  });

  it("multi-block: length 1000, x % 2 == 0", async () => {
    const input = new Float32Array(1000);
    for (let i = 0; i < input.length; i++) input[i] = i;
    const expected = Array.from(input).filter((x) => x % 2 === 0);
    const out = (await gpuFilter(...args, input, "x % 2 == 0")) as Float32Array;
    expect(out.length).toBe(expected.length);
    expect(Array.from(out)).toEqual(expected);
  });

  it("does not mutate a GPUArray input", async () => {
    const device = await deviceManager.getDevice();
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const buf = bufferPool.acquire(
      device,
      data.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    device.queue.writeBuffer(buf, 0, data.buffer);
    const gpuInput = new GPUArray(buf, data.length, "f32", device, bufferPool);

    const out = (await gpuFilter(...args, gpuInput, (x) => x > 3)) as Float32Array;
    expect(Array.from(out)).toEqual([4, 5, 6]);

    // The input GPUArray must be untouched by the filter.
    const after = await gpuInput.toArray();
    expect(Array.from(after)).toEqual([1, 2, 3, 4, 5, 6]);
    gpuInput.destroy();
  });

  // Compound boolean predicates: the emitter lowers && / || to WGSL bitwise & / | on bool
  // operands. Guard that codegen path (documented but otherwise untested here).
  it("supports a compound && predicate (f32)", async () => {
    const input = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    const out = (await gpuFilter(...args, input, (x) => x > 2 && x < 7)) as Float32Array;
    expect(Array.from(out)).toEqual([3, 4, 5, 6]);
  });

  it("supports a compound || predicate (i32)", async () => {
    const input = new Int32Array([-3, -2, -1, 0, 1, 2, 3]);
    const out = (await gpuFilter(...args, input, "x < -1 || x > 1")) as Int32Array;
    expect(Array.from(out)).toEqual([-3, -2, 2, 3]);
  });
});
