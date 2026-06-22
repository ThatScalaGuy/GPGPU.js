import { describe, it, expect } from "vitest";
import { gpuGather } from "../../src/ops/gather";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";

// Drive the GPU gather implementation directly (no CPU fallback wrapper), so a broken
// gather shader fails the test loudly instead of silently passing through the fallback.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

describe("gather (real GPU)", () => {
  it("looks up src by idx", async () => {
    const src = new Float32Array([10, 20, 30, 40, 50]);
    const idx = new Uint32Array([4, 0, 2]);
    const out = (await gpuGather(...args, src, idx)) as Float32Array;
    expect(Array.from(out)).toEqual([50, 10, 30]);
  });

  it("output length follows idx, not src", async () => {
    const src = new Float32Array([1, 2, 3]);
    const idx = new Uint32Array([0, 0, 1, 2, 2, 2]);
    const out = (await gpuGather(...args, src, idx)) as Float32Array;
    expect(Array.from(out)).toEqual([1, 1, 2, 3, 3, 3]);
  });

  it("realises a permutation", async () => {
    const src = new Float32Array([0, 1, 2, 3, 4]);
    const idx = new Uint32Array([2, 4, 0, 3, 1]);
    const out = (await gpuGather(...args, src, idx)) as Float32Array;
    expect(Array.from(out)).toEqual([2, 4, 0, 3, 1]);
  });

  // An out-of-range index clamps to the last src element (the shader can't throw).
  it("clamps an out-of-range index to the last element", async () => {
    const src = new Float32Array([7, 8, 9]);
    const idx = new Uint32Array([0, 99, 2]);
    const out = (await gpuGather(...args, src, idx)) as Float32Array;
    expect(Array.from(out)).toEqual([7, 9, 9]);
  });

  it("output dtype follows src dtype (i32)", async () => {
    const src = new Int32Array([-5, -6, -7]);
    const idx = new Uint32Array([2, 1, 0]);
    const out = (await gpuGather(...args, src, idx)) as Int32Array;
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([-7, -6, -5]);
  });

  it("keepOnGpu returns a GPUArray of the gathered values", async () => {
    const src = new Float32Array([100, 200, 300]);
    const idx = new Uint32Array([2, 1, 0, 0]);
    const out = await gpuGather(...args, src, idx, { keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.length).toBe(4);
    expect(out.dtype).toBe("f32");
    const arr = await out.toArray();
    expect(Array.from(arr)).toEqual([300, 200, 100, 100]);
    out.destroy();
  });
});
