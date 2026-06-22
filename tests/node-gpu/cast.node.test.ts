import { describe, it, expect } from "vitest";
import { gpuCast } from "../../src/ops/cast";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";

// Drive the GPU cast implementation directly (no CPU fallback wrapper), so a broken
// cast shader fails loudly instead of silently passing through the fallback.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

describe("cast (real GPU)", () => {
  it("f32 -> i32 truncates toward zero", async () => {
    const input = new Float32Array([1.9, 2.1, -3.7, 0.5, -0.5]);
    const out = (await gpuCast(...args, input, "i32")) as Int32Array;
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([1, 2, -3, 0, 0]);
  });

  it("i32 -> f32 round-trips", async () => {
    const input = new Int32Array([-5, 0, 7, 42]);
    const out = (await gpuCast(...args, input, "f32")) as Float32Array;
    expect(out).toBeInstanceOf(Float32Array);
    expect(Array.from(out)).toEqual([-5, 0, 7, 42]);
  });

  it("u32 -> f32 round-trips", async () => {
    const input = new Uint32Array([0, 1, 1000, 65535]);
    const out = (await gpuCast(...args, input, "f32")) as Float32Array;
    expect(out).toBeInstanceOf(Float32Array);
    expect(Array.from(out)).toEqual([0, 1, 1000, 65535]);
  });

  it("f32 -> u32 (non-negative) round-trips", async () => {
    const input = new Float32Array([0, 1.9, 2.1, 100.7]);
    const out = (await gpuCast(...args, input, "u32")) as Uint32Array;
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([0, 1, 2, 100]);
  });

  it("i32 -> u32 (non-negative) round-trips", async () => {
    const input = new Int32Array([0, 1, 2, 123456]);
    const out = (await gpuCast(...args, input, "u32")) as Uint32Array;
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([0, 1, 2, 123456]);
  });

  it("keepOnGpu returns a GPUArray with the new dtype", async () => {
    const input = new Float32Array([1.5, 2.5, 3.5]);
    const out = await gpuCast(...args, input, "i32", { keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.dtype).toBe("i32");
    expect(out.length).toBe(3);
    const arr = await out.toArray();
    expect(arr).toBeInstanceOf(Int32Array);
    expect(Array.from(arr)).toEqual([1, 2, 3]);
    out.destroy();
  });

  it("casts a multi-block array (length > 64)", async () => {
    const n = 100;
    const input = new Float32Array(n);
    for (let i = 0; i < n; i++) input[i] = i + 0.7;
    const out = (await gpuCast(...args, input, "i32")) as Int32Array;
    expect(out.length).toBe(n);
    for (let i = 0; i < n; i++) expect(out[i]).toBe(i);
  });
});
