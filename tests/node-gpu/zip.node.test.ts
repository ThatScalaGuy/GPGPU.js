import { describe, it, expect } from "vitest";
import { gpuZip } from "../../src/ops/elementwise";
import { GPU } from "../../src/gpu";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { uploadBuffer } from "../../src/core/command";

// Drive gpuZip directly (no CPU fallback) so a broken zip shader fails loudly instead of
// silently falling back to CPU and matching by coincidence.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

function gpuArrayFrom(
  device: GPUDevice,
  data: Float32Array | Int32Array | Uint32Array,
  dtype: "f32" | "i32" | "u32"
): GPUArray {
  const buffer = uploadBuffer(device, data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, bufferPool);
  return new GPUArray(buffer, data.length, dtype, device, bufferPool);
}

describe("zip (real GPU)", () => {
  it("computes complex magnitude in one dispatch", async () => {
    const re = new Float32Array([3, 6, 5, 0]);
    const im = new Float32Array([4, 8, 12, 7]);
    const out = await gpuZip(...args, re, im, (r, i) => Math.sqrt(r * r + i * i));
    expect(out).toBeInstanceOf(Float32Array);
    const expected = [5, 10, 13, 7];
    for (let k = 0; k < expected.length; k++) expect(out[k]).toBeCloseTo(expected[k], 4);
  });

  it("works on integer inputs", async () => {
    const out = await gpuZip(...args, new Int32Array([1, 2, 3]), new Int32Array([10, 20, 30]), (a, b) => a + b);
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([11, 22, 33]);
  });

  it("throws on a length mismatch between two GPUArray inputs", async () => {
    const device = await deviceManager.getDevice();
    const a = gpuArrayFrom(device, new Float32Array([1, 2, 3]), "f32");
    const b = gpuArrayFrom(device, new Float32Array([1, 2]), "f32");
    await expect(gpuZip(...args, a, b, (x, y) => x + y)).rejects.toThrow("same length");
    a.destroy();
    b.destroy();
  });

  it("throws on a dtype mismatch between two GPUArray inputs", async () => {
    const device = await deviceManager.getDevice();
    const a = gpuArrayFrom(device, new Float32Array([1, 2, 3]), "f32");
    const b = gpuArrayFrom(device, new Int32Array([1, 2, 3]), "i32");
    await expect(gpuZip(...args, a, b, (x, y) => x + y)).rejects.toThrow("share a data type");
    a.destroy();
    b.destroy();
  });
});

describe("zip via public API (real GPU)", () => {
  const gpu = new GPU();

  it("keepOnGpu: true returns a GPUArray", async () => {
    const r = await gpu.zip([1, 2, 3], [4, 5, 6], (a, b) => a * b, { keepOnGpu: true });
    expect(r).toBeInstanceOf(GPUArray);
    expect(Array.from(await r.toArray())).toEqual([4, 10, 18]);
    r.destroy();
  });

  it("CPU-array inputs return a TypedArray matching the GPU result", async () => {
    const re = [3, 6, 5];
    const im = [4, 8, 12];
    const out = await gpu.zip(re, im, (r, i) => Math.sqrt(r * r + i * i));
    expect(out).toBeInstanceOf(Float32Array);
    const arr = out as Float32Array;
    const expected = [5, 10, 13];
    for (let k = 0; k < expected.length; k++) expect(arr[k]).toBeCloseTo(expected[k], 4);
  });
});
