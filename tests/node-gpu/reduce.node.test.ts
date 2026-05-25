import { describe, it, expect } from "vitest";
import { gpuSum, gpuMin, gpuMax, gpuProduct, gpuReduce } from "../../src/ops/reduce";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";

// Drive the GPU reduce implementations directly (these have no CPU fallback —
// that wrapper lives in the public gpu.* API), so a broken reduce shader fails
// the test loudly instead of silently passing.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

describe("reduce (real GPU)", () => {
  const data = [3, 1, 4, 1, 5, 9, 2, 6];

  it("sum", async () => {
    expect(await gpuSum(...args, data)).toBe(31);
  });

  it("product", async () => {
    expect(await gpuProduct(...args, [1, 2, 3, 4, 5])).toBe(120);
  });

  // Regression: gpuMin/gpuMax seeded the reduction with ±3.4028235e+38, which
  // `toFixed` renders just above the f32 max → shader compile error → CPU fallback.
  it("min", async () => {
    expect(await gpuMin(...args, data)).toBe(1);
  });

  it("max", async () => {
    expect(await gpuMax(...args, data)).toBe(9);
  });

  it("min/max handle negatives", async () => {
    const neg = [-3, -1, -7, -2];
    expect(await gpuMin(...args, neg)).toBe(-7);
    expect(await gpuMax(...args, neg)).toBe(-1);
  });

  it("reduces across many workgroups", async () => {
    // sum stays < 2^24 so it's f32-exact while spanning a multi-pass reduction.
    const n = 5000;
    const input = Array.from({ length: n }, (_, i) => i + 1);
    expect(await gpuReduce(...args, input, (a, b) => a + b, 0)).toBe((n * (n + 1)) / 2);
  });
});
