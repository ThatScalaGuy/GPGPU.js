import { describe, it, expect } from "vitest";
import { Pipeline } from "../../src/pipeline/pipeline";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";

// Drive Pipeline directly against the real GPU (no CPU fallback), so a broken
// reduce/map shader fails the test loudly instead of silently passing.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();

const pipeline = () => new Pipeline(deviceManager, bufferPool, shaderCache);

describe("pipeline (real GPU)", () => {
  it("runs a map-only chain", async () => {
    const result = await pipeline()
      .map((x) => x * 2)
      .map((x) => x + 1)
      .run([1, 2, 3, 4, 5]);
    expect(Array.from(result as Float32Array)).toEqual([3, 5, 7, 9, 11]);
  });

  // Regression: the reduce shader used `shared` (a reserved WGSL keyword), so
  // `pipeline-reduce` failed createShaderModule and run() threw instead of returning a value.
  it("runs map -> reduce (sum)", async () => {
    const result = await pipeline()
      .map((x) => x * 2)
      .map((x) => x + 1)
      .reduce((a, b) => a + b, 0)
      .run([1, 2, 3, 4, 5]);
    expect(result).toBe(35); // [3,5,7,9,11] -> 35
  });

  it("reduces across many workgroups", async () => {
    // n chosen so the sum (12,502,500) stays within f32's exact-integer range
    // (< 2^24) while still spanning many workgroups + a multi-pass reduction.
    const n = 5000;
    const input = Array.from({ length: n }, (_, i) => i + 1);
    const result = await pipeline().reduce((a, b) => a + b, 0).run(input);
    expect(result).toBe((n * (n + 1)) / 2);
  });

  it("reduces with a max expression", async () => {
    const result = await pipeline()
      .reduce((a, b) => Math.max(a, b), -3.4e38)
      .run([3, 1, 4, 1, 5, 9, 2, 6]);
    expect(result).toBe(9);
  });
});
