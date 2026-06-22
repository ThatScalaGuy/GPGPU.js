import { describe, it, expect } from "vitest";
import { gpuArgmin, gpuArgmax } from "../../src/ops/reduce";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";

// Drive the GPU argmin/argmax implementations directly (no CPU fallback wrapper, which
// lives in the public gpu.* API), so a broken arg-reduce shader fails the test loudly
// instead of silently passing through the fallback.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

describe("argmin/argmax (real GPU)", () => {
  const data = [3, 1, 4, 1, 5, 9, 2, 6];

  it("argmin returns the index of the minimum", async () => {
    expect(await gpuArgmin(...args, data)).toBe(1);
  });

  it("argmax returns the index of the maximum", async () => {
    expect(await gpuArgmax(...args, data)).toBe(5);
  });

  // 1 appears at indices 1 and 3; the FIRST occurrence must win (NumPy tie-break).
  it("argmin breaks ties toward the first index", async () => {
    expect(await gpuArgmin(...args, data)).toBe(1);
  });

  // A duplicated maximum: 9 at indices 2 and 5 — the first (2) must win.
  it("argmax breaks ties toward the first index", async () => {
    expect(await gpuArgmax(...args, [1, 2, 9, 4, 9, 3])).toBe(2);
  });

  // Larger than REDUCE_WORKGROUP_SIZE (256) so the cross-block combine pass runs. The
  // extremum sits in a non-first block to prove blocks past block 0 reach the final answer.
  it("finds the extremum across many workgroups", async () => {
    const n = 100000;
    const f32 = new Float32Array(n);
    for (let i = 0; i < n; i++) f32[i] = i % 1000;
    const maxIdx = 70123; // i % 1000 == 123, but we overwrite it to a unique global max
    f32[maxIdx] = 5000;
    const minIdx = 88456;
    f32[minIdx] = -5000;
    expect(await gpuArgmax(...args, f32)).toBe(maxIdx);
    expect(await gpuArgmin(...args, f32)).toBe(minIdx);
  });

  // Tie that straddles two workgroups: equal maxima at a low and a high index — the
  // low index must survive the cross-block combine.
  it("multi-block ties keep the smaller index", async () => {
    const n = 100000;
    const f32 = new Float32Array(n); // all zeros
    f32[300] = 7; // block 1
    f32[90000] = 7; // block ~351
    expect(await gpuArgmax(...args, f32)).toBe(300);
  });

  it("handles i32 input (including negatives)", async () => {
    const i32 = new Int32Array([10, -4, 7, -4, 3]);
    expect(await gpuArgmin(...args, i32)).toBe(1);
    expect(await gpuArgmax(...args, i32)).toBe(0);
  });

  it("handles u32 input", async () => {
    const u32 = new Uint32Array([5, 0, 9, 9, 2]);
    expect(await gpuArgmin(...args, u32)).toBe(1);
    expect(await gpuArgmax(...args, u32)).toBe(2);
  });
});
