import { describe, it, expect } from "vitest";
import { gpuScatter } from "../../src/ops/scatter";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";
import { expectClose } from "../shared/tolerance";

// Drive the GPU scatter implementation directly (no CPU fallback wrapper), so a broken
// scatter shader / atomics path fails the test loudly instead of silently passing through
// the fallback.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

describe("scatter (real GPU)", () => {
  it("set: writes vals at distinct indices (deterministic)", async () => {
    const dst = new Float32Array([0, 0, 0, 0]);
    const idx = new Uint32Array([3, 1]);
    const vals = new Float32Array([9, 5]);
    const out = (await gpuScatter(...args, dst, idx, vals)) as Float32Array;
    expect(Array.from(out)).toEqual([0, 5, 0, 9]);
  });

  it("set: overwrites at an index and leaves the rest of dst intact", async () => {
    const dst = new Float32Array([10, 20, 30, 40, 50]);
    const idx = new Uint32Array([0, 2, 4]);
    const vals = new Float32Array([1, 3, 5]);
    const out = (await gpuScatter(...args, dst, idx, vals)) as Float32Array;
    expect(Array.from(out)).toEqual([1, 20, 3, 40, 5]);
  });

  it("set: output dtype follows dst dtype (i32)", async () => {
    const dst = new Int32Array([-1, -2, -3]);
    const idx = new Uint32Array([0, 2]);
    const vals = new Int32Array([-9, -7]);
    const out = (await gpuScatter(...args, dst, idx, vals)) as Int32Array;
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([-9, -2, -7]);
  });

  it("add (i32): duplicate indices accumulate", async () => {
    const dst = new Int32Array([0, 0, 0]);
    const idx = new Uint32Array([1, 1, 1]);
    const vals = new Int32Array([1, 1, 1]);
    const out = (await gpuScatter(...args, dst, idx, vals, { mode: "add" })) as Int32Array;
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([0, 3, 0]);
  });

  it("add (u32): duplicate indices accumulate onto a non-zero base", async () => {
    const dst = new Uint32Array([5, 5, 5]);
    const idx = new Uint32Array([0, 0, 2]);
    const vals = new Uint32Array([1, 2, 10]);
    const out = (await gpuScatter(...args, dst, idx, vals, { mode: "add" })) as Uint32Array;
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([8, 5, 15]);
  });

  it("add (f32): the CAS path accumulates duplicates", async () => {
    const dst = new Float32Array([0, 0]);
    const idx = new Uint32Array([0, 0, 1]);
    const vals = new Float32Array([1.5, 2.5, 4.0]);
    const out = (await gpuScatter(...args, dst, idx, vals, { mode: "add" })) as Float32Array;
    expect(out).toBeInstanceOf(Float32Array);
    expectClose(out[0], 4.0);
    expectClose(out[1], 4.0);
  });

  // An out-of-range index clamps to the last element (the shader can't throw).
  // Indices are distinct here (the clamped 99 is the sole writer of the last slot)
  // so the result is deterministic — duplicate-index set writes would race.
  it("set: clamps an out-of-range index to the last element", async () => {
    const dst = new Float32Array([0, 0, 0]);
    const idx = new Uint32Array([0, 99]);
    const vals = new Float32Array([7, 8]);
    const out = (await gpuScatter(...args, dst, idx, vals)) as Float32Array;
    // idx 99 clamps to the last slot (2).
    expect(Array.from(out)).toEqual([7, 0, 8]);
  });

  it("keepOnGpu returns a GPUArray of length N", async () => {
    const dst = new Float32Array([0, 0, 0, 0]);
    const idx = new Uint32Array([3, 1]);
    const vals = new Float32Array([9, 5]);
    const out = await gpuScatter(...args, dst, idx, vals, { keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.length).toBe(4);
    expect(out.dtype).toBe("f32");
    const arr = await out.toArray();
    expect(Array.from(arr)).toEqual([0, 5, 0, 9]);
    out.destroy();
  });

  // M well over the 64-wide workgroup: scatter-add 1000 ones into 10 bins to exercise
  // multiple workgroups and atomic contention. Each bin gets exactly 100.
  it("add: multi-block with atomic contention sums exactly", async () => {
    const bins = 10;
    const m = 1000;
    const dst = new Int32Array(bins); // zeros
    const idx = new Uint32Array(m);
    const vals = new Int32Array(m).fill(1);
    for (let i = 0; i < m; i++) idx[i] = i % bins;
    const out = (await gpuScatter(...args, dst, idx, vals, { mode: "add" })) as Int32Array;
    expect(Array.from(out)).toEqual(Array.from({ length: bins }, () => 100));
  });

  it("accepts GPUArray inputs for dst, idx, and vals", async () => {
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

    const dst = mk(new Float32Array([0, 0, 0, 0]), "f32");
    const idx = mk(new Uint32Array([3, 1]), "u32");
    const vals = mk(new Float32Array([9, 5]), "f32");

    const out = await gpuScatter(...args, dst, idx, vals, { keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(Array.from(await out.toArray())).toEqual([0, 5, 0, 9]);
    // inputs are reused in place, not consumed
    expect(Array.from(await dst.toArray())).toEqual([0, 0, 0, 0]);
    dst.destroy();
    idx.destroy();
    vals.destroy();
    out.destroy();
  });
});
