import { describe, it, expect } from "vitest";
import { GPU } from "../../src/gpu";
import { GPUArray } from "../../src/pipeline/gpu-array";

// reshape is a pure metadata op living on the GPU facade; exercise it against the real GPU.
const gpu = new GPU();

describe("reshape (real GPU)", () => {
  it("reshapes a GPUArray to a 2D shape as a zero-copy view", async () => {
    const src = await gpu.upload([1, 2, 3, 4, 5, 6]);
    const view = await gpu.reshape(src, [2, 3]);
    expect(view).toBeInstanceOf(GPUArray);
    expect(view.shape).toEqual([2, 3]);
    expect(view.length).toBe(6);
    expect(view.dtype).toBe("f32");
    // The view shares the source buffer (zero copy) and reads back the same flat data.
    expect(view.buffer).toBe(src.buffer);
    expect(Array.from(await view.toArray())).toEqual([1, 2, 3, 4, 5, 6]);
    expect(Array.from(await src.toArray())).toEqual([1, 2, 3, 4, 5, 6]);
    src.destroy();
  });

  it("reshapes a CPU array into an owning GPUArray with the shape", async () => {
    const arr = await gpu.reshape(new Float32Array([1, 2, 3, 4]), [2, 2]);
    expect(arr).toBeInstanceOf(GPUArray);
    expect(arr.shape).toEqual([2, 2]);
    expect(arr.length).toBe(4);
    expect(Array.from(await arr.toArray())).toEqual([1, 2, 3, 4]);
    arr.destroy();
  });

  it("throws on a length mismatch", async () => {
    const src = await gpu.upload([1, 2, 3, 4, 5, 6]);
    await expect(gpu.reshape(src, [2, 2])).rejects.toThrow(/4 elements but array has 6/);
    await expect(gpu.reshape(new Float32Array([1, 2, 3]), [2, 2])).rejects.toThrow(
      /4 elements but array has 3/
    );
    src.destroy();
  });

  it("a reshaped view feeds transpose with dims inferred from its shape", async () => {
    const src = await gpu.upload([1, 2, 3, 4, 5, 6]);
    const view = await gpu.reshape(src, [2, 3]);
    const out = (await gpu.transpose(view, { keepOnGpu: true })) as GPUArray;
    expect(out.shape).toEqual([3, 2]);
    expect(Array.from(await out.toArray())).toEqual([1, 4, 2, 5, 3, 6]);
    src.destroy();
    out.destroy();
  });

  it("destroying the view does NOT free the source buffer", async () => {
    const src = await gpu.upload([10, 20, 30, 40]);
    const view = await gpu.reshape(src, [2, 2]);
    // Destroying a non-owning view must leave the borrowed buffer intact.
    view.destroy();
    expect(view.isDestroyed).toBe(true);
    // The source still owns and can still be read.
    expect(Array.from(await src.toArray())).toEqual([10, 20, 30, 40]);
    // Destroying the source is the owner's job.
    src.destroy();
  });
});
