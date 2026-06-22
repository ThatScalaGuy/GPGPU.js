import { describe, it, expect } from "vitest";
import { gpuTranspose } from "../../src/ops/transpose";
import { GPU } from "../../src/gpu";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";

// Drive the GPU transpose implementation directly (no CPU fallback wrapper), so a broken
// transpose kernel fails loudly instead of silently passing through the fallback.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

// CPU reference: transpose a row-major rows×cols array into cols×rows.
function refTranspose(arr: ArrayLike<number>, rows: number, cols: number): number[] {
  const out = new Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out[c * rows + r] = arr[r * cols + c];
    }
  }
  return out;
}

describe("transpose (real GPU)", () => {
  it("2x3 -> 3x2 known matrix", async () => {
    // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    const input = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = (await gpuTranspose(...args, input, { rows: 2, cols: 3 })) as Float32Array;
    expect(Array.from(out)).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it("17x5 (rows not a multiple of TILE) matches a CPU reference", async () => {
    const rows = 17;
    const cols = 5;
    const input = new Float32Array(rows * cols);
    for (let i = 0; i < input.length; i++) input[i] = i;
    const out = (await gpuTranspose(...args, input, { rows, cols })) as Float32Array;
    expect(Array.from(out)).toEqual(refTranspose(input, rows, cols));
  });

  it("5x17 (cols not a multiple of TILE) matches a CPU reference", async () => {
    const rows = 5;
    const cols = 17;
    const input = new Float32Array(rows * cols);
    for (let i = 0; i < input.length; i++) input[i] = i;
    const out = (await gpuTranspose(...args, input, { rows, cols })) as Float32Array;
    expect(Array.from(out)).toEqual(refTranspose(input, rows, cols));
  });

  it("1xN (row vector) -> Nx1", async () => {
    const input = new Float32Array([1, 2, 3, 4, 5]);
    const out = (await gpuTranspose(...args, input, { rows: 1, cols: 5 })) as Float32Array;
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5]);
  });

  it("Nx1 (column vector) -> 1xN", async () => {
    const input = new Float32Array([1, 2, 3, 4, 5]);
    const out = (await gpuTranspose(...args, input, { rows: 5, cols: 1 })) as Float32Array;
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5]);
  });

  it("preserves i32 dtype", async () => {
    const input = new Int32Array([-1, -2, -3, -4, -5, -6]);
    const out = (await gpuTranspose(...args, input, { rows: 2, cols: 3 })) as Int32Array;
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([-1, -4, -2, -5, -3, -6]);
  });

  it("keepOnGpu returns a GPUArray of length rows*cols with shape [cols, rows]", async () => {
    const input = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = await gpuTranspose(...args, input, { rows: 2, cols: 3, keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.length).toBe(6);
    expect(out.dtype).toBe("f32");
    expect(out.shape).toEqual([3, 2]);
    expect(Array.from(await out.toArray())).toEqual([1, 4, 2, 5, 3, 6]);
    out.destroy();
  });

  it("throws when dims are missing for a plain array", async () => {
    const input = new Float32Array([1, 2, 3, 4]);
    await expect(gpuTranspose(...args, input)).rejects.toThrow(
      /provide \{ rows, cols \} or pass a reshaped 2D GPUArray/
    );
  });

  it("infers dims from a reshaped 2D GPUArray (no explicit rows/cols)", async () => {
    const gpu = new GPU();
    const reshaped = await gpu.reshape(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const out = (await gpu.transpose(reshaped, { keepOnGpu: true })) as GPUArray;
    expect(out.shape).toEqual([3, 2]);
    expect(Array.from(await out.toArray())).toEqual([1, 4, 2, 5, 3, 6]);
    reshaped.destroy();
    out.destroy();
  });

  it("double transpose returns the original", async () => {
    const rows = 5;
    const cols = 17;
    const input = new Float32Array(rows * cols);
    for (let i = 0; i < input.length; i++) input[i] = i * 1.5;
    const gpu = new GPU();
    const once = (await gpu.transpose(input, { rows, cols, keepOnGpu: true })) as GPUArray;
    // `once` is logically cols×rows; transpose again using its inferred shape.
    const twice = (await gpu.transpose(once, { keepOnGpu: true })) as GPUArray;
    expect(twice.shape).toEqual([rows, cols]);
    expect(Array.from(await twice.toArray())).toEqual(Array.from(input));
    once.destroy();
    twice.destroy();
  });
});
