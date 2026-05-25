import { describe, it, expect } from "vitest";
import { gpuMap, gpuElementwiseBinary, gpuScalarBroadcast } from "../../src/ops/elementwise";
import { gpuSum, gpuMin, gpuMax } from "../../src/ops/reduce";
import { gpuScan } from "../../src/ops/scan";
import { gpuSort } from "../../src/ops/sort";
import { gpuMatmul } from "../../src/ops/matmul";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";

// Drive the GPU implementations directly (no CPU fallback) so a broken integer
// shader fails loudly instead of silently falling back to CPU.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

describe("i32/u32 end-to-end (real GPU)", () => {
  describe("map", () => {
    it("integer arithmetic returns Int32Array", async () => {
      const out = await gpuMap(...args, new Int32Array([1, 2, 3, 4]), (x) => x * 3 + 1);
      expect(out).toBeInstanceOf(Int32Array);
      expect(Array.from(out)).toEqual([4, 7, 10, 13]);
    });

    it("bitwise AND on i32", async () => {
      const out = await gpuMap(...args, new Int32Array([1, 2, 3, 4, 5]), "x & 1");
      expect(out).toBeInstanceOf(Int32Array);
      expect(Array.from(out)).toEqual([1, 0, 1, 0, 1]);
    });

    it("left shift on u32 returns Uint32Array", async () => {
      const out = await gpuMap(...args, new Uint32Array([1, 2, 3, 4]), "x << 1");
      expect(out).toBeInstanceOf(Uint32Array);
      expect(Array.from(out)).toEqual([2, 4, 6, 8]);
    });

    it("bitwise XOR + OR + NOT on u32", async () => {
      const out = await gpuMap(...args, new Uint32Array([0, 1, 2, 3]), "(x ^ 1) | 4");
      expect(out).toBeInstanceOf(Uint32Array);
      expect(Array.from(out)).toEqual([5, 4, 7, 6]);
    });
  });

  describe("elementwise", () => {
    it("i32 add (array + array)", async () => {
      const out = await gpuElementwiseBinary(
        ...args, new Int32Array([1, 2, 3]), new Int32Array([10, 20, 30]), "+"
      );
      expect(out).toBeInstanceOf(Int32Array);
      expect(Array.from(out)).toEqual([11, 22, 33]);
    });

    it("u32 scalar multiply", async () => {
      const out = await gpuScalarBroadcast(...args, new Uint32Array([1, 2, 3]), 4, "*");
      expect(out).toBeInstanceOf(Uint32Array);
      expect(Array.from(out)).toEqual([4, 8, 12]);
    });
  });

  describe("reduce", () => {
    it("i32 sum", async () => {
      expect(await gpuSum(...args, new Int32Array([3, 1, 4, 1, 5, 9, 2, 6]))).toBe(31);
    });

    it("u32 sum", async () => {
      expect(await gpuSum(...args, new Uint32Array([10, 20, 30, 40]))).toBe(100);
    });

    it("i32 min/max with negatives", async () => {
      const neg = new Int32Array([-3, -1, -7, -2]);
      expect(await gpuMin(...args, neg)).toBe(-7);
      expect(await gpuMax(...args, neg)).toBe(-1);
    });

    it("u32 min/max", async () => {
      const u = new Uint32Array([5, 1, 9, 3]);
      expect(await gpuMin(...args, u)).toBe(1);
      expect(await gpuMax(...args, u)).toBe(9);
    });
  });

  describe("scan", () => {
    it("i32 prefix sum", async () => {
      const out = await gpuScan(...args, new Int32Array([1, 2, 3, 4, 5]));
      expect(out).toBeInstanceOf(Int32Array);
      expect(Array.from(out)).toEqual([1, 3, 6, 10, 15]);
    });

    it("u32 prefix sum across multiple blocks", async () => {
      const n = 200; // > workgroup width (64) to exercise the block-offset pass
      const input = new Uint32Array(n).fill(1);
      const out = await gpuScan(...args, input);
      expect(out).toBeInstanceOf(Uint32Array);
      expect(Array.from(out)).toEqual(Array.from({ length: n }, (_, i) => i + 1));
    });
  });

  describe("sort", () => {
    it("i32 ascending with negatives", async () => {
      const out = await gpuSort(...args, new Int32Array([3, -1, 4, -7, 5, 2]));
      expect(out).toBeInstanceOf(Int32Array);
      expect(Array.from(out)).toEqual([-7, -1, 2, 3, 4, 5]);
    });

    it("u32 ascending (non-power-of-2 length)", async () => {
      const out = await gpuSort(...args, new Uint32Array([10, 2, 33, 4, 100]));
      expect(out).toBeInstanceOf(Uint32Array);
      expect(Array.from(out)).toEqual([2, 4, 10, 33, 100]);
    });
  });

  describe("matmul", () => {
    it("i32 2x2 matrix multiply", async () => {
      const a = new Int32Array([1, 2, 3, 4]);
      const b = new Int32Array([5, 6, 7, 8]);
      const out = await gpuMatmul(...args, a, b, { rowsA: 2, colsA: 2, colsB: 2 });
      expect(out).toBeInstanceOf(Int32Array);
      expect(Array.from(out)).toEqual([19, 22, 43, 50]);
    });
  });
});
