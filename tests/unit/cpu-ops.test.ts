import { describe, it, expect } from "vitest";
import {
  cpuAdd, cpuSubtract, cpuMultiply, cpuDivide,
  cpuMap, cpuReduce, cpuSum, cpuMin, cpuMax, cpuProduct,
  cpuMatmul, cpuScan, cpuSort,
} from "../../src/fallback/cpu-ops";

describe("CPU Fallback Operations", () => {
  describe("cpuAdd", () => {
    it("adds two arrays", () => {
      const result = cpuAdd([1, 2, 3], [4, 5, 6]);
      expect(Array.from(result)).toEqual([5, 7, 9]);
    });

    it("adds array and scalar", () => {
      const result = cpuAdd([1, 2, 3], 10);
      expect(Array.from(result)).toEqual([11, 12, 13]);
    });

    it("works with Float32Array", () => {
      const result = cpuAdd(new Float32Array([1, 2]), new Float32Array([3, 4]));
      expect(Array.from(result)).toEqual([4, 6]);
    });
  });

  describe("cpuSubtract", () => {
    it("subtracts two arrays", () => {
      const result = cpuSubtract([10, 20, 30], [1, 2, 3]);
      expect(Array.from(result)).toEqual([9, 18, 27]);
    });

    it("subtracts scalar", () => {
      const result = cpuSubtract([10, 20, 30], 5);
      expect(Array.from(result)).toEqual([5, 15, 25]);
    });
  });

  describe("cpuMultiply", () => {
    it("multiplies two arrays", () => {
      const result = cpuMultiply([2, 3, 4], [5, 6, 7]);
      expect(Array.from(result)).toEqual([10, 18, 28]);
    });

    it("multiplies by scalar", () => {
      const result = cpuMultiply([1, 2, 3], 3);
      expect(Array.from(result)).toEqual([3, 6, 9]);
    });
  });

  describe("cpuDivide", () => {
    it("divides two arrays", () => {
      const result = cpuDivide([10, 20, 30], [2, 4, 5]);
      expect(Array.from(result)).toEqual([5, 5, 6]);
    });

    it("divides by scalar", () => {
      const result = cpuDivide([10, 20, 30], 10);
      expect(Array.from(result)).toEqual([1, 2, 3]);
    });
  });

  describe("cpuMap", () => {
    it("maps with function", () => {
      const result = cpuMap([1, 2, 3], (x) => x * 2);
      expect(Array.from(result)).toEqual([2, 4, 6]);
    });

    it("maps with string expression", () => {
      const result = cpuMap([1, 2, 3], "x * 2");
      expect(Array.from(result)).toEqual([2, 4, 6]);
    });

    it("maps with Math functions", () => {
      const result = cpuMap([4, 9, 16], Math.sqrt);
      expect(Array.from(result)).toEqual([2, 3, 4]);
    });
  });

  describe("cpuReduce", () => {
    it("reduces with sum", () => {
      const result = cpuReduce([1, 2, 3, 4], (a, b) => a + b, 0);
      expect(result).toBe(10);
    });

    it("reduces with string expression", () => {
      const result = cpuReduce([1, 2, 3, 4], "a + b", 0);
      expect(result).toBe(10);
    });

    it("reduces product", () => {
      const result = cpuReduce([1, 2, 3, 4], (a, b) => a * b, 1);
      expect(result).toBe(24);
    });
  });

  describe("convenience reductions", () => {
    it("cpuSum", () => {
      expect(cpuSum([1, 2, 3, 4, 5])).toBe(15);
    });

    it("cpuMin", () => {
      expect(cpuMin([3, 1, 4, 1, 5])).toBe(1);
    });

    it("cpuMax", () => {
      expect(cpuMax([3, 1, 4, 1, 5])).toBe(5);
    });

    it("cpuProduct", () => {
      expect(cpuProduct([1, 2, 3, 4])).toBe(24);
    });
  });

  describe("cpuMatmul", () => {
    it("multiplies 2x2 matrices", () => {
      const a = [1, 2, 3, 4];
      const b = [5, 6, 7, 8];
      const result = cpuMatmul(a, b, { rowsA: 2, colsA: 2, colsB: 2 });
      expect(Array.from(result)).toEqual([19, 22, 43, 50]);
    });

    it("multiplies non-square matrices", () => {
      const a = [1, 2, 3, 4, 5, 6];
      const b = [7, 8, 9, 10, 11, 12];
      const result = cpuMatmul(a, b, { rowsA: 2, colsA: 3, colsB: 2 });
      expect(Array.from(result)).toEqual([58, 64, 139, 154]);
    });
  });

  describe("cpuScan", () => {
    it("computes prefix sum", () => {
      const result = cpuScan([1, 2, 3, 4], (a, b) => a + b, 0);
      expect(Array.from(result)).toEqual([1, 3, 6, 10]);
    });

    it("computes prefix product", () => {
      const result = cpuScan([1, 2, 3, 4], (a, b) => a * b, 1);
      expect(Array.from(result)).toEqual([1, 2, 6, 24]);
    });
  });

  describe("cpuSort", () => {
    it("sorts ascending", () => {
      const result = cpuSort([3, 1, 4, 1, 5, 9, 2, 6]);
      expect(Array.from(result)).toEqual([1, 1, 2, 3, 4, 5, 6, 9]);
    });

    it("handles already sorted", () => {
      const result = cpuSort([1, 2, 3, 4]);
      expect(Array.from(result)).toEqual([1, 2, 3, 4]);
    });

    it("handles single element", () => {
      const result = cpuSort([42]);
      expect(Array.from(result)).toEqual([42]);
    });

    it("handles empty array", () => {
      const result = cpuSort([]);
      expect(Array.from(result)).toEqual([]);
    });

    it("sorts integers numerically, not lexicographically", () => {
      const result = cpuSort(new Int32Array([10, 2, 33, 4]));
      expect(result).toBeInstanceOf(Int32Array);
      expect(Array.from(result)).toEqual([2, 4, 10, 33]);
    });
  });

  describe("typed-array dtype parity", () => {
    it("returns Int32Array for Int32Array input", () => {
      const result = cpuAdd(new Int32Array([1, 2, 3]), new Int32Array([4, 5, 6]));
      expect(result).toBeInstanceOf(Int32Array);
      expect(Array.from(result)).toEqual([5, 7, 9]);
    });

    it("returns Uint32Array for Uint32Array input", () => {
      const result = cpuMultiply(new Uint32Array([1, 2, 3]), 3);
      expect(result).toBeInstanceOf(Uint32Array);
      expect(Array.from(result)).toEqual([3, 6, 9]);
    });

    it("evaluates bitwise map on integers", () => {
      const result = cpuMap(new Int32Array([1, 2, 3, 4]), "x & 1");
      expect(result).toBeInstanceOf(Int32Array);
      expect(Array.from(result)).toEqual([1, 0, 1, 0]);
    });

    it("evaluates shift map on u32", () => {
      const result = cpuMap(new Uint32Array([1, 2, 3]), "x << 1");
      expect(result).toBeInstanceOf(Uint32Array);
      expect(Array.from(result)).toEqual([2, 4, 6]);
    });

    it("scan preserves integer dtype", () => {
      const result = cpuScan(new Int32Array([1, 2, 3, 4]), (a, b) => a + b, 0);
      expect(result).toBeInstanceOf(Int32Array);
      expect(Array.from(result)).toEqual([1, 3, 6, 10]);
    });
  });
});
