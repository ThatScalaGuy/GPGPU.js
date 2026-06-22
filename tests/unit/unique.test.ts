import { describe, it, expect } from "vitest";
import { cpuUnique } from "../../src/ops/unique";

// CPU reference for unique: sorted distinct values (NumPy np.unique semantics). No GPU here —
// this is the oracle the real-GPU test compares against.
describe("cpuUnique", () => {
  it("returns sorted distinct values", () => {
    expect(Array.from(cpuUnique([3, 1, 2, 3, 1, 2, 3]))).toEqual([1, 2, 3]);
  });

  it("sorts even when input is already distinct", () => {
    expect(Array.from(cpuUnique([5, 4, 3, 2, 1]))).toEqual([1, 2, 3, 4, 5]);
  });

  it("collapses an all-equal input to one element", () => {
    expect(Array.from(cpuUnique([7, 7, 7, 7]))).toEqual([7]);
  });

  it("handles an empty input", () => {
    expect(Array.from(cpuUnique([]))).toEqual([]);
  });

  it("handles a single element", () => {
    expect(Array.from(cpuUnique([42]))).toEqual([42]);
  });

  it("preserves i32 dtype and sorts negatives correctly", () => {
    const out = cpuUnique(new Int32Array([3, -5, 0, -1, 2, -5, 0, 3]));
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([-5, -1, 0, 2, 3]);
  });

  it("preserves u32 dtype", () => {
    const out = cpuUnique(new Uint32Array([10, 5, 10, 5, 20, 5]));
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([5, 10, 20]);
  });

  it("preserves f32 dtype and dedupes fractional values", () => {
    const out = cpuUnique(new Float32Array([1.5, 0.25, 1.5, 0.25, 2.75]));
    expect(out).toBeInstanceOf(Float32Array);
    expect(Array.from(out)).toEqual([0.25, 1.5, 2.75]);
  });

  it("a plain number[] is treated as f32", () => {
    const out = cpuUnique([2, 1, 2, 1]);
    expect(out).toBeInstanceOf(Float32Array);
    expect(Array.from(out)).toEqual([1, 2]);
  });
});
