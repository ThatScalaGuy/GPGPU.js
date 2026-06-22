import { describe, it, expect } from "vitest";
import { cpuSegmentedReduce } from "../../src/ops/segmented-reduce";

describe("cpuSegmentedReduce", () => {
  it("sums values by segment id", () => {
    const out = cpuSegmentedReduce([10, 1, 20, 2, 30], [0, 1, 0, 1, 2], { numSegments: 3 });
    expect(Array.from(out)).toEqual([30, 3, 30]);
  });

  it("defaults op to sum", () => {
    const out = cpuSegmentedReduce([1, 2, 3, 4], [0, 0, 1, 1], { numSegments: 2 });
    expect(Array.from(out)).toEqual([3, 7]);
  });

  it("empty segments hold the sum identity (0)", () => {
    const out = cpuSegmentedReduce([5, 7], [0, 3], { numSegments: 4 });
    expect(Array.from(out)).toEqual([5, 0, 0, 7]);
  });

  it("max with an empty segment at the identity", () => {
    const out = cpuSegmentedReduce([3, -1, 9, 2, 4], [0, 0, 1, 1, 1], { numSegments: 3, op: "max" });
    expect(out[0]).toBe(3);
    expect(out[1]).toBe(9);
    expect(out[2]).toBe(Math.fround(-3.4e38)); // empty max segment = -∞ identity
  });

  it("min with an empty segment at the identity", () => {
    const out = cpuSegmentedReduce([3, -1, 9, 2], [0, 0, 2, 2], { numSegments: 3, op: "min" });
    expect(out[0]).toBe(-1);
    expect(out[1]).toBe(Math.fround(3.4e38)); // empty min segment = +∞ identity
    expect(out[2]).toBe(2);
  });

  it("product over segments", () => {
    const out = cpuSegmentedReduce([2, 3, 4, 5], [0, 0, 1, 1], { numSegments: 2, op: "product" });
    expect(Array.from(out)).toEqual([6, 20]);
  });

  it("empty product segment holds 1", () => {
    const out = cpuSegmentedReduce([2, 3], [0, 0], { numSegments: 2, op: "product" });
    expect(Array.from(out)).toEqual([6, 1]);
  });

  it("out-of-range segment id clamps to the last segment", () => {
    const out = cpuSegmentedReduce([1, 2, 3], [0, 99, 1], { numSegments: 2 });
    // id 99 clamps to seg 1: seg0=1, seg1=2+3=5
    expect(Array.from(out)).toEqual([1, 5]);
  });

  it("preserves i32 dtype", () => {
    const out = cpuSegmentedReduce(new Int32Array([-3, -2, 10, 1]), new Uint32Array([0, 0, 1, 1]), {
      numSegments: 2,
    });
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([-5, 11]);
  });

  it("preserves u32 dtype", () => {
    const out = cpuSegmentedReduce(new Uint32Array([1, 2, 3, 4, 5, 6]), new Uint32Array([0, 1, 2, 0, 1, 2]), {
      numSegments: 3,
    });
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([5, 7, 9]);
  });

  it("preserves f32 dtype for a plain number[] input", () => {
    const out = cpuSegmentedReduce([1, 2, 3], [0, 0, 0], { numSegments: 1 });
    expect(out).toBeInstanceOf(Float32Array);
    expect(Array.from(out)).toEqual([6]);
  });

  it("throws when numSegments <= 0", () => {
    expect(() => cpuSegmentedReduce([1, 2], [0, 0], { numSegments: 0 })).toThrow(/numSegments/);
  });

  it("handles empty values (all segments at identity)", () => {
    const out = cpuSegmentedReduce([], [], { numSegments: 3 });
    expect(Array.from(out)).toEqual([0, 0, 0]);
  });
});
