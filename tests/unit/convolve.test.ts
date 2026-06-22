import { describe, it, expect } from "vitest";
import { cpuConvolve, convolveOutputLength } from "../../src/ops/convolve";

// Pure-CPU reference tests for the convolution math, independent of any GPU. Expected
// values are cross-checked against numpy.convolve.
describe("cpuConvolve", () => {
  it("full: reverses the kernel (true convolution, not correlation)", () => {
    // np.convolve([1,2,3], [0,1,0.5]) = [0, 1, 2.5, 4, 1.5]
    const out = cpuConvolve([1, 2, 3], [0, 1, 0.5], "full");
    expect(Array.from(out)).toEqual([0, 1, 2.5, 4, 1.5]);
  });

  it("full is the default mode", () => {
    expect(Array.from(cpuConvolve([1, 2, 3], [1, 1]))).toEqual(
      Array.from(cpuConvolve([1, 2, 3], [1, 1], "full"))
    );
  });

  it("full: unit impulse stamps the un-reversed kernel", () => {
    expect(Array.from(cpuConvolve([1, 0, 0, 0], [1, 2, 3], "full"))).toEqual([1, 2, 3, 0, 0, 0]);
  });

  it("same: returns the centred window of length max(N,M)", () => {
    // centre of [0,1,2.5,4,1.5] -> [1, 2.5, 4]
    expect(Array.from(cpuConvolve([1, 2, 3], [0, 1, 0.5], "same"))).toEqual([1, 2.5, 4]);
  });

  it("valid: only fully-overlapping positions", () => {
    // np.convolve([1,2,3,4,5],[1,1,1],'valid') = [6, 9, 12]
    expect(Array.from(cpuConvolve([1, 2, 3, 4, 5], [1, 1, 1], "valid"))).toEqual([6, 9, 12]);
  });

  it("preserves i32 dtype", () => {
    const out = cpuConvolve(new Int32Array([1, -2, 3]), new Int32Array([2, -1]), "full");
    expect(out).toBeInstanceOf(Int32Array);
    // np.convolve([1,-2,3],[2,-1]) = [2, -5, 8, -3]
    expect(Array.from(out)).toEqual([2, -5, 8, -3]);
  });

  it("preserves u32 dtype", () => {
    const out = cpuConvolve(new Uint32Array([1, 2, 3]), new Uint32Array([1, 1]), "full");
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([1, 3, 5, 3]);
  });

  it("handles a length-1 kernel as plain scaling", () => {
    expect(Array.from(cpuConvolve([1, 2, 3, 4], [2], "full"))).toEqual([2, 4, 6, 8]);
    expect(Array.from(cpuConvolve([1, 2, 3, 4], [2], "same"))).toEqual([2, 4, 6, 8]);
    expect(Array.from(cpuConvolve([1, 2, 3, 4], [2], "valid"))).toEqual([2, 4, 6, 8]);
  });

  it("empty input or kernel yields an empty result", () => {
    expect(cpuConvolve([], [1, 2], "full").length).toBe(0);
    expect(cpuConvolve([1, 2], [], "full").length).toBe(0);
  });
});

describe("convolveOutputLength", () => {
  it("full = n + m - 1", () => {
    expect(convolveOutputLength(5, 3, "full")).toBe(7);
    expect(convolveOutputLength(1, 1, "full")).toBe(1);
  });

  it("same = max(n, m)", () => {
    expect(convolveOutputLength(5, 3, "same")).toBe(5);
    expect(convolveOutputLength(3, 5, "same")).toBe(5);
  });

  it("valid = max(n,m) - min(n,m) + 1", () => {
    expect(convolveOutputLength(5, 3, "valid")).toBe(3);
    expect(convolveOutputLength(3, 5, "valid")).toBe(3); // symmetric
    expect(convolveOutputLength(5, 1, "valid")).toBe(5);
  });
});
