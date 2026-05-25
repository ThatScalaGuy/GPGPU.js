import { describe, it, expect } from "vitest";
import { inferDataType } from "../../src/core/types";
import { toTypedArray } from "../../src/utils/data-conversion";

describe("inferDataType", () => {
  it("infers i32 from Int32Array", () => {
    expect(inferDataType(new Int32Array([1, 2]))).toBe("i32");
  });

  it("infers u32 from Uint32Array", () => {
    expect(inferDataType(new Uint32Array([1, 2]))).toBe("u32");
  });

  it("infers f32 from Float32Array", () => {
    expect(inferDataType(new Float32Array([1, 2]))).toBe("f32");
  });

  it("defaults a plain number[] to f32", () => {
    expect(inferDataType([1, 2, 3])).toBe("f32");
  });
});

describe("toTypedArray", () => {
  it("returns the same reference when dtype already matches", () => {
    const src = new Int32Array([1, 2, 3]);
    expect(toTypedArray(src, "i32")).toBe(src);
  });

  it("converts a number[] to the requested dtype", () => {
    const out = toTypedArray([1, 2, 3], "u32");
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([1, 2, 3]);
  });
});
