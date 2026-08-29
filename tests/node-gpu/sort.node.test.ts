import { describe, it, expect } from "vitest";
import { GPU } from "../../src/gpu";
import { cpuSort } from "../../src/fallback/cpu-ops";

// The bitonic sort had no dedicated GPU suite (only sortByKey). Sorting permutes
// exact values — no float arithmetic — so GPU and CPU must agree bit for bit.
const gpu = new GPU({ fallback: "throw" });

function fill(n: number, seed: number): Float32Array {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = ((s >>> 8) / 16777216) * 200 - 100; // [-100, 100), with duplicates
  }
  return out;
}

describe("sort (real GPU)", () => {
  it("small array with duplicates and negatives", async () => {
    const input = [3, -1, 3, 0, -7, 2, 2, -1];
    const got = (await gpu.sort(input)) as Float32Array;
    expect(Array.from(got)).toEqual([-7, -1, -1, 0, 2, 2, 3, 3]);
  });

  it("1,000 elements (non-power-of-two, padded lanes must not leak)", async () => {
    const input = fill(1_000, 7);
    const got = (await gpu.sort(input)) as Float32Array;
    expect(got.length).toBe(1_000);
    expect(Array.from(got)).toEqual(Array.from(cpuSort(input)));
  });

  it("100,000 elements matches the CPU sort exactly", async () => {
    const input = fill(100_000, 11);
    const got = (await gpu.sort(input)) as Float32Array;
    expect(got.length).toBe(100_000);
    const want = cpuSort(input);
    let mismatches = 0;
    for (let i = 0; i < got.length; i++) if (got[i] !== want[i]) mismatches++;
    expect(mismatches).toBe(0);
  });

  it("i32 input sorts as signed integers", async () => {
    const input = new Int32Array([5, -3, 2147483647, -2147483648, 0, -3]);
    const got = (await gpu.sort(input)) as Int32Array;
    expect(got).toBeInstanceOf(Int32Array);
    expect(Array.from(got)).toEqual([-2147483648, -3, -3, 0, 5, 2147483647]);
  });

  it("u32 input sorts as unsigned integers", async () => {
    const input = new Uint32Array([5, 4294967295, 0, 7, 4294967295]);
    const got = (await gpu.sort(input)) as Uint32Array;
    expect(got).toBeInstanceOf(Uint32Array);
    expect(Array.from(got)).toEqual([0, 5, 7, 4294967295, 4294967295]);
  });
});
