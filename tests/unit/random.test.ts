import { describe, it, expect } from "vitest";
import { cpuRandom } from "../../src/ops/random";

describe("cpuRandom (counter-based PRNG reference)", () => {
  it("defaults to f32 in [0,1)", () => {
    const out = cpuRandom(1000, { seed: 1 });
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(1000);
    for (let i = 0; i < out.length; i++) {
      expect(out[i]).toBeGreaterThanOrEqual(0);
      expect(out[i]).toBeLessThan(1);
    }
  });

  it("is deterministic: same (n, seed) -> same values", () => {
    const a = cpuRandom(128, { seed: 99 });
    const b = cpuRandom(128, { seed: 99 });
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it("default seed is 0", () => {
    const a = cpuRandom(64);
    const b = cpuRandom(64, { seed: 0 });
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it("different seeds produce different streams", () => {
    const a = cpuRandom(256, { seed: 1 });
    const b = cpuRandom(256, { seed: 2 });
    let same = 0;
    for (let i = 0; i < a.length; i++) if (a[i] === b[i]) same++;
    expect(same).toBe(0);
  });

  it("u32 dtype spans the full 32-bit range", () => {
    const out = cpuRandom(10000, { seed: 7, dtype: "u32" });
    expect(out).toBeInstanceOf(Uint32Array);
    const max = out.reduce((m, x) => Math.max(m, x), 0);
    const min = out.reduce((m, x) => Math.min(m, x), 0xffffffff);
    // With 10k draws we expect values in both the low and high halves of the u32 range.
    expect(max).toBeGreaterThan(0x80000000);
    expect(min).toBeLessThan(0x80000000);
  });

  it("i32 dtype is the u32 bits reinterpreted as signed", () => {
    const u = cpuRandom(256, { seed: 3, dtype: "u32" }) as Uint32Array;
    const s = cpuRandom(256, { seed: 3, dtype: "i32" }) as Int32Array;
    expect(s).toBeInstanceOf(Int32Array);
    for (let i = 0; i < u.length; i++) {
      expect(s[i]).toBe(u[i] | 0);
    }
  });

  it("f32 output is the top 24 bits of the u32 stream / 2^24", () => {
    const u = cpuRandom(256, { seed: 11, dtype: "u32" }) as Uint32Array;
    const f = cpuRandom(256, { seed: 11 }) as Float32Array;
    for (let i = 0; i < u.length; i++) {
      expect(f[i]).toBe((u[i] >>> 8) * (1 / 16777216));
    }
  });

  it("mean of a large f32 sample is ~0.5", () => {
    const n = 100000;
    const out = cpuRandom(n, { seed: 2024 });
    let sum = 0;
    for (let i = 0; i < n; i++) sum += out[i];
    expect(Math.abs(sum / n - 0.5)).toBeLessThan(0.01);
  });

  it("n=0 returns an empty array of the requested dtype", () => {
    expect(cpuRandom(0)).toBeInstanceOf(Float32Array);
    expect(cpuRandom(0).length).toBe(0);
    expect(cpuRandom(0, { dtype: "u32" })).toBeInstanceOf(Uint32Array);
    expect(cpuRandom(0, { dtype: "i32" })).toBeInstanceOf(Int32Array);
  });
});
