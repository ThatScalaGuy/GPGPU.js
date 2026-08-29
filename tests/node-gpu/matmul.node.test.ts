import { describe, it, expect } from "vitest";
import { GPU } from "../../src/gpu";
import { cpuMatmul } from "../../src/fallback/cpu-ops";

// matmul was previously GPU-tested only at 2x2 against MATMUL_TILE_SIZE = 8 — the
// tiled kernel had never crossed a tile boundary in a test. These sizes cover the
// exact-tile, multi-tile, and ragged (non-multiple-of-tile) cases.
const gpu = new GPU({ fallback: "throw" });

// Deterministic pseudo-random fill (no Math.random: reproducible failures).
function fill(n: number, seed: number): Float32Array {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s >>> 8) / 16777216; // [0, 1)
  }
  return out;
}

async function check(rowsA: number, colsA: number, colsB: number) {
  const a = fill(rowsA * colsA, 1);
  const b = fill(colsA * colsB, 2);
  const opts = { rowsA, colsA, colsB };
  const got = (await gpu.matmul(a, b, opts)) as Float32Array;
  const want = cpuMatmul(a, b, opts);
  expect(got.length).toBe(rowsA * colsB);
  // f32 dot products reassociate on the GPU; compare with a small absolute tolerance.
  let maxDiff = 0;
  for (let i = 0; i < got.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(got[i] - want[i]));
  }
  expect(maxDiff).toBeLessThan(1e-3);
}

describe("matmul across tile boundaries (real GPU)", () => {
  it("2x2 (baseline)", () => check(2, 2, 2));
  it("8x8 (exactly one tile)", () => check(8, 8, 8));
  it("16x16 (2x2 tiles)", () => check(16, 16, 16));
  it("13x9 * 9x7 (ragged, no dimension is a tile multiple)", () => check(13, 9, 7));
  it("64x64 (8x8 tiles)", () => check(64, 64, 64));
  it("1x64 * 64x1 (matrix-vector shapes)", () => check(1, 64, 1));

  it("identity leaves the matrix unchanged (exact)", async () => {
    const n = 12; // crosses the 8-wide tile
    const eye = new Float32Array(n * n);
    for (let i = 0; i < n; i++) eye[i * n + i] = 1;
    const m = fill(n * n, 3);
    const got = (await gpu.matmul(m, eye, { rowsA: n, colsA: n, colsB: n })) as Float32Array;
    expect(Array.from(got)).toEqual(Array.from(m));
  });
});
