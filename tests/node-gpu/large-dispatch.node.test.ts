import { describe, it, expect, vi } from "vitest";
import { GPU } from "../../src/gpu";
import { cpuRandom } from "../../src/ops/random";

// Regression: computeWorkgroupCount never clamped against maxComputeWorkgroupsPerDimension
// (65535), so any per-element dispatch over 65535 × 64 = 4,194,240 elements was a WebGPU
// validation error — and with no error scopes it never became a JS exception: the fallback
// couldn't fire and the op returned recycled pool-buffer contents.
const CEIL = 65535 * 64;

describe("dispatch beyond the 65535-workgroup ceiling (real GPU)", () => {
  // Failures must throw — a silent CPU fallback would mask a broken GPU path.
  const gpu = new GPU({ fallback: "throw" });

  it("map over 4.3M elements runs every lane with the right global index", async () => {
    const n = CEIL + 100_000;
    // "i" makes every lane write its own global index — this verifies the 2-D
    // grid linearization exactly (any wrong stride shows up as a mismatch).
    const r = (await gpu.map(new Float32Array(n), "i")) as Float32Array;
    expect(r.length).toBe(n);
    expect(r[0]).toBe(0);
    expect(r[CEIL - 1]).toBe(CEIL - 1);
    expect(r[CEIL]).toBe(CEIL); // first element beyond the old ceiling
    expect(r[n - 1]).toBe(n - 1);
    let mismatches = 0;
    for (let k = 0; k < n; k++) if (r[k] !== k) mismatches++;
    expect(mismatches).toBe(0);
  });

  it("elementwise add over 4.3M elements", async () => {
    const n = CEIL + 1_000;
    const a = new Float32Array(n).fill(2);
    const r = (await gpu.add(a, 3)) as Float32Array;
    expect(r[0]).toBe(5);
    expect(r[CEIL]).toBe(5);
    expect(r[n - 1]).toBe(5);
    expect(r.every((x) => x === 5)).toBe(true);
  });

  it("random over 4.5M elements still matches the counter-based CPU reference", async () => {
    const n = CEIL + 300_000;
    const r = (await gpu.random(n)) as Float32Array;
    expect(r.length).toBe(n);
    // The generator is a pure hash of (seed, index), so the head must agree with a
    // short CPU reference; non-degenerate tail = the lanes beyond the ceiling ran.
    const head = cpuRandom(1024) as Float32Array;
    expect(Array.from(r.slice(0, 1024))).toEqual(Array.from(head));
    const tail = r.slice(n - 1024);
    expect(new Set(tail).size).toBeGreaterThan(1000);
  });
});

describe("error scopes turn oversized structural dispatches into a clean CPU fallback (real GPU)", () => {
  it("scan beyond the ceiling falls back to CPU with correct results", async () => {
    // scan's block passes are one workgroup per 64-element block and are not
    // grid-split, so > 4.19M elements exceeds the dispatch limit. Without error
    // scopes that returned garbage; now it must reject inside the GPU path and
    // fall back to the CPU implementation.
    const fallbacks: unknown[] = [];
    const gpu = new GPU({ fallback: "silent", onFallback: (f) => fallbacks.push(f) });
    const n = CEIL + 1_000;
    const stats: { backend: string }[] = [];
    gpu.onStats = (s) => stats.push(s);

    const r = (await gpu.scan(new Float32Array(n).fill(1))) as Float32Array;
    expect(r[0]).toBe(1);
    expect(r[999]).toBe(1000);
    expect(r[n - 1]).toBe(n); // exact: integer-valued f32 sums < 2^24
    expect(fallbacks.length).toBe(1);
    expect(stats[stats.length - 1].backend).toBe("cpu");
  });
});

describe("Math.clamp works on the CPU fallback", () => {
  it("string expression with Math.clamp evaluates on CPU", async () => {
    const noGpu = new GPU();
    // Force the CPU path by driving cpuMap through a GPU instance with a dead device
    // manager stub — simplest is to call the op on a tiny array with fallback and a
    // stubbed unavailable device.
    (noGpu as unknown as { deviceManager: { isAvailable(): boolean } }).deviceManager = {
      isAvailable: () => false,
    } as never;
    const r = (await noGpu.map([-5, 0.5, 9], "Math.clamp(x, 0, 1)")) as Float32Array;
    expect(Array.from(r)).toEqual([0, 0.5, 1]);
  });
});
