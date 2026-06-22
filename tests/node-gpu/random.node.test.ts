import { describe, it, expect } from "vitest";
import { gpuRandom, cpuRandom } from "../../src/ops/random";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";
import { expectClose } from "../shared/tolerance";

// Drive gpuRandom directly (no CPU fallback wrapper) so a broken PRNG shader fails loudly
// instead of silently passing through the fallback. The generator is counter-based, so the
// GPU and the cpuRandom reference compute the same wrapping-u32 hash and must agree exactly.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

// Sizes spanning the 64-wide workgroup boundaries plus a 1e6 sample (cheap: one pass).
const SIZES = [1, 63, 64, 65, 127, 128, 129, 1000, 4096, 65536, 1_000_000];

describe("random (real GPU)", () => {
  // f32 in [0,1). The float is the top 24 bits / 2^24 — a power-of-two divide, exact in f32 —
  // so GPU and CPU agree bit-for-bit (assert exact here, not just expectClose).
  describe("f32 uniform [0,1) matches the CPU reference exactly", () => {
    for (const n of SIZES) {
      it(`n=${n}`, async () => {
        const g = (await gpuRandom(...args, n, { seed: 12345 })) as Float32Array;
        const c = cpuRandom(n, { seed: 12345 }) as Float32Array;
        expect(g).toBeInstanceOf(Float32Array);
        expect(g.length).toBe(n);
        expect(Array.from(g)).toEqual(Array.from(c));
        // Every value is a valid uniform in [0,1).
        for (let i = 0; i < g.length; i++) {
          expect(g[i]).toBeGreaterThanOrEqual(0);
          expect(g[i]).toBeLessThan(1);
        }
      });
    }
  });

  // Raw u32 generator output: full 32-bit range, exact integer comparison.
  describe("u32 raw output matches the CPU reference exactly", () => {
    for (const n of SIZES) {
      it(`n=${n}`, async () => {
        const g = (await gpuRandom(...args, n, { seed: 777, dtype: "u32" })) as Uint32Array;
        const c = cpuRandom(n, { seed: 777, dtype: "u32" }) as Uint32Array;
        expect(g).toBeInstanceOf(Uint32Array);
        expect(g.length).toBe(n);
        expect(Array.from(g)).toEqual(Array.from(c));
      });
    }
  });

  // i32 = the same bits reinterpreted as signed (bitcast on GPU, |0 on CPU), exact.
  describe("i32 bitcast output matches the CPU reference exactly", () => {
    for (const n of SIZES) {
      it(`n=${n}`, async () => {
        const g = (await gpuRandom(...args, n, { seed: 9, dtype: "i32" })) as Int32Array;
        const c = cpuRandom(n, { seed: 9, dtype: "i32" }) as Int32Array;
        expect(g).toBeInstanceOf(Int32Array);
        expect(g.length).toBe(n);
        expect(Array.from(g)).toEqual(Array.from(c));
      });
    }
  });

  it("default seed is 0 and reproducible across calls", async () => {
    const a = (await gpuRandom(...args, 256)) as Float32Array;
    const b = (await gpuRandom(...args, 256, { seed: 0 })) as Float32Array;
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it("different seeds give different streams", async () => {
    const a = (await gpuRandom(...args, 256, { seed: 1 })) as Float32Array;
    const b = (await gpuRandom(...args, 256, { seed: 2 })) as Float32Array;
    // Not a single element should coincide for independent seeds over 256 draws.
    let same = 0;
    for (let i = 0; i < a.length; i++) if (a[i] === b[i]) same++;
    expect(same).toBe(0);
  });

  it("n=0 returns an empty array of the requested dtype", async () => {
    const f = (await gpuRandom(...args, 0)) as Float32Array;
    expect(f).toBeInstanceOf(Float32Array);
    expect(f.length).toBe(0);
    const u = (await gpuRandom(...args, 0, { dtype: "u32" })) as Uint32Array;
    expect(u).toBeInstanceOf(Uint32Array);
    expect(u.length).toBe(0);
  });

  it("keepOnGpu returns a GPUArray with the right dtype and length", async () => {
    const out = await gpuRandom(...args, 1000, { seed: 42, keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.dtype).toBe("f32");
    expect(out.length).toBe(1000);
    const arr = await out.toArray();
    expect(Array.from(arr)).toEqual(Array.from(cpuRandom(1000, { seed: 42 })));
    out.destroy();
  });

  it("keepOnGpu preserves a u32 dtype", async () => {
    const out = await gpuRandom(...args, 64, { seed: 5, dtype: "u32", keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.dtype).toBe("u32");
    expect(out.length).toBe(64);
    out.destroy();
  });

  // Distribution sanity on a large sample: a uniform [0,1) stream has mean ~0.5 and roughly
  // even decile occupancy. This guards against a degenerate / constant generator.
  it("1e6 f32 samples are uniformly distributed", async () => {
    const n = 1_000_000;
    const g = (await gpuRandom(...args, n, { seed: 2024 })) as Float32Array;
    let sum = 0;
    const deciles = new Array(10).fill(0);
    for (let i = 0; i < n; i++) {
      sum += g[i];
      deciles[Math.min(9, Math.floor(g[i] * 10))]++;
    }
    expectClose(sum / n, 0.5, { eps: 0.005 });
    for (const d of deciles) expectClose(d / n, 0.1, { eps: 0.005 });
  });
});
