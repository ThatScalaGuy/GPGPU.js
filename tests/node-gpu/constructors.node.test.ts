import { describe, it, expect } from "vitest";
import {
  gpuZeros,
  gpuFull,
  gpuArange,
  gpuLinspace,
  cpuZeros,
  cpuFull,
  cpuArange,
  cpuLinspace,
} from "../../src/ops/constructors";
import { gpuGather } from "../../src/ops/gather";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";
import { expectClose } from "../shared/tolerance";

// Drive the GPU constructors directly (no CPU fallback wrapper) so a broken fill /
// sequence shader fails loudly instead of silently passing through the fallback.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

describe("constructors (real GPU)", () => {
  describe("zeros", () => {
    // Sizes spanning the 64-wide workgroup boundaries.
    for (const n of [1, 63, 64, 65, 1000]) {
      it(`n=${n} is all zeros and matches cpuZeros`, async () => {
        const g = (await gpuZeros(...args, n)) as Float32Array;
        expect(g).toBeInstanceOf(Float32Array);
        expect(g.length).toBe(n);
        expect(Array.from(g)).toEqual(Array.from(cpuZeros(n)));
        for (let i = 0; i < n; i++) expect(g[i]).toBe(0);
      });
    }

    it("supports i32 and u32 dtypes", async () => {
      const i = (await gpuZeros(...args, 100, { dtype: "i32" })) as Int32Array;
      expect(i).toBeInstanceOf(Int32Array);
      expect(Array.from(i)).toEqual(Array.from(cpuZeros(100, { dtype: "i32" })));
      const u = (await gpuZeros(...args, 100, { dtype: "u32" })) as Uint32Array;
      expect(u).toBeInstanceOf(Uint32Array);
      expect(Array.from(u)).toEqual(Array.from(cpuZeros(100, { dtype: "u32" })));
    });

    it("n=0 returns an empty array of the requested dtype", async () => {
      const f = (await gpuZeros(...args, 0)) as Float32Array;
      expect(f).toBeInstanceOf(Float32Array);
      expect(f.length).toBe(0);
      const i = (await gpuZeros(...args, 0, { dtype: "i32" })) as Int32Array;
      expect(i).toBeInstanceOf(Int32Array);
      expect(i.length).toBe(0);
    });

    // The pool hands buffers back without clearing them, so zeros must WRITE zeros,
    // not rely on fresh-buffer zero-init. Fill a buffer with junk, return it to the
    // pool, then request a same-sized zeros — the pool recycles that exact buffer.
    it("really writes zeros over a pool-recycled junk buffer", async () => {
      const junk = await gpuFull(...args, 1024, 123.5, { keepOnGpu: true });
      const j = await junk.toArray();
      expect(j[0]).toBe(123.5);
      expect(j[1023]).toBe(123.5);
      junk.destroy(); // junk-filled buffer goes back to the pool
      const z = (await gpuZeros(...args, 1024)) as Float32Array;
      expect(z.length).toBe(1024);
      for (let i = 0; i < z.length; i++) expect(z[i]).toBe(0);
    });
  });

  describe("full", () => {
    it("fills with a negative fractional value (f32, exact)", async () => {
      const g = (await gpuFull(...args, 7, -3.25)) as Float32Array;
      expect(g).toBeInstanceOf(Float32Array);
      expect(Array.from(g)).toEqual([-3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25]);
      expect(Array.from(g)).toEqual(Array.from(cpuFull(7, -3.25)));
    });

    it("rounds a non-dyadic value to f32 like the CPU reference", async () => {
      const g = (await gpuFull(...args, 3, 0.1)) as Float32Array;
      for (const v of g) expect(v).toBe(Math.fround(0.1));
      expect(Array.from(g)).toEqual(Array.from(cpuFull(3, 0.1)));
    });

    it("fills i32 with a negative value", async () => {
      const g = (await gpuFull(...args, 5, -7, { dtype: "i32" })) as Int32Array;
      expect(g).toBeInstanceOf(Int32Array);
      expect(Array.from(g)).toEqual([-7, -7, -7, -7, -7]);
      expect(Array.from(g)).toEqual(Array.from(cpuFull(5, -7, { dtype: "i32" })));
    });

    it("fills u32 with the max 32-bit value", async () => {
      const g = (await gpuFull(...args, 4, 4294967295, { dtype: "u32" })) as Uint32Array;
      expect(g).toBeInstanceOf(Uint32Array);
      expect(Array.from(g)).toEqual([4294967295, 4294967295, 4294967295, 4294967295]);
      expect(Array.from(g)).toEqual(Array.from(cpuFull(4, 4294967295, { dtype: "u32" })));
    });

    it("n=1 and n=0", async () => {
      const one = (await gpuFull(...args, 1, 42)) as Float32Array;
      expect(Array.from(one)).toEqual([42]);
      const none = (await gpuFull(...args, 0, 42)) as Float32Array;
      expect(none).toBeInstanceOf(Float32Array);
      expect(none.length).toBe(0);
    });
  });

  describe("arange", () => {
    it("counts up (f32, exact)", async () => {
      const g = (await gpuArange(...args, 0, 10, 1)) as Float32Array;
      expect(g).toBeInstanceOf(Float32Array);
      expect(Array.from(g)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      expect(Array.from(g)).toEqual(Array.from(cpuArange(0, 10, 1)));
    });

    it("counts down (f32, exact)", async () => {
      const g = (await gpuArange(...args, 10, 0, -1)) as Float32Array;
      expect(Array.from(g)).toEqual([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
      expect(Array.from(g)).toEqual(Array.from(cpuArange(10, 0, -1)));
    });

    // A dyadic step makes i * step exact in f32, so GPU and CPU agree bit-for-bit.
    it("dyadic fractional step is exact", async () => {
      const g = (await gpuArange(...args, 0, 1, 0.25)) as Float32Array;
      expect(Array.from(g)).toEqual([0, 0.25, 0.5, 0.75]);
      expect(Array.from(g)).toEqual(Array.from(cpuArange(0, 1, 0.25)));
    });

    // Non-dyadic step: f32 rounding (and possible fused multiply-add on the GPU)
    // can differ in the last bits, so compare with a tolerance.
    it("non-dyadic fractional step matches within tolerance", async () => {
      const g = (await gpuArange(...args, 0, 2, 0.3)) as Float32Array;
      const c = cpuArange(0, 2, 0.3) as Float32Array;
      expect(g.length).toBe(7); // ceil(2 / 0.3)
      for (let i = 0; i < g.length; i++) {
        expectClose(g[i], c[i], { eps: 1e-6 });
        expectClose(g[i], i * 0.3, { eps: 1e-6 });
      }
    });

    it("i32 with a negative step (exact integer arithmetic)", async () => {
      const g = (await gpuArange(...args, 5, -5, -2, { dtype: "i32" })) as Int32Array;
      expect(g).toBeInstanceOf(Int32Array);
      expect(Array.from(g)).toEqual([5, 3, 1, -1, -3]);
      expect(Array.from(g)).toEqual(Array.from(cpuArange(5, -5, -2, { dtype: "i32" })));
    });

    it("u32 counts up and wraps down identically to the CPU reference", async () => {
      const up = (await gpuArange(...args, 0, 5, 1, { dtype: "u32" })) as Uint32Array;
      expect(up).toBeInstanceOf(Uint32Array);
      expect(Array.from(up)).toEqual([0, 1, 2, 3, 4]);
      // Negative step in u32 uses wrapping arithmetic: 3, 2, 1, 0.
      const down = (await gpuArange(...args, 3, -1, -1, { dtype: "u32" })) as Uint32Array;
      expect(Array.from(down)).toEqual([3, 2, 1, 0]);
      expect(Array.from(down)).toEqual(Array.from(cpuArange(3, -1, -1, { dtype: "u32" })));
    });

    it("a large range crossing workgroup boundaries matches exactly", async () => {
      const g = (await gpuArange(...args, 0, 1000, 1)) as Float32Array;
      expect(g.length).toBe(1000);
      expect(Array.from(g)).toEqual(Array.from(cpuArange(0, 1000, 1)));
    });

    it("empty ranges return an empty array", async () => {
      const a = (await gpuArange(...args, 0, 0, 1)) as Float32Array;
      expect(a.length).toBe(0);
      const b = (await gpuArange(...args, 5, 1, 1)) as Float32Array; // stop < start, step > 0
      expect(b.length).toBe(0);
      const c = (await gpuArange(...args, 1, 5, -1, { dtype: "i32" })) as Int32Array;
      expect(c).toBeInstanceOf(Int32Array);
      expect(c.length).toBe(0);
      expect(cpuArange(5, 1, 1).length).toBe(0);
    });

    it("throws on step = 0", async () => {
      await expect(gpuArange(...args, 0, 10, 0)).rejects.toThrow(/step must be nonzero/);
      expect(() => cpuArange(0, 10, 0)).toThrow(/step must be nonzero/);
    });
  });

  describe("linspace", () => {
    it("dyadic spacing is exact, endpoints included", async () => {
      const g = (await gpuLinspace(...args, 0, 1, 5)) as Float32Array;
      expect(g).toBeInstanceOf(Float32Array);
      expect(Array.from(g)).toEqual([0, 0.25, 0.5, 0.75, 1]);
      expect(Array.from(g)).toEqual(Array.from(cpuLinspace(0, 1, 5)));
    });

    it("integer spacing over a negative range is exact", async () => {
      const g = (await gpuLinspace(...args, 5, -5, 11)) as Float32Array;
      expect(Array.from(g)).toEqual([5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]);
      expect(Array.from(g)).toEqual(Array.from(cpuLinspace(5, -5, 11)));
    });

    // Non-dyadic step: interior points compare with a tolerance, but both endpoints
    // are exact — the first element is start itself and the last is pinned to stop.
    it("non-dyadic spacing: exact endpoints, close interior", async () => {
      const g = (await gpuLinspace(...args, 0, 1, 7)) as Float32Array;
      const c = cpuLinspace(0, 1, 7);
      expect(g.length).toBe(7);
      expect(g[0]).toBe(0);
      expect(g[6]).toBe(1);
      for (let i = 0; i < g.length; i++) {
        expectClose(g[i], c[i], { eps: 1e-6 });
        expectClose(g[i], i / 6, { eps: 1e-6 });
      }
    });

    it("endpoints are exact even when the formula would drift", async () => {
      const g = (await gpuLinspace(...args, 0.1, 0.7, 13)) as Float32Array;
      expect(g[0]).toBe(Math.fround(0.1));
      expect(g[12]).toBe(Math.fround(0.7));
    });

    it("num=1 yields [start]", async () => {
      const g = (await gpuLinspace(...args, 3.5, 99, 1)) as Float32Array;
      expect(Array.from(g)).toEqual([3.5]);
      expect(Array.from(cpuLinspace(3.5, 99, 1))).toEqual([3.5]);
    });

    it("num=2 yields exactly [start, stop]", async () => {
      const g = (await gpuLinspace(...args, -2.5, 7.5, 2)) as Float32Array;
      expect(Array.from(g)).toEqual([-2.5, 7.5]);
    });

    it("num=0 yields an empty array", async () => {
      const g = (await gpuLinspace(...args, 0, 1, 0)) as Float32Array;
      expect(g).toBeInstanceOf(Float32Array);
      expect(g.length).toBe(0);
      expect(cpuLinspace(0, 1, 0).length).toBe(0);
    });

    it("throws on negative or non-integer num", async () => {
      await expect(gpuLinspace(...args, 0, 1, -1)).rejects.toThrow(/nonnegative integer/);
      await expect(gpuLinspace(...args, 0, 1, 2.5)).rejects.toThrow(/nonnegative integer/);
      expect(() => cpuLinspace(0, 1, -1)).toThrow(/nonnegative integer/);
    });
  });

  describe("keepOnGpu", () => {
    it("zeros round-trips through a GPUArray", async () => {
      const out = await gpuZeros(...args, 256, { dtype: "u32", keepOnGpu: true });
      expect(out).toBeInstanceOf(GPUArray);
      expect(out.dtype).toBe("u32");
      expect(out.length).toBe(256);
      const arr = await out.toArray();
      expect(Array.from(arr)).toEqual(new Array(256).fill(0));
      out.destroy();
    });

    it("n=0 with keepOnGpu returns an empty GPUArray", async () => {
      const out = await gpuFull(...args, 0, 9, { keepOnGpu: true });
      expect(out).toBeInstanceOf(GPUArray);
      expect(out.length).toBe(0);
      const arr = await out.toArray();
      expect(arr.length).toBe(0);
      out.destroy();
    });

    it("a u32 arange chains into gather as indices (and is not consumed)", async () => {
      // 3, 2, 1, 0 — a reversal permutation, kept on the GPU.
      const idx = await gpuArange(...args, 3, -1, -1, { dtype: "u32", keepOnGpu: true });
      expect(idx).toBeInstanceOf(GPUArray);
      expect(idx.dtype).toBe("u32");
      expect(idx.length).toBe(4);

      const src = new Float32Array([10, 20, 30, 40]);
      const out = (await gpuGather(...args, src, idx)) as Float32Array;
      expect(Array.from(out)).toEqual([40, 30, 20, 10]);

      // gather must not have mutated or destroyed the caller's GPUArray.
      expect(idx.isDestroyed).toBe(false);
      const idxArr = await idx.toArray();
      expect(Array.from(idxArr)).toEqual([3, 2, 1, 0]);
      idx.destroy();
    });

    it("linspace keeps f32 dtype on the GPU", async () => {
      const out = await gpuLinspace(...args, 0, 1, 5, { keepOnGpu: true });
      expect(out).toBeInstanceOf(GPUArray);
      expect(out.dtype).toBe("f32");
      expect(out.length).toBe(5);
      const arr = await out.toArray();
      expect(Array.from(arr)).toEqual([0, 0.25, 0.5, 0.75, 1]);
      out.destroy();
    });
  });
});
