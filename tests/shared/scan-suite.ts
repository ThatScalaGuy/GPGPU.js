import { describe, it, expect } from "vitest";
import { gpuScan } from "../../src/ops/scan";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";
import type { NumericArray } from "../../src/core/types";

// A large, safely f32-representable negative number to seed a max-scan (WGSL has no -Infinity literal).
const F32_MIN = -3.4e38;

// Drive the GPU implementation directly (not the public gpu.scan, which has a CPU
// fallback) so a broken shader fails the test loudly instead of silently passing.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();

function scan(
  input: NumericArray,
  fn?: ((a: number, b: number) => number) | string,
  identity?: number
): Promise<Float32Array> {
  return gpuScan(deviceManager, bufferPool, shaderCache, input, fn, identity);
}

type BinFn = (a: number, b: number) => number;

function refScan(arr: ArrayLike<number>, fn: BinFn = (a, b) => a + b, identity = 0): Float32Array {
  const out = new Float32Array(arr.length);
  let acc = identity;
  for (let i = 0; i < arr.length; i++) {
    acc = fn(acc, arr[i]);
    out[i] = acc;
  }
  return out;
}

// Sizes spanning a single block, exact block multiples, partial tails, and many blocks up to ~1M.
const SIZES = [1, 2, 63, 64, 65, 127, 128, 129, 1000, 4096, 65536, 1_000_000];

/**
 * Registers the GPU prefix-scan test suite. Async because it probes for a real adapter and
 * skips the whole suite when none is available (e.g. GPU-less CI). The test file must
 * `await` this at top level so Vitest waits for collection.
 */
export async function registerScanSuite(): Promise<void> {
  const hasGPU =
    typeof navigator !== "undefined" &&
    !!navigator.gpu &&
    !!(await navigator.gpu.requestAdapter());

  const suite = hasGPU ? describe : describe.skip;

  suite("gpu.scan multi-block prefix sum", () => {
    for (const n of SIZES) {
      it(`is correct for size ${n} (all-ones input)`, async () => {
        // All-ones prefix sums are 1..n, exact in f32 since n <= 2^24.
        const input = new Float32Array(n).fill(1);
        const out = await scan(input);

        expect(out.length).toBe(n);
        expect(out[0]).toBe(1);
        expect(out[n - 1]).toBe(n);
        const mid = Math.floor(n / 2);
        expect(out[mid]).toBe(mid + 1);
      });
    }

    it("matches a JS reference on random input across blocks", async () => {
      const n = 500;
      const input = new Float32Array(n);
      for (let i = 0; i < n; i++) input[i] = Math.random();

      const out = await scan(input);
      const expected = refScan(input);

      for (let i = 0; i < n; i++) {
        // f32 GPU vs f64 reference, with tree-reassociated accumulation.
        expect(Math.abs(out[i] - expected[i])).toBeLessThan(0.05);
      }
    });

    it("propagates block offsets for product scan", async () => {
      // 1.0 everywhere except two 2.0s straddling the first block boundary (block size 64),
      // so the running product is 1 then 4 — only correct if offsets cross blocks.
      const n = 130;
      const input = new Float32Array(n).fill(1);
      input[64] = 2;
      input[65] = 2;

      const out = await scan(input, (a, b) => a * b, 1);
      const expected = refScan(input, (a, b) => a * b, 1);

      expect(Array.from(out)).toEqual(Array.from(expected));
      expect(out[63]).toBe(1);
      expect(out[64]).toBe(2);
      expect(out[n - 1]).toBe(4);
    });

    it("propagates block offsets for max scan", async () => {
      // Running max stays 0 until a spike in the second block, then holds.
      const n = 130;
      const input = new Float32Array(n).fill(0);
      input[100] = 5;

      const out = await scan(input, "Math.max(a, b)", F32_MIN);

      expect(out[99]).toBe(0);
      expect(out[100]).toBe(5);
      expect(out[n - 1]).toBe(5);
    });
  });
}
