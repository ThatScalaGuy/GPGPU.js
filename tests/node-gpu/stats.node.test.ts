import { describe, it, expect } from "vitest";
import {
  gpuMean,
  gpuVariance,
  gpuStd,
  gpuDot,
  gpuNorm,
  gpuCosineSimilarity,
  gpuSoftmax,
  cpuMean,
  cpuVariance,
  cpuStd,
  cpuDot,
  cpuNorm,
  cpuCosineSimilarity,
  cpuSoftmax,
} from "../../src/ops/stats";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";
import { uploadBuffer } from "../../src/core/command";
import { expectClose } from "../shared/tolerance";

// Drive the stats compositions directly (no CPU fallback wrapper), so a broken step in
// any underlying gpu op chain fails the test loudly instead of silently passing.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

function gpuArrayFrom(
  device: GPUDevice,
  data: Float32Array | Int32Array | Uint32Array,
  dtype: "f32" | "i32" | "u32"
): GPUArray {
  const buffer = uploadBuffer(
    device, data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, bufferPool
  );
  return new GPUArray(buffer, data.length, dtype, device, bufferPool);
}

// Deterministic pseudo-random values in [-1, 1), stored as f32 so the GPU and the CPU
// reference see bit-identical inputs.
function pseudoRandom(n: number, seed = 42): Float32Array {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 2 ** 31) - 1;
  }
  return out;
}

describe("stats (real GPU)", () => {
  describe("mean", () => {
    it("mean of 1..100 is exactly 50.5 (integer-valued f32 sums are exact)", async () => {
      const data = Array.from({ length: 100 }, (_, i) => i + 1);
      expect(await gpuMean(...args, data)).toBe(50.5);
    });

    it("empty input yields NaN", async () => {
      expect(await gpuMean(...args, new Float32Array(0))).toBeNaN();
    });

    it("single element", async () => {
      expect(await gpuMean(...args, [7])).toBe(7);
    });
  });

  describe("variance / std", () => {
    it("population variance of [1,2,3,4] is 1.25 (ddof=0)", async () => {
      expect(await gpuVariance(...args, [1, 2, 3, 4])).toBe(1.25);
      expect(await gpuStd(...args, [1, 2, 3, 4])).toBe(Math.sqrt(1.25));
    });

    it("i32 input with a fractional mean goes through the f32 cast path", async () => {
      expect(await gpuVariance(...args, new Int32Array([1, 2, 3, 4]))).toBe(1.25);
    });

    it("u32 input (deviations below the mean must not wrap)", async () => {
      expect(await gpuVariance(...args, new Uint32Array([2, 4, 6, 8]))).toBe(5);
    });

    it("length 1 has zero variance", async () => {
      expect(await gpuVariance(...args, [42])).toBe(0);
      expect(await gpuStd(...args, [42])).toBe(0);
    });

    it("empty input yields NaN", async () => {
      expect(await gpuVariance(...args, [])).toBeNaN();
      expect(await gpuStd(...args, [])).toBeNaN();
    });
  });

  describe("dot / norm", () => {
    it("dot of small ints is exact", async () => {
      expect(await gpuDot(...args, [1, 2, 3], [4, 5, 6])).toBe(32);
    });

    it("i32 dtype with negatives is exact", async () => {
      const a = new Int32Array([-1, 2, -3]);
      const b = new Int32Array([4, -5, 6]);
      expect(await gpuDot(...args, a, b)).toBe(cpuDot(a, b));
      expect(cpuDot(a, b)).toBe(-32);
    });

    it("throws on a length mismatch", async () => {
      await expect(gpuDot(...args, [1, 2, 3], [1, 2])).rejects.toThrow("same length");
      expect(() => cpuDot([1, 2, 3], [1, 2])).toThrow("same length");
    });

    it("dot of empty inputs is 0", async () => {
      expect(await gpuDot(...args, [], [])).toBe(0);
    });

    it("norm of [3,4] is 5", async () => {
      expect(await gpuNorm(...args, [3, 4])).toBe(5);
    });
  });

  describe("cosine similarity", () => {
    it("identical vectors are close to 1", async () => {
      const v = [1, 2, 3, 4];
      expectClose(await gpuCosineSimilarity(...args, v, v), 1, { eps: 1e-5 });
    });

    it("orthogonal vectors are close to 0", async () => {
      expectClose(await gpuCosineSimilarity(...args, [1, 0], [0, 1]), 0, { eps: 1e-6 });
    });

    it("a zero vector yields NaN", async () => {
      expect(await gpuCosineSimilarity(...args, [0, 0], [1, 2])).toBeNaN();
      expect(cpuCosineSimilarity([0, 0], [1, 2])).toBeNaN();
    });
  });

  describe("10k elements vs cpu reference", () => {
    const a = pseudoRandom(10000, 42);
    const b = pseudoRandom(10000, 1234);

    it("mean / variance / std", async () => {
      expectClose(await gpuMean(...args, a), cpuMean(a));
      expectClose(await gpuVariance(...args, a), cpuVariance(a));
      expectClose(await gpuStd(...args, a), cpuStd(a));
    });

    it("dot / norm / cosine", async () => {
      expectClose(await gpuDot(...args, a, b), cpuDot(a, b));
      expectClose(await gpuNorm(...args, a), cpuNorm(a));
      expectClose(await gpuCosineSimilarity(...args, a, b), cpuCosineSimilarity(a, b), {
        eps: 1e-3,
      });
    });
  });

  describe("softmax", () => {
    it("matches the cpu reference and sums to ~1", async () => {
      const data = new Float32Array([1, -2, 0.5, 3, -0.25, 2]);
      const out = (await gpuSoftmax(...args, data)) as Float32Array;
      expect(out).toBeInstanceOf(Float32Array);
      const ref = cpuSoftmax(data);
      for (let i = 0; i < data.length; i++) expectClose(out[i], ref[i], { eps: 1e-5 });
      let sum = 0;
      for (let i = 0; i < out.length; i++) sum += out[i];
      expectClose(sum, 1, { eps: 1e-4 });
    });

    it("is shift-invariant (softmax(x + 100) == softmax(x))", async () => {
      const base = new Float32Array([-1.5, 0, 0.75, 2, -3, 1.25]);
      const shiftedInput = Float32Array.from(base, (v) => v + 100);
      const out = (await gpuSoftmax(...args, base)) as Float32Array;
      const outShifted = (await gpuSoftmax(...args, shiftedInput)) as Float32Array;
      for (let i = 0; i < base.length; i++) {
        expectClose(outShifted[i], out[i], { eps: 1e-4 });
      }
    });

    it("i32 input produces the f32 softmax of the numeric values", async () => {
      const out = (await gpuSoftmax(...args, new Int32Array([1, 2, 3]))) as Float32Array;
      expect(out).toBeInstanceOf(Float32Array);
      const ref = cpuSoftmax([1, 2, 3]);
      for (let i = 0; i < 3; i++) expectClose(out[i], ref[i], { eps: 1e-5 });
    });

    it("single element yields [1]", async () => {
      const out = (await gpuSoftmax(...args, [123.5])) as Float32Array;
      expect(Array.from(out)).toEqual([1]);
    });

    it("empty input yields a length-0 result", async () => {
      const out = (await gpuSoftmax(...args, new Float32Array(0))) as Float32Array;
      expect(out.length).toBe(0);
    });

    it("keepOnGpu round-trips through a GPUArray", async () => {
      const out = await gpuSoftmax(...args, [0, Math.log(3)], { keepOnGpu: true });
      expect(out).toBeInstanceOf(GPUArray);
      expect(out.dtype).toBe("f32");
      expect(out.length).toBe(2);
      const arr = await out.toArray();
      expectClose(arr[0], 0.25, { eps: 1e-5 });
      expectClose(arr[1], 0.75, { eps: 1e-5 });
      out.destroy();
    });
  });

  describe("GPUArray inputs", () => {
    it("resident input is accepted and not mutated", async () => {
      const device = await deviceManager.getDevice();
      const data = new Float32Array([3, 1, 4, 1, 5, 9, 2, 6]);
      const resident = gpuArrayFrom(device, data, "f32");

      expectClose(await gpuMean(...args, resident), cpuMean(data), { eps: 1e-5 });
      expectClose(await gpuVariance(...args, resident), cpuVariance(data), { eps: 1e-4 });
      expectClose(await gpuNorm(...args, resident), cpuNorm(data), { eps: 1e-4 });

      const soft = (await gpuSoftmax(...args, resident)) as Float32Array;
      const ref = cpuSoftmax(data);
      for (let i = 0; i < data.length; i++) expectClose(soft[i], ref[i], { eps: 1e-5 });

      // The chained ops read the resident buffer in place — it must be untouched.
      const after = await resident.toArray();
      expect(Array.from(after)).toEqual(Array.from(data));
      resident.destroy();
    });

    it("resident i32 input goes through the cast path without mutation", async () => {
      const device = await deviceManager.getDevice();
      const data = new Int32Array([1, 2, 3, 4]);
      const resident = gpuArrayFrom(device, data, "i32");

      expect(await gpuVariance(...args, resident)).toBe(1.25);
      const soft = (await gpuSoftmax(...args, resident)) as Float32Array;
      const ref = cpuSoftmax([1, 2, 3, 4]);
      for (let i = 0; i < 4; i++) expectClose(soft[i], ref[i], { eps: 1e-5 });

      const after = await resident.toArray();
      expect(Array.from(after)).toEqual([1, 2, 3, 4]);
      resident.destroy();
    });
  });
});
