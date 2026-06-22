import { describe, it, expect } from "vitest";
import { gpuFft, cpuFft } from "../../src/ops/fft";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";
import { uploadBuffer } from "../../src/core/command";
import { expectClose } from "../shared/tolerance";

// Drive the GPU radix-2 FFT directly (no CPU fallback wrapper) so a broken
// bit-reversal / butterfly path fails loudly instead of passing through fallback.
// The spectrum is complex (interleaved f32), so every magnitude assertion uses
// expectClose against the O(n^2) DFT oracle in cpuFft.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

// FFT magnitudes grow with n (a single tone of amplitude A in an n-point FFT has
// a bin of magnitude ~A*n/2), so scale the tolerance with n rather than using the
// default 0.05.
function expectSpectrumClose(actual: Float32Array, expected: Float32Array, n: number): void {
  expect(actual.length).toBe(expected.length);
  const eps = 1e-3 * Math.max(n, 1);
  for (let i = 0; i < expected.length; i++) {
    expectClose(actual[i], expected[i], { eps });
  }
}

describe("fft (real GPU)", () => {
  it("DC signal: all-ones -> single spike at bin 0", async () => {
    const n = 8;
    const input = new Float32Array(n).fill(1);
    const out = (await gpuFft(...args, input)) as Float32Array;
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(2 * n);
    // X[0] = n (real), all other bins ~0.
    expectClose(out[0], n);
    expectClose(out[1], 0);
    for (let k = 1; k < n; k++) {
      expectClose(out[2 * k], 0);
      expectClose(out[2 * k + 1], 0);
    }
  });

  it("single sample (n=1) passes through unchanged", async () => {
    const out = (await gpuFft(...args, new Float32Array([3.5]))) as Float32Array;
    expect(Array.from(out)).toHaveLength(2);
    expectClose(out[0], 3.5);
    expectClose(out[1], 0);
  });

  it("real cosine tone: energy at the expected conjugate bins", async () => {
    const n = 16;
    const freq = 2;
    const input = new Float32Array(n);
    for (let i = 0; i < n; i++) input[i] = Math.cos((2 * Math.PI * freq * i) / n);
    const out = (await gpuFft(...args, input)) as Float32Array;
    const expected = cpuFft(input);
    expectSpectrumClose(out, expected, n);
  });

  it("matches CPU DFT on a non-power-of-frequency mixed signal", async () => {
    const n = 32;
    const input = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      input[i] = 0.5 * Math.sin((2 * Math.PI * 3 * i) / n) + 0.25 * Math.cos((2 * Math.PI * 7 * i) / n) + 0.1;
    }
    const out = (await gpuFft(...args, input)) as Float32Array;
    expectSpectrumClose(out, cpuFft(input), n);
  });

  // Size sweep across power-of-two block boundaries (the workgroup size is 64,
  // so 64/128 cross the per-workgroup boundary for both the bit-reverse and the
  // n/2 butterfly dispatch). FFT requires power-of-two lengths.
  const sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 65536];
  for (const n of sizes) {
    it(`matches CPU DFT for n=${n}`, async () => {
      const input = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        input[i] = Math.sin((2 * Math.PI * 3 * i) / n) + 0.5 * Math.cos((2 * Math.PI * 11 * i) / n);
      }
      const out = (await gpuFft(...args, input)) as Float32Array;
      // For the larger sizes the O(n^2) CPU DFT is the bottleneck; cap it.
      if (n <= 4096) {
        expectSpectrumClose(out, cpuFft(input), n);
      } else {
        // n=65536: spot-check the DC bin (sum of the signal) and that the length is right.
        expect(out.length).toBe(2 * n);
        let sum = 0;
        for (let i = 0; i < n; i++) sum += input[i];
        expectClose(out[0], sum, { eps: 1e-2 * n });
        expectClose(out[1], 0, { eps: 1e-2 * n });
      }
    });
  }

  it("preserves f32 output dtype for an i32 input (numeric cast)", async () => {
    const n = 8;
    const input = new Int32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const out = (await gpuFft(...args, input)) as Float32Array;
    expect(out).toBeInstanceOf(Float32Array);
    const ref = cpuFft(input);
    expectSpectrumClose(out, ref, n);
  });

  it("keepOnGpu returns a GPUArray of f32 with length 2*n", async () => {
    const n = 16;
    const input = new Float32Array(n).fill(1);
    const out = await gpuFft(...args, input, { keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.dtype).toBe("f32");
    expect(out.length).toBe(2 * n);
    const arr = await out.toArray();
    expectClose(arr[0], n);
    out.destroy();
  });

  it("accepts an f32 GPUArray input and reads it in place", async () => {
    const device = await deviceManager.getDevice();
    const n = 8;
    const data = new Float32Array(n).fill(1);
    const gpuIn = new GPUArray(
      uploadBuffer(device, data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, bufferPool),
      n,
      "f32",
      device,
      bufferPool
    );
    const out = (await gpuFft(...args, gpuIn)) as Float32Array;
    expectClose(out[0], n);
    // input reused, not consumed
    expect(Array.from(await gpuIn.toArray())).toEqual(Array.from(data));
    gpuIn.destroy();
  });

  it("throws on a non-power-of-two length", async () => {
    await expect(gpuFft(...args, new Float32Array(6))).rejects.toThrow(/power of two/);
  });

  it("re-runs over pooled buffers without stale state", async () => {
    const n = 32;
    const input = new Float32Array(n);
    for (let i = 0; i < n; i++) input[i] = Math.sin((2 * Math.PI * 5 * i) / n);
    const first = (await gpuFft(...args, input)) as Float32Array;
    const second = (await gpuFft(...args, input)) as Float32Array;
    for (let i = 0; i < first.length; i++) expectClose(second[i], first[i], { eps: 1e-3 * n });
  });
});
