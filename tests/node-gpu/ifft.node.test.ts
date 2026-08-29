import { describe, it, expect } from "vitest";
import { gpuFft, gpuIfft, cpuFft, cpuIfft } from "../../src/ops/fft";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";
import { uploadBuffer } from "../../src/core/command";
import { expectClose } from "../shared/tolerance";

// Drive gpuIfft (and gpuFft's complexInput path) directly, no fallback wrapper,
// so a broken conjugate/seed/scale pass fails loudly.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

// After the 1/n scaling an ifft output is O(signal amplitude), but the butterfly
// tree reassociates f32 adds and GPU sin/cos carry ~2^-11 absolute error per
// stage, so the error grows with the stage count log2(n), not with n itself.
function roundTripEps(n: number): number {
  return 1e-3 + 3e-3 * Math.log2(Math.max(n, 2));
}

describe("ifft (real GPU)", () => {
  // Round trip: fft of a real signal, then ifft, must recover the signal
  // (real parts close to the input, imaginary parts close to zero).
  for (const n of [8, 64, 1024]) {
    it(`ifft(fft(x)) recovers a real signal for n=${n}`, async () => {
      const signal = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        signal[i] =
          Math.sin((2 * Math.PI * 3 * i) / n) +
          0.5 * Math.cos((2 * Math.PI * 7 * i) / n) +
          0.25;
      }
      const spec = await gpuFft(...args, signal, { keepOnGpu: true });
      const time = (await gpuIfft(...args, spec)) as Float32Array;
      spec.destroy();
      expect(time).toBeInstanceOf(Float32Array);
      expect(time.length).toBe(2 * n);
      const eps = roundTripEps(n);
      for (let i = 0; i < n; i++) {
        expectClose(time[2 * i], signal[i], { eps });
        expectClose(time[2 * i + 1], 0, { eps });
      }
    });
  }

  it("complex-input fft matches the CPU reference", async () => {
    const n = 32;
    const interleaved = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) {
      interleaved[2 * i] = Math.sin(i * 0.7) + (0.2 * i) / n;
      interleaved[2 * i + 1] = Math.cos(i * 1.3) - 0.1;
    }
    const out = (await gpuFft(...args, interleaved, { complexInput: true })) as Float32Array;
    const ref = cpuFft(interleaved, { complexInput: true });
    expect(out.length).toBe(2 * n);
    // Spectrum bins are O(n), same tolerance scaling as the fft suite.
    for (let i = 0; i < ref.length; i++) expectClose(out[i], ref[i], { eps: 1e-3 * n });
  });

  it("complex input with zero imaginary parts matches the real-input fft", async () => {
    const n = 16;
    const real = new Float32Array(n);
    const interleaved = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) {
      real[i] = Math.cos((2 * Math.PI * 2 * i) / n) + 0.5;
      interleaved[2 * i] = real[i];
    }
    const fromComplex = (await gpuFft(...args, interleaved, { complexInput: true })) as Float32Array;
    const fromReal = (await gpuFft(...args, real)) as Float32Array;
    for (let i = 0; i < fromReal.length; i++) {
      expectClose(fromComplex[i], fromReal[i], { eps: 1e-3 * n });
    }
  });

  it("ifft of a single-bin spectrum is the expected complex exponential", async () => {
    const n = 16;
    const k = 3;
    const spec = new Float32Array(2 * n);
    spec[2 * k] = n; // X[k] = n  ->  x[t] = exp(2*pi*i*k*t/n)
    const time = (await gpuIfft(...args, spec)) as Float32Array;
    for (let t = 0; t < n; t++) {
      const angle = (2 * Math.PI * k * t) / n;
      expectClose(time[2 * t], Math.cos(angle), { eps: 5e-3 });
      expectClose(time[2 * t + 1], Math.sin(angle), { eps: 5e-3 });
    }
  });

  it("matches cpuIfft on an arbitrary spectrum", async () => {
    const n = 64;
    const spec = new Float32Array(2 * n);
    for (let i = 0; i < 2 * n; i++) spec[i] = Math.sin(i * 2.1) * 3 + Math.cos(i * 0.4);
    const out = (await gpuIfft(...args, spec)) as Float32Array;
    const ref = cpuIfft(spec);
    expect(out.length).toBe(2 * n);
    for (let i = 0; i < ref.length; i++) expectClose(out[i], ref[i], { eps: roundTripEps(n) });
  });

  it("linearity: ifft(a*X + b*Y) = a*ifft(X) + b*ifft(Y)", async () => {
    const n = 16;
    const a = 2;
    const b = -0.5;
    const X = new Float32Array(2 * n);
    const Y = new Float32Array(2 * n);
    const combined = new Float32Array(2 * n);
    for (let i = 0; i < 2 * n; i++) {
      X[i] = Math.sin(i * 1.7);
      Y[i] = Math.cos(i * 0.9) * 2;
      combined[i] = a * X[i] + b * Y[i];
    }
    const outC = (await gpuIfft(...args, combined)) as Float32Array;
    const outX = (await gpuIfft(...args, X)) as Float32Array;
    const outY = (await gpuIfft(...args, Y)) as Float32Array;
    for (let i = 0; i < 2 * n; i++) {
      expectClose(outC[i], a * outX[i] + b * outY[i], { eps: 5e-3 });
    }
  });

  it("ifft of a single complex point (n=1) passes through unchanged", async () => {
    const time = (await gpuIfft(...args, new Float32Array([2.5, -1]))) as Float32Array;
    expect(time.length).toBe(2);
    expectClose(time[0], 2.5);
    expectClose(time[1], -1);
  });

  it("complex-input fft of a single point (n=1) passes through unchanged", async () => {
    const out = (await gpuFft(...args, new Float32Array([1.5, 0.5]), {
      complexInput: true,
    })) as Float32Array;
    expect(out.length).toBe(2);
    expectClose(out[0], 1.5);
    expectClose(out[1], 0.5);
  });

  it("numerically casts an integer spectrum to f32", async () => {
    // n = 4, X[0] = 8 -> constant signal of 8/4 = 2.
    const spec = new Int32Array([8, 0, 0, 0, 0, 0, 0, 0]);
    const out = (await gpuIfft(...args, spec)) as Float32Array;
    expect(out).toBeInstanceOf(Float32Array);
    for (let t = 0; t < 4; t++) {
      expectClose(out[2 * t], 2);
      expectClose(out[2 * t + 1], 0);
    }
  });

  it("keepOnGpu returns a GPUArray of f32 with length 2*n", async () => {
    const n = 8;
    const spec = new Float32Array(2 * n);
    spec[0] = 3 * n; // DC only -> constant real signal of 3
    const out = await gpuIfft(...args, spec, { keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.dtype).toBe("f32");
    expect(out.length).toBe(2 * n);
    const arr = await out.toArray();
    for (let t = 0; t < n; t++) {
      expectClose(arr[2 * t], 3);
      expectClose(arr[2 * t + 1], 0);
    }
    out.destroy();
  });

  it("accepts a GPUArray spectrum and does not mutate it", async () => {
    const device = await deviceManager.getDevice();
    const n = 8;
    const data = new Float32Array(2 * n);
    data[0] = n;
    data[2 * 2] = 4;
    data[2 * 2 + 1] = -4;
    const gpuIn = new GPUArray(
      uploadBuffer(device, data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, bufferPool),
      2 * n,
      "f32",
      device,
      bufferPool
    );
    const out = (await gpuIfft(...args, gpuIn)) as Float32Array;
    const ref = cpuIfft(data);
    for (let i = 0; i < ref.length; i++) expectClose(out[i], ref[i], { eps: 5e-3 });
    // input reused, not consumed or overwritten
    expect(Array.from(await gpuIn.toArray())).toEqual(Array.from(data));
    gpuIn.destroy();
  });

  it("ifft throws on lengths that are not 2*n with n a power of two", async () => {
    await expect(gpuIfft(...args, new Float32Array(6))).rejects.toThrow(/2\*n/); // n=3, not a power of two
    await expect(gpuIfft(...args, new Float32Array(5))).rejects.toThrow(/2\*n/); // odd, not pairs
    await expect(gpuIfft(...args, new Float32Array(0))).rejects.toThrow(/2\*n/); // empty
  });

  it("complex-input fft throws on lengths that are not 2*n with n a power of two", async () => {
    await expect(
      gpuFft(...args, new Float32Array(6), { complexInput: true })
    ).rejects.toThrow(/2\*n/);
    await expect(
      gpuFft(...args, new Float32Array(3), { complexInput: true })
    ).rejects.toThrow(/2\*n/);
  });

  it("re-runs over pooled buffers without stale state", async () => {
    const n = 32;
    const spec = new Float32Array(2 * n);
    for (let i = 0; i < 2 * n; i++) spec[i] = Math.sin(i * 0.3);
    const first = (await gpuIfft(...args, spec)) as Float32Array;
    const second = (await gpuIfft(...args, spec)) as Float32Array;
    for (let i = 0; i < first.length; i++) expectClose(second[i], first[i], { eps: 1e-4 });
  });
});
