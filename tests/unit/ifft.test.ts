import { describe, it, expect } from "vitest";
import { cpuFft, cpuIfft } from "../../src/ops/fft";

// cpuIfft is the O(n^2) inverse-DFT oracle the GPU ifft is checked against, and
// cpuFft's complexInput option is the oracle for the complex-input forward path.
// These tests pin their contracts: interleaved [re, im] pairs of length 2*n, the
// 2*n power-of-two guard, the 1/n scaling, and round-trip consistency.
function close(a: number, b: number, eps = 1e-4): void {
  expect(Math.abs(a - b)).toBeLessThan(eps);
}

describe("cpuIfft (inverse DFT reference)", () => {
  it("returns interleaved complex of length 2*n", () => {
    const out = cpuIfft([1, 0, 2, 0, 3, 0, 4, 0]); // n = 4
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(8);
  });

  it("cpuIfft(cpuFft(x)) recovers a real signal", () => {
    const n = 8;
    const signal = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      signal[i] = Math.sin((2 * Math.PI * 2 * i) / n) + 0.5;
    }
    const time = cpuIfft(cpuFft(signal));
    for (let i = 0; i < n; i++) {
      close(time[2 * i], signal[i], 1e-3);
      close(time[2 * i + 1], 0, 1e-3);
    }
  });

  it("single-bin spectrum -> complex exponential with positive twiddle sign", () => {
    const n = 8;
    const k = 1;
    const spec = new Float32Array(2 * n);
    spec[2 * k] = n; // X[1] = n
    const time = cpuIfft(spec);
    for (let t = 0; t < n; t++) {
      const angle = (2 * Math.PI * k * t) / n;
      close(time[2 * t], Math.cos(angle), 1e-3);
      close(time[2 * t + 1], Math.sin(angle), 1e-3);
    }
  });

  it("scales by 1/n (DC bin of n -> constant 1)", () => {
    const n = 4;
    const spec = new Float32Array(2 * n);
    spec[0] = n;
    const time = cpuIfft(spec);
    for (let t = 0; t < n; t++) {
      close(time[2 * t], 1);
      close(time[2 * t + 1], 0);
    }
  });

  it("single complex point (n=1) passes through unchanged", () => {
    const time = cpuIfft([3.5, -2]);
    expect(time.length).toBe(2);
    close(time[0], 3.5);
    close(time[1], -2);
  });

  it("throws on lengths that are not 2*n with n a power of two", () => {
    expect(() => cpuIfft([1, 2, 3, 4, 5, 6])).toThrow(/2\*n/); // n=3
    expect(() => cpuIfft([1, 2, 3])).toThrow(/2\*n/); // odd
    expect(() => cpuIfft([])).toThrow(/2\*n/); // empty
  });
});

describe("cpuFft complexInput", () => {
  it("complex exponential -> single bin of n", () => {
    const n = 8;
    const f = 2;
    const interleaved = new Float32Array(2 * n);
    for (let t = 0; t < n; t++) {
      const angle = (2 * Math.PI * f * t) / n;
      interleaved[2 * t] = Math.cos(angle);
      interleaved[2 * t + 1] = Math.sin(angle);
    }
    const out = cpuFft(interleaved, { complexInput: true });
    expect(out.length).toBe(2 * n);
    for (let k = 0; k < n; k++) {
      close(out[2 * k], k === f ? n : 0, 1e-3);
      close(out[2 * k + 1], 0, 1e-3);
    }
  });

  it("zero imaginary parts match the real-input path", () => {
    const n = 8;
    const real = [3, 1, 4, 1, 5, 9, 2, 6];
    const interleaved = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) interleaved[2 * i] = real[i];
    const fromComplex = cpuFft(interleaved, { complexInput: true });
    const fromReal = cpuFft(real);
    for (let i = 0; i < fromReal.length; i++) close(fromComplex[i], fromReal[i], 1e-3);
  });

  it("throws on lengths that are not 2*n with n a power of two", () => {
    expect(() => cpuFft([1, 2, 3, 4, 5, 6], { complexInput: true })).toThrow(/2\*n/);
    expect(() => cpuFft([1, 2, 3], { complexInput: true })).toThrow(/2\*n/);
  });

  it("real-input path is unchanged when the option is absent", () => {
    const out = cpuFft([1, 1, 1, 1]);
    close(out[0], 4);
    close(out[1], 0);
    expect(() => cpuFft([1, 2, 3])).toThrow(/power of two/);
  });
});
