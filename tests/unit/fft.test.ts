import { describe, it, expect } from "vitest";
import { cpuFft } from "../../src/ops/fft";

// cpuFft is the O(n^2) DFT oracle the GPU FFT is checked against. These tests pin
// its contract: interleaved complex output [re0, im0, ...] of length 2*n, the
// power-of-two guard, and a few closed-form spectra.
function close(a: number, b: number, eps = 1e-4): void {
  expect(Math.abs(a - b)).toBeLessThan(eps);
}

describe("cpuFft (DFT reference)", () => {
  it("returns interleaved complex of length 2*n", () => {
    const out = cpuFft([1, 2, 3, 4]);
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(8);
  });

  it("all-ones -> spike at bin 0 equal to n, zeros elsewhere", () => {
    const n = 8;
    const out = cpuFft(new Array(n).fill(1));
    close(out[0], n);
    close(out[1], 0);
    for (let k = 1; k < n; k++) {
      close(out[2 * k], 0);
      close(out[2 * k + 1], 0);
    }
  });

  it("single sample passes through (n=1)", () => {
    const out = cpuFft([2.5]);
    expect(out.length).toBe(2);
    close(out[0], 2.5);
    close(out[1], 0);
  });

  it("real cosine tone: conjugate-symmetric pair of n/2", () => {
    const n = 8;
    const freq = 1;
    const input = new Float32Array(n);
    for (let i = 0; i < n; i++) input[i] = Math.cos((2 * Math.PI * freq * i) / n);
    const out = cpuFft(input);
    // cos -> real spikes of n/2 at bins freq and n-freq.
    close(out[2 * freq], n / 2);
    close(out[2 * freq + 1], 0);
    close(out[2 * (n - freq)], n / 2);
    close(out[2 * (n - freq) + 1], 0);
    // bin 0 is ~0 for a zero-mean cosine.
    close(out[0], 0);
  });

  it("real sine tone: imaginary, antisymmetric pair of n/2", () => {
    const n = 8;
    const freq = 1;
    const input = new Float32Array(n);
    for (let i = 0; i < n; i++) input[i] = Math.sin((2 * Math.PI * freq * i) / n);
    const out = cpuFft(input);
    // sin -> imag spikes of -n/2 at +freq and +n/2 at n-freq.
    close(out[2 * freq], 0);
    close(out[2 * freq + 1], -n / 2);
    close(out[2 * (n - freq)], 0);
    close(out[2 * (n - freq) + 1], n / 2);
  });

  it("bin 0 equals the sum of the signal (DC term)", () => {
    const input = [3, 1, 4, 1, 5, 9, 2, 6];
    const out = cpuFft(input);
    const sum = input.reduce((a, b) => a + b, 0);
    close(out[0], sum);
    close(out[1], 0);
  });

  it("numerically casts an Int32Array input to f32", () => {
    const out = cpuFft(new Int32Array([1, 2, 3, 4]));
    expect(out).toBeInstanceOf(Float32Array);
    const sum = 1 + 2 + 3 + 4;
    close(out[0], sum);
  });

  it("throws on a non-power-of-two length", () => {
    expect(() => cpuFft([1, 2, 3])).toThrow(/power of two/);
  });
});
