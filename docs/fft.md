# FFT: complex as `vec2<f32>` and the power-of-two contract

`gpu.fft(input)` computes the forward discrete Fourier transform of a
**real-valued** signal and returns its **complex** spectrum. WGSL has no native
complex type (and no `f64`), so a complex number is carried as a `vec2<f32>` of
`(real, imag)`. The returned `Float32Array` is that pair stream **interleaved**:

```
[re0, im0, re1, im1, ..., re(n-1), im(n-1)]   // length 2*n
```

```javascript
await gpu.fft([1, 1, 1, 1, 1, 1, 1, 1]);
// Float32Array(16): X[0] = (8, 0), all other bins ~ (0, 0)
// -> [8,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0]
```

Reading a single bin `k` is `re = out[2*k]`, `im = out[2*k + 1]`; its magnitude
is `Math.hypot(out[2*k], out[2*k+1])`.

## Input length must be a power of two

`fft` uses an iterative **radix-2 Cooley-Tukey** transform, so the input length
`n` must be a power of two (1, 2, 4, 8, … 65536, …). A non-power-of-two length
throws:

```javascript
await gpu.fft(new Float32Array(6)); // Error: fft: input length must be a power of two, got 6
```

If your data isn't a power of two, zero-pad it to the next one yourself before
calling (zero-padding changes the frequency resolution but not the contract).

## Output is always complex `f32`

The spectrum is complex regardless of the input's dtype, so the result is always
`f32`. An `i32`/`u32` input is numerically cast to `f32` and treated as a real
signal. With `{ keepOnGpu: true }` the returned `GPUArray` has `dtype === "f32"`
and `length === 2 * n` (the interleaved length, not `n`).

```javascript
const spec = await gpu.fft(signal, { keepOnGpu: true });
spec.length; // 2 * signal.length
```

## How it works: bit-reversal + log₂(n) butterfly stages

`fft` is a multi-pass op (like `scan`), all passes recorded into a single command
encoder so the transform never round-trips through the CPU between stages:

1. **Bit-reversal permutation** — one thread per sample scatters the real input
   into a complex buffer at its bit-reversed index, with imaginary part `0`. This
   is the decimation-in-time pre-ordering that makes the following stages operate
   on contiguous half-blocks.
2. **`log₂(n)` butterfly stages** — each stage launches `n/2` butterfly threads.
   A thread combines a pair `(a, b)` separated by the stage's half-size with the
   twiddle factor `w = exp(-2πi·k / (2·half))`, writing `a + w·b` and `a − w·b`.
   The complex multiply is done by hand on the `vec2<f32>` lanes. The two complex
   buffers ping-pong between stages.

After the final stage the live buffer holds the spectrum; the other ping-pong
buffer is returned to the pool.

## Numerics

The butterfly tree reassociates floating-point adds and multiplies, and the
twiddles are computed in `f32`, so the GPU spectrum will not match a sequential
CPU DFT bit-for-bit — compare with a tolerance, not strict equality (see
[numerics.md](./numerics.md)). The error also grows with `n`: a single tone of
amplitude `A` produces a bin of magnitude `~A·n/2`, so absolute tolerances should
scale with `n`. This op's tests compare against an O(n²) DFT reference with an
`eps` proportional to `n`.
