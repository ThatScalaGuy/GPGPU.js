# Convolve: 1-D discrete convolution, kernel reversal, and the three modes

`gpu.convolve(input, kernel)` computes the discrete 1-D convolution of a signal
with a kernel — the GPU equivalent of NumPy's `numpy.convolve`. Each output
element is one dot product of the signal against the **reversed** kernel, and
every output element is computed by its own GPU thread (embarrassingly parallel,
no shared state, no scan).

```javascript
await gpu.convolve([1, 2, 3], [0, 1, 0.5]);                  // [0, 1, 2.5, 4, 1.5]
await gpu.convolve([1, 2, 3], [0, 1, 0.5], { mode: "same" }); // [1, 2.5, 4]
await gpu.convolve([1, 2, 3, 4, 5], [1, 1, 1], { mode: "valid" }); // [6, 9, 12]
```

The result dtype follows `input` (the first argument); the kernel is read in the
same dtype. The kernel is typically much shorter than the signal (a filter tap
list), but any non-empty lengths are accepted.

## Convolution, not correlation: the kernel is reversed

`convolve` implements the mathematical convolution sum

```
out[p] = Σ_k input[p - k] * kernel[k]
```

so the kernel is applied **reversed** relative to the signal — exactly like
`numpy.convolve`. This is the difference from cross-correlation (a sliding dot
product with the kernel un-reversed). The cleanest way to see it: convolving a
unit impulse `[1, 0, 0, 0]` with `[1, 2, 3]` stamps the **un-reversed** kernel
into the output (`[1, 2, 3, 0, 0, 0]`), because the impulse selects a single
shifted copy of the kernel. For a symmetric kernel (a box blur, a Gaussian) the
distinction vanishes; for an asymmetric one it matters. Reverse your kernel
before passing it if you actually want correlation.

## The three modes

`mode` selects which slice of the full convolution is returned. The full result
has length `N + M - 1` (signal length `N`, kernel length `M`); `same` and `valid`
are that result trimmed to a centred or fully-overlapping window — the kernel
maths is identical, only the returned range changes.

- **`"full"`** (the default) — every position where the two signals overlap at
  all, including the partial overlaps at both ends. Length `N + M - 1`.
- **`"same"`** — the centred window of length `max(N, M)`, so the output lines up
  with the input. This is what you want for filtering a signal in place.
- **`"valid"`** — only the positions where the kernel lies **entirely** within
  the signal (no zero-padding at the edges). Length `max(N, M) - min(N, M) + 1`.

Boundaries are treated as zero padding: where the reversed kernel hangs off the
end of the signal, the missing samples contribute nothing. There is no `mode`
that wraps around or reflects.

## How it works

One thread computes one output element. The thread maps its output index to a
position `p` in the full-mode output (adding a fixed per-mode `offset`, so
`same`/`valid` reuse the same kernel without a second pass), then loops `k` over
just the kernel taps that land inside the signal — `k ∈ [max(0, p-(N-1)),
min(p, M-1)]` — accumulating `input[p-k] * kernel[k]`. The signal and kernel are
two read-only storage buffers; the four sizes (`N`, `M`, output length, offset)
ride in a uniform. The output length is known up front from `N`, `M`, and `mode`,
so — unlike `filter` — there is no GPU→CPU readback.

`GPUArray` inputs for either the signal or the kernel are read in place and never
mutated, so a convolved result can feed straight back into another op with
`{ keepOnGpu: true }`.

## dtype and numerics

`input` and `kernel` share the dtype inferred from `input` (`f32`, `i32`, or
`u32`); `i32` correctly handles negative taps. For integer dtypes the GPU and
CPU-fallback paths agree **exactly**. For `f32`, each output element is a sum of
up to `min(N, M)` products, and that sum is accumulated in a different order on
the GPU than on the CPU, so the last bits can differ — compare with a tolerance
(see [numerics.md](./numerics.md)), not strict equality. Longer kernels (more
summed terms) accumulate slightly more reassociation error.
