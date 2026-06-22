# Histogram: edge-clamp behaviour and u32 counts

`gpu.histogram(input, { bins, min, max })` counts `input` into `bins` equal-width
buckets spanning `[min, max]` and returns a `Uint32Array` of length `bins`. Each
input element does one atomic increment into its bucket, so the counts are exact
regardless of how the GPU threads interleave.

```javascript
await gpu.histogram([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], { bins: 5, min: 0, max: 10 });
// Uint32Array [2, 2, 2, 2, 2]
```

## Binning rule

For each value `x`:

```
bin = floor((x - min) / (max - min) * bins)
```

then the bin is **clamped** into `[0, bins - 1]`. The bins are half-open on the
left, `[edge, edge + width)`, so a value exactly on an interior edge falls into
the higher bin — except `x == max`, which clamps into the last bin (`bins - 1`)
rather than spilling past the end.

## Out-of-range values clamp to the edge bins

A value below `min` lands in bin `0`; a value at or above `max` lands in bin
`bins - 1`. **Nothing is dropped.** This differs from NumPy's `histogram`, which
discards values outside the range. If you need NumPy's drop semantics, filter the
input before calling `histogram`.

## `max == min` guard

When `max == min` the range is zero, so every value falls in bin `0`.

```javascript
await gpu.histogram([3, 3, 3], { bins: 4, min: 3, max: 3 });
// Uint32Array [3, 0, 0, 0]
```

## Counts are always u32

The output dtype is always `u32` (bin counts), regardless of the input dtype.
With `keepOnGpu: true` the returned `GPUArray` has `dtype === "u32"` and
`length === bins`.

## Input is cast to f32 for the bucket math

The bucket arithmetic runs in `f32`, so `i32` and `u32` inputs work as well as
`f32`. For inputs whose magnitude exceeds the `f32` mantissa (about 2^24), the
cast can round values onto a neighbouring bin edge — keep `min`, `max`, and the
data within `f32` precision when exact edges matter.
