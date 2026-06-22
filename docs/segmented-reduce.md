# Segmented reduce: per-segment (group-by) reductions

`gpu.segmentedReduce(values, segmentIds, { numSegments, op })` reduces `values`
(length `N`) into `numSegments` groups keyed by `segmentIds` (length `N`, read as
`u32`). Output element `s` is the reduction of every `values[i]` whose
`segmentIds[i] === s`. It is the group-by / per-segment building block — the GPU
form of a pandas `groupby(...).sum()` once your rows are tagged with a segment id.

```javascript
// Sum each group. seg0: 10+20=30, seg1: 1+2=3, seg2: 30.
await gpu.segmentedReduce([10, 1, 20, 2, 30], [0, 1, 0, 1, 2], { numSegments: 3 });
// Float32Array [30, 3, 30]

await gpu.segmentedReduce(values, segIds, { numSegments: 4, op: "max" });
```

The output length is always `numSegments`, and its dtype follows `values`
(`segmentIds` is always read as `u32`).

## Reductions

`op` selects the associative combine (default `"sum"`):

- **`"sum"`** — `Σ values` per segment.
- **`"product"`** — `Π values` per segment.
- **`"min"` / `"max"`** — the extremum per segment.

Every reduction is **collision-safe**: each input element folds into its segment
through a GPU atomic, so duplicate segment ids accumulate correctly regardless of
how the threads interleave — the same guarantee `scatter` with `mode: "add"`
gives. Integer `sum` uses the native `atomicAdd`; the other cases
(`product`/`min`/`max` on integers, and **all** `f32` reductions, since WGSL has
no `atomic<f32>`) use a portable compare-and-swap loop over the slot's bits. Under
heavy contention (many rows hammering one hot segment) those CAS retries
serialize, so spreading rows across more segments helps throughput.

## Empty segments hold the identity

A segment that no input targets reads back the op's identity:

| op | identity |
|---|---|
| `sum` | `0` |
| `product` | `1` |
| `min` | `+3.4e38` (largest finite f32 / type max) |
| `max` | `-3.4e38` (smallest finite f32 / type min) |

```javascript
await gpu.segmentedReduce([5, 7], [0, 3], { numSegments: 4 });
// Float32Array [5, 0, 0, 7]  — segments 1 and 2 are empty → 0
```

The `min`/`max` identities are the same safe finite extremes the `min`/`max`
reductions use, so a padding/empty segment never beats a real value.

## Out-of-range segment ids clamp

The GPU cannot throw, so a segment id `>= numSegments` is **clamped** to the last
segment (`numSegments - 1`), matching the `scatter`/`gather` shaders. Tag your
rows with ids in `[0, numSegments)` if you need a clean error instead of a silent
clamp. `numSegments` must be `> 0`.

## Pairs with sort-by-key / searchsorted

`segmentIds` is whatever assigns each row to a group. Two common sources:

- Sort rows by a key with `sortByKey`, then turn run boundaries into ids.
- Bucket a sorted axis with `searchsorted` to get a bin index per row, then
  `segmentedReduce` to aggregate within each bin.

## Floating-point caveat

`f32` `sum`/`product` reassociate across the atomic accumulation order, so the
**last bits** of a segment's result can differ from a sequential CPU reduction —
the same non-associativity caveat that applies to GPU reductions and
`scatter`-add (see [numerics.md](./numerics.md)). Compare `f32` results with a
tolerance. Integer `sum`/`product` are exact as long as the true result fits the
type; `min`/`max` are exact for every dtype.

## `keepOnGpu`

With `{ keepOnGpu: true }` the result stays on the device as a `GPUArray` of
`length === numSegments` and `dtype` matching `values`, ready to chain into the
next op without a readback.
