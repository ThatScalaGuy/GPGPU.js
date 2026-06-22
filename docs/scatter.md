# Scatter: duplicate-index semantics and the f32 atomic-add caveat

`gpu.scatter(dst, idx, vals, { mode })` returns a copy of `dst` (length `N`) with
`vals` (length `M`) written at the positions named by `idx` (length `M`, read as
`u32`). It is the inverse of `gather`: where gather *reads* `src[idx[i]]`,
scatter *writes* `out[idx[i]]`. The output dtype follows `dst`.

```javascript
await gpu.scatter([0, 0, 0, 0], [3, 1], [9, 5]);                 // [0, 5, 0, 9]
await gpu.scatter(dst, idx, vals, { mode: "add" });             // out[idx[i]] += vals[i]
```

## Two modes

- **`mode: "set"`** (default) does `out[idx[i]] = vals[i]`.
- **`mode: "add"`** does `out[idx[i]] += vals[i]` atomically.

The distinction only matters when `idx` contains **duplicate** indices — when two
or more elements of `vals` target the same output slot.

## Duplicate indices

Because every element of `idx` is handled by its own GPU thread, duplicate
indices land on the same memory location concurrently.

- **`set`**: the writes **race**. Exactly one of the colliding values survives,
  and *which* one is **nondeterministic** — it depends on thread scheduling, not
  on array order. There is no "last in `idx` wins" guarantee on the GPU. Use
  `set` only when you know `idx` has no duplicates (e.g. a permutation), or when
  you genuinely do not care which duplicate wins.
- **`add`**: the updates are **atomic**, so every contribution is counted exactly
  once. The result is **deterministic** regardless of how the threads interleave
  — `[1, 1, 1]` scattered with `mode: "add"` onto the same bin adds `3`. This is
  the right mode for histograms, segment sums, and any accumulation.

The CPU fallback matches these rules as closely as a single-threaded loop can:
`set` is last-write-wins in `idx` order, and `add` accumulates. For `add` the two
backends agree; for `set` with duplicates, treat the winner as unspecified.

## Out-of-range indices clamp

The GPU cannot throw, so an index `>= N` is clamped to the last element
(`N - 1`), matching the gather shader and the CPU fallback. Pass indices in range
if you need a clean error instead of a silent clamp.

## The f32 atomic-add caveat

WGSL's atomics are defined only for `i32` and `u32`. Integer scatter-add
therefore uses the native `atomicAdd`, which is both correct and fast.

`f32` has **no** native atomic add. The shader instead reinterprets each slot's
bits as `u32` and runs a compare-and-swap loop with
`atomicCompareExchangeWeak`: read the current bits, compute
`bitcast<f32>(bits) + v`, and try to swap the new bits in; on a lost race, retry
with the freshly observed value. This is portable, stays within core WGSL (no
extensions), and produces the correct sum — but under **heavy collision** (many
threads hammering the same slot) the retries serialize, so an `f32` scatter-add
into a few hot bins can be slower than the integer path. Spreading writes across
more bins, or using an integer dtype where the data allows, avoids the
contention.

Floating-point addition is not associative, so the order in which contributions
land can shift the **last bits** of an `f32` scatter-add result relative to a
sequential CPU sum — the same reassociation caveat that applies to GPU
reductions (see [numerics.md](./numerics.md)). Integer scatter-add is exact as
long as the true sum fits the type.
