# Searchsorted: left vs right, out-of-range queries, and the ascending requirement

`gpu.searchsorted(sorted, queries, { side })` returns, for each query, the index
at which it would be inserted into `sorted` to keep the array ordered. It is the
GPU equivalent of NumPy's `numpy.searchsorted`: one binary search per query, run
by its own GPU thread. `sorted` and `queries` share the input dtype; the result
is always a `Uint32Array` of length `queries.length`.

```javascript
await gpu.searchsorted([1, 3, 5, 7], [0, 1, 2, 3, 8]);                  // [0, 0, 1, 1, 4]
await gpu.searchsorted([1, 3, 5, 7], [0, 1, 3, 8], { side: "right" }); // [0, 1, 2, 4]
```

## `left` vs `right`

The `side` option chooses *which* insertion point is returned when the query is
equal to one or more elements of `sorted`:

- **`side: "left"`** (the default) returns the **leftmost** valid position — the
  number of elements **strictly less than** `q`. This is a lower bound: inserting
  `q` here places it before any equal elements.
- **`side: "right"`** returns the **rightmost** valid position — the number of
  elements **less than or equal to** `q`. This is an upper bound: inserting `q`
  here places it after any equal elements.

They differ **only** when `q` is present in `sorted`. For a run of duplicates the
two sides bracket the run: with `sorted = [2, 2, 2]` and `q = 2`, `left` is `0`
(before the run) and `right` is `3` (after it). For `q` not in the array, both
sides return the same index.

The count interpretation is the easiest mental model: `left` counts elements
`< q`, `right` counts elements `<= q`.

## Out-of-range queries

A query outside the range of `sorted` is not an error — it simply lands at an
end:

- `q` below the minimum returns `0`.
- `q` above the maximum returns `sorted.length`.

This holds for both sides and matches NumPy.

## `sorted` must be ascending

The search assumes `sorted` is in **ascending** order and never verifies it (an
O(n) check would defeat the point of a binary search). If `sorted` is unsorted or
descending, the returned indices are **undefined** — the algorithm still
terminates and returns *some* `u32`, but it is meaningless. Sort the array first
(e.g. with `gpu.sort`) if you are not certain it is ordered.

Ties in `sorted` are fine — duplicates are exactly what `left`/`right` are there
to disambiguate.

## dtype and backends

`sorted` and `queries` are read in the same dtype (`f32`, `i32`, or `u32`),
inferred from `sorted`; `i32` correctly handles negative values. The output is
`u32` regardless. Because the result is integer insertion indices, the GPU and
CPU-fallback paths agree exactly — there is no floating-point reassociation
caveat here (unlike reductions; see [numerics.md](./numerics.md)).
