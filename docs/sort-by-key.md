# Sort by key: stability and the type-max sentinel caveat

`gpu.sortByKey(keys, values)` returns `[sortedKeys, sortedValues]`: it sorts
`keys` into ascending order and applies the same permutation to `values`, so
each value stays paired with its original key. It is the keyed extension of
`gpu.sort` — same GPU bitonic sort, with a parallel `values` payload that is
swapped whenever the comparison swaps two keys.

```javascript
const [keys, values] = await gpu.sortByKey([3, 1, 2], [30, 10, 20]);
// keys   -> [1, 2, 3]
// values -> [10, 20, 30]
```

`keys` and `values` must be the same length. The `values` dtype is independent
of the `keys` dtype (both are 4-byte elements): you can carry, say, `u32`
payload ids alongside `f32` sort keys. Only the keys are compared; the values
are pure passengers.

## Not stable

The sort is **not stable**. For a run of **equal** keys, the relative order of
their paired values is **unspecified** — it depends on the bitonic network and
GPU thread scheduling, not on the input order. A keys-only `gpu.sort` hides this
(equal keys are indistinguishable once sorted), but with a payload it becomes
visible: two equal keys may emerge with their values in either order.

```javascript
// keys [1, 1] with values [10, 11] may come back as values [10, 11] OR [11, 10].
await gpu.sortByKey([1, 1], [10, 11]);
```

If you need values in a defined order within equal keys, sort on a composite key
that breaks ties (e.g. fold a secondary ordinal into the key), or pre-sort and
use a stable CPU sort. Do not rely on, or test for, a particular per-duplicate
value order.

## The type-max key sentinel caveat

To sort an array whose length is not a power of two, the implementation pads
both buffers up to the next power of two. Pad **keys** are set to the largest
value of the key type — `+Infinity` for `f32`, `2147483647` for `i32`,
`4294967295` for `u32` — so padding always sorts to the tail and is trimmed off
on readback (pad **values** are set to `0` and ride along, then get trimmed with
their sentinel keys).

The consequence — identical to `gpu.sort` — is that a **real** key equal to that
type-max sentinel is indistinguishable from padding and may be trimmed away
wrongly when the length is not already a power of two. In practice: avoid using
`+Infinity` (f32) or the exact integer max as a genuine key. Powers-of-two
lengths use no padding and are unaffected.
