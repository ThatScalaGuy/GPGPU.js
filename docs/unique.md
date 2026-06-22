# Unique: sorted distinct values via sort → boundary flags → compaction

`gpu.unique(input)` returns the **distinct** values of `input` in **ascending
sorted order** — the same semantics as NumPy's `np.unique`. Like `filter`, it is
a variable-length-output op: the result is shorter than (or equal to) the input,
and the length is data-dependent.

```javascript
await gpu.unique([3, 1, 2, 3, 1, 2, 3]);        // [1, 2, 3]
await gpu.unique([5, 4, 3, 2, 1]);              // [1, 2, 3, 4, 5]  (all distinct → just sorted)
await gpu.unique([7, 7, 7, 7]);                 // [7]
await gpu.unique(new Int32Array([3, -5, 0, -5])); // Int32Array [-5, 0, 3]
```

The output dtype follows the input dtype (`f32` / `i32` / `u32`), so an `i32`
input dedupes integers — including negatives — and yields an `Int32Array`.

## How it works: sort → flags → scan → compaction

`unique` composes the existing building blocks:

1. **sort** — reuse the multi-block `gpu.sort` so equal values become adjacent
   runs. A `GPUArray` input is never mutated (sort copies it).
2. **flags** — one thread per element writes `1u` at each **run boundary**
   (`i == 0` or `sorted[i] != sorted[i-1]`), else `0u`.
3. **scan** — an inclusive prefix sum of the flags (reusing the multi-block
   `gpu.scan`), so each first-of-run element learns its 1-based output position.
4. **count** — the **last** scan element is the number of distinct values.
5. **compact** — each boundary element `i` writes `sorted[i]` to
   `output[scanned[i] - 1]`.

Because compaction copies values verbatim into per-element slots that never
collide, there is no floating-point reassociation: the GPU and CPU-fallback paths
agree **exactly**, even for `f32` (unlike reductions; see
[numerics.md](./numerics.md)). Equality is bit-exact — two `f32` values are "the
same" only if their bits match. `NaN` is not handled (it never compares equal),
matching `gpu.sort`.

## The one small GPU→CPU readback

As with `filter`, the output length is not known up front. After the scan,
`unique` copies a single `u32` (the last scan element) back to the CPU and waits
on it to size the output buffer and the returned array — one tiny `mapAsync`.
This is the only synchronisation point; the sort, flags, scan, and compaction
passes themselves stay entirely on the GPU. An **empty input** short-circuits to
a length-0 result without dispatching anything.

## `keepOnGpu`

With `{ keepOnGpu: true }` the result stays on the device as a `GPUArray` whose
`.length` is the distinct count; read it later with `.toArray()`. The count
readback still happens (the length must be known to build the handle), but the
distinct values are never copied to the CPU. A `GPUArray` input is never
mutated — the sort copies it and the flags pass writes to its own buffer.
