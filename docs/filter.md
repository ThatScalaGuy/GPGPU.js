# Filter: the boolean-predicate rule and the count readback

`gpu.filter(input, predicate)` returns the elements of `input` for which
`predicate(x, i, len)` is true, keeping their original order. The output is
shorter than (or equal to) the input — this is the library's first
variable-length-output op, and learning that length is what makes `filter`
slightly different from every other op.

```javascript
await gpu.filter([1, 2, 3, 4, 5, 6], x => x > 3);       // [4, 5, 6]
await gpu.filter([1, 2, 3, 4, 5, 6], "x % 2 == 0");     // [2, 4, 6]
await gpu.filter([10, 20, 30, 40], (x, i) => i % 2 == 0); // [10, 30]
await gpu.filter(data, (x, i, len) => i < len / 2);     // first half
```

## The predicate must be a boolean expression

Unlike `map`, whose function returns a **number** that is written straight to the
output, `filter`'s predicate must be a **boolean** expression — built from the
comparison and logical operators `< > <= >= == != && ||` (and `x`, the element
index `i`, and the length `len`):

```javascript
await gpu.filter(a, x => x > 5);          // ok
await gpu.filter(a, "x % 2 == 0");        // ok
await gpu.filter(a, (x, i, len) => i < len / 2); // ok
await gpu.filter(a, x => x * 2);          // WGSL type error — not a boolean
```

The expression is compiled with the same codegen as `map`, then wrapped in
`select(0u, 1u, (<expr>))` to turn the boolean into the `1u`/`0u` keep-flag. A
non-boolean expression (e.g. `x * 2`) makes the `select` ill-typed and fails to
compile rather than silently misbehaving.

`&&` and `||` lower to WGSL bitwise `&` / `|`, which is correct for the boolean
operands comparisons produce. The predicate dtype follows the input dtype, so an
`i32`/`u32` input correctly compares integers (including negatives for `i32`).

## How it works: flags → scan → compaction

`filter` composes four GPU passes:

1. **flags** — one thread per element writes `1u` where the predicate holds, else
   `0u`.
2. **scan** — an inclusive prefix sum of the flags (reusing the multi-block
   `gpu.scan`), so each kept element learns its 1-based output position.
3. **count** — the **last** scan element is the number of kept elements.
4. **compact** — each kept element `i` writes `input[i]` to
   `output[scanned[i] - 1]`.

Because compaction is a scatter into per-element slots that never collide,
order is preserved exactly and values are copied verbatim — there is no
floating-point reassociation, so the GPU and CPU-fallback paths agree exactly
(unlike reductions; see [numerics.md](./numerics.md)).

## The one small GPU→CPU readback

Every other array op knows its output length up front. `filter` does not: the
count is data-dependent, computed on the GPU by the scan. To size the output
buffer and the returned array, `filter` copies that single `u32` (the last scan
element) back to the CPU and waits on it — one tiny `mapAsync`. This is the only
synchronisation point, and it is unavoidable for a variable-length result; the
flags, scan, and compaction passes themselves stay entirely on the GPU.

Two boundary cases short-circuit around the readback's surroundings:

- **Empty input** returns a length-0 result without dispatching anything.
- **Nothing kept** (count `0`) skips the compaction dispatch and returns a
  length-0 array.

## `keepOnGpu`

With `{ keepOnGpu: true }` the result stays on the device as a `GPUArray` whose
`.length` is the kept count; read it later with `.toArray()`. The count readback
still happens (the length must be known to build the handle), but the kept values
are never copied to the CPU. A `GPUArray` input is never mutated — the flags pass
writes to its own buffer and the scan copies the flags, so the original data is
untouched.
