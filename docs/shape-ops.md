# Shape ops: cast, transpose, reshape

Three ergonomics ops for changing an array's dtype or shape. `cast` runs a kernel,
`transpose` runs a tiled kernel, and `reshape` is pure metadata (no kernel).

## Cast: out-of-range and negative-to-unsigned are implementation-defined

`gpu.cast(input, dtype)` converts each element with WGSL's value-conversion
constructors (`i32()`, `u32()`, `f32()`). For **in-range** values the result
matches a plain TypedArray assignment, which is what the CPU fallback does:

```javascript
await gpu.cast([1.9, 2.1, -3.7], "i32");   // [1, 2, -3] — truncates toward zero
```

What is **not** portable:

- **Float out of the integer range.** Converting an `f32` whose value exceeds the
  target integer's range (e.g. `1e30 -> i32`) is implementation-defined in WGSL.
- **Negative to unsigned.** Converting a negative number to `u32` (e.g.
  `-1 -> u32`) is implementation-defined — do not rely on a particular wrap.

Stay within the destination type's range (and non-negative for `u32`) for results
that agree across GPUs and with the CPU fallback. The tests only assert the
well-defined cases (truncation of in-range floats, non-negative integer
round-trips).

## Transpose: the dims describe the input

`gpu.transpose(input, { rows, cols })` treats `input` as a **row-major `rows × cols`**
matrix and returns its transpose, the logically `cols × rows` matrix, flattened
row-major (`output[c*rows + r] = input[r*cols + c]`).

```javascript
// 2x3 -> 3x2
await gpu.transpose([1, 2, 3, 4, 5, 6], { rows: 2, cols: 3 }); // [1, 4, 2, 5, 3, 6]
```

The kernel is tiled with a `TILE+1`-wide shared tile (to avoid shared-memory bank
conflicts), the same shape of kernel as `matmul`. Dimensions that are **not** a
multiple of the tile size are handled by bounds checks — the result is exact.

A returned `GPUArray` carries `shape: [cols, rows]`. Two consequences:

- **Inferred dims.** If you pass a 2D `GPUArray` (e.g. straight from `reshape`)
  with no `{ rows, cols }`, the dims come from its `shape`.
- **Double transpose.** Transposing the result again (with its inferred shape)
  reproduces the original array.

A plain CPU array carries no shape, so the CPU fallback requires `{ rows, cols }`.

## Reshape: a zero-copy, non-owning view

`gpu.reshape(input, shape)` reinterprets the element count under a new shape
without moving any data. It always returns a `GPUArray`, and the element count of
`shape` must equal the array's length or it throws.

- **`GPUArray` input → a non-owning view.** The view shares the **same buffer** as
  the source (`view.buffer === source.buffer`). It is a zero-copy alias with a new
  `shape`.
- **CPU array input → a fresh owning `GPUArray`** with the shape (one upload).

### Ownership footgun

The view does **not** own its buffer — the **source** array does. So:

- Destroying the view (`view.destroy()`) leaves the source's buffer intact; the
  source still reads correctly afterwards.
- Destroying the **source** frees the buffer. Any view still pointing at it is now
  dangling — **don't `destroy()` the source while a view is in use.**

This matches the buffer-ownership rules in the README's "GPU-Resident Arrays"
section: a `GPUArray` you receive is yours to manage, but a reshaped view borrows
its source's buffer rather than owning a new one.
