# GPGPU.js

[![npm version](https://img.shields.io/npm/v/@thatscalaguy/gpgpu.js.svg)](https://www.npmjs.com/package/@thatscalaguy/gpgpu.js)
[![CI](https://github.com/ThatScalaGuy/GPGPU.js/actions/workflows/ci.yml/badge.svg)](https://github.com/ThatScalaGuy/GPGPU.js/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

General-Purpose GPU Computing in JavaScript using WebGPU. Write GPU-accelerated code with zero boilerplate — no shader knowledge required.

> **Try it now:** experiment with GPGPU.js right in your browser at the [hosted playground](https://thatscalaguy.github.io/GPGPU.js/).

## Features

- **Minimal API** — `gpu.add(a, b)`, `gpu.map(arr, x => x * 2)`, `gpu.sum(arr)`
- **Auto WGSL codegen** — Arrow functions compile to GPU shaders automatically
- **Pipeline chaining** — Chain ops without CPU-GPU roundtrips
- **GPU-resident arrays** — `gpu.upload()` + `keepOnGpu` keep data on the GPU across ops
- **CPU fallback** — Runs everywhere, accelerates where WebGPU is available
- **TypeScript** — Full type safety with zero runtime dependencies
- **Tree-shakeable** — ESM + CJS, import only what you use

## Install

```bash
npm install @thatscalaguy/gpgpu.js
```

## Quick Start

```javascript
import { gpu } from "@thatscalaguy/gpgpu.js";

// Element-wise operations
const sum = await gpu.add([1, 2, 3], [4, 5, 6]);       // [5, 7, 9]
const scaled = await gpu.multiply([1, 2, 3], 10);       // [10, 20, 30]

// Map with JS arrow functions (auto-compiled to GPU shaders)
const doubled = await gpu.map([1, 2, 3, 4], x => x * 2);           // [2, 4, 6, 8]
const transformed = await gpu.map(data, x => Math.sqrt(x) + 1);    // sqrt(x) + 1

// Reduce
const total = await gpu.sum([1, 2, 3, 4, 5]);           // 15
const maximum = await gpu.max([3, 1, 4, 1, 5, 9]);      // 9

// Matrix multiply
const result = await gpu.matmul(matA, matB, { rowsA: 64, colsA: 64, colsB: 64 });

// Pipeline — data stays on GPU between steps
const result = await gpu.pipeline()
  .map(x => x * 2)
  .map(x => x + 1)
  .reduce((a, b) => a + b, 0)
  .run(data);

// Keep data on the GPU across ops — one upload, one readback
const g = await gpu.upload([1, 2, 3, 4]);   // GPUArray
const scaled = await gpu.multiply(g, 2);     // GPUArray (input was on GPU)
const shifted = await gpu.add(scaled, 1);    // GPUArray
const out = await shifted.toArray();         // [3, 5, 7, 9]
g.destroy(); scaled.destroy(); shifted.destroy();

// Cleanup
gpu.destroy();
```

## API Reference

### Element-wise Operations

```javascript
await gpu.add(a, b)        // a + b (arrays or array + scalar)
await gpu.subtract(a, b)   // a - b
await gpu.multiply(a, b)   // a * b
await gpu.divide(a, b)     // a / b
```

**Broadcasting.** When two operands have different shapes, `add`/`subtract`/
`multiply`/`divide` (and `zip`) apply [NumPy broadcasting](./docs/broadcasting.md):
shapes align from the trailing dimension, and a dimension of size 1 stretches to
match. Shapes come from [`reshape`](#reshape); an unshaped array is read as 1-D.

```javascript
const mat = await gpu.reshape([1, 2, 3, 4, 5, 6], [2, 3]); // shape [2, 3]
const row = await gpu.reshape([10, 20, 30], [3]);          // shape [3]
await gpu.add(mat, row, { keepOnGpu: true });              // row added to every row → [2, 3]
```

Equal-length operands keep the plain element-wise fast path. Shapes that can't be
broadcast (e.g. `[2,3]` vs `[2]`) throw. See [docs/broadcasting.md](./docs/broadcasting.md).

### Map

```javascript
// Arrow function (auto-compiled to WGSL)
await gpu.map(array, x => x * 2 + 1)

// String expression (minifier-safe)
await gpu.map(array, "x * x + 1")

// Index-aware: the expression also sees the element index `i` and the length `len`
await gpu.map(array, (x, i, len) => x * i)
await gpu.map(array, "i / len")               // 0, 1/n, 2/n, ...
```

**Supported in expressions:**
- Arithmetic: `+ - * / %`
- Comparisons: `< > <= >= == !=`
- Ternary: `a > 0 ? a : -a`
- Math: `Math.abs`, `Math.sqrt`, `Math.pow`, `Math.min`, `Math.max`, `Math.floor`, `Math.ceil`, `Math.round`, `Math.sign`, `Math.clamp`, `Math.sin`, `Math.cos`, `Math.tan`, `Math.exp`, `Math.log`

**Captured arrays (`consts`).** A map expression can read extra arrays by name —
the escape hatch for windowing, lookup tables, and n-ary elementwise math that
`zip` (two inputs) can't express. A `GPUArray` const binds in place (no upload);
a plain array uploads per call:

```javascript
// Apply a window function: out[i] = x * hann[i]
await gpu.map(samples, "x * hann[i]", { consts: { hann } });

// Three-array elementwise: out[i] = x * gains[i] + offsets[i]
await gpu.map(signal, "x * gains[i] + offsets[i]", { consts: { gains, offsets } });
```

### Zip

Combine two arrays element by element with a custom function — `map` with two
inputs. Accepts the same arrow-function or string forms as `map`, and applies
[NumPy broadcasting](./docs/broadcasting.md) when the shapes differ:

```javascript
await gpu.zip(a, b, (a, b) => a * b + 1);
await gpu.zip(prices, quantities, "a * b");
```

### Reduce

```javascript
await gpu.reduce(array, (a, b) => a + b, 0)  // custom reduce
await gpu.sum(array)                           // sum
await gpu.min(array)                           // minimum
await gpu.max(array)                           // maximum
await gpu.product(array)                       // product
await gpu.argmin(array)                        // index of the minimum
await gpu.argmax(array)                        // index of the maximum
```

`argmin`/`argmax` return the **first** index on ties (NumPy tie-break) and `-1`
for an empty array.

### Gather

Indexed reads: `out[k] = src[idx[k]]` — permutations, lookups, and reordering.
`idx` is read as `u32`; the result length is `idx.length` and the dtype follows
`src`. An out-of-range index clamps to the last element (the GPU can't throw).

```javascript
await gpu.gather([10, 20, 30, 40], [3, 0, 1]);  // [40, 10, 20]
```

### Matrix Multiply

```javascript
await gpu.matmul(a, b, { rowsA, colsA, colsB })
```

Flat arrays with explicit dimensions. Uses tiled GPU algorithm with shared memory.

### Sort

```javascript
await gpu.sort(array)  // GPU-accelerated bitonic sort
```

### Sort by key

Sort `keys` ascending and permute `values` to follow, returning `[sortedKeys,
sortedValues]`. `keys` and `values` must be the same length; the `values` dtype
is independent of the `keys` dtype.

```javascript
const [keys, values] = await gpu.sortByKey([3, 1, 2], [30, 10, 20]);
// keys   -> [1, 2, 3]
// values -> [10, 20, 30]
```

- **Not stable.** For equal keys the relative order of their values is
  unspecified (bitonic sort is not stable) — see
  [docs/sort-by-key.md](./docs/sort-by-key.md).

### Prefix Sum (Scan)

```javascript
await gpu.scan(array)                          // default: addition
await gpu.scan(array, (a, b) => a + b, 0)     // custom scan
```

### Filter

Keep the elements for which `predicate(x, i, len)` holds, in order. The result is
shorter than (or equal to) the input — `filter` is the library's first
variable-length-output op.

```javascript
await gpu.filter([1, 2, 3, 4, 5, 6], x => x > 3);   // [4, 5, 6]
await gpu.filter([1, 2, 3, 4, 5, 6], "x % 2 == 0"); // [2, 4, 6]
await gpu.filter(data, (x, i, len) => i < len / 2); // first half
```

- The predicate must be a **boolean** expression (`< > <= >= == != && ||`),
  unlike `map`, whose function returns a **number**. Internally the expression is
  wrapped in `select(0u, 1u, (<expr>))`, so a non-boolean expression is a WGSL
  type error.
- Order-preserving: kept elements stay in their original relative order, with
  exact values (no floating-point reassociation). The output dtype follows the
  input.
- Built from a flags pass → prefix-sum scan → compaction, with **one small
  GPU→CPU readback** to learn the result length (see [docs/filter.md](./docs/filter.md)).

### Scatter

Write values into a copy of `dst` at the positions given by `idx` (the inverse of
a gather). `idx` is read as `u32`; the output dtype follows `dst`.

```javascript
// set (default): out[idx[i]] = vals[i]
await gpu.scatter([0, 0, 0, 0], [3, 1], [9, 5]);              // [0, 5, 0, 9]

// add: out[idx[i]] += vals[i], atomically
await gpu.scatter(bins, idx, ones, { mode: "add" });          // histogram-style accumulation
```

- **`set`** (default) overwrites. On **duplicate** indices the writes race and the
  surviving value is nondeterministic — use `set` only for duplicate-free indices
  (e.g. a permutation).
- **`add`** accumulates atomically, so duplicates are summed exactly and the result
  is deterministic. Integer add uses native atomics; `f32` add uses a portable
  compare-and-swap loop (see [docs/scatter.md](./docs/scatter.md)).
- An out-of-range index clamps to the last element (the GPU can't throw).

### Searchsorted

Binary-search each query's insertion point into an **ascending** `sorted` array,
one thread per query. NumPy-compatible. `sorted` and `queries` share the input
dtype; the result is always a `Uint32Array` of length `queries.length`.

```javascript
// left (default): count of elements strictly < q
await gpu.searchsorted([1, 3, 5, 7], [0, 1, 2, 3, 8]);                  // Uint32Array [0, 0, 1, 1, 4]

// right: count of elements <= q
await gpu.searchsorted([1, 3, 5, 7], [0, 1, 3, 8], { side: "right" }); // Uint32Array [0, 1, 2, 4]
```

- **`side: "left"`** (default) returns the leftmost insertion point — the count of
  elements `< q`. **`side: "right"`** returns the rightmost — the count of
  elements `<= q`. They differ only when `q` equals an element of `sorted`.
- Out-of-range queries return `0` (below the minimum) or `sorted.length` (above
  the maximum).
- `sorted` **must** be ascending; results are undefined otherwise (not checked).
  See [docs/searchsorted.md](./docs/searchsorted.md).

### Histogram

Count values into `bins` equal-width buckets over `[min, max]`. Each element does
one atomic increment, so counts are exact. The result is always a `Uint32Array`
of length `bins`, whatever the input dtype.

```javascript
await gpu.histogram([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], { bins: 5, min: 0, max: 10 });
// Uint32Array [2, 2, 2, 2, 2]
```

- The bin is `floor((x - min) / (max - min) * bins)`, clamped into `[0, bins-1]`.
- **Out-of-range values clamp to the edge bins** — they are *not* dropped (unlike
  NumPy). `x == max` lands in the last bin; `max == min` puts everything in bin 0.
- Binning casts the input to `f32`, so `i32`/`u32` inputs work too (see
  [docs/histogram.md](./docs/histogram.md)).

### Cast

Convert an array to another dtype, element by element. The output dtype is the one
you pass; the length is unchanged.

```javascript
await gpu.cast([1.9, 2.1, -3.7], "i32");   // Int32Array [1, 2, -3] (truncates toward zero)
await gpu.cast(new Int32Array([1, 2, 3]), "f32");  // Float32Array [1, 2, 3]
```

- Uses WGSL's value constructors (`i32()`/`u32()`/`f32()`), so `f32 -> i32`/`u32`
  truncates toward zero.
- **Out-of-range and negative-to-unsigned conversions are
  implementation-defined** — see [docs/shape-ops.md](./docs/shape-ops.md).

### Transpose

Transpose a flat row-major matrix. `{ rows, cols }` describe the **input**; the
result is the logically `cols × rows` transpose, flattened.

```javascript
// [[1,2,3],[4,5,6]] (2x3) -> [[1,4],[2,5],[3,6]] (3x2)
await gpu.transpose([1, 2, 3, 4, 5, 6], { rows: 2, cols: 3 });
// Float32Array [1, 4, 2, 5, 3, 6]
```

- Tiled kernel with a shared, bank-conflict-padded tile (modeled on `matmul`).
- A returned `GPUArray` carries `shape: [cols, rows]`, so you can `transpose` it
  again with no explicit dims (double transpose returns the original).
- For a 2D `GPUArray` from `reshape`, the dims are inferred from its `shape` —
  pass no `{ rows, cols }`.

### Reshape

Reinterpret an array's shape without moving data. Returns a `GPUArray`.

```javascript
const g = await gpu.upload([1, 2, 3, 4, 5, 6]);
const m = await gpu.reshape(g, [2, 3]);     // zero-copy view, shape [2, 3]
const t = await gpu.transpose(m);            // dims inferred from the shape
```

- A `GPUArray` input returns a **zero-copy, non-owning view** that shares the
  source buffer. The **source** array owns the buffer — don't `destroy()` the
  source while a view is in use (see [docs/shape-ops.md](./docs/shape-ops.md)).
- A CPU array is uploaded to a fresh, owning `GPUArray` with the shape.
- The element count must match the new shape, or it throws.

### `gpu.unique(input, opts?)`

Returns the **distinct** values of `input` in **ascending sorted order** (NumPy
`np.unique` semantics). Variable-length output: the result is shorter than (or
equal to) the input, and the length is data-dependent. Output dtype follows the
input dtype.

```javascript
await gpu.unique([3, 1, 2, 3, 1, 2, 3]);          // [1, 2, 3]
await gpu.unique([5, 4, 3, 2, 1]);                // [1, 2, 3, 4, 5]
await gpu.unique(new Int32Array([3, -5, 0, -5])); // Int32Array [-5, 0, 3]
```

Implemented as `sort` → run-boundary flags → `scan` → stream compaction (reusing
the multi-block `gpu.sort` and `gpu.scan`). Values are copied verbatim, so the
GPU and CPU paths agree exactly for every dtype. A `GPUArray` input is never
mutated. With `{ keepOnGpu: true }` the result stays on the device as a
`GPUArray`. See [docs/unique.md](./docs/unique.md).

### `gpu.segmentedReduce(values, segmentIds, { numSegments, op? })`

Per-segment (group-by) reduction. Reduces `values` (length `N`) into `numSegments`
groups keyed by `segmentIds` (length `N`, read as `u32`): output `s` is the
reduction of every `values[i]` where `segmentIds[i] === s`. The GPU form of
`groupby(...).agg()`. Output dtype follows `values`; length is `numSegments`.

```javascript
// Sum each group. seg0: 10+20, seg1: 1+2, seg2: 30.
await gpu.segmentedReduce([10, 1, 20, 2, 30], [0, 1, 0, 1, 2], { numSegments: 3 });
// Float32Array [30, 3, 30]

await gpu.segmentedReduce(values, segIds, { numSegments: 4, op: "max" });
```

- `op`: `"sum"` (default) | `"product"` | `"min"` | `"max"`. Collision-safe via GPU atomics.
- Empty segments hold the op's identity (`0` / `1` / `+3.4e38` / `-3.4e38`).
- Out-of-range ids clamp to the last segment; `numSegments` must be `> 0`.
- `f32` `sum`/`product` reassociate (compare with a tolerance); integer ops and `min`/`max` are exact.

See [docs/segmented-reduce.md](./docs/segmented-reduce.md).

### `gpu.random(n, opts?)`

Generate `n` reproducible pseudo-random values on the GPU (no input array). The GPU counterpart to `numpy.random` for Monte-Carlo, initialisation, and sampling.

```javascript
await gpu.random(5);                    // Float32Array, 5 uniforms in [0, 1)
await gpu.random(5, { seed: 42 });      // a different, fixed stream
await gpu.random(5, { dtype: "u32" });  // raw 32-bit integers
await gpu.random(1_000_000, { keepOnGpu: true }); // stays GPU-resident for chaining
```

The generator is stateless and counter-based: each output is a pure hash of `(seed, index)`, so the same `(n, seed, dtype)` always yields the same values — bit-for-bit on the GPU and the CPU fallback alike (no float-reassociation caveat). `seed` defaults to `0`. `dtype` (default `"f32"`) picks the output: `"f32"` → uniform `[0, 1)`; `"u32"` → raw 32-bit; `"i32"` → those bits as signed. See [docs/random.md](./docs/random.md).

### `gpu.fft(input, opts?)`

Forward FFT of a **real-valued** signal. Returns the **complex** spectrum as an
interleaved `Float32Array` `[re0, im0, re1, im1, ...]` of length `2 * n`. WGSL has
no native complex type, so a complex number is carried as a `vec2<f32>`.

- `input` length **must be a power of two** (radix-2 Cooley-Tukey); a
  non-power-of-two length throws.
- Output dtype is always `f32` (the spectrum is complex); `i32`/`u32` inputs are
  numerically cast to a real `f32` signal.
- `{ keepOnGpu: true }` returns a `GPUArray` with `dtype === "f32"` and
  `length === 2 * n`.

```javascript
await gpu.fft([1, 1, 1, 1, 1, 1, 1, 1]);
// Float32Array(16): X[0] = (8, 0), all other bins ~ (0, 0)
// bin k:  re = out[2*k],  im = out[2*k + 1]
```

See [docs/fft.md](./docs/fft.md) for the interleaved-complex layout, the
power-of-two contract, and the per-stage butterfly design.

### `gpu.convolve(input, kernel, opts?)`

Discrete 1-D convolution of `input` with `kernel`, matching `numpy.convolve`. The
kernel is reversed relative to the signal (true convolution, not correlation).

- `opts.mode` selects the output length: `"full"` (default, `n + m - 1`),
  `"same"` (`max(n, m)`, centred), or `"valid"` (`max(n, m) - min(n, m) + 1`,
  full overlap only).
- Output dtype follows `input` (`f32`/`i32`/`u32`).
- `{ keepOnGpu: true }` returns a `GPUArray`.

```javascript
await gpu.convolve([1, 2, 3, 4], [1, 1, 1]);                   // [1, 3, 6, 9, 7, 4]
await gpu.convolve([1, 2, 3, 4], [1, 1, 1], { mode: "same" });  // [3, 6, 9, 7]
await gpu.convolve([1, 2, 3, 4], [1, 1, 1], { mode: "valid" }); // [6, 9]
```

See [docs/convolve.md](./docs/convolve.md) for the mode conventions and the
per-output-element kernel design.

### Pipeline

Chain operations to keep data on the GPU between steps — every stage reads the
previous stage's buffer, so there is one upload at the start and one readback at the
end:

```javascript
const result = await gpu.pipeline()
  .map(x => x * 2)
  .map(x => x + 1)
  .reduce((a, b) => a + b, 0)
  .run(inputData);
```

Available steps: `.map(fn)`, `.scan(fn?, identity?)` (defaults to an inclusive prefix
sum), `.filter(predicate)`, `.sort()`, `.cast(dtype)`, `.unique()`,
`.histogram({ bins, min, max })`, `.convolve(kernel, { mode })`, `.gather(indices)`, and
`.reduce(fn, identity)`. `filter`/`unique`/`gather`/`convolve` produce a length that
differs from the input; `cast` changes the element type for every subsequent step, and
`histogram` makes the stream `bins` u32 counts. `reduce` collapses the stream to a scalar
and must be the terminal step.

```javascript
// histogram -> scan = cumulative distribution, entirely on the GPU
const cdf = await gpu.pipeline()
  .histogram({ bins: 5, min: 0, max: 10 })
  .scan()
  .run(samples);
```

```javascript
// keep evens, running total, then sum — all on the GPU
const total = await gpu.pipeline()
  .map(x => x * 2)
  .filter(x => x > 5)
  .scan()
  .reduce((a, b) => a + b, 0)
  .run([1, 2, 3, 4, 5, 6]);
```

Consecutive `.map()` steps are fused into a single kernel (no intermediate buffers or
dispatches). `run()` honours `keepOnGpu` like any other op, returning a `GPUArray` you
can feed into the next pipeline.

### GPU-Resident Arrays

By default every op uploads its input, computes, and reads the result back to the
CPU. When you chain several ops on the same data, you can keep it on the GPU and
pay for just one upload and one readback.

`gpu.upload()` returns a `GPUArray` — a handle to GPU-resident data:

```javascript
const g = await gpu.upload([1, 2, 3, 4]);   // GPUArray
const result = await g.toArray();             // read back to a TypedArray
g.destroy();                                  // free the GPU buffer
```

Ops accept a `GPUArray` anywhere they accept an array. **Auto mode**: if any input
is a `GPUArray`, the result is returned as a `GPUArray` (no readback); otherwise
you get a `TypedArray` as before:

```javascript
const g = await gpu.upload(data);
const a = await gpu.multiply(g, 2);   // GPUArray (input was on GPU)
const b = await gpu.map(a, x => x + 1);
const out = await b.toArray();        // read back once, at the end
g.destroy(); a.destroy(); b.destroy();
```

Override auto mode per call with `keepOnGpu`:

```javascript
await gpu.add([1, 2, 3], 1, { keepOnGpu: true });  // force a GPUArray from CPU input
await gpu.add(g, 1, { keepOnGpu: false });          // force readback to a TypedArray
```

`keepOnGpu` works on every array-returning op: `add`/`subtract`/`multiply`/`divide`,
`map`, `zip`, `gather`, `filter`, `searchsorted`, `cast`, `transpose`, `scatter`,
`histogram`, `matmul`, `scan`, `sort`, `unique`, `segmentedReduce`, `random`,
`fft`, `convolve`, `pipeline().run()`, and `createKernel().run()` (`sortByKey`
returns a pair of `GPUArray`s under `keepOnGpu`; `reshape` always returns a
`GPUArray`). Reductions (`sum`/`min`/`max`/`product`/`reduce`/`argmin`/`argmax`)
accept a `GPUArray` input but always return a scalar `number`. In-place ops
(`scan`, `sort`) never mutate a `GPUArray` input.

> **Ownership:** a `GPUArray` you receive is yours to manage — read it with
> `toArray()` (which keeps it alive) or release it with `destroy()`. Leaking
> handles ties up pooled GPU buffers.

### Custom Kernel

For power users who want to write raw WGSL:

```javascript
const kernel = await gpu.createKernel({
  workgroupSize: 64,
  shader: `
    @group(0) @binding(0) var<storage, read> input0: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
      let idx = gid.x;
      output[idx] = input0[idx] * input0[idx];
    }
  `,
  inputs: [{ type: "f32", size: 1024 }],
  output: { type: "f32", size: 1024 },
});

const result = await kernel.run(inputData);
```

### Instance Management

```javascript
import { GPU } from "@thatscalaguy/gpgpu.js";

// Use the default singleton
import { gpu } from "@thatscalaguy/gpgpu.js";

// Or create your own instance
const myGpu = new GPU();
myGpu.destroy(); // cleanup when done
```

### Backend & Fallback Control

When an op can't run on the GPU it falls back to a CPU implementation. By default
that's silent apart from a `console.warn`, so callers can't tell which backend
ran, how long it took, or stop the fallback from happening. Configure these on
the instance — at construction or via mutable fields — and every op honours them:

```javascript
import { GPU } from "@thatscalaguy/gpgpu.js";

const gpu = new GPU({
  // "warn" (default) logs + falls back, "silent" falls back quietly,
  // "throw" re-throws the GPU error instead of falling back.
  fallback: "throw",
  // Fires after every op with the backend that ran and how long it took.
  onStats: ({ op, backend, ms }) => console.log(`${op} ran on ${backend} in ${ms.toFixed(2)}ms`),
  // Fires when a GPU op throws, before the fallback policy is applied.
  onFallback: ({ op, error }) => report(op, error),
});

// Fields are mutable too:
gpu.onStats = (s) => metrics.record(s);
gpu.fallback = "silent";
```

- `onStats` reports `{ op, backend: "gpu" | "cpu", ms }` for **every** op,
  including GPU-resident and `createKernel`/`pipeline` runs (always `"gpu"`).
  The reduce family (`reduce`/`sum`/`min`/`max`/`product`) still returns a
  scalar `number` but reports stats just like the rest.
- `onFallback` and the `fallback` policy apply only to ops with a CPU fallback;
  forced-GPU paths (a `GPUArray` input, `keepOnGpu`, custom kernels) throw on
  failure regardless.

## How It Works

1. **You write JavaScript** — `x => x * 2 + 1`
2. **Parser extracts the expression** — builds an intermediate representation
3. **WGSL emitter generates GPU shader code** — `output[idx] = (x * 2.0) + 1.0;`
4. **WebGPU compiles and dispatches** — runs on thousands of GPU cores in parallel
5. **Results returned as Float32Array** — ready to use

The library manages GPU device initialization, buffer pooling, shader caching, and data transfer automatically.

## Troubleshooting

When a shader fails to compile, the thrown error includes a code frame of the generated WGSL (with a caret under the failing column) and, for codegen ops, a `from expression:` note echoing your JS. See [docs/debugging.md](./docs/debugging.md).

## Numerical Precision

Reductions (`sum`, `reduce`, `scan`) and the index reductions (`argmin`,
`argmax`) run as a **parallel tree** on the GPU. Because floating-point addition
is not associative, reassociating the additions can make a GPU float result
differ from a sequential CPU result **in the last bits** — even though every
individual IEEE-754 op is deterministic. Tests compare GPU floats against a CPU
reference with a small tolerance rather than strict equality. Integer reductions
are exact. See [docs/numerics.md](./docs/numerics.md) for details.

## Browser Support

WebGPU is supported in:
- Chrome 113+ / Edge 113+
- Firefox 141+ (Windows), 145+ (macOS)
- Safari 18+

When WebGPU is unavailable, all operations automatically fall back to CPU implementations.

## Node.js

GPGPU.js also runs on **Node** via Google Dawn. Node has no built-in
`navigator.gpu`, so install the optional [`webgpu`](https://www.npmjs.com/package/webgpu)
package and the library auto-detects it — no configuration:

```bash
npm install @thatscalaguy/gpgpu.js webgpu
```

`webgpu` is an optional peer dependency; the import is lazy and Node-gated, so it
never enters the browser bundle. Deno and Web Workers expose `navigator.gpu`
natively and work with no extra setup. See [docs/runtimes.md](./docs/runtimes.md)
for details and the `f32`/`i32`/`u32` dtype boundary.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for setup
and workflow, and [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) for community
guidelines. Security issues: see [SECURITY.md](./SECURITY.md).

## License

[MIT](./LICENSE) © Sven Herrmann
