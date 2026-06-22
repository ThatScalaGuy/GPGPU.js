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

### Map

```javascript
// Arrow function (auto-compiled to WGSL)
await gpu.map(array, x => x * 2 + 1)

// String expression (minifier-safe)
await gpu.map(array, "x * x + 1")
```

**Supported in expressions:**
- Arithmetic: `+ - * / %`
- Comparisons: `< > <= >= == !=`
- Ternary: `a > 0 ? a : -a`
- Math: `Math.abs`, `Math.sqrt`, `Math.pow`, `Math.min`, `Math.max`, `Math.floor`, `Math.ceil`, `Math.sin`, `Math.cos`, `Math.tan`, `Math.exp`, `Math.log`

### Reduce

```javascript
await gpu.reduce(array, (a, b) => a + b, 0)  // custom reduce
await gpu.sum(array)                           // sum
await gpu.min(array)                           // minimum
await gpu.max(array)                           // maximum
await gpu.product(array)                       // product
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

### Pipeline

Chain operations to keep data on the GPU between steps:

```javascript
const result = await gpu.pipeline()
  .map(x => x * 2)
  .map(x => x + 1)
  .reduce((a, b) => a + b, 0)
  .run(inputData);
```

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

`keepOnGpu` works on `add`/`subtract`/`multiply`/`divide`, `map`, `matmul`,
`scan`, `sort`, `pipeline().run()`, and `createKernel().run()` (`sortByKey`
returns a pair of `GPUArray`s under `keepOnGpu`). Reductions
(`sum`/`min`/`max`/`product`/`reduce`) accept a `GPUArray` input but always return
a scalar `number`. In-place ops (`scan`, `sort`) never mutate a `GPUArray` input.

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
