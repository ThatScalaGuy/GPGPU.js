# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-08-29

A correctness-hardening pass plus the most-requested convenience ops. A review
of v0.4.0 turned up several silent-corruption bugs (all with regression tests
now), and the new ops close the NumPy-family gaps: statistics, `topK`, an
inverse FFT, GPU-side constructors, and `slice`. There is also a benchmark
harness with published numbers.

### Added

- **Statistics:** `mean`, `variance`, `std` (population, like NumPy), `dot`,
  `norm` (L2), `cosineSimilarity`, and `softmax` (numerically stable
  `exp(x - max)`, always `f32`). All are compositions of existing primitives —
  they accept `GPUArray` inputs, never mutate them, and `softmax` supports
  `keepOnGpu`.
- **`topK(input, k, { largest })`** — the `k` largest (default) or smallest
  elements together with their **original indices**
  (`input[indices[j]] === values[j]`, torch.topk-like). Built from a descending
  key–value sort against an on-GPU `arange` iota plus `slice`, so taking the top
  10 of 100k scores no longer reads the whole sorted array back. Tie order is
  unspecified (the underlying bitonic sort is unstable).
- **`ifft(spectrum)`** — inverse FFT of an interleaved complex spectrum, scaled
  by `1/n`, so `ifft(fft(x))` recovers `x`. `fft` also accepts interleaved
  complex input via `{ complexInput: true }`. The forward path is unchanged
  bit-for-bit.
- **Constructors:** `zeros`, `full`, `arange` (NumPy semantics, integer dtypes
  and negative steps included), and `linspace` (both endpoints exact). With
  `keepOnGpu` the result is born on the GPU — e.g. `arange` as `u32` indices
  feeding `gather` without an upload.
- **`slice(input, begin, end)`** — sub-range copy with JS
  `Array.prototype.slice` semantics (negative indices, clamping). On the GPU it
  is a single buffer copy, no compute pass.
- **`{ descending: true }`** for `sort` and `sortByKey` (flipped bitonic compare
  with flipped pad sentinels; cached pipelines carry the direction).
- **Benchmark harness:** `npm run bench` measures GPU vs plain JS end-to-end
  (upload/readback included), and `bench/README.md` publishes measured numbers —
  one-shot element-wise ops are transfer-bound (~1×), GPU-resident chains and
  compute-heavy ops win big (resident map 177× at 4M, sort 32×, matmul ~310×).
- README now documents previously invisible shipped features: `zip`, `gather`,
  `argmin`/`argmax`, the index-aware `map` form `(x, i, len)`, and the `consts`
  capture (named arrays readable inside a map expression).

### Fixed

- **The reduce family corrupted a resident `GPUArray` input.** `reduce` / `sum`
  / `min` / `max` / `product` / `argmin` / `argmax` used the caller's buffer as
  a ping-pong target from the second pass on, silently overwriting any
  GPU-resident input longer than one workgroup (256 elements) with partial
  results. The ping-pong partner is now a pooled scratch buffer.
- **Per-element ops over ~4.19M elements returned garbage.** Dispatches never
  clamped against `maxComputeWorkgroupsPerDimension` (65535), and the resulting
  WebGPU validation error never became a JS exception, so the op returned
  whatever was left in the recycled pool buffer. Per-element shaders now
  dispatch a 2-D workgroup grid (tested at 4.3M elements).
- **GPU errors now actually reach the CPU fallback.** Every GPU op runs inside
  `pushErrorScope`/`popErrorScope`, so validation and out-of-memory failures
  reject instead of passing silently — an oversized multi-pass op (e.g. a very
  large `scan`) now falls back to the CPU with correct results.
- **`pipeline().run()` had no CPU fallback** — the recommended chaining API
  hard-threw on machines without WebGPU. It now honours the instance's fallback
  policy with a CPU interpreter over the recorded steps; forced-GPU paths
  (`GPUArray` input, `keepOnGpu`) still throw, like standalone ops.
- **`GPUArray` dtype mismatches were silently reinterpreted.** Feeding an `f32`
  array where `u32` indices are expected (`gather`, `scatter`, `searchsorted`,
  …) read the raw bits as the wrong type; it now throws with a hint to
  `gpu.cast()`.
- **The Node auto-detect runtime could segfault.** Nothing rooted the lazily
  imported Dawn `GPU` instance after init, so V8 could collect it while the
  device was still in use — multi-MB workloads (which trigger GC) crashed the
  process. The device manager now retains the instance for the device's
  lifetime.
- `Math.clamp` compiled to WGSL fine but crashed the CPU expression evaluator
  (JS has no `Math.clamp`); string expressions now evaluate against a Math that
  includes it.

### Changed

- The device is requested with the adapter's real buffer limits instead of the
  128 MiB spec default, so multi-hundred-MB datasets can bind.
- The publish workflow runs the real-GPU suite (Dawn + Mesa lavapipe, same as
  CI) before `npm publish`, so a WGSL regression can't ship in a release.
- The hosted playground follows the current release again (it was pinned to
  0.2.0, hiding two releases of ops from the README's "Try it now" link).
- The optional `webgpu` peer dependency range is widened to
  `^0.4.0 || ^0.5.0 || ^0.6.0`.

### Tests

- 172 new real-GPU tests (442 → 614): resident-input reduce regressions,
  beyond-the-dispatch-ceiling coverage, matmul across tile boundaries (it was
  previously only tested at 2×2 against a tile size of 8), a dedicated sort
  suite, per-op suites for every new op, end-to-end facade smoke tests, and a
  plain-Node subprocess test for the auto-detect runtime.

## [0.4.0] - 2026-06-23

A richer `pipeline`: the fluent builder now chains most of the op set while keeping
data on the GPU between steps, and folds adjacent maps into a single kernel.

### Added

- **Chainable pipeline steps.** `gpu.pipeline()` now supports `scan`, `filter`,
  `sort`, `cast`, `unique`, `histogram`, `convolve`, and `gather` alongside the
  existing `map` and `reduce`. Each step keeps the stream GPU-resident by delegating
  to the matching op, so a chain pays for one upload and one readback (e.g.
  `histogram(...).scan()` produces a cumulative distribution entirely on the GPU).
  `filter` / `unique` / `gather` / `convolve` change the stream length; `cast`
  changes its element type and `histogram` makes it `bins` `u32` counts for every
  following step.

### Changed

- **Map fusion.** Consecutive `.map()` steps now compile to a single GPU dispatch
  with no intermediate buffers, instead of one dispatch (and one buffer) per map.
- `reduce` is enforced as the terminal pipeline step — the builder throws if a step
  is chained after it — and reducing an empty (e.g. fully filtered-out) stream
  returns the identity.

### Tests

- Expanded the real-GPU pipeline suite with multi-step compositions
  (`map → filter → scan → reduce`, `sort → unique`, `histogram → scan`,
  `map → gather`), dtype-changing `cast`, map-fusion equivalence, empty-stream
  handling, and GPU-resident round-trips.

## [0.3.0] - 2026-06-22

A large batch of new operations (Tiers 1–3), a Node runtime via Google Dawn,
NumPy-style broadcasting, and clearer shader-compile diagnostics. Every new op
ships with real-GPU tests (Dawn) and CPU-fallback unit tests plus docs.

### Added

- **New operations:**
  - `argmin` / `argmax` — index of the minimum/maximum via an index-carrying
    multi-block reduction (first-occurrence tie-break).
  - `gather(src, idx)` — `output[k] = src[idx[k]]`; output dtype follows `src`,
    out-of-range indices are clamped to the last element.
  - `scatter(dst, idx, vals)` — scatter writes backed by GPU atomics, with `set`
    and `add` modes.
  - `histogram` — `atomic<u32>` bin counts.
  - `sortByKey(keys, values)` — bitonic sort of a key array carrying a parallel
    value array.
  - `cast`, `transpose`, `reshape` — shape ops, with optional shape metadata on
    `GPUArray`.
  - `searchsorted(sorted, queries)` — per-query binary search (`left` / `right`).
  - `filter` — stream compaction (predicate → scan → compact).
  - `unique` — sorted distinct values (sort → adjacent-diff → scan → compact).
  - `segmentedReduce(values, segmentIds)` — group-by reductions over segment ids.
  - `random(n)` — counter-based Philox-4×32-10 RNG; deterministic, and bit-for-bit
    reproducible between the GPU and CPU paths for `u32`.
  - `convolve(input, kernel)` — 1-D convolution with NumPy `full` / `same` /
    `valid` modes (true convolution; the kernel is reversed).
  - `fft` — radix-2 Cooley–Tukey forward and inverse transforms; complex values
    carried as interleaved `vec2<f32>`.
  - `zip(a, b, fn)` — combine two arrays element-wise with a custom expression.
- **NumPy-style broadcasting** for `add` / `subtract` / `multiply` / `divide` /
  `zip`, built on `GPUArray` shapes.
- **Index-aware `map`** — the mapping function can take the element index, and
  capture constants from the surrounding scope.
- **Node runtime via Google Dawn.** Under Node, the device manager lazily
  dynamic-imports the optional `webgpu` peer dependency so the same API runs
  headless; it is kept out of the browser bundle (tsup external + optional peer
  dependency).

### Changed

- WGSL compile errors are now framed with the generated shader source, plus a
  note tying the error back to the originating JS expression when the shader was
  generated from one.
- Code generation validates `Math.*` call arity for the general `scan` / `reduce`
  operator.

### Tests

- Real-GPU test suites running on Google Dawn for the new ops, plus a dedicated
  CI `gpu` job that runs them headless via a software Vulkan driver (Mesa
  lavapipe).

## [0.2.0] - 2026-05-26

### Added

- **`GPUArray` as a first-class input/output type.** Ops now accept
  `NumericArray | GPUArray`, so a GPU-resident array can flow from one op to the
  next without re-uploading. They can also *return* a `GPUArray` (no readback):
  this happens automatically when any input is a `GPUArray`, and can be forced or
  overridden per call with `{ keepOnGpu: true | false }`. New `gpu.upload(arr)`
  uploads a CPU array and hands back a `GPUArray`; read it with `array.toArray()`
  and free it with `array.destroy()`. Covers `add`/`subtract`/`multiply`/`divide`
  (array and scalar), `map`, `matmul`, `scan`, `sort`, `pipeline().run()`, and
  `createKernel().run()`. The reduce family (`reduce`/`sum`/`min`/`max`/`product`)
  accepts a `GPUArray` input but still returns a scalar `number`.
  - Chaining several ops on the same data now costs a single upload and a single
    readback instead of one round-trip per op.
  - In-place ops (`scan`, `sort`) copy a `GPUArray` input into their own working
    buffer, so the caller's array is never mutated.
  - A `keepOnGpu`/auto `GPUArray` result is owned by the caller — read it via
    `toArray()` (which does not free it) or release it with `destroy()`.
- **End-to-end `i32` / `u32` support.** The data type is inferred from the input
  array's runtime type — pass an `Int32Array`/`Uint32Array` and the matching typed
  array comes back; `Float32Array` and plain `number[]` continue to mean `f32`. No
  new API parameters. Covers `map`, `add`/`subtract`/`multiply`/`divide` (array and
  scalar), `reduce`/`sum`/`min`/`max`/`product`, `scan`, `sort`, `matmul`,
  `pipeline()`, `GPUArray.toArray()`, and `createKernel` (which now honours the
  `DataType` already declared on each `BufferSpec`).
- **Bitwise operators in the expression language**: `&`, `|`, `^`, `~`, `<<`, `>>`,
  legal only for integer dtypes (`i32`/`u32`). Enables hashing, bitmasks, and exact
  integer counters that were previously impossible.
- Exported a `TypedArray` type (`Float32Array | Int32Array | Uint32Array`).
- **Backend observability and fallback control.** A GPU op that fails no longer
  *only* falls back via `console.warn` — callers can now see which backend ran,
  time each op, and opt out of the silent fallback. Configure on the instance
  (`new GPU(opts)` or mutable `gpu.fallback` / `gpu.onStats` / `gpu.onFallback`
  fields):
  - `onStats({ op, backend, ms })` fires after **every** op (`backend` is
    `"gpu"` or `"cpu"`), including GPU-resident, `createKernel`, and `pipeline`
    runs. The reduce family still returns a `number` but reports stats too.
  - `onFallback({ op, error })` fires when a GPU op throws, before the policy is
    applied.
  - `fallback` policy: `"warn"` (default, unchanged behavior), `"silent"`
    (fall back without logging), or `"throw"` (re-throw instead of falling back).
  - New exported types: `Backend`, `OpStats`, `FallbackInfo`, `FallbackMode`,
    `GPUOptions`.

### Changed

- Code generation is now type-aware: the WGSL emitter selects the element type,
  formats literals per type (`1.0` for f32, `1` for i32, `1u` for u32), and rejects
  illegal combinations (bitwise ops on f32, non-integer/negative literals for
  integer types). Ops returning element arrays now return `Float32Array |
Int32Array | Uint32Array`; reductions still return `number`.
- `min`/`max` and `sort` padding now use type-correct sentinels (e.g. `2147483647`
  for `i32`, `4294967295` for `u32`) instead of f32-only values.

### Fixed

- `cpuSort` (the CPU fallback for `sort`) now sorts numerically rather than
  lexicographically, matching the GPU bitonic sort for every dtype.

## [0.1.5]

### Fixed

- `scan` now produces correct prefix sums for arrays larger than the workgroup
  size. The previous shader only scanned within a single workgroup, so results
  for arrays beyond 64 elements were wrong. It now performs a true multi-block
  scan (per-block scan → scan of block sums → add block offsets), recursing on
  the block sums so any size is handled, all within a single command encoder.
- Renamed the scan shader's workgroup variable off the reserved WGSL keyword
  `shared`, which had caused the scan shader to fail compilation and silently
  fall back to the CPU.
- `reduce`, `sum`, `min`, `max`, `product`, and `pipeline().reduce()` now run on
  the GPU instead of silently falling back to the CPU. The reduce shader used
  the same reserved keyword `shared` (renamed to `sdata`). On top of that,
  `min`/`max` passed the bare expressions `min(a, b)`/`max(a, b)` (which the
  expression parser rejects — only `Math.min`/`Math.max` are valid) and a
  non-f32-representable identity (`±3.4028235e+38`); these are now
  `Math.min`/`Math.max` with `±3.4e38`.
- `pipeline().reduce()` no longer throws an invalid-shader-module error. Unlike
  the standalone reductions it has no CPU fallback, so the broken reduce shader
  surfaced as a hard error rather than a silent fallback.

### Changed

- `reduce` (and `sum`/`min`/`max`/`product`) now chain buffers on-device with
  ping-pong storage buffers, eliminating the per-pass CPU round-trip. The
  multi-pass reduction uploads once, dispatches all passes in a single command
  encoder, and reads back only the final value.
- Replaced the Node-only `examples/` scripts with an interactive browser
  `playground/` — a zero-build showcase of every feature using import maps +
  esm.sh, Lit web components, and CSS design tokens. See `playground/README.md`.

### Tests

- Added real-GPU test suites for `scan` covering sizes from 1 to ~1M, run in
  both Node (via Google Dawn / `webgpu`) and headless Chromium (Vitest browser
  mode), in addition to the existing CPU unit tests.
- Added real-GPU test suites for `reduce` (`sum`/`min`/`max`/`product`, including
  negatives and multi-workgroup inputs) and for `pipeline` (map-only chains and
  `map → reduce`), which drive the GPU path directly so a broken shader fails
  loudly instead of silently falling back.

## [0.1.0]

### Added

- Initial release.
- Element-wise operations: `add`, `subtract`, `multiply`, `divide` (array + array or array + scalar).
- `map` with automatic JavaScript-to-WGSL compilation of arrow functions and string expressions.
- Reductions: `reduce`, `sum`, `min`, `max`, `product`.
- Tiled matrix multiplication (`matmul`) using shared memory.
- Prefix sum (`scan`) and GPU bitonic `sort`.
- Pipeline chaining to keep data on the GPU between operations.
- Custom WGSL kernels via `createKernel`.
- Automatic CPU fallback when WebGPU is unavailable.

[Unreleased]: https://github.com/ThatScalaGuy/GPGPU.js/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/ThatScalaGuy/GPGPU.js/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ThatScalaGuy/GPGPU.js/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ThatScalaGuy/GPGPU.js/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ThatScalaGuy/GPGPU.js/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/ThatScalaGuy/GPGPU.js/compare/v0.1.0...v0.1.5
[0.1.0]: https://github.com/ThatScalaGuy/GPGPU.js/releases/tag/v0.1.0
