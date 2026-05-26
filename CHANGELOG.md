# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/ThatScalaGuy/GPGPU.js/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ThatScalaGuy/GPGPU.js/releases/tag/v0.1.0
