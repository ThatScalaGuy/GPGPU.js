# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
