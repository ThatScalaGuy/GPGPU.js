# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
