# Contributing to GPGPU.js

Thanks for your interest in contributing! This document explains how to get set up and the workflow for proposing changes.

## Development Setup

Requires Node.js 18+.

```bash
git clone https://github.com/ThatScalaGuy/GPGPU.js.git
cd GPGPU.js
npm ci
```

## Common Commands

```bash
npm run build      # Build the library with tsup (ESM + CJS + .d.ts)
npm run dev        # Build in watch mode
npm test           # Run the test suite once (vitest)
npm run test:watch # Run tests in watch mode
npm run typecheck  # Type-check without emitting
```

## Running Examples

The examples in `examples/` import directly from `src/`. Run them with a
TypeScript runner, e.g.:

```bash
npx tsx examples/basic-add.ts
```

WebGPU is used when available; otherwise the library falls back to CPU
implementations, so examples run in plain Node as well.

## Workflow

1. Fork the repository and create a feature branch from `main`.
2. Make your change. Keep changes focused and match the existing code style.
3. Add or update tests under `tests/` for any behavior change.
4. Make sure `npm run typecheck`, `npm test`, and `npm run build` all pass.
5. Open a pull request against `main` with a clear description of the change
   and the motivation behind it.

## Code Style

- TypeScript, strict mode. No new runtime dependencies without discussion.
- Keep the public API minimal and consistent with the existing surface
  (`GPU`, `Pipeline`, `GPUArray`).
- Prefer small, well-named functions over large ones.

## Reporting Bugs & Requesting Features

Please use the GitHub issue templates. Include a minimal reproduction for bugs.

By contributing, you agree that your contributions will be licensed under the
MIT License.
