# Runtimes

GPGPU.js targets the WebGPU API, so it runs anywhere WebGPU is available.

## Browser

The primary target. When `navigator.gpu` is present it is used directly. WebGPU
ships in:

- Chrome 113+ / Edge 113+
- Firefox 141+ (Windows), 145+ (macOS)
- Safari 18+

A secure context (HTTPS or `localhost`) is required. No extra setup — the browser
bundle has zero runtime dependencies.

## Node.js (via Google Dawn)

Node has no built-in `navigator.gpu`, so GPGPU.js obtains a device through the
[`webgpu`](https://www.npmjs.com/package/webgpu) package, which bundles Google's
Dawn implementation.

Install it alongside the library:

```bash
npm install @thatscalaguy/gpgpu.js webgpu
```

`webgpu` is declared as an **optional peer dependency** — the library works in
the browser without it, and Node users add it explicitly.

### How auto-detection works

`DeviceManager` resolves a device with no configuration:

1. If `navigator.gpu` exists (browser), it is used exactly as in the browser.
2. Otherwise, if the code is running under Node (`process.versions.node`), the
   library performs a lazy **dynamic** `import("webgpu")`, installs Dawn's
   globals, and requests an adapter/device from it.
3. If neither applies (or the `webgpu` package is not installed), GPU
   initialization fails. With the default fallback policy, ops then run on the
   CPU; a clear error explains how to install `webgpu` for GPU execution.

The `webgpu` import is dynamic and Node-gated, so it never enters the browser
bundle (it is also marked `external` in the bundler config).

## Deno and Web Workers

Both are WebGPU-capable and expose `navigator.gpu` natively, so they take the
same path as the browser (step 1 above) with no extra setup. They are not part
of the test matrix yet.

## The dtype boundary

WGSL — the shader language WebGPU compiles to — natively supports only three
scalar numeric types:

- `f32` — 32-bit float
- `i32` — 32-bit signed integer
- `u32` — 32-bit unsigned integer

This is independent of the runtime: it is a property of WebGPU/WGSL itself.
GPGPU.js maps JavaScript arrays and `TypedArray`s onto these three types. There
is no native `f64`/`f16`/`i64` on the GPU path, so float results carry 32-bit
precision regardless of whether the input was a `Float64Array`. See
[numerics.md](./numerics.md) for the precision and reduction-order implications.
