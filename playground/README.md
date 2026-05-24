# GPGPU.js Playground

An interactive, **zero-build** browser showcase of every feature in
[`@thatscalaguy/gpgpu.js`](https://www.npmjs.com/package/@thatscalaguy/gpgpu.js).
Dependencies load straight from [esm.sh](https://esm.sh) via an
[import map](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/script/type/importmap),
the UI is built with [Lit](https://lit.dev) web components, and all styling
comes from CSS custom-property design tokens.

## Running it

ES modules and import maps don't work over `file://`, so the playground must be
served over HTTP. From the repository root:

```bash
npx serve playground
```

(or any static server, e.g. `python3 -m http.server -d playground`)

Then open the printed URL in a **WebGPU-capable browser**: Chrome/Edge 113+,
Safari 18+, or a recent Firefox. When WebGPU is unavailable the library falls
back to CPU automatically and the demos still work — the header chip shows which
backend is active.

## What it demos

| Demo | Feature | Real-world framing |
| --- | --- | --- |
| Element-wise math | `add` / `subtract` / `multiply` / `divide` | mix two signals, apply a gain |
| Map / activation | `map` | ReLU / sigmoid / transforms (JS→WGSL) |
| Reduce / statistics | `sum` / `min` / `max` / `product` / `reduce` | dataset statistics |
| Scan / running total | `scan` | cumulative sums & products |
| Matrix multiply | `matmul` | neural-net layer / 2D transform |
| Sort + timing | `sort` | sort N values, timed vs `Array.sort` |
| Pipeline / chaining | `pipeline().map().map().run()` | chained transforms, no CPU round-trips |
| Custom WGSL kernel | `createKernel` | raw compute-shader escape hatch |

## Using your local build

The import map in `index.html` points at the published package on esm.sh. To
test your **local** `dist/` build instead, run `npm run build` from the repo
root, then change the `@thatscalaguy/gpgpu.js` entry in `index.html` to:

```json
"@thatscalaguy/gpgpu.js": "../dist/index.js"
```

## Structure

```
playground/
├── index.html            # import map + mounts <gpgpu-app>
├── styles/tokens.css      # design tokens (:root + [data-theme="dark"])
├── lib/shared.js          # gpu singleton + parse/format/timing helpers
└── components/            # Lit components — one per feature + a shared card shell
```
