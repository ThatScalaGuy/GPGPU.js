# GPGPU.js

[![npm version](https://img.shields.io/npm/v/@thatscalaguy/gpgpu.js.svg)](https://www.npmjs.com/package/@thatscalaguy/gpgpu.js)
[![CI](https://github.com/ThatScalaGuy/GPGPU.js/actions/workflows/ci.yml/badge.svg)](https://github.com/ThatScalaGuy/GPGPU.js/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

General-Purpose GPU Computing in JavaScript using WebGPU. Write GPU-accelerated code with zero boilerplate — no shader knowledge required.

## Features

- **Minimal API** — `gpu.add(a, b)`, `gpu.map(arr, x => x * 2)`, `gpu.sum(arr)`
- **Auto WGSL codegen** — Arrow functions compile to GPU shaders automatically
- **Pipeline chaining** — Chain ops without CPU-GPU roundtrips
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

### Prefix Sum (Scan)

```javascript
await gpu.scan(array)                          // default: addition
await gpu.scan(array, (a, b) => a + b, 0)     // custom scan
```

### Pipeline

Chain operations to keep data on the GPU between steps:

```javascript
const result = await gpu.pipeline()
  .map(x => x * 2)
  .map(x => x + 1)
  .reduce((a, b) => a + b, 0)
  .run(inputData);
```

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

## How It Works

1. **You write JavaScript** — `x => x * 2 + 1`
2. **Parser extracts the expression** — builds an intermediate representation
3. **WGSL emitter generates GPU shader code** — `output[idx] = (x * 2.0) + 1.0;`
4. **WebGPU compiles and dispatches** — runs on thousands of GPU cores in parallel
5. **Results returned as Float32Array** — ready to use

The library manages GPU device initialization, buffer pooling, shader caching, and data transfer automatically.

## Browser Support

WebGPU is supported in:
- Chrome 113+ / Edge 113+
- Firefox 141+ (Windows), 145+ (macOS)
- Safari 18+

When WebGPU is unavailable, all operations automatically fall back to CPU implementations.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for setup
and workflow, and [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) for community
guidelines. Security issues: see [SECURITY.md](./SECURITY.md).

## License

[MIT](./LICENSE) © Sven Herrmann
