import { describe, it, expect } from "vitest";
import { gpuReduce } from "../../src/ops/reduce";
import { gpuScan } from "../../src/ops/scan";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";
import { expectClose } from "../shared/tolerance";
import type { NumericArray } from "../../src/core/types";

// Tier 3: a fully-general associative operator passed to gpu.reduce / gpu.scan must inline
// through the JS->WGSL codegen and produce correct results. Drive the GPU ops directly (no
// CPU fallback) so a broken operator inlining fails loudly instead of silently passing.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

type BinFn = (a: number, b: number) => number;

// Sequential CPU reference scan (inclusive) and reduce, applied with `identity` as the seed.
function refScan(arr: ArrayLike<number>, fn: BinFn, identity: number): number[] {
  const out: number[] = [];
  let acc = identity;
  for (let i = 0; i < arr.length; i++) {
    acc = fn(acc, arr[i]);
    out[i] = acc;
  }
  return out;
}

function refReduce(arr: ArrayLike<number>, fn: BinFn, identity: number): number {
  let acc = identity;
  for (let i = 0; i < arr.length; i++) acc = fn(acc, arr[i]);
  return acc;
}

// Each case is a (JS arrow | WGSL string) operator with its identity, a CPU reference fn,
// an input (whose dtype drives the WGSL element type), and how to compare results.
interface OpCase {
  name: string;
  fn: ((a: number, b: number) => number) | string;
  identity: number;
  ref: BinFn;
  input: NumericArray;
  exact: boolean; // integer ops compare bit-exact; float ops use expectClose
}

// A large finite f32 (NOT a non-round-tripping sentinel) seeds min/max so padding lanes
// never win — mirrors src/ops/reduce.ts MIN/MAX_IDENTITY.
const F32_MAX = 3.4e38;

// Spread inputs across several blocks (workgroup width is 64 / 256) so the multi-block scan
// and multi-pass reduce both exercise the inter-block stitching for the custom operator.
const floats = Array.from({ length: 300 }, (_, i) => ((i * 37 + 11) % 97) / 13 + 0.5);
const ints = Array.from({ length: 300 }, (_, i) => ((i * 1103515245 + 12345) >>> 8) & 0x3ff);

const CASES: OpCase[] = [
  {
    name: "sum (arrow a+b)",
    fn: (a, b) => a + b,
    identity: 0,
    ref: (a, b) => a + b,
    input: new Float32Array(floats),
    exact: false,
  },
  {
    name: "product (arrow a*b)",
    fn: (a, b) => a * b,
    identity: 1,
    ref: (a, b) => a * b,
    // Keep the running product bounded: values near 1 so 300 multiplies stay finite.
    input: new Float32Array(Array.from({ length: 130 }, (_, i) => 1 + ((i % 7) - 3) / 100)),
    exact: false,
  },
  {
    name: "min (Math.min string)",
    fn: "Math.min(a, b)",
    identity: F32_MAX,
    ref: (a, b) => Math.min(a, b),
    input: new Float32Array(floats),
    exact: false,
  },
  {
    name: "max (arrow Math.max)",
    fn: (a, b) => Math.max(a, b),
    identity: -F32_MAX,
    ref: (a, b) => Math.max(a, b),
    input: new Float32Array(floats),
    exact: false,
  },
  {
    name: "u32 bitwise-or",
    fn: (a, b) => a | b,
    identity: 0,
    ref: (a, b) => (a | b) >>> 0,
    input: new Uint32Array(ints),
    exact: true,
  },
  {
    name: "u32 bitwise-and",
    fn: (a, b) => a & b,
    identity: 0xffffffff,
    ref: (a, b) => (a & b) >>> 0,
    input: new Uint32Array(ints),
    exact: true,
  },
];

export async function registerGeneralOperatorSuite(): Promise<void> {
  const hasGPU =
    typeof navigator !== "undefined" &&
    !!navigator.gpu &&
    !!(await navigator.gpu.requestAdapter());

  const suite = hasGPU ? describe : describe.skip;

  suite("general associative operator (reduce + scan, real GPU)", () => {
    for (const c of CASES) {
      it(`reduce: ${c.name}`, async () => {
        const got = await gpuReduce(...args, c.input, c.fn, c.identity);
        const want = refReduce(c.input, c.ref, c.identity);
        if (c.exact) expect(got).toBe(want);
        else expectClose(got, want, { eps: Math.max(0.05, Math.abs(want) * 1e-4) });
      });

      it(`scan: ${c.name}`, async () => {
        const got = await gpuScan(...args, c.input, c.fn, c.identity);
        const want = refScan(c.input, c.ref, c.identity);
        expect(got.length).toBe(c.input.length);
        for (let i = 0; i < want.length; i++) {
          if (c.exact) expect(got[i]).toBe(want[i]);
          else expectClose(got[i], want[i], { eps: Math.max(0.05, Math.abs(want[i]) * 1e-4) });
        }
      });
    }

    // A composite (non-trivial) associative arrow: max-magnitude via a ternary + Math.abs,
    // proving the parser/emitter inline arbitrary expression trees, not just single ops.
    it("reduce: composite ternary (keep larger magnitude)", async () => {
      const fn = (a: number, b: number) => (Math.abs(a) > Math.abs(b) ? a : b);
      const input = new Float32Array([1, -7, 3, -2, 6, -9, 4]);
      const got = await gpuReduce(...args, input, fn, 0);
      const want = refReduce(input, fn, 0);
      expect(got).toBe(want); // integer-valued floats, exact
    });
  });
}

await registerGeneralOperatorSuite();
