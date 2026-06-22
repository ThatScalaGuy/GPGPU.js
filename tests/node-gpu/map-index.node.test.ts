import { describe, it, expect } from "vitest";
import { gpuMap } from "../../src/ops/elementwise";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { uploadBuffer } from "../../src/core/command";

// `len` is a map-expression builtin resolved by codegen, not a JS value. Declare it so a
// `(x) => x / len` arrow type-checks; its body is only stringified and parsed, never run.
declare const len: number;

// Drive gpuMap directly (no CPU fallback) so a broken index-aware shader fails loudly
// instead of silently falling back to CPU and matching by coincidence.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

describe("index-aware map (real GPU)", () => {
  // The `i` builtin in f32 arithmetic must be float-cast or the shader won't compile.
  it("x + i on f32 uses the float-cast path", async () => {
    const data = new Float32Array([10, 20, 30, 40]);
    const out = await gpuMap(...args, data, (x, i) => x + i);
    expect(out).toBeInstanceOf(Float32Array);
    expect(Array.from(out)).toEqual([10, 21, 32, 43]);
  });

  // The `i` builtin stays u32 inside an array subscript.
  it("x * hann[i % N] with a const array matches a CPU reference", async () => {
    const N = 8;
    const hann = Float32Array.from({ length: N }, (_, k) =>
      0.5 - 0.5 * Math.cos((2 * Math.PI * k) / (N - 1))
    );
    const samples = Float32Array.from({ length: 20 }, (_, k) => k + 1);

    const out = await gpuMap(...args, samples, `x * hann[i % ${N}]`, { consts: { hann } });
    expect(out).toBeInstanceOf(Float32Array);

    const expected = Array.from(samples, (x, i) => x * hann[i % N]);
    for (let k = 0; k < samples.length; k++) {
      expect(out[k]).toBeCloseTo(expected[k], 5);
    }
  });

  it("len resolves to the input length", async () => {
    const data = new Float32Array([5, 5, 5, 5, 5]);
    // every element divided by len (=5) -> all ones
    const out = await gpuMap(...args, data, (x) => x / len);
    expect(Array.from(out)).toEqual([1, 1, 1, 1, 1]);
  });

  // A GPUArray const binds in place: the same buffer is reused across calls (no re-upload).
  it("a GPUArray const is bound in place and not re-uploaded across calls", async () => {
    const device = await deviceManager.getDevice();
    const N = 4;
    const gainsData = new Float32Array([1, 2, 3, 4]);
    const gainsBuf = uploadBuffer(
      device, gainsData, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, bufferPool
    );
    const gains = new GPUArray(gainsBuf, gainsData.length, "f32", device, bufferPool);

    const a = await gpuMap(...args, new Float32Array([10, 10, 10, 10]), `x * gains[i % ${N}]`, {
      consts: { gains },
    });
    const b = await gpuMap(...args, new Float32Array([1, 1, 1, 1]), `x * gains[i % ${N}]`, {
      consts: { gains },
    });

    expect(Array.from(a)).toEqual([10, 20, 30, 40]);
    expect(Array.from(b)).toEqual([1, 2, 3, 4]);
    // The const handle survived both calls — it was bound in place, never destroyed.
    expect(gains.isDestroyed).toBe(false);
    expect(gains.buffer).toBe(gainsBuf);

    gains.destroy();
  });
});
