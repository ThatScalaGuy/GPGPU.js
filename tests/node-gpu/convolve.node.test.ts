import { describe, it, expect } from "vitest";
import { gpuConvolve, cpuConvolve, type ConvolveMode } from "../../src/ops/convolve";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";
import { uploadBuffer } from "../../src/core/command";
import { expectClose } from "../shared/tolerance";

// Drive the GPU convolve implementation directly (no CPU fallback wrapper), so a broken
// convolve kernel fails the test loudly instead of silently passing through the fallback.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

const MODES: ConvolveMode[] = ["full", "same", "valid"];

describe("convolve (real GPU)", () => {
  // Known values cross-checked against numpy.convolve.
  it("full: [1,2,3] * [0,1,0.5]", async () => {
    const a = new Float32Array([1, 2, 3]);
    const v = new Float32Array([0, 1, 0.5]);
    const out = (await gpuConvolve(...args, a, v)) as Float32Array;
    // np.convolve([1,2,3],[0,1,0.5]) = [0, 1, 2.5, 4, 1.5]
    expect(out.length).toBe(5);
    [0, 1, 2.5, 4, 1.5].forEach((e, i) => expectClose(out[i], e));
  });

  it("same: [1,2,3] * [0,1,0.5] returns the centred window", async () => {
    const a = new Float32Array([1, 2, 3]);
    const v = new Float32Array([0, 1, 0.5]);
    const out = (await gpuConvolve(...args, a, v, { mode: "same" })) as Float32Array;
    // length max(3,3)=3, centre of [0,1,2.5,4,1.5] -> [1, 2.5, 4]
    expect(out.length).toBe(3);
    [1, 2.5, 4].forEach((e, i) => expectClose(out[i], e));
  });

  it("valid: [1,2,3,4,5] * [1,1,1] returns only fully-overlapping sums", async () => {
    const a = new Float32Array([1, 2, 3, 4, 5]);
    const v = new Float32Array([1, 1, 1]);
    const out = (await gpuConvolve(...args, a, v, { mode: "valid" })) as Float32Array;
    // np.convolve(...,'valid') = [6, 9, 12]
    expect(out.length).toBe(3);
    [6, 9, 12].forEach((e, i) => expectClose(out[i], e));
  });

  it("reversal: convolution (not correlation) reverses the kernel", async () => {
    // An asymmetric kernel exposes a correlation-vs-convolution mistake.
    const a = new Float32Array([1, 0, 0, 0]); // unit impulse
    const v = new Float32Array([1, 2, 3]);
    const out = (await gpuConvolve(...args, a, v)) as Float32Array;
    // Impulse at index 0 stamps the (un-reversed) kernel into the output head.
    [1, 2, 3, 0, 0, 0].forEach((e, i) => expectClose(out[i], e));
  });

  // Size sweep across block boundaries (workgroup size is 64). The kernel size also
  // straddles a block so the per-thread loop is exercised both short and long.
  const sizes = [1, 63, 64, 65, 127, 128, 129, 1000, 4096, 65536];
  for (const n of sizes) {
    for (const km of [1, 3, 64, 65]) {
      // Skip kernels larger than the signal for 'valid' is fine (numpy swaps), but keep
      // the sweep meaningful: only run kernels <= n plus a couple that exceed it.
      it(`matches CPU reference for n=${n}, m=${km} across modes`, async () => {
        const a = new Float32Array(n);
        for (let i = 0; i < n; i++) a[i] = Math.sin(i * 0.1) * 3 - 1;
        const v = new Float32Array(km);
        for (let i = 0; i < km; i++) v[i] = (i % 5) - 2 + 0.25;

        for (const mode of MODES) {
          const ref = cpuConvolve(a, v, mode);
          const out = (await gpuConvolve(...args, a, v, { mode })) as Float32Array;
          expect(out.length).toBe(ref.length);
          // Tolerance scales with the number of summed terms (longer dot products
          // accumulate more f32 reassociation error).
          const eps = Math.max(0.05, Math.min(n, km) * 1e-3);
          for (let i = 0; i < ref.length; i++) expectClose(out[i], ref[i], { eps });
        }
      });
    }
  }

  it("matches CPU reference for n=1e6 (full)", async () => {
    const n = 1_000_000;
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = ((i * 31) % 7) - 3;
    const v = new Float32Array([1, -2, 1]); // discrete second derivative
    const ref = cpuConvolve(a, v, "full");
    const out = (await gpuConvolve(...args, a, v, { mode: "full" })) as Float32Array;
    expect(out.length).toBe(ref.length);
    // Spot-check a spread of indices (full readback of 1e6 is fine but the assert loop
    // is the cost; integer-valued data here means exact agreement is expected).
    for (const i of [0, 1, 2, n >> 1, n - 1, n, n + 1]) {
      expectClose(out[i], ref[i], { eps: 1e-3 });
    }
  });

  it("preserves i32 dtype with exact integer agreement", async () => {
    const a = new Int32Array([1, -2, 3, -4, 5, -6, 7]);
    const v = new Int32Array([2, -1, 3]);
    const ref = cpuConvolve(a, v, "full");
    const out = (await gpuConvolve(...args, a, v)) as Int32Array;
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual(Array.from(ref));
  });

  it("preserves u32 dtype with exact integer agreement", async () => {
    const a = new Uint32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const v = new Uint32Array([1, 2, 1]);
    const ref = cpuConvolve(a, v, "full");
    const out = (await gpuConvolve(...args, a, v)) as Uint32Array;
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual(Array.from(ref));
  });

  it("i32 across modes matches CPU exactly (block-boundary signal)", async () => {
    const n = 130;
    const a = new Int32Array(n);
    for (let i = 0; i < n; i++) a[i] = ((i * 7) % 11) - 5;
    const v = new Int32Array([3, -1, 2, -4, 1]);
    for (const mode of MODES) {
      const ref = cpuConvolve(a, v, mode);
      const out = (await gpuConvolve(...args, a, v, { mode })) as Int32Array;
      expect(out).toBeInstanceOf(Int32Array);
      expect(Array.from(out)).toEqual(Array.from(ref));
    }
  });

  it("keepOnGpu returns a GPUArray of length outLen", async () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const v = new Float32Array([1, 1]);
    const out = await gpuConvolve(...args, a, v, { keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.length).toBe(5); // n+m-1 = 4+2-1
    expect(out.dtype).toBe("f32");
    const arr = await out.toArray();
    [1, 3, 5, 7, 4].forEach((e, i) => expectClose(arr[i], e));
    out.destroy();
  });

  it("accepts GPUArray inputs and reuses them in place", async () => {
    const device = await deviceManager.getDevice();
    const mk = (data: Float32Array) =>
      new GPUArray(
        uploadBuffer(device, data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, bufferPool),
        data.length,
        "f32",
        device,
        bufferPool
      );
    const a = mk(new Float32Array([1, 2, 3]));
    const v = mk(new Float32Array([1, 1, 1]));
    const out = (await gpuConvolve(...args, a, v, { keepOnGpu: true })) as GPUArray;
    expect(Array.from(await out.toArray())).toEqual([1, 3, 6, 5, 3]);
    // inputs survive (reused, not consumed)
    expect(Array.from(await a.toArray())).toEqual([1, 2, 3]);
    expect(Array.from(await v.toArray())).toEqual([1, 1, 1]);
    a.destroy();
    v.destroy();
    out.destroy();
  });
});
