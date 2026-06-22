import { describe, it, expect } from "vitest";
import { gpuSearchsorted } from "../../src/ops/searchsorted";
import { cpuSearchsorted } from "../../src/fallback/cpu-ops";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";

// Drive the GPU searchsorted implementation directly (no CPU fallback wrapper), so a broken
// shader fails the test loudly instead of silently passing through the fallback.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

describe("searchsorted (real GPU)", () => {
  it("left (default): count of elements strictly < q", async () => {
    const sorted = new Float32Array([1, 3, 5, 7]);
    const queries = new Float32Array([0, 1, 2, 3, 8]);
    const out = (await gpuSearchsorted(...args, sorted, queries)) as Uint32Array;
    expect(Array.from(out)).toEqual([0, 0, 1, 1, 4]);
  });

  it("right: count of elements <= q", async () => {
    const sorted = new Float32Array([1, 3, 5, 7]);
    const queries = new Float32Array([0, 1, 3, 8]);
    const out = (await gpuSearchsorted(...args, sorted, queries, { side: "right" })) as Uint32Array;
    expect(Array.from(out)).toEqual([0, 1, 2, 4]);
  });

  it("duplicates: left lands before the run, right after it", async () => {
    const sorted = new Float32Array([2, 2, 2]);
    const left = (await gpuSearchsorted(...args, sorted, new Float32Array([2]))) as Uint32Array;
    expect(Array.from(left)).toEqual([0]);
    const right = (await gpuSearchsorted(...args, sorted, new Float32Array([2]), { side: "right" })) as Uint32Array;
    expect(Array.from(right)).toEqual([3]);
  });

  it("out-of-range queries return 0 (below min) or length (above max)", async () => {
    const sorted = new Float32Array([10, 20, 30]);
    const queries = new Float32Array([-5, 100]);
    const left = (await gpuSearchsorted(...args, sorted, queries)) as Uint32Array;
    expect(Array.from(left)).toEqual([0, 3]);
    const right = (await gpuSearchsorted(...args, sorted, queries, { side: "right" })) as Uint32Array;
    expect(Array.from(right)).toEqual([0, 3]);
  });

  it("i32 inputs including negatives", async () => {
    const sorted = new Int32Array([-7, -3, 0, 4, 9]);
    const queries = new Int32Array([-8, -3, -2, 0, 9, 10]);
    const left = (await gpuSearchsorted(...args, sorted, queries)) as Uint32Array;
    expect(Array.from(left)).toEqual([0, 1, 2, 2, 4, 5]);
    const right = (await gpuSearchsorted(...args, sorted, queries, { side: "right" })) as Uint32Array;
    expect(Array.from(right)).toEqual([0, 2, 2, 3, 5, 5]);
  });

  it("u32 inputs", async () => {
    const sorted = new Uint32Array([2, 4, 6, 8]);
    const queries = new Uint32Array([1, 4, 5, 9]);
    const left = (await gpuSearchsorted(...args, sorted, queries)) as Uint32Array;
    expect(Array.from(left)).toEqual([0, 1, 2, 4]);
    const right = (await gpuSearchsorted(...args, sorted, queries, { side: "right" })) as Uint32Array;
    expect(Array.from(right)).toEqual([0, 2, 2, 4]);
  });

  it("keepOnGpu returns a u32 GPUArray of length === queries.length", async () => {
    const sorted = new Float32Array([1, 3, 5, 7]);
    const queries = new Float32Array([0, 2, 4, 6, 8]);
    const out = await gpuSearchsorted(...args, sorted, queries, { keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.dtype).toBe("u32");
    expect(out.length).toBe(queries.length);
    const arr = await out.toArray();
    expect(Array.from(arr)).toEqual([0, 1, 2, 3, 4]);
    out.destroy();
  });

  it("multi-block: > 64 queries against a sorted ramp, both sides match the CPU reference", async () => {
    const n = 200;
    const sorted = new Float32Array(n);
    for (let i = 0; i < n; i++) sorted[i] = i * 2; // 0, 2, 4, ... (so odd queries fall between)
    const m = 257; // forces > 1 workgroup of 64
    const queries = new Float32Array(m);
    for (let i = 0; i < m; i++) queries[i] = i - 8; // span below 0, through the ramp, past the top

    const left = (await gpuSearchsorted(...args, sorted, queries)) as Uint32Array;
    expect(Array.from(left)).toEqual(Array.from(cpuSearchsorted(sorted, queries, "left")));

    const right = (await gpuSearchsorted(...args, sorted, queries, { side: "right" })) as Uint32Array;
    expect(Array.from(right)).toEqual(Array.from(cpuSearchsorted(sorted, queries, "right")));
  });

  it("GPUArray inputs: upload sorted + queries, then search", async () => {
    const device = await deviceManager.getDevice();
    const { uploadBuffer } = await import("../../src/core/command");
    const mk = (data: Float32Array) =>
      new GPUArray(
        uploadBuffer(device, data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, bufferPool),
        data.length,
        "f32",
        device,
        bufferPool
      );
    const sorted = mk(new Float32Array([1, 3, 5, 7]));
    const queries = mk(new Float32Array([0, 1, 2, 3, 8]));
    const out = await gpuSearchsorted(...args, sorted, queries, { keepOnGpu: true });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.dtype).toBe("u32");
    const arr = await out.toArray();
    expect(Array.from(arr)).toEqual([0, 0, 1, 1, 4]);
    out.destroy();
    sorted.destroy();
    queries.destroy();
  });
});
