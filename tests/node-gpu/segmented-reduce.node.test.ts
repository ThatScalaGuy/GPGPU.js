import { describe, it, expect } from "vitest";
import { gpuSegmentedReduce, cpuSegmentedReduce } from "../../src/ops/segmented-reduce";
import type { SegmentedReduceOp } from "../../src/ops/segmented-reduce";
import { GPUArray } from "../../src/pipeline/gpu-array";
import { DeviceManager } from "../../src/core/device";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";
import { expectClose } from "../shared/tolerance";

// Drive the GPU segmented-reduce (atomic per-segment accumulation) directly — no CPU
// fallback wrapper — so a broken atomics / CAS path fails loudly instead of silently
// passing through the fallback.
const deviceManager = new DeviceManager();
const bufferPool = new BufferPool();
const shaderCache = new ShaderCache();
const args = [deviceManager, bufferPool, shaderCache] as const;

describe("segmentedReduce (real GPU)", () => {
  it("sum: groups values by segment id", async () => {
    const values = new Float32Array([10, 1, 20, 2, 30]);
    const segIds = new Uint32Array([0, 1, 0, 1, 2]);
    const out = (await gpuSegmentedReduce(...args, values, segIds, {
      numSegments: 3,
    })) as Float32Array;
    // seg0: 10+20=30, seg1: 1+2=3, seg2: 30
    expect(Array.from(out)).toEqual([30, 3, 30]);
  });

  it("empty segments read back the identity (sum → 0)", async () => {
    const values = new Float32Array([5, 7]);
    const segIds = new Uint32Array([0, 3]);
    const out = (await gpuSegmentedReduce(...args, values, segIds, {
      numSegments: 4,
    })) as Float32Array;
    expect(Array.from(out)).toEqual([5, 0, 0, 7]);
  });

  it("max over segments (f32)", async () => {
    const values = new Float32Array([3, -1, 9, 2, 4]);
    const segIds = new Uint32Array([0, 0, 1, 1, 1]);
    const out = (await gpuSegmentedReduce(...args, values, segIds, {
      numSegments: 2,
      op: "max",
    })) as Float32Array;
    expect(Array.from(out)).toEqual([3, 9]);
  });

  it("min over segments leaves an empty segment at the identity", async () => {
    const values = new Float32Array([3, -1, 9, 2]);
    const segIds = new Uint32Array([0, 0, 2, 2]);
    const out = (await gpuSegmentedReduce(...args, values, segIds, {
      numSegments: 3,
      op: "min",
    })) as Float32Array;
    expect(out[0]).toBe(-1);
    expect(out[1]).toBe(Math.fround(3.4e38)); // empty segment = min identity (f32-rounded)
    expect(out[2]).toBe(2);
  });

  it("product over segments (i32)", async () => {
    const values = new Int32Array([2, 3, 4, 5]);
    const segIds = new Uint32Array([0, 0, 1, 1]);
    const out = (await gpuSegmentedReduce(...args, values, segIds, {
      numSegments: 2,
      op: "product",
    })) as Int32Array;
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([6, 20]);
  });

  it("sum: output dtype follows values (i32)", async () => {
    const values = new Int32Array([-3, -2, 10, 1]);
    const segIds = new Uint32Array([0, 0, 1, 1]);
    const out = (await gpuSegmentedReduce(...args, values, segIds, {
      numSegments: 2,
    })) as Int32Array;
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([-5, 11]);
  });

  it("sum: u32 values accumulate", async () => {
    const values = new Uint32Array([1, 2, 3, 4, 5, 6]);
    const segIds = new Uint32Array([0, 1, 2, 0, 1, 2]);
    const out = (await gpuSegmentedReduce(...args, values, segIds, {
      numSegments: 3,
    })) as Uint32Array;
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([5, 7, 9]);
  });

  it("out-of-range segment id clamps to the last segment", async () => {
    const values = new Float32Array([1, 2, 3]);
    const segIds = new Uint32Array([0, 99, 1]);
    const out = (await gpuSegmentedReduce(...args, values, segIds, {
      numSegments: 2,
    })) as Float32Array;
    // id 99 clamps to seg 1: seg0=1, seg1=2+3=5
    expect(Array.from(out)).toEqual([1, 5]);
  });

  it("keepOnGpu returns a GPUArray of length numSegments", async () => {
    const values = new Float32Array([10, 1, 20, 2, 30]);
    const segIds = new Uint32Array([0, 1, 0, 1, 2]);
    const out = await gpuSegmentedReduce(...args, values, segIds, {
      numSegments: 3,
      keepOnGpu: true,
    });
    expect(out).toBeInstanceOf(GPUArray);
    expect(out.length).toBe(3);
    expect(out.dtype).toBe("f32");
    const arr = await out.toArray();
    expect(Array.from(arr)).toEqual([30, 3, 30]);
    out.destroy();
  });

  it("does not mutate GPUArray inputs", async () => {
    const device = await deviceManager.getDevice();
    const { uploadBuffer } = await import("../../src/core/command");
    const mk = (data: Float32Array | Uint32Array, dtype: "f32" | "u32") =>
      new GPUArray(
        uploadBuffer(device, data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, bufferPool),
        data.length,
        dtype,
        device,
        bufferPool
      );

    const values = mk(new Float32Array([10, 1, 20, 2, 30]), "f32");
    const segIds = mk(new Uint32Array([0, 1, 0, 1, 2]), "u32");

    const out = await gpuSegmentedReduce(...args, values, segIds, {
      numSegments: 3,
      keepOnGpu: true,
    });
    expect(Array.from(await out.toArray())).toEqual([30, 3, 30]);
    // inputs reused in place, not consumed
    expect(Array.from(await values.toArray())).toEqual([10, 1, 20, 2, 30]);
    expect(Array.from(await segIds.toArray())).toEqual([0, 1, 0, 1, 2]);
    values.destroy();
    segIds.destroy();
    out.destroy();
  });

  it("throws when numSegments <= 0", async () => {
    const values = new Float32Array([1, 2]);
    const segIds = new Uint32Array([0, 0]);
    await expect(
      gpuSegmentedReduce(...args, values, segIds, { numSegments: 0 })
    ).rejects.toThrow(/numSegments/);
  });

  // ---- Size sweep across block boundaries vs the CPU reference. ----
  const sizes = [1, 63, 64, 65, 127, 128, 129, 1000, 4096, 65536, 1_000_000];
  const ops: SegmentedReduceOp[] = ["sum", "max", "min", "product"];

  for (const op of ops) {
    for (const n of sizes) {
      it(`${op}: f32 matches CPU at n=${n}`, async () => {
        const numSegments = Math.max(1, Math.min(257, Math.floor(n / 3) + 1));
        const values = new Float32Array(n);
        const segIds = new Uint32Array(n);
        for (let i = 0; i < n; i++) {
          // Keep f32 sums/products within precision: small magnitudes, product near 1.
          values[i] =
            op === "product" ? 0.95 + ((i * 7) % 11) * 0.01 : ((i * 31 + 7) % 97) - 48;
          segIds[i] = (i * 13 + 5) % numSegments;
        }
        const expected = cpuSegmentedReduce(values, segIds, { numSegments, op });
        const out = (await gpuSegmentedReduce(...args, values, segIds, {
          numSegments,
          op,
        })) as Float32Array;
        expect(out.length).toBe(numSegments);
        for (let s = 0; s < numSegments; s++) {
          // Empty min/max segments sit at ±3.4e38; compare those exactly, real sums loosely.
          if (Math.abs(expected[s]) > 1e30) {
            expect(out[s]).toBe(expected[s]);
          } else {
            expectClose(out[s], expected[s], { eps: 0.05 });
          }
        }
      });

      it(`${op}: i32 matches CPU exactly at n=${n}`, async () => {
        const numSegments = Math.max(1, Math.min(257, Math.floor(n / 3) + 1));
        const values = new Int32Array(n);
        const segIds = new Uint32Array(n);
        for (let i = 0; i < n; i++) {
          // Bound integer products so they don't overflow i32 across a segment.
          values[i] = op === "product" ? ((i % 3) - 1 || 1) : ((i * 17 + 3) % 50) - 25;
          segIds[i] = (i * 13 + 5) % numSegments;
        }
        const expected = cpuSegmentedReduce(values, segIds, { numSegments, op });
        const out = (await gpuSegmentedReduce(...args, values, segIds, {
          numSegments,
          op,
        })) as Int32Array;
        expect(out).toBeInstanceOf(Int32Array);
        expect(Array.from(out)).toEqual(Array.from(expected));
      });
    }
  }
});
