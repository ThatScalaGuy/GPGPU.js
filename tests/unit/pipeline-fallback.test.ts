import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Pipeline } from "../../src/pipeline/pipeline";
import type { DeviceManager } from "../../src/core/device";
import type { FallbackConfig } from "../../src/fallback/index";
import type { OpStats } from "../../src/core/types";
import { BufferPool } from "../../src/core/buffer-pool";
import { ShaderCache } from "../../src/core/shader-cache";

// Regression: Pipeline.run used to call getDevice() bare — no CPU fallback — so the
// recommended chaining API hard-threw on machines without WebGPU while every
// standalone op fell back transparently.

const noGpu = {
  isAvailable: () => false,
  getDevice: () => Promise.reject(new Error("no adapter")),
} as unknown as DeviceManager;

const brokenGpu = {
  isAvailable: () => true,
  getDevice: () => Promise.reject(new Error("boom")),
} as unknown as DeviceManager;

const make = (dm: DeviceManager, cfg?: FallbackConfig) =>
  new Pipeline(dm, new BufferPool(), new ShaderCache(), cfg ? () => cfg : undefined);

describe("Pipeline CPU fallback", () => {
  let warn: ReturnType<typeof vi.spyOn>;
  beforeEach(() => {
    warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  });
  afterEach(() => warn.mockRestore());

  it("runs a full chain on the CPU when WebGPU is unavailable", async () => {
    const r = await make(noGpu)
      .map((x) => x * 2)
      .filter((x) => x > 5)
      .scan()
      .reduce((a, b) => a + b, 0)
      .run([1, 2, 3, 4, 5, 6]);
    // doubled: [2,4,6,8,10,12] -> filtered: [6,8,10,12] -> scan: [6,14,24,36] -> sum: 80
    expect(r).toBe(80);
  });

  it("array-result chain returns a TypedArray and honours cast/histogram dtypes", async () => {
    const r = await make(noGpu)
      .histogram({ bins: 2, min: 0, max: 10 })
      .scan()
      .run([1, 2, 3, 9]);
    expect(r).toBeInstanceOf(Uint32Array);
    expect(Array.from(r as Uint32Array)).toEqual([3, 4]);
  });

  it("reports backend cpu via onStats and fires onFallback on GPU failure", async () => {
    const stats: OpStats[] = [];
    const fallbacks: unknown[] = [];
    const cfg: FallbackConfig = {
      mode: "warn",
      onStats: (s) => stats.push(s),
      onFallback: (f) => fallbacks.push(f),
    };
    const r = await make(brokenGpu, cfg)
      .map((x) => x + 1)
      .run([1, 2, 3]);
    expect(Array.from(r as Float32Array)).toEqual([2, 3, 4]);
    expect(stats).toEqual([{ op: "pipeline", backend: "cpu", ms: expect.any(Number) }]);
    expect(fallbacks).toHaveLength(1);
    expect(warn).toHaveBeenCalledOnce();
  });

  it('mode "throw" re-throws the GPU error instead of falling back', async () => {
    const cfg: FallbackConfig = { mode: "throw" };
    await expect(
      make(brokenGpu, cfg)
        .map((x) => x + 1)
        .run([1, 2, 3])
    ).rejects.toThrow("boom");
  });

  it("keepOnGpu is a forced-GPU path and never falls back", async () => {
    await expect(
      make(noGpu)
        .map((x) => x + 1)
        .run([1, 2, 3], { keepOnGpu: true })
    ).rejects.toThrow("no adapter");
  });
});
