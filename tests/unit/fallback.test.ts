import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { withFallback, type FallbackConfig } from "../../src/fallback/index";
import type { DeviceManager } from "../../src/core/device";
import type { OpStats } from "../../src/core/types";

// withFallback only ever calls deviceManager.isAvailable(), so a tiny stub suffices.
const deviceManager = (available: boolean) =>
  ({ isAvailable: () => available }) as unknown as DeviceManager;

const gpuOk = () => Promise.resolve("gpu-result");
const gpuFail = () => Promise.reject(new Error("boom"));
const cpu = () => "cpu-result";

describe("withFallback", () => {
  let stats: OpStats[];
  let fallbacks: { op: string; error: unknown }[];
  let warn: ReturnType<typeof vi.spyOn>;

  const cfg = (mode: FallbackConfig["mode"]): FallbackConfig => ({
    mode,
    onStats: (s) => stats.push(s),
    onFallback: (f) => fallbacks.push(f),
  });

  beforeEach(() => {
    stats = [];
    fallbacks = [];
    warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  });
  afterEach(() => warn.mockRestore());

  it("runs the GPU path and reports gpu stats on success", async () => {
    const r = await withFallback(deviceManager(true), "add", gpuOk, cpu, cfg("warn"));
    expect(r).toBe("gpu-result");
    expect(stats).toEqual([{ op: "add", backend: "gpu", ms: expect.any(Number) }]);
    expect(fallbacks).toHaveLength(0);
    expect(warn).not.toHaveBeenCalled();
  });

  it('mode "warn": GPU failure warns, falls back to CPU, fires onFallback + cpu stats', async () => {
    const r = await withFallback(deviceManager(true), "sum", gpuFail, cpu, cfg("warn"));
    expect(r).toBe("cpu-result");
    expect(fallbacks).toEqual([{ op: "sum", error: expect.any(Error) }]);
    expect(stats).toEqual([{ op: "sum", backend: "cpu", ms: expect.any(Number) }]);
    expect(warn).toHaveBeenCalledOnce();
  });

  it('mode "silent": GPU failure falls back without warning', async () => {
    const r = await withFallback(deviceManager(true), "sort", gpuFail, cpu, cfg("silent"));
    expect(r).toBe("cpu-result");
    expect(fallbacks).toHaveLength(1);
    expect(stats).toEqual([{ op: "sort", backend: "cpu", ms: expect.any(Number) }]);
    expect(warn).not.toHaveBeenCalled();
  });

  it('mode "throw": GPU failure re-throws, CPU never runs', async () => {
    const cpuFn = vi.fn(cpu);
    await expect(
      withFallback(deviceManager(true), "map", gpuFail, cpuFn, cfg("throw"))
    ).rejects.toThrow("boom");
    expect(cpuFn).not.toHaveBeenCalled();
    expect(fallbacks).toEqual([{ op: "map", error: expect.any(Error) }]);
    expect(stats).toHaveLength(0);
  });

  it("runs the CPU path with cpu stats when no GPU is available", async () => {
    const gpuFn = vi.fn(gpuOk);
    const r = await withFallback(deviceManager(false), "max", gpuFn, cpu, cfg("warn"));
    expect(r).toBe("cpu-result");
    expect(gpuFn).not.toHaveBeenCalled();
    expect(stats).toEqual([{ op: "max", backend: "cpu", ms: expect.any(Number) }]);
    expect(fallbacks).toHaveLength(0);
  });

  it("works without optional hooks", async () => {
    const r = await withFallback(deviceManager(true), "add", gpuOk, cpu, { mode: "warn" });
    expect(r).toBe("gpu-result");
  });
});
