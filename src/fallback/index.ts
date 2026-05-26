import { DeviceManager } from "../core/device";
import type { FallbackInfo, FallbackMode, OpStats } from "../core/types";

export interface FallbackConfig {
  mode: FallbackMode;
  onFallback?: (info: FallbackInfo) => void;
  onStats?: (stats: OpStats) => void;
}

export async function withFallback<T>(
  deviceManager: DeviceManager,
  op: string,
  gpuFn: () => Promise<T>,
  cpuFn: () => T,
  cfg: FallbackConfig
): Promise<T> {
  if (deviceManager.isAvailable()) {
    const t = performance.now();
    try {
      const r = await gpuFn();
      cfg.onStats?.({ op, backend: "gpu", ms: performance.now() - t });
      return r;
    } catch (e) {
      cfg.onFallback?.({ op, error: e });
      if (cfg.mode === "throw") throw e;
      if (cfg.mode === "warn")
        console.warn(`GPU execution failed for "${op}", falling back to CPU:`, e);
    }
  }
  const t = performance.now();
  const r = cpuFn();
  cfg.onStats?.({ op, backend: "cpu", ms: performance.now() - t });
  return r;
}
