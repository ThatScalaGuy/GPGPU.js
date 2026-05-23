import { DeviceManager } from "../core/device";

export async function withFallback<T>(
  deviceManager: DeviceManager,
  gpuFn: () => Promise<T>,
  cpuFn: () => T
): Promise<T> {
  if (deviceManager.isAvailable()) {
    try {
      return await gpuFn();
    } catch (e) {
      console.warn("GPU execution failed, falling back to CPU:", e);
    }
  }
  return cpuFn();
}
