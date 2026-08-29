export class GPUNotAvailableError extends Error {
  constructor() {
    super(
      "WebGPU is not available. Ensure your browser supports WebGPU and you are using a secure context (HTTPS)."
    );
    this.name = "GPUNotAvailableError";
  }
}

// Minimal local shape so we can probe for Node without pulling @types/node into
// the library's (browser-first) type surface — see tsconfig `types`.
declare const process: { versions?: { node?: string } } | undefined;

/** True when running under Node.js (not a browser, Deno, or Worker). */
function isNode(): boolean {
  return (
    typeof process !== "undefined" &&
    !!process.versions &&
    !!process.versions.node
  );
}

export class DeviceManager {
  private device: GPUDevice | null = null;
  private initPromise: Promise<GPUDevice> | null = null;

  async getDevice(): Promise<GPUDevice> {
    if (this.device) return this.device;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this.init();
    return this.initPromise;
  }

  private async init(): Promise<GPUDevice> {
    const gpu = await this.resolveGpu();

    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      throw new GPUNotAvailableError();
    }

    // The spec's default limits cap storage bindings at 128 MiB even on adapters
    // that support far more — request the adapter's actual buffer limits so large
    // datasets (e.g. multi-hundred-MB matrices) can bind.
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });

    device.lost.then(() => {
      this.device = null;
      this.initPromise = null;
    });

    this.device = device;
    return device;
  }

  /**
   * Resolve a `GPU` object. In the browser this is `navigator.gpu`, used exactly
   * as before. When `navigator.gpu` is absent and we are under Node, lazily load
   * the optional `webgpu` (Google Dawn) bindings — via a *dynamic* import so the
   * browser bundle never statically depends on the native package — and use the
   * `GPU` it provides.
   */
  private async resolveGpu(): Promise<GPU> {
    if (typeof navigator !== "undefined" && navigator.gpu) {
      return navigator.gpu;
    }

    if (isNode()) {
      let webgpu: typeof import("webgpu");
      try {
        webgpu = await import("webgpu");
      } catch {
        throw new Error(
          "WebGPU is not available under Node. Install the optional `webgpu` package (npm install webgpu) to run GPGPU.js on Node via Google Dawn."
        );
      }
      Object.assign(globalThis, webgpu.globals);
      return webgpu.create([]);
    }

    throw new GPUNotAvailableError();
  }

  isAvailable(): boolean {
    return (
      (typeof navigator !== "undefined" && !!navigator.gpu) || isNode()
    );
  }

  reset(): void {
    this.device = null;
    this.initPromise = null;
  }
}
