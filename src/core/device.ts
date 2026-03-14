export class GPUNotAvailableError extends Error {
  constructor() {
    super(
      "WebGPU is not available. Ensure your browser supports WebGPU and you are using a secure context (HTTPS)."
    );
    this.name = "GPUNotAvailableError";
  }
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
    if (!this.isAvailable()) {
      throw new GPUNotAvailableError();
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new GPUNotAvailableError();
    }

    const device = await adapter.requestDevice();

    device.lost.then(() => {
      this.device = null;
      this.initPromise = null;
    });

    this.device = device;
    return device;
  }

  isAvailable(): boolean {
    return typeof navigator !== "undefined" && !!navigator.gpu;
  }

  reset(): void {
    this.device = null;
    this.initPromise = null;
  }
}
