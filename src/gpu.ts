import type { NumericArray, TypedArray, MatMulOpts, KernelConfig } from "./core/types";
import { DeviceManager } from "./core/device";
import { BufferPool } from "./core/buffer-pool";
import { ShaderCache } from "./core/shader-cache";
import { withFallback } from "./fallback/index";
import {
  cpuAdd, cpuSubtract, cpuMultiply, cpuDivide,
  cpuMap, cpuReduce, cpuSum, cpuMin, cpuMax, cpuProduct,
  cpuMatmul, cpuScan, cpuSort,
} from "./fallback/cpu-ops";
import {
  gpuElementwiseBinary, gpuScalarBroadcast, gpuMap,
} from "./ops/elementwise";
import { gpuReduce, gpuSum, gpuMin, gpuMax, gpuProduct } from "./ops/reduce";
import { gpuMatmul } from "./ops/matmul";
import { gpuScan } from "./ops/scan";
import { gpuSort } from "./ops/sort";
import { Pipeline } from "./pipeline/pipeline";

export class GPU {
  private deviceManager = new DeviceManager();
  private bufferPool = new BufferPool();
  private shaderCache = new ShaderCache();

  /** Check if WebGPU is available */
  isAvailable(): boolean {
    return this.deviceManager.isAvailable();
  }

  // --- Elementwise operations ---

  async add(a: NumericArray, b: NumericArray | number): Promise<TypedArray> {
    return withFallback(
      this.deviceManager,
      () =>
        typeof b === "number"
          ? gpuScalarBroadcast(this.deviceManager, this.bufferPool, this.shaderCache, a, b, "+")
          : gpuElementwiseBinary(this.deviceManager, this.bufferPool, this.shaderCache, a, b, "+"),
      () => cpuAdd(a, b)
    );
  }

  async subtract(a: NumericArray, b: NumericArray | number): Promise<TypedArray> {
    return withFallback(
      this.deviceManager,
      () =>
        typeof b === "number"
          ? gpuScalarBroadcast(this.deviceManager, this.bufferPool, this.shaderCache, a, b, "-")
          : gpuElementwiseBinary(this.deviceManager, this.bufferPool, this.shaderCache, a, b, "-"),
      () => cpuSubtract(a, b)
    );
  }

  async multiply(a: NumericArray, b: NumericArray | number): Promise<TypedArray> {
    return withFallback(
      this.deviceManager,
      () =>
        typeof b === "number"
          ? gpuScalarBroadcast(this.deviceManager, this.bufferPool, this.shaderCache, a, b, "*")
          : gpuElementwiseBinary(this.deviceManager, this.bufferPool, this.shaderCache, a, b, "*"),
      () => cpuMultiply(a, b)
    );
  }

  async divide(a: NumericArray, b: NumericArray | number): Promise<TypedArray> {
    return withFallback(
      this.deviceManager,
      () =>
        typeof b === "number"
          ? gpuScalarBroadcast(this.deviceManager, this.bufferPool, this.shaderCache, a, b, "/")
          : gpuElementwiseBinary(this.deviceManager, this.bufferPool, this.shaderCache, a, b, "/"),
      () => cpuDivide(a, b)
    );
  }

  // --- Map ---

  async map(
    input: NumericArray,
    fn: ((x: number) => number) | string
  ): Promise<TypedArray> {
    return withFallback(
      this.deviceManager,
      () => gpuMap(this.deviceManager, this.bufferPool, this.shaderCache, input, fn),
      () => cpuMap(input, fn)
    );
  }

  // --- Reduce ---

  async reduce(
    input: NumericArray,
    fn: ((a: number, b: number) => number) | string,
    identity: number
  ): Promise<number> {
    return withFallback(
      this.deviceManager,
      () => gpuReduce(this.deviceManager, this.bufferPool, this.shaderCache, input, fn, identity),
      () => cpuReduce(input, fn, identity)
    );
  }

  async sum(input: NumericArray): Promise<number> {
    return withFallback(
      this.deviceManager,
      () => gpuSum(this.deviceManager, this.bufferPool, this.shaderCache, input),
      () => cpuSum(input)
    );
  }

  async min(input: NumericArray): Promise<number> {
    return withFallback(
      this.deviceManager,
      () => gpuMin(this.deviceManager, this.bufferPool, this.shaderCache, input),
      () => cpuMin(input)
    );
  }

  async max(input: NumericArray): Promise<number> {
    return withFallback(
      this.deviceManager,
      () => gpuMax(this.deviceManager, this.bufferPool, this.shaderCache, input),
      () => cpuMax(input)
    );
  }

  async product(input: NumericArray): Promise<number> {
    return withFallback(
      this.deviceManager,
      () => gpuProduct(this.deviceManager, this.bufferPool, this.shaderCache, input),
      () => cpuProduct(input)
    );
  }

  // --- Matrix multiply ---

  async matmul(
    a: NumericArray,
    b: NumericArray,
    opts: MatMulOpts
  ): Promise<TypedArray> {
    return withFallback(
      this.deviceManager,
      () => gpuMatmul(this.deviceManager, this.bufferPool, this.shaderCache, a, b, opts),
      () => cpuMatmul(a, b, opts)
    );
  }

  // --- Scan (prefix sum) ---

  async scan(
    input: NumericArray,
    fn?: ((a: number, b: number) => number) | string,
    identity?: number
  ): Promise<TypedArray> {
    return withFallback(
      this.deviceManager,
      () =>
        gpuScan(
          this.deviceManager, this.bufferPool, this.shaderCache,
          input, fn, identity
        ),
      () => cpuScan(input, fn ?? ((a, b) => a + b), identity ?? 0)
    );
  }

  // --- Sort ---

  async sort(input: NumericArray): Promise<TypedArray> {
    return withFallback(
      this.deviceManager,
      () => gpuSort(this.deviceManager, this.bufferPool, this.shaderCache, input),
      () => cpuSort(input)
    );
  }

  // --- Pipeline builder ---

  pipeline(): Pipeline {
    return new Pipeline(this.deviceManager, this.bufferPool, this.shaderCache);
  }

  // --- Custom kernel ---

  async createKernel(config: KernelConfig): Promise<{
    run: (...inputs: NumericArray[]) => Promise<TypedArray>;
  }> {
    const device = await this.deviceManager.getDevice();
    const pipeline = await this.shaderCache.getOrCreate(device, config.shader, "custom-kernel");
    const workgroupSize = config.workgroupSize ?? 64;

    return {
      run: async (...inputs: NumericArray[]) => {
        const { toTypedArray } = await import("./utils/data-conversion");
        const { uploadBuffer, createOutputBuffer, dispatchAndRead } = await import("./core/command");
        const { computeWorkgroupCount } = await import("./utils/workgroup");

        const inputBuffers = inputs.map((input, i) => {
          const arr = toTypedArray(input, config.inputs[i].type);
          return uploadBuffer(device, arr, GPUBufferUsage.STORAGE, this.bufferPool);
        });

        const outputSize = config.output.size * 4;
        const outputBuffer = createOutputBuffer(device, outputSize, this.bufferPool);

        const entries: GPUBindGroupEntry[] = inputBuffers.map((buffer, i) => ({
          binding: i,
          resource: { buffer, size: inputs[i].length * 4 },
        }));
        entries.push({
          binding: inputBuffers.length,
          resource: { buffer: outputBuffer, size: outputSize },
        });

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries,
        });

        const workgroupCount = computeWorkgroupCount(config.output.size, workgroupSize);
        const result = await dispatchAndRead(
          device, pipeline, bindGroup,
          [workgroupCount], outputBuffer, outputSize, this.bufferPool, config.output.type
        );

        for (const buf of inputBuffers) this.bufferPool.release(buf);
        this.bufferPool.release(outputBuffer);

        return result.slice(0, config.output.size);
      },
    };
  }

  // --- Cleanup ---

  destroy(): void {
    this.bufferPool.destroy();
    this.shaderCache.clear();
    this.deviceManager.reset();
  }
}
