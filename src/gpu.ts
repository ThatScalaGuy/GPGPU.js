import type {
  NumericArray, TypedArray, MatMulOpts, KernelConfig,
  FallbackInfo, FallbackMode, GPUOptions, OpStats,
} from "./core/types";
import { inferDataType } from "./core/types";
import { DeviceManager } from "./core/device";
import { BufferPool } from "./core/buffer-pool";
import { ShaderCache } from "./core/shader-cache";
import { uploadBuffer } from "./core/command";
import { toTypedArray } from "./utils/data-conversion";
import {
  type OpInput,
  type OpOptions,
  isGPUArray,
  resolveInput,
  finalize,
} from "./core/io";
import { withFallback, type FallbackConfig } from "./fallback/index";
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
import { GPUArray } from "./pipeline/gpu-array";

export class GPU {
  private deviceManager = new DeviceManager();
  private bufferPool = new BufferPool();
  private shaderCache = new ShaderCache();

  /** What to do when a GPU op fails and a CPU fallback exists. Default `"warn"`. */
  fallback: FallbackMode;
  /** Called when a GPU op throws, before the fallback policy is applied. */
  onFallback?: (info: FallbackInfo) => void;
  /** Called after every op with the backend that ran and how long it took. */
  onStats?: (stats: OpStats) => void;

  constructor(opts?: GPUOptions) {
    this.fallback = opts?.fallback ?? "warn";
    this.onFallback = opts?.onFallback;
    this.onStats = opts?.onStats;
  }

  /** Check if WebGPU is available */
  isAvailable(): boolean {
    return this.deviceManager.isAvailable();
  }

  private fallbackConfig(): FallbackConfig {
    return { mode: this.fallback, onFallback: this.onFallback, onStats: this.onStats };
  }

  // Time a forced-GPU op (GPUArray input, keepOnGpu, or custom kernel) and report
  // its stats. These paths have no CPU alternative, so the fallback policy and
  // onFallback hook do not apply — a GPU failure simply throws.
  private async timedGpu<T>(op: string, fn: () => Promise<T>): Promise<T> {
    const t = performance.now();
    const r = await fn();
    this.onStats?.({ op, backend: "gpu", ms: performance.now() - t });
    return r;
  }

  /** Upload a CPU array to the GPU and keep it resident as a GPUArray. */
  async upload(input: NumericArray): Promise<GPUArray> {
    const device = await this.deviceManager.getDevice();
    const dtype = inferDataType(input);
    const arr = toTypedArray(input, dtype);
    const buffer = uploadBuffer(
      device, arr, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, this.bufferPool
    );
    return new GPUArray(buffer, arr.length, dtype, device, this.bufferPool);
  }

  // Decide GPU-vs-CPU path for an array-returning op. A GPUArray input or an explicit
  // keepOnGpu forces the GPU path (CPU can't produce/consume a GPUArray); otherwise the
  // op runs through the CPU fallback as before. `keepOnGpu` defaults to "auto": true when
  // any input is already a GPUArray.
  private runArrayOp(
    op: string,
    gpuFn: (keepOnGpu: boolean) => Promise<TypedArray | GPUArray>,
    cpuFn: () => TypedArray,
    hasGpuInput: boolean,
    keepOnGpu: boolean
  ): Promise<TypedArray | GPUArray> {
    if (hasGpuInput || keepOnGpu) {
      return this.timedGpu(op, () => gpuFn(keepOnGpu));
    }
    return withFallback(this.deviceManager, op, () => gpuFn(false), cpuFn, this.fallbackConfig());
  }

  // A GPUArray input forces the GPU path for a scalar-returning op (reduce family).
  private runScalarOp(
    op: string,
    gpuFn: () => Promise<number>,
    cpuFn: () => number,
    hasGpuInput: boolean
  ): Promise<number> {
    if (hasGpuInput) return this.timedGpu(op, gpuFn);
    return withFallback(this.deviceManager, op, gpuFn, cpuFn, this.fallbackConfig());
  }

  // --- Elementwise operations ---

  add(a: NumericArray, b: NumericArray | number): Promise<TypedArray>;
  add(a: OpInput, b: OpInput | number, opts: { keepOnGpu: true }): Promise<GPUArray>;
  add(a: OpInput, b: OpInput | number, opts?: OpOptions): Promise<TypedArray | GPUArray>;
  add(a: OpInput, b: OpInput | number, opts?: OpOptions): Promise<TypedArray | GPUArray> {
    return this.binaryOp("add", a, b, "+", cpuAdd, opts);
  }

  subtract(a: NumericArray, b: NumericArray | number): Promise<TypedArray>;
  subtract(a: OpInput, b: OpInput | number, opts: { keepOnGpu: true }): Promise<GPUArray>;
  subtract(a: OpInput, b: OpInput | number, opts?: OpOptions): Promise<TypedArray | GPUArray>;
  subtract(a: OpInput, b: OpInput | number, opts?: OpOptions): Promise<TypedArray | GPUArray> {
    return this.binaryOp("subtract", a, b, "-", cpuSubtract, opts);
  }

  multiply(a: NumericArray, b: NumericArray | number): Promise<TypedArray>;
  multiply(a: OpInput, b: OpInput | number, opts: { keepOnGpu: true }): Promise<GPUArray>;
  multiply(a: OpInput, b: OpInput | number, opts?: OpOptions): Promise<TypedArray | GPUArray>;
  multiply(a: OpInput, b: OpInput | number, opts?: OpOptions): Promise<TypedArray | GPUArray> {
    return this.binaryOp("multiply", a, b, "*", cpuMultiply, opts);
  }

  divide(a: NumericArray, b: NumericArray | number): Promise<TypedArray>;
  divide(a: OpInput, b: OpInput | number, opts: { keepOnGpu: true }): Promise<GPUArray>;
  divide(a: OpInput, b: OpInput | number, opts?: OpOptions): Promise<TypedArray | GPUArray>;
  divide(a: OpInput, b: OpInput | number, opts?: OpOptions): Promise<TypedArray | GPUArray> {
    return this.binaryOp("divide", a, b, "/", cpuDivide, opts);
  }

  private binaryOp(
    name: string,
    a: OpInput,
    b: OpInput | number,
    op: string,
    cpuFn: (a: NumericArray, b: NumericArray | number) => TypedArray,
    opts?: OpOptions
  ): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(a) || isGPUArray(b);
    const keep = opts?.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      name,
      (k) =>
        typeof b === "number"
          ? gpuScalarBroadcast(this.deviceManager, this.bufferPool, this.shaderCache, a, b, op, { keepOnGpu: k } as { keepOnGpu: true })
          : gpuElementwiseBinary(this.deviceManager, this.bufferPool, this.shaderCache, a, b, op, { keepOnGpu: k } as { keepOnGpu: true }),
      () => cpuFn(a as NumericArray, b as NumericArray | number),
      hasGpu,
      keep
    );
  }

  // --- Map ---

  map(input: NumericArray, fn: ((x: number) => number) | string): Promise<TypedArray>;
  map(input: OpInput, fn: ((x: number) => number) | string, opts: { keepOnGpu: true }): Promise<GPUArray>;
  map(input: OpInput, fn: ((x: number) => number) | string, opts?: OpOptions): Promise<TypedArray | GPUArray>;
  map(
    input: OpInput,
    fn: ((x: number) => number) | string,
    opts?: OpOptions
  ): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(input);
    const keep = opts?.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "map",
      (k) => gpuMap(this.deviceManager, this.bufferPool, this.shaderCache, input, fn, { keepOnGpu: k } as { keepOnGpu: true }),
      () => cpuMap(input as NumericArray, fn),
      hasGpu,
      keep
    );
  }

  // --- Reduce ---

  reduce(
    input: OpInput,
    fn: ((a: number, b: number) => number) | string,
    identity: number
  ): Promise<number> {
    return this.runScalarOp(
      "reduce",
      () => gpuReduce(this.deviceManager, this.bufferPool, this.shaderCache, input, fn, identity),
      () => cpuReduce(input as NumericArray, fn, identity),
      isGPUArray(input)
    );
  }

  sum(input: OpInput): Promise<number> {
    return this.runScalarOp(
      "sum",
      () => gpuSum(this.deviceManager, this.bufferPool, this.shaderCache, input),
      () => cpuSum(input as NumericArray),
      isGPUArray(input)
    );
  }

  min(input: OpInput): Promise<number> {
    return this.runScalarOp(
      "min",
      () => gpuMin(this.deviceManager, this.bufferPool, this.shaderCache, input),
      () => cpuMin(input as NumericArray),
      isGPUArray(input)
    );
  }

  max(input: OpInput): Promise<number> {
    return this.runScalarOp(
      "max",
      () => gpuMax(this.deviceManager, this.bufferPool, this.shaderCache, input),
      () => cpuMax(input as NumericArray),
      isGPUArray(input)
    );
  }

  product(input: OpInput): Promise<number> {
    return this.runScalarOp(
      "product",
      () => gpuProduct(this.deviceManager, this.bufferPool, this.shaderCache, input),
      () => cpuProduct(input as NumericArray),
      isGPUArray(input)
    );
  }

  // --- Matrix multiply ---

  matmul(a: NumericArray, b: NumericArray, opts: MatMulOpts): Promise<TypedArray>;
  matmul(a: OpInput, b: OpInput, opts: MatMulOpts & { keepOnGpu: true }): Promise<GPUArray>;
  matmul(a: OpInput, b: OpInput, opts: MatMulOpts & OpOptions): Promise<TypedArray | GPUArray>;
  matmul(
    a: OpInput,
    b: OpInput,
    opts: MatMulOpts & OpOptions
  ): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(a) || isGPUArray(b);
    const keep = opts.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "matmul",
      (k) => gpuMatmul(this.deviceManager, this.bufferPool, this.shaderCache, a, b, { ...opts, keepOnGpu: k } as MatMulOpts & { keepOnGpu: true }),
      () => cpuMatmul(a as NumericArray, b as NumericArray, opts),
      hasGpu,
      keep
    );
  }

  // --- Scan (prefix sum) ---

  scan(input: NumericArray, fn?: ((a: number, b: number) => number) | string, identity?: number): Promise<TypedArray>;
  scan(input: OpInput, fn: ((a: number, b: number) => number) | string | undefined, identity: number | undefined, opts: { keepOnGpu: true }): Promise<GPUArray>;
  scan(input: OpInput, fn?: ((a: number, b: number) => number) | string, identity?: number, opts?: OpOptions): Promise<TypedArray | GPUArray>;
  scan(
    input: OpInput,
    fn?: ((a: number, b: number) => number) | string,
    identity?: number,
    opts?: OpOptions
  ): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(input);
    const keep = opts?.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "scan",
      (k) => gpuScan(this.deviceManager, this.bufferPool, this.shaderCache, input, fn, identity, { keepOnGpu: k } as { keepOnGpu: true }),
      () => cpuScan(input as NumericArray, fn ?? ((a, b) => a + b), identity ?? 0),
      hasGpu,
      keep
    );
  }

  // --- Sort ---

  sort(input: NumericArray): Promise<TypedArray>;
  sort(input: OpInput, opts: { keepOnGpu: true }): Promise<GPUArray>;
  sort(input: OpInput, opts?: OpOptions): Promise<TypedArray | GPUArray>;
  sort(input: OpInput, opts?: OpOptions): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(input);
    const keep = opts?.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "sort",
      (k) => gpuSort(this.deviceManager, this.bufferPool, this.shaderCache, input, { keepOnGpu: k } as { keepOnGpu: true }),
      () => cpuSort(input as NumericArray),
      hasGpu,
      keep
    );
  }

  // --- Pipeline builder ---

  pipeline(): Pipeline {
    return new Pipeline(this.deviceManager, this.bufferPool, this.shaderCache, this.onStats);
  }

  // --- Custom kernel ---

  async createKernel(config: KernelConfig): Promise<{
    run: (...args: (OpInput | OpOptions)[]) => Promise<TypedArray | GPUArray>;
  }> {
    const device = await this.deviceManager.getDevice();
    const pipeline = await this.shaderCache.getOrCreate(device, config.shader, "custom-kernel");
    const workgroupSize = config.workgroupSize ?? 64;

    return {
      run: async (...args: (OpInput | OpOptions)[]) => {
        const { createOutputBuffer, dispatchOnly } = await import("./core/command");
        const { computeWorkgroupCount } = await import("./utils/workgroup");

        // A trailing options object (not an array / typed array / GPUArray) is the opts arg.
        let opts: OpOptions | undefined;
        const last = args[args.length - 1];
        if (last && !Array.isArray(last) && !isGPUArray(last) && !ArrayBuffer.isView(last as ArrayBufferView)) {
          opts = args.pop() as OpOptions;
        }
        const inputs = args as OpInput[];
        const keepOnGpu = opts?.keepOnGpu ?? inputs.some(isGPUArray);

        const resolved = inputs.map((input, i) =>
          resolveInput(input, device, this.bufferPool, config.inputs[i].type, GPUBufferUsage.STORAGE)
        );

        const outputSize = config.output.size * 4;
        const outputBuffer = createOutputBuffer(device, outputSize, this.bufferPool);

        const entries: GPUBindGroupEntry[] = resolved.map((r, i) => ({
          binding: i,
          resource: { buffer: r.buffer, size: r.length * 4 },
        }));
        entries.push({
          binding: resolved.length,
          resource: { buffer: outputBuffer, size: outputSize },
        });

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries,
        });

        return this.timedGpu("kernel", () => {
          const workgroupCount = computeWorkgroupCount(config.output.size, workgroupSize);
          dispatchOnly(device, pipeline, bindGroup, [workgroupCount]);

          for (const r of resolved) r.release();

          return finalize(
            new GPUArray(outputBuffer, config.output.size, config.output.type, device, this.bufferPool),
            keepOnGpu
          );
        });
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
