import type {
  NumericArray, TypedArray, DataType, MatMulOpts, ScatterOpts, SearchSortedOpts, HistogramOpts, TransposeOpts, KernelConfig,
  FallbackInfo, FallbackMode, GPUOptions, OpStats,
} from "./core/types";
import { inferDataType } from "./core/types";
import { DeviceManager } from "./core/device";
import { BufferPool } from "./core/buffer-pool";
import { ShaderCache } from "./core/shader-cache";
import { uploadBuffer, withErrorScope } from "./core/command";
import { toTypedArray } from "./utils/data-conversion";
import {
  type OpInput,
  type OpOptions,
  type MapOptions,
  isGPUArray,
  resolveInput,
  finalize,
} from "./core/io";
import { withFallback, type FallbackConfig } from "./fallback/index";
import {
  cpuAdd, cpuSubtract, cpuMultiply, cpuDivide,
  cpuMap, cpuZip, cpuReduce, cpuSum, cpuMin, cpuMax, cpuProduct,
  cpuArgmin, cpuArgmax, cpuGather, cpuSearchsorted, cpuCast, cpuTranspose, cpuScatter, cpuHistogram,
  cpuMatmul, cpuScan, cpuSort, cpuSortByKey, cpuFilter,
  cpuUnique, cpuSegmentedReduce, cpuRandom, cpuFft, cpuConvolve,
} from "./fallback/cpu-ops";
import {
  gpuElementwiseBinary, gpuScalarBroadcast, gpuMap, gpuZip,
} from "./ops/elementwise";
import { gpuReduce, gpuSum, gpuMin, gpuMax, gpuProduct, gpuArgmin, gpuArgmax } from "./ops/reduce";
import { gpuGather } from "./ops/gather";
import { gpuSearchsorted } from "./ops/searchsorted";
import { gpuCast } from "./ops/cast";
import { gpuTranspose } from "./ops/transpose";
import { gpuScatter } from "./ops/scatter";
import { gpuHistogram } from "./ops/histogram";
import { gpuMatmul } from "./ops/matmul";
import { gpuScan } from "./ops/scan";
import { gpuSort } from "./ops/sort";
import { gpuSortByKey } from "./ops/sort-by-key";
import { gpuFilter } from "./ops/filter";
import { gpuUnique } from "./ops/unique";
import { gpuSegmentedReduce } from "./ops/segmented-reduce";
import type { SegmentedReduceOpts } from "./ops/segmented-reduce";
import { gpuRandom } from "./ops/random";
import type { RandomOpts } from "./ops/random";
import { gpuFft } from "./ops/fft";
import { gpuConvolve } from "./ops/convolve";
import type { ConvolveOpts } from "./ops/convolve";
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

  // Run a GPU op inside error scopes so WebGPU validation/OOM failures reject
  // instead of silently returning pooled-buffer garbage (see withErrorScope).
  private async guarded<T>(fn: () => Promise<T>): Promise<T> {
    const device = await this.deviceManager.getDevice();
    return withErrorScope(device, fn);
  }

  // Time a forced-GPU op (GPUArray input, keepOnGpu, or custom kernel) and report
  // its stats. These paths have no CPU alternative, so the fallback policy and
  // onFallback hook do not apply — a GPU failure simply throws.
  private async timedGpu<T>(op: string, fn: () => Promise<T>): Promise<T> {
    const t = performance.now();
    const r = await this.guarded(fn);
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
    return withFallback(
      this.deviceManager, op, () => this.guarded(() => gpuFn(false)), cpuFn, this.fallbackConfig()
    );
  }

  // A GPUArray input forces the GPU path for a scalar-returning op (reduce family).
  private runScalarOp(
    op: string,
    gpuFn: () => Promise<number>,
    cpuFn: () => number,
    hasGpuInput: boolean
  ): Promise<number> {
    if (hasGpuInput) return this.timedGpu(op, gpuFn);
    return withFallback(
      this.deviceManager, op, () => this.guarded(gpuFn), cpuFn, this.fallbackConfig()
    );
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

  map(input: NumericArray, fn: ((x: number, i: number, len: number) => number) | string, opts?: MapOptions): Promise<TypedArray>;
  map(input: OpInput, fn: ((x: number, i: number, len: number) => number) | string, opts: MapOptions & { keepOnGpu: true }): Promise<GPUArray>;
  map(input: OpInput, fn: ((x: number, i: number, len: number) => number) | string, opts?: MapOptions): Promise<TypedArray | GPUArray>;
  map(
    input: OpInput,
    fn: ((x: number, i: number, len: number) => number) | string,
    opts?: MapOptions
  ): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(input);
    const keep = opts?.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "map",
      (k) => gpuMap(this.deviceManager, this.bufferPool, this.shaderCache, input, fn, { ...opts, keepOnGpu: k } as MapOptions & { keepOnGpu: true }),
      () => cpuMap(input as NumericArray, fn, opts?.consts as Record<string, NumericArray> | undefined),
      hasGpu,
      keep
    );
  }

  // --- Zip (two-input map) ---

  zip(a: NumericArray, b: NumericArray, fn: ((a: number, b: number) => number) | string): Promise<TypedArray>;
  zip(a: OpInput, b: OpInput, fn: ((a: number, b: number) => number) | string, opts: { keepOnGpu: true }): Promise<GPUArray>;
  zip(a: OpInput, b: OpInput, fn: ((a: number, b: number) => number) | string, opts?: OpOptions): Promise<TypedArray | GPUArray>;
  zip(
    a: OpInput,
    b: OpInput,
    fn: ((a: number, b: number) => number) | string,
    opts?: OpOptions
  ): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(a) || isGPUArray(b);
    const keep = opts?.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "zip",
      (k) => gpuZip(this.deviceManager, this.bufferPool, this.shaderCache, a, b, fn, { keepOnGpu: k } as { keepOnGpu: true }),
      () => cpuZip(a as NumericArray, b as NumericArray, fn),
      hasGpu,
      keep
    );
  }

  // --- Filter (stream compaction) ---

  filter(input: NumericArray, predicate: ((x: number, i: number, len: number) => boolean) | string): Promise<TypedArray>;
  filter(input: OpInput, predicate: ((x: number, i: number, len: number) => boolean) | string, opts: { keepOnGpu: true }): Promise<GPUArray>;
  filter(input: OpInput, predicate: ((x: number, i: number, len: number) => boolean) | string, opts?: OpOptions): Promise<TypedArray | GPUArray>;
  filter(
    input: OpInput,
    predicate: ((x: number, i: number, len: number) => boolean) | string,
    opts?: OpOptions
  ): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(input);
    const keep = opts?.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "filter",
      (k) => gpuFilter(this.deviceManager, this.bufferPool, this.shaderCache, input, predicate, { keepOnGpu: k } as { keepOnGpu: true }),
      () => cpuFilter(input as NumericArray, predicate),
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

  /** Index of the minimum element. Ties resolve to the first (smallest) index. */
  argmin(input: OpInput): Promise<number> {
    return this.runScalarOp(
      "argmin",
      () => gpuArgmin(this.deviceManager, this.bufferPool, this.shaderCache, input),
      () => cpuArgmin(input as NumericArray),
      isGPUArray(input)
    );
  }

  /** Index of the maximum element. Ties resolve to the first (smallest) index. */
  argmax(input: OpInput): Promise<number> {
    return this.runScalarOp(
      "argmax",
      () => gpuArgmax(this.deviceManager, this.bufferPool, this.shaderCache, input),
      () => cpuArgmax(input as NumericArray),
      isGPUArray(input)
    );
  }

  // --- Gather ---

  gather(src: NumericArray, idx: NumericArray): Promise<TypedArray>;
  gather(src: OpInput, idx: OpInput, opts: { keepOnGpu: true }): Promise<GPUArray>;
  gather(src: OpInput, idx: OpInput, opts?: OpOptions): Promise<TypedArray | GPUArray>;
  gather(
    src: OpInput,
    idx: OpInput,
    opts?: OpOptions
  ): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(src) || isGPUArray(idx);
    const keep = opts?.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "gather",
      (k) => gpuGather(this.deviceManager, this.bufferPool, this.shaderCache, src, idx, { keepOnGpu: k } as { keepOnGpu: true }),
      () => cpuGather(src as NumericArray, idx as NumericArray),
      hasGpu,
      keep
    );
  }

  // --- Searchsorted ---

  searchsorted(sorted: NumericArray, queries: NumericArray, opts?: SearchSortedOpts): Promise<Uint32Array>;
  searchsorted(sorted: OpInput, queries: OpInput, opts: SearchSortedOpts & { keepOnGpu: true }): Promise<GPUArray>;
  searchsorted(sorted: OpInput, queries: OpInput, opts?: SearchSortedOpts & OpOptions): Promise<TypedArray | GPUArray>;
  searchsorted(
    sorted: OpInput,
    queries: OpInput,
    opts?: SearchSortedOpts & OpOptions
  ): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(sorted) || isGPUArray(queries);
    const keep = opts?.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "searchsorted",
      (k) => gpuSearchsorted(this.deviceManager, this.bufferPool, this.shaderCache, sorted, queries, { ...opts, keepOnGpu: k } as SearchSortedOpts & { keepOnGpu: true }),
      () => cpuSearchsorted(sorted as NumericArray, queries as NumericArray, opts?.side),
      hasGpu,
      keep
    );
  }

  // --- Transpose ---

  transpose(input: NumericArray, opts: TransposeOpts): Promise<TypedArray>;
  transpose(input: OpInput, opts: TransposeOpts & { keepOnGpu: true }): Promise<GPUArray>;
  transpose(input: OpInput, opts?: TransposeOpts & OpOptions): Promise<TypedArray | GPUArray>;
  transpose(input: OpInput, opts?: TransposeOpts & OpOptions): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(input);
    const keep = opts?.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "transpose",
      (k) => gpuTranspose(this.deviceManager, this.bufferPool, this.shaderCache, input, { ...opts, keepOnGpu: k } as TransposeOpts & { keepOnGpu: true }),
      () => cpuTranspose(input as NumericArray, opts?.rows, opts?.cols),
      hasGpu,
      keep
    );
  }

  // --- Cast (output-dtype conversion) ---

  cast(input: NumericArray, dtype: DataType): Promise<TypedArray>;
  cast(input: OpInput, dtype: DataType, opts: { keepOnGpu: true }): Promise<GPUArray>;
  cast(input: OpInput, dtype: DataType, opts?: OpOptions): Promise<TypedArray | GPUArray>;
  cast(input: OpInput, dtype: DataType, opts?: OpOptions): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(input);
    const keep = opts?.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "cast",
      (k) => gpuCast(this.deviceManager, this.bufferPool, this.shaderCache, input, dtype, { keepOnGpu: k } as { keepOnGpu: true }),
      () => cpuCast(input as NumericArray, dtype),
      hasGpu,
      keep
    );
  }

  // --- Reshape (zero-copy view) ---

  // Pure metadata op. A GPUArray input -> a NON-owning view sharing the same buffer with the
  // new shape (the SOURCE array still owns the buffer — don't destroy it while the view is in
  // use). A CPU array input -> uploaded to a fresh OWNING GPUArray with the shape.
  async reshape(input: OpInput, shape: number[]): Promise<GPUArray> {
    const device = await this.deviceManager.getDevice();
    const total = shape.reduce((a, b) => a * b, 1);
    if (isGPUArray(input)) {
      if (input.isDestroyed) throw new Error("GPUArray has been destroyed");
      if (input.length !== total) {
        throw new Error(`reshape: shape [${shape}] = ${total} elements but array has ${input.length}`);
      }
      return new GPUArray(input.buffer, input.length, input.dtype, device, this.bufferPool, { shape, owns: false });
    }
    const dtype = inferDataType(input);
    const arr = toTypedArray(input, dtype);
    if (arr.length !== total) {
      throw new Error(`reshape: shape [${shape}] = ${total} elements but array has ${arr.length}`);
    }
    const buffer = uploadBuffer(device, arr, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, this.bufferPool);
    return new GPUArray(buffer, arr.length, dtype, device, this.bufferPool, { shape });
  }

  // --- Scatter ---

  scatter(dst: NumericArray, idx: NumericArray, vals: NumericArray, opts?: ScatterOpts): Promise<TypedArray>;
  scatter(dst: OpInput, idx: OpInput, vals: OpInput, opts: ScatterOpts & { keepOnGpu: true }): Promise<GPUArray>;
  scatter(dst: OpInput, idx: OpInput, vals: OpInput, opts?: ScatterOpts & OpOptions): Promise<TypedArray | GPUArray>;
  scatter(
    dst: OpInput,
    idx: OpInput,
    vals: OpInput,
    opts?: ScatterOpts & OpOptions
  ): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(dst) || isGPUArray(idx) || isGPUArray(vals);
    const keep = opts?.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "scatter",
      (k) => gpuScatter(this.deviceManager, this.bufferPool, this.shaderCache, dst, idx, vals, { ...opts, keepOnGpu: k } as ScatterOpts & { keepOnGpu: true }),
      () => cpuScatter(dst as NumericArray, idx as NumericArray, vals as NumericArray, opts?.mode),
      hasGpu,
      keep
    );
  }

  // --- Histogram ---

  histogram(input: NumericArray, opts: HistogramOpts): Promise<Uint32Array>;
  histogram(input: OpInput, opts: HistogramOpts & { keepOnGpu: true }): Promise<GPUArray>;
  histogram(input: OpInput, opts: HistogramOpts & OpOptions): Promise<TypedArray | GPUArray>;
  histogram(
    input: OpInput,
    opts: HistogramOpts & OpOptions
  ): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(input);
    const keep = opts.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "histogram",
      (k) => gpuHistogram(this.deviceManager, this.bufferPool, this.shaderCache, input, { ...opts, keepOnGpu: k }),
      () => cpuHistogram(input as NumericArray, opts.bins, opts.min, opts.max),
      hasGpu,
      keep
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

  // --- Sort by key ---

  sortByKey(keys: NumericArray, values: NumericArray): Promise<[TypedArray, TypedArray]>;
  sortByKey(keys: OpInput, values: OpInput, opts: { keepOnGpu: true }): Promise<[GPUArray, GPUArray]>;
  sortByKey(keys: OpInput, values: OpInput, opts?: OpOptions): Promise<[TypedArray, TypedArray] | [GPUArray, GPUArray]>;
  sortByKey(
    keys: OpInput,
    values: OpInput,
    opts?: OpOptions
  ): Promise<[TypedArray, TypedArray] | [GPUArray, GPUArray]> {
    // Two-output op → cannot use runArrayOp (single-output). Route manually, mirroring
    // runArrayOp: forced-GPU when a GPUArray input or keepOnGpu; otherwise withFallback.
    const hasGpu = isGPUArray(keys) || isGPUArray(values);
    const keep = opts?.keepOnGpu ?? hasGpu;
    if (hasGpu || keep) {
      return this.timedGpu("sortByKey", () =>
        gpuSortByKey(this.deviceManager, this.bufferPool, this.shaderCache, keys, values, { keepOnGpu: keep } as { keepOnGpu: true })
      );
    }
    return withFallback(
      this.deviceManager,
      "sortByKey",
      () => gpuSortByKey(this.deviceManager, this.bufferPool, this.shaderCache, keys, values),
      () => cpuSortByKey(keys as NumericArray, values as NumericArray),
      this.fallbackConfig()
    );
  }

  // --- Unique (sorted distinct values) ---

  unique(input: NumericArray): Promise<TypedArray>;
  unique(input: OpInput, opts: { keepOnGpu: true }): Promise<GPUArray>;
  unique(input: OpInput, opts?: OpOptions): Promise<TypedArray | GPUArray>;
  unique(
    input: OpInput,
    opts?: OpOptions
  ): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(input);
    const keep = opts?.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "unique",
      (k) => gpuUnique(this.deviceManager, this.bufferPool, this.shaderCache, input, { keepOnGpu: k } as { keepOnGpu: true }),
      () => cpuUnique(input as NumericArray),
      hasGpu,
      keep
    );
  }

  // --- Segmented reduce (group-by) ---

  segmentedReduce(values: NumericArray, segmentIds: NumericArray, opts: SegmentedReduceOpts): Promise<TypedArray>;
  segmentedReduce(values: OpInput, segmentIds: OpInput, opts: SegmentedReduceOpts & { keepOnGpu: true }): Promise<GPUArray>;
  segmentedReduce(values: OpInput, segmentIds: OpInput, opts: SegmentedReduceOpts & OpOptions): Promise<TypedArray | GPUArray>;
  segmentedReduce(
    values: OpInput,
    segmentIds: OpInput,
    opts: SegmentedReduceOpts & OpOptions
  ): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(values) || isGPUArray(segmentIds);
    const keep = opts.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "segmentedReduce",
      (k) => gpuSegmentedReduce(this.deviceManager, this.bufferPool, this.shaderCache, values, segmentIds, { ...opts, keepOnGpu: k } as SegmentedReduceOpts & { keepOnGpu: true }),
      () => cpuSegmentedReduce(values as NumericArray, segmentIds as NumericArray, opts),
      hasGpu,
      keep
    );
  }

  // --- Random (counter-based generator) ---

  random(n: number, opts?: RandomOpts): Promise<TypedArray>;
  random(n: number, opts: RandomOpts & { keepOnGpu: true }): Promise<GPUArray>;
  random(n: number, opts?: RandomOpts): Promise<TypedArray | GPUArray>;
  random(n: number, opts?: RandomOpts): Promise<TypedArray | GPUArray> {
    // Generator: no array input, so the GPU path is never forced by an input. keepOnGpu still
    // forces it; otherwise withFallback runs the GPU op first and cpuRandom on failure.
    const keep = opts?.keepOnGpu ?? false;
    return this.runArrayOp(
      "random",
      (k) => gpuRandom(this.deviceManager, this.bufferPool, this.shaderCache, n, { ...opts, keepOnGpu: k } as RandomOpts & { keepOnGpu: true }),
      () => cpuRandom(n, opts),
      false,
      keep
    );
  }

  // --- FFT (forward, real input -> interleaved complex spectrum) ---

  fft(input: NumericArray): Promise<TypedArray>;
  fft(input: OpInput, opts: { keepOnGpu: true }): Promise<GPUArray>;
  fft(input: OpInput, opts?: OpOptions): Promise<TypedArray | GPUArray>;
  fft(
    input: OpInput,
    opts?: OpOptions
  ): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(input);
    const keep = opts?.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "fft",
      (k) => gpuFft(this.deviceManager, this.bufferPool, this.shaderCache, input, { keepOnGpu: k } as { keepOnGpu: true }),
      () => cpuFft(input as NumericArray),
      hasGpu,
      keep
    );
  }

  // --- Convolution (1-D, numpy.convolve modes) ---

  convolve(input: NumericArray, kernel: NumericArray, opts?: ConvolveOpts): Promise<TypedArray>;
  convolve(input: OpInput, kernel: OpInput, opts: ConvolveOpts & { keepOnGpu: true }): Promise<GPUArray>;
  convolve(input: OpInput, kernel: OpInput, opts?: ConvolveOpts): Promise<TypedArray | GPUArray>;
  convolve(
    input: OpInput,
    kernel: OpInput,
    opts?: ConvolveOpts
  ): Promise<TypedArray | GPUArray> {
    const hasGpu = isGPUArray(input) || isGPUArray(kernel);
    const keep = opts?.keepOnGpu ?? hasGpu;
    return this.runArrayOp(
      "convolve",
      (k) => gpuConvolve(this.deviceManager, this.bufferPool, this.shaderCache, input, kernel, { ...opts, keepOnGpu: k } as ConvolveOpts & { keepOnGpu: true }),
      () => cpuConvolve(input as NumericArray, kernel as NumericArray, opts?.mode),
      hasGpu,
      keep
    );
  }

  // --- Pipeline builder ---

  pipeline(): Pipeline {
    return new Pipeline(this.deviceManager, this.bufferPool, this.shaderCache, () => this.fallbackConfig());
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
