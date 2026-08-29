import type { DataType, HistogramOpts, NumericArray, OpStats, TypedArray } from "../core/types";
import { inferDataType } from "../core/types";
import { toTypedArray } from "../utils/data-conversion";
import type { FallbackConfig } from "../fallback/index";
import {
  cpuCast,
  cpuConvolve,
  cpuFilter,
  cpuGather,
  cpuHistogram,
  cpuMap,
  cpuReduce,
  cpuScan,
  cpuSort,
  cpuUnique,
} from "../fallback/cpu-ops";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { uploadBuffer, createOutputBuffer, viewFor, withErrorScope } from "../core/command";
import { GPUArray } from "./gpu-array";
import { type OpInput, type OpOptions, inputDtype, isGPUArray, finalize } from "../core/io";
import { computeWorkgroupGrid } from "../utils/workgroup";
import { parseExpression } from "../codegen/expression-parser";
import { emitWGSL, formatLiteral } from "../codegen/wgsl-emitter";
import { fusedMapShader, reduceShader } from "../codegen/templates";
import { REDUCE_WORKGROUP_SIZE } from "../core/types";
import { gpuScan } from "../ops/scan";
import { gpuFilter } from "../ops/filter";
import { gpuSort } from "../ops/sort";
import { gpuCast } from "../ops/cast";
import { gpuUnique } from "../ops/unique";
import { gpuHistogram } from "../ops/histogram";
import { gpuConvolve, type ConvolveMode } from "../ops/convolve";
import { gpuGather } from "../ops/gather";

interface MapStep {
  type: "map";
  fn: ((x: number) => number) | string;
}

interface ReduceStep {
  type: "reduce";
  fn: ((a: number, b: number) => number) | string;
  identity: number;
}

interface ScanStep {
  type: "scan";
  fn?: ((a: number, b: number) => number) | string;
  identity?: number;
}

interface FilterStep {
  type: "filter";
  predicate: ((x: number, i: number, len: number) => boolean) | string;
}

interface SortStep {
  type: "sort";
}

interface CastStep {
  type: "cast";
  toDtype: DataType;
}

interface UniqueStep {
  type: "unique";
}

interface HistogramStep {
  type: "histogram";
  bins: number;
  min: number;
  max: number;
}

interface ConvolveStep {
  type: "convolve";
  kernel: OpInput;
  mode?: ConvolveMode;
}

interface GatherStep {
  type: "gather";
  indices: OpInput;
}

type PipelineStep =
  | MapStep
  | ReduceStep
  | ScanStep
  | FilterStep
  | SortStep
  | CastStep
  | UniqueStep
  | HistogramStep
  | ConvolveStep
  | GatherStep;

// Execution form: consecutive `map` steps are folded into one fused dispatch (see `run`).
type FusedMapStep = { type: "fusedmap"; fns: (((x: number) => number) | string)[] };
type ExecStep =
  | FusedMapStep
  | ReduceStep
  | ScanStep
  | FilterStep
  | SortStep
  | CastStep
  | UniqueStep
  | HistogramStep
  | ConvolveStep
  | GatherStep;

export class Pipeline {
  private steps: PipelineStep[] = [];
  private deviceManager: DeviceManager;
  private bufferPool: BufferPool;
  private shaderCache: ShaderCache;
  // Read at run() time (not captured at construction) so mutations of the owning
  // GPU instance's `fallback`/`onStats`/`onFallback` fields are honoured.
  private getFallbackConfig?: () => FallbackConfig;

  constructor(
    deviceManager: DeviceManager,
    bufferPool: BufferPool,
    shaderCache: ShaderCache,
    getFallbackConfig?: () => FallbackConfig
  ) {
    this.deviceManager = deviceManager;
    this.bufferPool = bufferPool;
    this.shaderCache = shaderCache;
    this.getFallbackConfig = getFallbackConfig;
  }

  // A reduce collapses the stream to a scalar, so no step may follow it.
  private add(step: PipelineStep): Pipeline {
    if (this.steps[this.steps.length - 1]?.type === "reduce") {
      throw new Error("reduce must be the terminal pipeline step");
    }
    this.steps.push(step);
    return this;
  }

  map(fn: ((x: number) => number) | string): Pipeline {
    return this.add({ type: "map", fn });
  }

  reduce(
    fn: ((a: number, b: number) => number) | string,
    identity: number
  ): Pipeline {
    return this.add({ type: "reduce", fn, identity });
  }

  /** Prefix scan over the stream (defaults to an inclusive prefix sum). */
  scan(
    fn?: ((a: number, b: number) => number) | string,
    identity?: number
  ): Pipeline {
    return this.add({ type: "scan", fn, identity });
  }

  /** Keep only elements for which `predicate` is true (variable-length result). */
  filter(
    predicate: ((x: number, i: number, len: number) => boolean) | string
  ): Pipeline {
    return this.add({ type: "filter", predicate });
  }

  /** Sort the stream ascending. */
  sort(): Pipeline {
    return this.add({ type: "sort" });
  }

  /** Convert the stream's element type; subsequent steps run in the new dtype. */
  cast(toDtype: DataType): Pipeline {
    return this.add({ type: "cast", toDtype });
  }

  /** Sorted distinct values of the stream (variable-length result). */
  unique(): Pipeline {
    return this.add({ type: "unique" });
  }

  /** Bucket the stream into `bins` u32 counts over `[min, max]` (length becomes `bins`,
   *  dtype becomes u32). Followed by `.scan()` this yields a CDF. */
  histogram(opts: HistogramOpts): Pipeline {
    return this.add({ type: "histogram", bins: opts.bins, min: opts.min, max: opts.max });
  }

  /** 1-D convolution with a fixed `kernel` (numpy.convolve semantics; default mode "full"). */
  convolve(kernel: OpInput, opts?: { mode?: ConvolveMode }): Pipeline {
    return this.add({ type: "convolve", kernel, mode: opts?.mode });
  }

  /** Reorder/select elements by `indices` (result length becomes `indices.length`). */
  gather(indices: OpInput): Pipeline {
    return this.add({ type: "gather", indices });
  }

  async run(input: OpInput, opts?: OpOptions): Promise<TypedArray | GPUArray | number> {
    const cfg: FallbackConfig = this.getFallbackConfig?.() ?? { mode: "warn" };
    // Forced-GPU paths never fall back (same rule as standalone ops): a GPUArray
    // input or a requested GPUArray result must live on a device, and a GPUArray
    // captured operand can't be read by the CPU interpreter.
    const forcedGpu =
      isGPUArray(input) ||
      opts?.keepOnGpu === true ||
      this.steps.some(
        (s) =>
          (s.type === "convolve" && isGPUArray(s.kernel)) ||
          (s.type === "gather" && isGPUArray(s.indices))
      );

    if (forcedGpu || this.deviceManager.isAvailable()) {
      try {
        const device = await this.deviceManager.getDevice();
        return await withErrorScope(device, () => this.runGpu(input, opts, cfg.onStats));
      } catch (e) {
        if (forcedGpu) throw e;
        cfg.onFallback?.({ op: "pipeline", error: e });
        if (cfg.mode === "throw") throw e;
        if (cfg.mode === "warn")
          console.warn(`GPU execution failed for "pipeline", falling back to CPU:`, e);
      }
    }

    const t = performance.now();
    const r = this.runCpu(input as NumericArray);
    cfg.onStats?.({ op: "pipeline", backend: "cpu", ms: performance.now() - t });
    return r;
  }

  // CPU interpreter over the recorded steps — the same cpu* implementations the
  // standalone ops fall back to, so GPU and CPU pipelines agree step by step.
  private runCpu(input: NumericArray): TypedArray | number {
    let current: NumericArray = input;
    for (const step of this.steps) {
      switch (step.type) {
        case "map":
          current = cpuMap(current, step.fn);
          break;
        case "reduce":
          return cpuReduce(current, step.fn, step.identity);
        case "scan":
          current = cpuScan(current, step.fn ?? ((a, b) => a + b), step.identity ?? 0);
          break;
        case "filter":
          current = cpuFilter(current, step.predicate);
          break;
        case "sort":
          current = cpuSort(current);
          break;
        case "cast":
          current = cpuCast(current, step.toDtype);
          break;
        case "unique":
          current = cpuUnique(current);
          break;
        case "histogram":
          current = cpuHistogram(current, step.bins, step.min, step.max);
          break;
        case "convolve":
          current = cpuConvolve(current, step.kernel as NumericArray, step.mode);
          break;
        case "gather":
          current = cpuGather(current, step.indices as NumericArray);
          break;
      }
    }
    return toTypedArray(current, inferDataType(current));
  }

  private async runGpu(
    input: OpInput,
    opts: OpOptions | undefined,
    onStats: ((stats: OpStats) => void) | undefined
  ): Promise<TypedArray | GPUArray | number> {
    const t0 = performance.now();
    const device = await this.deviceManager.getDevice();
    let dtype = inputDtype(input);
    const hasGpuInput = isGPUArray(input);
    const keepOnGpu = opts?.keepOnGpu ?? hasGpuInput;

    // A GPUArray input is reused in place as the initial buffer (the first step reads it
    // and writes a fresh buffer, so it is never mutated) and is never released here.
    const buffersToRelease: GPUBuffer[] = [];
    let currentSize: number;
    let currentBuffer: GPUBuffer;
    if (isGPUArray(input)) {
      if (input.isDestroyed) throw new Error("GPUArray has been destroyed");
      currentSize = input.length;
      currentBuffer = input.buffer;
    } else {
      const arr = toTypedArray(input, dtype);
      currentSize = arr.length;
      currentBuffer = uploadBuffer(
        device, arr, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, this.bufferPool
      );
      buffersToRelease.push(currentBuffer);
    }
    let currentByteSize = currentSize * 4;

    // Fold consecutive `map` steps into one fused dispatch. dtype only ever changes on a
    // `cast` (a different step type), so every fused run shares a single dtype.
    const execSteps: ExecStep[] = [];
    for (const step of this.steps) {
      const last = execSteps[execSteps.length - 1];
      if (step.type === "map" && last?.type === "fusedmap") {
        last.fns.push(step.fn);
      } else if (step.type === "map") {
        execSteps.push({ type: "fusedmap", fns: [step.fn] });
      } else {
        execSteps.push(step);
      }
    }

    // Delegate a step to an existing op, keeping the result on the GPU. The current stream
    // is wrapped as a NON-OWNING input so the op never releases the caller's/intermediate
    // buffer; the op's freshly allocated output becomes the new current buffer (tracked for
    // release). The returned handle is intentionally not destroyed — its buffer lives on as
    // `currentBuffer` and is freed via `buffersToRelease`.
    const delegate = async (runOp: (gin: GPUArray) => Promise<GPUArray>): Promise<void> => {
      const gin = new GPUArray(
        currentBuffer, currentSize, dtype, device, this.bufferPool, { owns: false }
      );
      const res = await runOp(gin);
      currentBuffer = res.buffer;
      currentSize = res.length;
      currentByteSize = currentSize * 4;
      buffersToRelease.push(currentBuffer);
    };

    let lastStepIsReduce = false;
    let reduceIdentity = 0;

    for (const step of execSteps) {
      if (step.type === "fusedmap") {
        lastStepIsReduce = false;
        if (currentSize === 0) continue; // empty stream maps to itself
        const expressions = step.fns.map((fn) => emitWGSL(parseExpression(fn, ["x"]), dtype));
        const shader = fusedMapShader(expressions, dtype);
        const pipeline = await this.shaderCache.getOrCreate(device, shader, `pipeline-map-${dtype}`);

        const outBuffer = createOutputBuffer(device, currentByteSize, this.bufferPool);
        buffersToRelease.push(outBuffer);

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: currentBuffer, size: currentByteSize } },
            { binding: 1, resource: { buffer: outBuffer, size: currentByteSize } },
          ],
        });

        const [wgX, wgY] = computeWorkgroupGrid(currentSize);
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(wgX, wgY);
        pass.end();
        device.queue.submit([encoder.finish()]);

        currentBuffer = outBuffer;
      } else if (step.type === "scan") {
        lastStepIsReduce = false;
        await delegate((gin) =>
          gpuScan(this.deviceManager, this.bufferPool, this.shaderCache, gin, step.fn, step.identity, { keepOnGpu: true })
        );
      } else if (step.type === "filter") {
        lastStepIsReduce = false;
        await delegate((gin) =>
          gpuFilter(this.deviceManager, this.bufferPool, this.shaderCache, gin, step.predicate, { keepOnGpu: true })
        );
      } else if (step.type === "sort") {
        lastStepIsReduce = false;
        await delegate((gin) =>
          gpuSort(this.deviceManager, this.bufferPool, this.shaderCache, gin, { keepOnGpu: true })
        );
      } else if (step.type === "cast") {
        lastStepIsReduce = false;
        await delegate((gin) =>
          gpuCast(this.deviceManager, this.bufferPool, this.shaderCache, gin, step.toDtype, { keepOnGpu: true })
        );
        dtype = step.toDtype;
      } else if (step.type === "unique") {
        lastStepIsReduce = false;
        await delegate((gin) =>
          gpuUnique(this.deviceManager, this.bufferPool, this.shaderCache, gin, { keepOnGpu: true })
        );
      } else if (step.type === "histogram") {
        lastStepIsReduce = false;
        await delegate((gin) =>
          gpuHistogram(this.deviceManager, this.bufferPool, this.shaderCache, gin, { bins: step.bins, min: step.min, max: step.max, keepOnGpu: true })
        );
        dtype = "u32"; // counts are u32 regardless of input dtype
      } else if (step.type === "convolve") {
        lastStepIsReduce = false;
        await delegate((gin) =>
          gpuConvolve(this.deviceManager, this.bufferPool, this.shaderCache, gin, step.kernel, { mode: step.mode, keepOnGpu: true })
        );
      } else if (step.type === "gather") {
        lastStepIsReduce = false;
        await delegate((gin) =>
          gpuGather(this.deviceManager, this.bufferPool, this.shaderCache, gin, step.indices, { keepOnGpu: true })
        );
      } else if (step.type === "reduce") {
        lastStepIsReduce = true;
        reduceIdentity = step.identity;
        const ir = parseExpression(step.fn, ["a", "b"]);
        const reduceExpr = emitWGSL(ir, dtype);
        const identityStr = formatLiteral(step.identity, dtype);
        const shader = reduceShader(reduceExpr, identityStr, dtype);
        const pipeline = await this.shaderCache.getOrCreate(device, shader, `pipeline-reduce-${dtype}`);

        while (currentSize > 1) {
          const workgroupCount = Math.ceil(currentSize / REDUCE_WORKGROUP_SIZE);
          const outputByteSize = workgroupCount * 4;

          const outBuffer = createOutputBuffer(device, outputByteSize, this.bufferPool);
          buffersToRelease.push(outBuffer);

          const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: currentBuffer, size: currentSize * 4 } },
              { binding: 1, resource: { buffer: outBuffer, size: outputByteSize } },
            ],
          });

          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(workgroupCount);
          pass.end();
          device.queue.submit([encoder.finish()]);

          currentBuffer = outBuffer;
          currentSize = workgroupCount;
          currentByteSize = outputByteSize;
        }
      }
    }

    // A reduce-terminated pipeline always yields a scalar; read it back directly.
    if (lastStepIsReduce) {
      // An empty stream (e.g. a filter that kept nothing) has no element to read back —
      // the reduction of nothing is the identity.
      if (currentSize === 0) {
        for (const buf of buffersToRelease) this.bufferPool.release(buf);
        onStats?.({ op: "pipeline", backend: "gpu", ms: performance.now() - t0 });
        return reduceIdentity;
      }
      const staging = this.bufferPool.acquire(
        device, 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      );
      const copyEncoder = device.createCommandEncoder();
      copyEncoder.copyBufferToBuffer(currentBuffer, 0, staging, 0, 4);
      device.queue.submit([copyEncoder.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      const result = viewFor(dtype, staging.getMappedRange().slice(0))[0];
      staging.unmap();
      this.bufferPool.release(staging);
      for (const buf of buffersToRelease) this.bufferPool.release(buf);
      onStats?.({ op: "pipeline", backend: "gpu", ms: performance.now() - t0 });
      return result;
    }

    // Array result: hand the final buffer to a GPUArray. `finalize` either keeps it on
    // the GPU or reads it back and frees it — so it must not also be released here.
    for (const buf of buffersToRelease) {
      if (buf !== currentBuffer) this.bufferPool.release(buf);
    }
    const out = await finalize(
      new GPUArray(currentBuffer, currentSize, dtype, device, this.bufferPool),
      keepOnGpu
    );
    onStats?.({ op: "pipeline", backend: "gpu", ms: performance.now() - t0 });
    return out;
  }
}
