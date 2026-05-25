import type { NumericArray, TypedArray } from "../core/types";
import { inferDataType } from "../core/types";
import { toTypedArray } from "../utils/data-conversion";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { uploadBuffer, createOutputBuffer, viewFor } from "../core/command";
import { computeWorkgroupCount } from "../utils/workgroup";
import { parseExpression } from "../codegen/expression-parser";
import { emitWGSL, formatLiteral } from "../codegen/wgsl-emitter";
import { mapShader, reduceShader } from "../codegen/templates";
import { REDUCE_WORKGROUP_SIZE } from "../core/types";

interface MapStep {
  type: "map";
  fn: ((x: number) => number) | string;
}

interface ReduceStep {
  type: "reduce";
  fn: ((a: number, b: number) => number) | string;
  identity: number;
}

type PipelineStep = MapStep | ReduceStep;

export class Pipeline {
  private steps: PipelineStep[] = [];
  private deviceManager: DeviceManager;
  private bufferPool: BufferPool;
  private shaderCache: ShaderCache;

  constructor(
    deviceManager: DeviceManager,
    bufferPool: BufferPool,
    shaderCache: ShaderCache
  ) {
    this.deviceManager = deviceManager;
    this.bufferPool = bufferPool;
    this.shaderCache = shaderCache;
  }

  map(fn: ((x: number) => number) | string): Pipeline {
    this.steps.push({ type: "map", fn });
    return this;
  }

  reduce(
    fn: ((a: number, b: number) => number) | string,
    identity: number
  ): Pipeline {
    this.steps.push({ type: "reduce", fn, identity });
    return this;
  }

  async run(input: NumericArray): Promise<TypedArray | number> {
    const device = await this.deviceManager.getDevice();
    const dtype = inferDataType(input);
    const arr = toTypedArray(input, dtype);
    let currentSize = arr.length;

    // Upload initial data
    let currentBuffer = uploadBuffer(
      device, arr, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, this.bufferPool
    );
    let currentByteSize = currentSize * 4;
    const buffersToRelease: GPUBuffer[] = [currentBuffer];

    let lastStepIsReduce = false;

    for (const step of this.steps) {
      if (step.type === "map") {
        lastStepIsReduce = false;
        const ir = parseExpression(step.fn, ["x"]);
        const expression = emitWGSL(ir, dtype);
        const shader = mapShader(expression, dtype);
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

        const workgroupCount = computeWorkgroupCount(currentSize);
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(workgroupCount);
        pass.end();
        device.queue.submit([encoder.finish()]);

        currentBuffer = outBuffer;
      } else if (step.type === "reduce") {
        lastStepIsReduce = true;
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

    // Read back final result
    const finalByteSize = currentSize * 4;
    const staging = this.bufferPool.acquire(
      device,
      finalByteSize,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    );

    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(currentBuffer, 0, staging, 0, finalByteSize);
    device.queue.submit([copyEncoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const result = viewFor(dtype, staging.getMappedRange().slice(0));
    staging.unmap();
    this.bufferPool.release(staging);

    // Release all intermediate buffers
    for (const buf of buffersToRelease) {
      this.bufferPool.release(buf);
    }

    if (lastStepIsReduce) {
      return result[0];
    }
    return result.slice(0, currentSize);
  }
}
