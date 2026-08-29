import type { DataType } from "./types";
import { BufferPool } from "./buffer-pool";

/** Wrap a copied ArrayBuffer in the typed-array view that matches `dtype`. */
export function viewFor(
  dtype: DataType,
  buffer: ArrayBuffer
): Float32Array | Int32Array | Uint32Array {
  if (dtype === "i32") return new Int32Array(buffer);
  if (dtype === "u32") return new Uint32Array(buffer);
  return new Float32Array(buffer);
}

/**
 * Run a GPU op inside WebGPU error scopes. Validation and out-of-memory errors
 * don't throw JS exceptions on their own — without a scope the op would return
 * whatever bytes were left in its (pooled) output buffer while the error goes to
 * the console at best. Capturing them turns a failed dispatch into a rejected
 * promise, so the CPU-fallback machinery can actually fire.
 *
 * Scopes are a per-device stack: this is correct for sequentially awaited ops
 * (the normal pattern); ops raced with `Promise.all` may attribute an error to
 * the wrong op, which still surfaces the failure and stays safe.
 */
export async function withErrorScope<T>(
  device: GPUDevice,
  fn: () => Promise<T>
): Promise<T> {
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  let popped = false;
  try {
    const result = await fn();
    popped = true;
    const validation = await device.popErrorScope();
    const oom = await device.popErrorScope();
    if (validation) throw new Error(`WebGPU validation error: ${validation.message}`);
    if (oom) throw new Error(`WebGPU out-of-memory error: ${oom.message}`);
    return result;
  } catch (e) {
    if (!popped) {
      // fn threw before the scopes were popped — rebalance the stack.
      await device.popErrorScope().catch(() => {});
      await device.popErrorScope().catch(() => {});
    }
    throw e;
  }
}

export function dispatchOnly(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroupCount: [number, number?, number?]
): void {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(...workgroupCount);
  pass.end();
  device.queue.submit([encoder.finish()]);
}

export function uploadBuffer(
  device: GPUDevice,
  data: ArrayBufferView,
  usage: number,
  bufferPool: BufferPool
): GPUBuffer {
  const buffer = bufferPool.acquire(
    device,
    data.byteLength,
    usage | GPUBufferUsage.COPY_DST
  );
  device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  return buffer;
}

export function createOutputBuffer(
  device: GPUDevice,
  size: number,
  bufferPool: BufferPool
): GPUBuffer {
  return bufferPool.acquire(
    device,
    size,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
}
