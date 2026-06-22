import type { DataType, TypedArray } from "../core/types";
import { BufferPool } from "../core/buffer-pool";
import { viewFor } from "../core/command";

export class GPUArray {
  readonly buffer: GPUBuffer;
  readonly length: number;
  readonly dtype: DataType;
  /** Optional logical shape (e.g. `[rows, cols]`). `undefined` = treat as 1-D by length. */
  readonly shape?: readonly number[];
  private device: GPUDevice;
  private pool: BufferPool;
  // Whether this handle owns its buffer. A non-owning view (e.g. from `reshape`) borrows
  // the source array's buffer and must NOT release it on destroy.
  private owns: boolean;
  private destroyed = false;

  constructor(
    buffer: GPUBuffer,
    length: number,
    dtype: DataType,
    device: GPUDevice,
    pool: BufferPool,
    opts?: { shape?: readonly number[]; owns?: boolean }
  ) {
    this.buffer = buffer;
    this.length = length;
    this.dtype = dtype;
    this.device = device;
    this.pool = pool;
    this.shape = opts?.shape;
    this.owns = opts?.owns ?? true;
  }

  get byteLength(): number {
    return this.length * 4;
  }

  get isDestroyed(): boolean {
    return this.destroyed;
  }

  async toArray(): Promise<TypedArray> {
    if (this.destroyed) throw new Error("GPUArray has been destroyed");

    const staging = this.pool.acquire(
      this.device,
      this.byteLength,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    );

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffer, 0, staging, 0, this.byteLength);
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const result = viewFor(this.dtype, staging.getMappedRange().slice(0));
    staging.unmap();
    this.pool.release(staging);

    return result.slice(0, this.length);
  }

  destroy(): void {
    if (!this.destroyed) {
      if (this.owns) this.pool.release(this.buffer);
      this.destroyed = true;
    }
  }
}
