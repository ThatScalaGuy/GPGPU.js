const MAX_POOL_SIZE = 32;

function nextPowerOf2(n: number): number {
  if (n <= 0) return 16;
  let p = 16; // minimum buffer size
  while (p < n) p *= 2;
  return p;
}

function poolKey(size: number, usage: number): string {
  return `${size}:${usage}`;
}

export class BufferPool {
  private pools = new Map<string, GPUBuffer[]>();

  acquire(device: GPUDevice, size: number, usage: number): GPUBuffer {
    const bucketSize = nextPowerOf2(size);
    const key = poolKey(bucketSize, usage);
    const pool = this.pools.get(key);

    if (pool && pool.length > 0) {
      return pool.pop()!;
    }

    return device.createBuffer({
      size: bucketSize,
      usage,
    });
  }

  release(buffer: GPUBuffer): void {
    if (buffer.mapState !== "unmapped") return;

    const key = poolKey(buffer.size, buffer.usage);
    let pool = this.pools.get(key);
    if (!pool) {
      pool = [];
      this.pools.set(key, pool);
    }

    if (pool.length < MAX_POOL_SIZE) {
      pool.push(buffer);
    } else {
      buffer.destroy();
    }
  }

  destroy(): void {
    for (const pool of this.pools.values()) {
      for (const buffer of pool) {
        buffer.destroy();
      }
    }
    this.pools.clear();
  }
}
