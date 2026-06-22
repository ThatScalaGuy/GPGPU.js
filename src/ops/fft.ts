import type { TypedArray } from "../core/types";
import { DEFAULT_WORKGROUP_SIZE } from "../core/types";
import { toTypedArray } from "../utils/data-conversion";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { GPUArray } from "../pipeline/gpu-array";
import {
  type OpInput,
  type OpOptions,
  inputDtype,
  isGPUArray,
  finalize,
} from "../core/io";
import { computeWorkgroupCount } from "../utils/workgroup";

const WG = DEFAULT_WORKGROUP_SIZE;

function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

function log2(n: number): number {
  return Math.round(Math.log2(n));
}

// ---------------------------------------------------------------------------
// WGSL templates (co-located so this op file is self-contained).
//
// A complex number is a vec2<f32> = (real, imag) — WGSL has no native complex
// or f64. The spectrum buffer holds `n` such vec2 values, laid out as the
// interleaved f32 array [re0, im0, re1, im1, ...] that the op returns.
// ---------------------------------------------------------------------------

// Pass 0: scatter the real input into a complex buffer at its bit-reversed
// position, with imaginary part 0. `params.logN` is the number of butterfly
// stages, i.e. the bit width of the reversal. This is the standard
// decimation-in-time pre-permutation that lets the later passes be in-place
// butterflies over contiguous half-blocks.
function fftBitReverseShader(workgroupSize = WG): string {
  return `
struct Params { n: u32, logN: u32 }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn reverseBits(v: u32, bits: u32) -> u32 {
  var x = v;
  var r = 0u;
  for (var i = 0u; i < bits; i = i + 1u) {
    r = (r << 1u) | (x & 1u);
    x = x >> 1u;
  }
  return r;
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let j = reverseBits(i, params.logN);
  output[j] = vec2<f32>(input[i], 0.0);
}
`;
}

// One radix-2 Cooley-Tukey butterfly stage (decimation in time). There are
// n/2 butterflies; thread `t` owns one. `half = 1 << (stage-1)` is the current
// sub-transform half-size; a butterfly pairs element `i0` with `i0 + half`
// within its size-`2*half` block. The twiddle factor is
// exp(-2*pi*i * k / (2*half)) with k the position inside the half-block.
// Reads `src`, writes `dst` (the host ping-pongs the two buffers per stage).
function fftButterflyShader(workgroupSize = WG): string {
  return `
struct Params { halfN: u32, half: u32 }

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= params.halfN) { return; }

  let half = params.half;
  let k = t % half;                 // position within the half-block
  let block = t / half;             // which size-(2*half) block
  let i0 = block * (2u * half) + k;  // even/lower index
  let i1 = i0 + half;                // odd/upper index

  let angle = -PI * f32(k) / f32(half);
  let w = vec2<f32>(cos(angle), sin(angle));

  let a = src[i0];
  let b = src[i1];
  // complex multiply w * b
  let wb = vec2<f32>(w.x * b.x - w.y * b.y, w.x * b.y + w.y * b.x);

  dst[i0] = a + wb;
  dst[i1] = a - wb;
}
`;
}

export function gpuFft(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts: { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuFft(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts?: OpOptions
): Promise<TypedArray>;
export async function gpuFft(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts?: OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const keepOnGpu = opts?.keepOnGpu ?? false;

  // FFT always produces an interleaved complex (vec2<f32>) spectrum, so the
  // result dtype is f32 regardless of the (real) input dtype.
  const inDtype = inputDtype(input);

  // Resolve length + how to seed the real input, without mutating a GPUArray.
  let n: number;
  let srcArr: TypedArray | null = null;
  if (isGPUArray(input)) {
    if (input.isDestroyed) throw new Error("GPUArray has been destroyed");
    n = input.length;
  } else {
    srcArr = toTypedArray(input, inDtype);
    n = srcArr.length;
  }

  if (!isPowerOfTwo(n)) {
    throw new Error(`fft: input length must be a power of two, got ${n}`);
  }

  const logN = log2(n);
  const complexBytes = n * 2 * 4; // n complex values, 2 f32 each

  const storageUsage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  // Real input as f32 on the GPU. A CPU array is uploaded to a temp buffer; a
  // GPUArray (already f32 by contract) is read in place.
  let realBuf: GPUBuffer;
  let releaseReal = false;
  if (srcArr) {
    const f32 = srcArr instanceof Float32Array ? srcArr : new Float32Array(srcArr);
    realBuf = bufferPool.acquire(device, n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(realBuf, 0, f32.buffer as ArrayBuffer, f32.byteOffset, f32.byteLength);
    releaseReal = true;
  } else {
    realBuf = (input as GPUArray).buffer;
  }

  const bitRevPipe = await shaderCache.getOrCreate(
    device,
    fftBitReverseShader(),
    "fft-bitreverse"
  );
  const butterflyPipe = await shaderCache.getOrCreate(
    device,
    fftButterflyShader(),
    "fft-butterfly"
  );

  // Two complex ping-pong buffers. `bufA` ends up holding the result after an
  // even number of butterfly stages; `cur`/`nxt` track which is live.
  const bufA = bufferPool.acquire(device, complexBytes, storageUsage);
  const bufB = bufferPool.acquire(device, complexBytes, storageUsage);

  const cleanup: GPUBuffer[] = [];
  if (releaseReal) cleanup.push(realBuf);
  cleanup.push(bufB);

  // Single encoder for every pass so the multi-stage transform never round-trips
  // through the CPU between stages (mirrors gpuScan).
  const encoder = device.createCommandEncoder();

  // Pass 0: bit-reversal permutation into bufA.
  {
    const params = bufferPool.acquire(
      device,
      8,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    device.queue.writeBuffer(params, 0, new Uint32Array([n, logN]));
    cleanup.push(params);

    const group = device.createBindGroup({
      layout: bitRevPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: realBuf, size: n * 4 } },
        { binding: 1, resource: { buffer: bufA, size: complexBytes } },
        { binding: 2, resource: { buffer: params } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(bitRevPipe);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(computeWorkgroupCount(n));
    pass.end();
  }

  // Butterfly stages 1..logN, ping-ponging bufA <-> bufB. Each stage launches
  // n/2 butterfly threads.
  const halfN = n / 2;
  let cur = bufA;
  let nxt = bufB;
  for (let stage = 1; stage <= logN; stage++) {
    const half = 1 << (stage - 1);
    const params = bufferPool.acquire(
      device,
      8,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    device.queue.writeBuffer(params, 0, new Uint32Array([halfN, half]));
    cleanup.push(params);

    const group = device.createBindGroup({
      layout: butterflyPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cur, size: complexBytes } },
        { binding: 1, resource: { buffer: nxt, size: complexBytes } },
        { binding: 2, resource: { buffer: params } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(butterflyPipe);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(computeWorkgroupCount(halfN));
    pass.end();

    const tmp = cur;
    cur = nxt;
    nxt = tmp;
  }

  device.queue.submit([encoder.finish()]);

  // `cur` holds the spectrum. Make sure the buffer we keep is bufA-or-bufB and
  // release the other (already queued in cleanup is only bufB; if the result is
  // in bufB, swap what we release).
  const resultBuf = cur;
  const otherBuf = cur === bufA ? bufB : bufA;
  // Replace the unconditionally-queued bufB with whichever buffer is NOT the result.
  const bIdx = cleanup.indexOf(bufB);
  if (bIdx !== -1) cleanup.splice(bIdx, 1);
  cleanup.push(otherBuf);

  for (const buf of cleanup) bufferPool.release(buf);

  // length is the interleaved-f32 length (2*n).
  return finalize(
    new GPUArray(resultBuf, n * 2, "f32", device, bufferPool),
    keepOnGpu
  );
}

/**
 * CPU reference: forward DFT of a real-valued signal, returned as the
 * interleaved complex spectrum [re0, im0, re1, im1, ...] (Float32Array of
 * length 2*n). Direct O(n^2) DFT — used only as the correctness oracle for the
 * GPU radix-2 FFT, so clarity beats speed. Length must be a power of two to
 * match the GPU op's contract.
 */
export function cpuFft(input: OpInput): Float32Array {
  const src =
    input instanceof GPUArray
      ? (() => {
          throw new Error("cpuFft does not accept a GPUArray input");
        })()
      : toTypedArray(input, "f32");
  const n = src.length;
  if (!isPowerOfTwo(n)) {
    throw new Error(`fft: input length must be a power of two, got ${n}`);
  }

  const out = new Float32Array(n * 2);
  for (let k = 0; k < n; k++) {
    let re = 0;
    let im = 0;
    for (let t = 0; t < n; t++) {
      const angle = (-2 * Math.PI * k * t) / n;
      const x = src[t];
      re += x * Math.cos(angle);
      im += x * Math.sin(angle);
    }
    out[2 * k] = re;
    out[2 * k + 1] = im;
  }
  return out;
}
