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
import { dispatchOnly, uploadBuffer } from "../core/command";
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

// Pass 0 variant for complex input: the input buffer already holds interleaved
// (re, im) pairs, i.e. an array<vec2<f32>> of n complex points, so the scatter
// moves whole pairs instead of promoting a real to (re, 0). Downstream stages
// are identical.
function fftBitReverseComplexShader(workgroupSize = WG): string {
  return `
struct Params { n: u32, logN: u32 }

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
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
  output[j] = input[i];
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

// ifft pass A: conjugate the spectrum into a fresh buffer (dst = conj(src)).
// Writing to a separate buffer keeps a caller's GPUArray input untouched. The
// bind-group entry sizes cap arrayLength at n even on an oversized pooled buffer.
function ifftConjShader(workgroupSize = WG): string {
  return `
@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&dst)) { return; }
  let v = src[i];
  dst[i] = vec2<f32>(v.x, -v.y);
}
`;
}

// ifft pass B: in-place conjugate + scale by 1/n, finishing the conjugate
// trick ifft(X) = conj(fft(conj(X))) / n.
function ifftConjScaleShader(workgroupSize = WG): string {
  return `
struct Params { n: u32, scale: f32 }

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let v = data[i];
  data[i] = vec2<f32>(v.x * params.scale, -v.y * params.scale);
}
`;
}

/** Options for `fft`. */
export interface FftOptions extends OpOptions {
  /**
   * Treat the input as an interleaved complex signal [re0, im0, re1, im1, ...]
   * of length 2*n (n complex points, n a power of two) instead of a real one.
   */
  complexInput?: boolean;
}

export function gpuFft(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts: FftOptions & { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuFft(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts?: FftOptions
): Promise<TypedArray>;
export async function gpuFft(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  input: OpInput,
  opts?: FftOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const keepOnGpu = opts?.keepOnGpu ?? false;
  const complexInput = opts?.complexInput ?? false;

  // FFT always produces an interleaved complex (vec2<f32>) spectrum, so the
  // result dtype is f32 regardless of the input dtype.
  const inDtype = inputDtype(input);

  // Resolve length + how to seed the input, without mutating a GPUArray.
  let rawLen: number;
  let srcArr: TypedArray | null = null;
  if (isGPUArray(input)) {
    if (input.isDestroyed) throw new Error("GPUArray has been destroyed");
    rawLen = input.length;
  } else {
    srcArr = toTypedArray(input, inDtype);
    rawLen = srcArr.length;
  }

  // `n` is the number of complex points of the transform. A real input has one
  // point per element; a complex input is interleaved (re, im) pairs, so its
  // raw length is 2*n.
  let n: number;
  if (complexInput) {
    if (rawLen % 2 !== 0 || !isPowerOfTwo(rawLen / 2)) {
      throw new Error(
        `fft: complex input is interleaved [re, im] pairs, so its length must be 2*n with n a power of two, got ${rawLen}`
      );
    }
    n = rawLen / 2;
  } else {
    if (!isPowerOfTwo(rawLen)) {
      throw new Error(`fft: input length must be a power of two, got ${rawLen}`);
    }
    n = rawLen;
  }

  const logN = log2(n);
  const complexBytes = n * 2 * 4; // n complex values, 2 f32 each

  const storageUsage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  // Input as f32 on the GPU (rawLen elements: real samples, or interleaved
  // pairs for complexInput). A CPU array is uploaded to a temp buffer; a
  // GPUArray (already f32 by contract) is read in place.
  let realBuf: GPUBuffer;
  let releaseReal = false;
  if (srcArr) {
    const f32 = srcArr instanceof Float32Array ? srcArr : new Float32Array(srcArr);
    realBuf = bufferPool.acquire(device, rawLen * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(realBuf, 0, f32.buffer as ArrayBuffer, f32.byteOffset, f32.byteLength);
    releaseReal = true;
  } else {
    realBuf = (input as GPUArray).buffer;
  }

  const bitRevPipe = await shaderCache.getOrCreate(
    device,
    complexInput ? fftBitReverseComplexShader() : fftBitReverseShader(),
    complexInput ? "fft-bitreverse-complex" : "fft-bitreverse"
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
        { binding: 0, resource: { buffer: realBuf, size: rawLen * 4 } },
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

export function gpuIfft(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  spectrum: OpInput,
  opts: { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuIfft(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  spectrum: OpInput,
  opts?: OpOptions
): Promise<TypedArray>;
/**
 * Inverse FFT of an interleaved complex spectrum [re0, im0, ...] of length 2*n
 * (n a power of two), returning the interleaved complex time signal of length
 * 2*n scaled by 1/n. Implemented as the conjugate trick over the forward
 * butterflies: ifft(X) = conj(fft(conj(X))) / n — a conjugate pass, the
 * complex-input forward FFT, and a fused conjugate+scale pass.
 */
export async function gpuIfft(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  spectrum: OpInput,
  opts?: OpOptions
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const keepOnGpu = opts?.keepOnGpu ?? false;

  let rawLen: number;
  let srcArr: TypedArray | null = null;
  if (isGPUArray(spectrum)) {
    if (spectrum.isDestroyed) throw new Error("GPUArray has been destroyed");
    rawLen = spectrum.length;
  } else {
    srcArr = toTypedArray(spectrum, "f32");
    rawLen = srcArr.length;
  }
  if (rawLen % 2 !== 0 || !isPowerOfTwo(rawLen / 2)) {
    throw new Error(
      `ifft: spectrum is interleaved [re, im] pairs, so its length must be 2*n with n a power of two, got ${rawLen}`
    );
  }
  const n = rawLen / 2;
  const complexBytes = n * 8;

  // Spectrum as f32 on the GPU: upload a CPU array; read a GPUArray in place.
  let specBuf: GPUBuffer;
  let releaseSpec = false;
  if (srcArr) {
    const f32 = srcArr instanceof Float32Array ? srcArr : new Float32Array(srcArr);
    specBuf = uploadBuffer(device, f32, GPUBufferUsage.STORAGE, bufferPool);
    releaseSpec = true;
  } else {
    specBuf = (spectrum as GPUArray).buffer;
  }

  // Pass A: conjugate into a buffer this op owns (the caller's input is only read).
  const conjPipe = await shaderCache.getOrCreate(device, ifftConjShader(), "ifft-conj");
  const conjBuf = bufferPool.acquire(
    device,
    complexBytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  );
  const conjGroup = device.createBindGroup({
    layout: conjPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: specBuf, size: complexBytes } },
      { binding: 1, resource: { buffer: conjBuf, size: complexBytes } },
    ],
  });
  dispatchOnly(device, conjPipe, conjGroup, [computeWorkgroupCount(n)]);
  if (releaseSpec) bufferPool.release(specBuf);

  // Forward FFT of the conjugated spectrum, staying on the GPU. The wrapper
  // GPUArray hands conjBuf to gpuFft without transferring ownership; destroy()
  // returns conjBuf to the pool once the forward transform is recorded.
  const conjArr = new GPUArray(conjBuf, rawLen, "f32", device, bufferPool);
  let result: GPUArray;
  try {
    result = await gpuFft(deviceManager, bufferPool, shaderCache, conjArr, {
      complexInput: true,
      keepOnGpu: true,
    });
  } finally {
    conjArr.destroy();
  }

  // Pass B: conjugate back + scale by 1/n, in place on the forward result.
  const scalePipe = await shaderCache.getOrCreate(
    device,
    ifftConjScaleShader(),
    "ifft-conj-scale"
  );
  const params = bufferPool.acquire(
    device,
    8,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  );
  const paramBytes = new ArrayBuffer(8);
  const paramView = new DataView(paramBytes);
  paramView.setUint32(0, n, true);
  paramView.setFloat32(4, 1 / n, true);
  device.queue.writeBuffer(params, 0, paramBytes);
  const scaleGroup = device.createBindGroup({
    layout: scalePipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: result.buffer, size: complexBytes } },
      { binding: 1, resource: { buffer: params } },
    ],
  });
  dispatchOnly(device, scalePipe, scaleGroup, [computeWorkgroupCount(n)]);
  bufferPool.release(params);

  return finalize(result, keepOnGpu);
}

/**
 * CPU reference: forward DFT of a real-valued signal (or, with
 * `{ complexInput: true }`, an interleaved complex signal of length 2*n),
 * returned as the interleaved complex spectrum [re0, im0, re1, im1, ...]
 * (Float32Array of length 2*n). Direct O(n^2) DFT — used only as the
 * correctness oracle for the GPU radix-2 FFT, so clarity beats speed. The
 * point count n must be a power of two to match the GPU op's contract.
 */
export function cpuFft(
  input: OpInput,
  opts?: { complexInput?: boolean }
): Float32Array {
  const src =
    input instanceof GPUArray
      ? (() => {
          throw new Error("cpuFft does not accept a GPUArray input");
        })()
      : toTypedArray(input, "f32");

  if (opts?.complexInput) {
    const rawLen = src.length;
    if (rawLen % 2 !== 0 || !isPowerOfTwo(rawLen / 2)) {
      throw new Error(
        `fft: complex input is interleaved [re, im] pairs, so its length must be 2*n with n a power of two, got ${rawLen}`
      );
    }
    const n = rawLen / 2;
    const out = new Float32Array(n * 2);
    for (let k = 0; k < n; k++) {
      let re = 0;
      let im = 0;
      for (let t = 0; t < n; t++) {
        const angle = (-2 * Math.PI * k * t) / n;
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        const xr = src[2 * t];
        const xi = src[2 * t + 1];
        re += xr * c - xi * s;
        im += xr * s + xi * c;
      }
      out[2 * k] = re;
      out[2 * k + 1] = im;
    }
    return out;
  }

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

/**
 * CPU reference: inverse DFT of an interleaved complex spectrum of length 2*n,
 * returning the interleaved complex time signal of length 2*n scaled by 1/n.
 * Direct O(n^2) sum with the positive twiddle sign — the correctness oracle
 * for gpuIfft.
 */
export function cpuIfft(spectrum: OpInput): Float32Array {
  const src =
    spectrum instanceof GPUArray
      ? (() => {
          throw new Error("cpuIfft does not accept a GPUArray input");
        })()
      : toTypedArray(spectrum, "f32");
  const rawLen = src.length;
  if (rawLen % 2 !== 0 || !isPowerOfTwo(rawLen / 2)) {
    throw new Error(
      `ifft: spectrum is interleaved [re, im] pairs, so its length must be 2*n with n a power of two, got ${rawLen}`
    );
  }
  const n = rawLen / 2;
  const out = new Float32Array(n * 2);
  for (let t = 0; t < n; t++) {
    let re = 0;
    let im = 0;
    for (let k = 0; k < n; k++) {
      const angle = (2 * Math.PI * k * t) / n;
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      const xr = src[2 * k];
      const xi = src[2 * k + 1];
      re += xr * c - xi * s;
      im += xr * s + xi * c;
    }
    out[2 * t] = re / n;
    out[2 * t + 1] = im / n;
  }
  return out;
}
