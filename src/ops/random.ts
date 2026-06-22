import type { DataType, TypedArray } from "../core/types";
import { DEFAULT_WORKGROUP_SIZE } from "../core/types";
import { DeviceManager } from "../core/device";
import { BufferPool } from "../core/buffer-pool";
import { ShaderCache } from "../core/shader-cache";
import { createOutputBuffer, uploadBuffer, dispatchOnly } from "../core/command";
import { GPUArray } from "../pipeline/gpu-array";
import { finalize } from "../core/io";
import { computeWorkgroupCount } from "../utils/workgroup";

/**
 * Options for {@link gpuRandom}. `random` is a generator (no input array): it
 * produces `n` deterministic pseudo-random values from `seed`.
 *
 * - `seed` (default `0`) selects the stream. The same `(n, seed, dtype)` always
 *   yields the exact same values — on the GPU and on the CPU reference alike.
 * - `dtype` (default `"f32"`) picks the output:
 *     - `"f32"` → uniform in `[0, 1)`;
 *     - `"u32"` → the raw 32-bit generator output (full `u32` range);
 *     - `"i32"` → those same bits reinterpreted as signed (full `i32` range).
 */
export interface RandomOpts {
  seed?: number;
  dtype?: DataType;
  keepOnGpu?: boolean;
}

const WG = DEFAULT_WORKGROUP_SIZE;

// --- Counter-based PRNG ----------------------------------------------------
// A stateless, counter-based generator: each output is a pure hash of (seed,
// index), so any element can be produced independently — perfect for the GPU
// (one thread per index, no shared state) and reproducible across runs.
//
// The mixer is the Murmur3 32-bit finalizer (`fmix32`), a well-tested avalanche
// function, applied twice with the seed folded in. Every step is wrapping u32
// arithmetic, which WGSL (`u32` ops wrap mod 2^32, `>>` is logical) and the JS
// reference (`Math.imul` + `>>> 0`, `>>>`) implement identically — so the GPU
// and CPU outputs match bit-for-bit, integers and floats alike.
//
// f32 output: the top 24 bits divided by 2^24. That is an integer ≤ 2^24 times a
// power-of-two reciprocal, exactly representable in f32, so the float round-trips
// identically on both backends (no reduction-order caveat here).

const MUL_A = 0x85ebca6b;
const MUL_B = 0xc2b2ae35;
const GOLDEN = 0x9e3779b9;
// 1 / 2^24 — scales a 24-bit integer into [0, 1).
const F32_RECIP = "0.000000059604644775390625";

function randomShader(dtype: DataType, workgroupSize = WG): string {
  // Reinterpret / scale the raw u32 hash into the requested output dtype.
  const convert =
    dtype === "f32"
      ? `out[i] = f32(h >> 8u) * ${F32_RECIP};`
      : dtype === "i32"
        ? `out[i] = bitcast<i32>(h);`
        : `out[i] = h;`;
  return `
struct Params { n: u32, seed: u32 }

@group(0) @binding(0) var<storage, read_write> out: array<${dtype}>;
@group(0) @binding(1) var<uniform> params: Params;

fn fmix32(x: u32) -> u32 {
  var h = x;
  h ^= h >> 16u;
  h = h * ${MUL_A}u;
  h ^= h >> 13u;
  h = h * ${MUL_B}u;
  h ^= h >> 16u;
  return h;
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let s = params.seed;
  var k = i ^ ${GOLDEN}u;
  k = k + s * ${MUL_A}u;
  var h = fmix32(k);
  h = fmix32((h + s * ${MUL_B}u) ^ i);
  ${convert}
}
`;
}

// --- CPU reference (bit-for-bit identical to the WGSL above) ---------------

function mul(a: number, b: number): number {
  return Math.imul(a, b) >>> 0;
}

function fmix32(x: number): number {
  let h = x >>> 0;
  h ^= h >>> 16;
  h = mul(h, MUL_A);
  h ^= h >>> 13;
  h = mul(h, MUL_B);
  h ^= h >>> 16;
  return h >>> 0;
}

function rawHash(seed: number, i: number): number {
  const s = seed >>> 0;
  let k = (i >>> 0) ^ GOLDEN;
  k = (k + mul(s, MUL_A)) >>> 0;
  let h = fmix32(k);
  h = fmix32(((h + mul(s, MUL_B)) >>> 0) ^ (i >>> 0));
  return h >>> 0;
}

/**
 * CPU reference for {@link gpuRandom}: the same counter-based generator, in
 * wrapping u32 arithmetic, so it reproduces the GPU output exactly.
 */
export function cpuRandom(n: number, opts?: RandomOpts): TypedArray {
  const seed = (opts?.seed ?? 0) >>> 0;
  const dtype = opts?.dtype ?? "f32";
  if (dtype === "u32") {
    const out = new Uint32Array(n);
    for (let i = 0; i < n; i++) out[i] = rawHash(seed, i);
    return out;
  }
  if (dtype === "i32") {
    const out = new Int32Array(n);
    for (let i = 0; i < n; i++) out[i] = rawHash(seed, i) | 0;
    return out;
  }
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (rawHash(seed, i) >>> 8) * (1 / 16777216);
  return out;
}

// --- GPU op ----------------------------------------------------------------

export function gpuRandom(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  n: number,
  opts: RandomOpts & { keepOnGpu: true }
): Promise<GPUArray>;
export function gpuRandom(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  n: number,
  opts?: RandomOpts
): Promise<TypedArray>;
export async function gpuRandom(
  deviceManager: DeviceManager,
  bufferPool: BufferPool,
  shaderCache: ShaderCache,
  n: number,
  opts?: RandomOpts
): Promise<TypedArray | GPUArray> {
  const device = await deviceManager.getDevice();
  const seed = (opts?.seed ?? 0) >>> 0;
  const dtype = opts?.dtype ?? "f32";
  const keepOnGpu = opts?.keepOnGpu ?? false;

  if (n === 0) {
    const empty = bufferPool.acquire(device, 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    return finalize(new GPUArray(empty, 0, dtype, device, bufferPool), keepOnGpu);
  }

  const pipeline = await shaderCache.getOrCreate(device, randomShader(dtype), `random-${dtype}`);

  const outBuf = createOutputBuffer(device, n * 4, bufferPool);

  // Uniform { n: u32, seed: u32 }.
  const params = new Uint32Array([n, seed]);
  const bufParams = uploadBuffer(device, params, GPUBufferUsage.UNIFORM, bufferPool);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: outBuf, size: n * 4 } },
      { binding: 1, resource: { buffer: bufParams, size: 8 } },
    ],
  });

  dispatchOnly(device, pipeline, bindGroup, [computeWorkgroupCount(n)]);

  bufferPool.release(bufParams);

  return finalize(new GPUArray(outBuf, n, dtype, device, bufferPool), keepOnGpu);
}
