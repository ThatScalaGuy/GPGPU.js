export type DataType = "f32" | "i32" | "u32";

export type NumericArray = Float32Array | Int32Array | Uint32Array | number[];

/** A concrete typed array — what ops return, matching the input's data type. */
export type TypedArray = Float32Array | Int32Array | Uint32Array;

export interface BufferSpec {
  type: DataType;
  size: number;
}

export interface KernelConfig {
  workgroupSize?: number;
  shader: string;
  inputs: BufferSpec[];
  output: BufferSpec;
}

export interface MatMulOpts {
  rowsA: number;
  colsA: number;
  colsB: number;
}

/** Which backend actually executed an op. */
export type Backend = "gpu" | "cpu";

/** Per-op telemetry reported to the `onStats` hook. */
export interface OpStats {
  op: string;
  backend: Backend;
  ms: number;
}

/** Passed to the `onFallback` hook when a GPU op fails. */
export interface FallbackInfo {
  op: string;
  error: unknown;
}

/**
 * What happens when a GPU op fails and a CPU implementation exists:
 * `"warn"` (default) logs and falls back, `"silent"` falls back quietly,
 * `"throw"` re-throws instead of falling back.
 */
export type FallbackMode = "warn" | "silent" | "throw";

/** Instance-level configuration for a {@link GPU}. */
export interface GPUOptions {
  fallback?: FallbackMode;
  onFallback?: (info: FallbackInfo) => void;
  onStats?: (stats: OpStats) => void;
}

export interface DispatchOpts {
  device: GPUDevice;
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  workgroupCount: [number, number?, number?];
  outputBuffer: GPUBuffer;
  outputSize: number;
}

export function toFloat32Array(input: NumericArray): Float32Array {
  if (input instanceof Float32Array) return input;
  return new Float32Array(input);
}

export function toInt32Array(input: NumericArray): Int32Array {
  if (input instanceof Int32Array) return input;
  return new Int32Array(input);
}

export function toUint32Array(input: NumericArray): Uint32Array {
  if (input instanceof Uint32Array) return input;
  return new Uint32Array(input);
}

export function inferDataType(input: NumericArray): DataType {
  if (input instanceof Int32Array) return "i32";
  if (input instanceof Uint32Array) return "u32";
  return "f32"; // Float32Array and number[]
}

export const DEFAULT_WORKGROUP_SIZE = 64;
export const REDUCE_WORKGROUP_SIZE = 256;
export const MATMUL_TILE_SIZE = 8;
