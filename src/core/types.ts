export type DataType = "f32" | "i32" | "u32";

export type NumericArray = Float32Array | Int32Array | Uint32Array | number[];

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

export const DEFAULT_WORKGROUP_SIZE = 64;
export const REDUCE_WORKGROUP_SIZE = 256;
export const MATMUL_TILE_SIZE = 8;
