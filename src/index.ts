export { GPU } from "./gpu";
export { Pipeline } from "./pipeline/pipeline";
export { GPUArray } from "./pipeline/gpu-array";
export type {
  DataType,
  NumericArray,
  TypedArray,
  KernelConfig,
  MatMulOpts,
  ScatterOpts,
  SearchSortedOpts,
  HistogramOpts,
  TransposeOpts,
  BufferSpec,
  Backend,
  OpStats,
  FallbackInfo,
  FallbackMode,
  GPUOptions,
} from "./core/types";
export type { SegmentedReduceOpts, SegmentedReduceOp } from "./ops/segmented-reduce";
export type { RandomOpts } from "./ops/random";
export type { ConvolveOpts, ConvolveMode } from "./ops/convolve";
export type { FftOptions } from "./ops/fft";
export type { SortOptions } from "./ops/sort";
export type { SortByKeyOptions } from "./ops/sort-by-key";
export type { TopKOptions } from "./ops/topk";
export type { ConstructorOpts } from "./ops/constructors";

import { GPU } from "./gpu";

/** Default GPU singleton for convenience */
const gpu = new GPU();
export { gpu };
export default gpu;
