export { GPU } from "./gpu";
export { Pipeline } from "./pipeline/pipeline";
export { GPUArray } from "./pipeline/gpu-array";
export type {
  DataType,
  NumericArray,
  KernelConfig,
  MatMulOpts,
  BufferSpec,
} from "./core/types";

import { GPU } from "./gpu";

/** Default GPU singleton for convenience */
const gpu = new GPU();
export { gpu };
export default gpu;
