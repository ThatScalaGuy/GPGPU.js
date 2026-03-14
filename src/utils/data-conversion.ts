import type { DataType, NumericArray } from "../core/types";

export function toTypedArray(
  input: NumericArray,
  dtype: DataType = "f32"
): Float32Array | Int32Array | Uint32Array {
  switch (dtype) {
    case "f32":
      return input instanceof Float32Array ? input : new Float32Array(input);
    case "i32":
      return input instanceof Int32Array ? input : new Int32Array(input);
    case "u32":
      return input instanceof Uint32Array ? input : new Uint32Array(input);
  }
}
