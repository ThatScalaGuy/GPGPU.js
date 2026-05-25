import type { DataType, NumericArray, TypedArray, MatMulOpts } from "../core/types";
import { inferDataType } from "../core/types";
import { toTypedArray } from "../utils/data-conversion";

/** Allocate a zero-filled result array matching `dtype`. */
function resultArray(dtype: DataType, len: number): TypedArray {
  if (dtype === "i32") return new Int32Array(len);
  if (dtype === "u32") return new Uint32Array(len);
  return new Float32Array(len);
}

export function cpuAdd(
  a: NumericArray,
  b: NumericArray | number
): TypedArray {
  const dtype = inferDataType(a);
  const arrA = toTypedArray(a, dtype);
  const result = resultArray(dtype, arrA.length);
  if (typeof b === "number") {
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] + b;
  } else {
    const arrB = toTypedArray(b, dtype);
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] + arrB[i];
  }
  return result;
}

export function cpuSubtract(
  a: NumericArray,
  b: NumericArray | number
): TypedArray {
  const dtype = inferDataType(a);
  const arrA = toTypedArray(a, dtype);
  const result = resultArray(dtype, arrA.length);
  if (typeof b === "number") {
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] - b;
  } else {
    const arrB = toTypedArray(b, dtype);
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] - arrB[i];
  }
  return result;
}

export function cpuMultiply(
  a: NumericArray,
  b: NumericArray | number
): TypedArray {
  const dtype = inferDataType(a);
  const arrA = toTypedArray(a, dtype);
  const result = resultArray(dtype, arrA.length);
  if (typeof b === "number") {
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] * b;
  } else {
    const arrB = toTypedArray(b, dtype);
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] * arrB[i];
  }
  return result;
}

export function cpuDivide(
  a: NumericArray,
  b: NumericArray | number
): TypedArray {
  const dtype = inferDataType(a);
  const arrA = toTypedArray(a, dtype);
  const result = resultArray(dtype, arrA.length);
  if (typeof b === "number") {
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] / b;
  } else {
    const arrB = toTypedArray(b, dtype);
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] / arrB[i];
  }
  return result;
}

export function cpuMap(
  input: NumericArray,
  fn: ((x: number) => number) | string
): TypedArray {
  const dtype = inferDataType(input);
  const arr = toTypedArray(input, dtype);
  const result = resultArray(dtype, arr.length);
  const mapFn =
    typeof fn === "string" ? new Function("x", `return ${fn}`) as (x: number) => number : fn;
  for (let i = 0; i < arr.length; i++) {
    result[i] = mapFn(arr[i]);
  }
  return result;
}

export function cpuReduce(
  input: NumericArray,
  fn: ((a: number, b: number) => number) | string,
  identity: number
): number {
  const arr = toTypedArray(input, inferDataType(input));
  const reduceFn =
    typeof fn === "string"
      ? (new Function("a", "b", `return ${fn}`) as (a: number, b: number) => number)
      : fn;
  let acc = identity;
  for (let i = 0; i < arr.length; i++) {
    acc = reduceFn(acc, arr[i]);
  }
  return acc;
}

export function cpuSum(input: NumericArray): number {
  const arr = toTypedArray(input, inferDataType(input));
  let sum = 0;
  for (let i = 0; i < arr.length; i++) sum += arr[i];
  return sum;
}

export function cpuMin(input: NumericArray): number {
  const arr = toTypedArray(input, inferDataType(input));
  let min = Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] < min) min = arr[i];
  return min;
}

export function cpuMax(input: NumericArray): number {
  const arr = toTypedArray(input, inferDataType(input));
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
  return max;
}

export function cpuProduct(input: NumericArray): number {
  const arr = toTypedArray(input, inferDataType(input));
  let prod = 1;
  for (let i = 0; i < arr.length; i++) prod *= arr[i];
  return prod;
}

export function cpuMatmul(
  a: NumericArray,
  b: NumericArray,
  opts: MatMulOpts
): TypedArray {
  const { rowsA, colsA, colsB } = opts;
  const dtype = inferDataType(a);
  const arrA = toTypedArray(a, dtype);
  const arrB = toTypedArray(b, dtype);
  const result = resultArray(dtype, rowsA * colsB);

  for (let row = 0; row < rowsA; row++) {
    for (let col = 0; col < colsB; col++) {
      let sum = 0;
      for (let k = 0; k < colsA; k++) {
        sum += arrA[row * colsA + k] * arrB[k * colsB + col];
      }
      result[row * colsB + col] = sum;
    }
  }
  return result;
}

export function cpuScan(
  input: NumericArray,
  fn: ((a: number, b: number) => number) | string,
  identity: number
): TypedArray {
  const dtype = inferDataType(input);
  const arr = toTypedArray(input, dtype);
  const scanFn =
    typeof fn === "string"
      ? (new Function("a", "b", `return ${fn}`) as (a: number, b: number) => number)
      : fn;
  const result = resultArray(dtype, arr.length);
  let acc = identity;
  for (let i = 0; i < arr.length; i++) {
    acc = scanFn(acc, arr[i]);
    result[i] = acc;
  }
  return result;
}

export function cpuSort(input: NumericArray): TypedArray {
  const dtype = inferDataType(input);
  const result = toTypedArray(input, dtype).slice();
  // Numeric ascending order (TypedArray.sort defaults to numeric, but be explicit
  // to match the GPU bitonic sort for all dtypes).
  result.sort((a, b) => a - b);
  return result;
}
