import type { NumericArray, MatMulOpts } from "../core/types";
import { toFloat32Array } from "../core/types";

export function cpuAdd(
  a: NumericArray,
  b: NumericArray | number
): Float32Array {
  const arrA = toFloat32Array(a);
  const result = new Float32Array(arrA.length);
  if (typeof b === "number") {
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] + b;
  } else {
    const arrB = toFloat32Array(b);
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] + arrB[i];
  }
  return result;
}

export function cpuSubtract(
  a: NumericArray,
  b: NumericArray | number
): Float32Array {
  const arrA = toFloat32Array(a);
  const result = new Float32Array(arrA.length);
  if (typeof b === "number") {
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] - b;
  } else {
    const arrB = toFloat32Array(b);
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] - arrB[i];
  }
  return result;
}

export function cpuMultiply(
  a: NumericArray,
  b: NumericArray | number
): Float32Array {
  const arrA = toFloat32Array(a);
  const result = new Float32Array(arrA.length);
  if (typeof b === "number") {
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] * b;
  } else {
    const arrB = toFloat32Array(b);
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] * arrB[i];
  }
  return result;
}

export function cpuDivide(
  a: NumericArray,
  b: NumericArray | number
): Float32Array {
  const arrA = toFloat32Array(a);
  const result = new Float32Array(arrA.length);
  if (typeof b === "number") {
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] / b;
  } else {
    const arrB = toFloat32Array(b);
    for (let i = 0; i < arrA.length; i++) result[i] = arrA[i] / arrB[i];
  }
  return result;
}

export function cpuMap(
  input: NumericArray,
  fn: ((x: number) => number) | string
): Float32Array {
  const arr = toFloat32Array(input);
  const result = new Float32Array(arr.length);
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
  const arr = toFloat32Array(input);
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
  const arr = toFloat32Array(input);
  let sum = 0;
  for (let i = 0; i < arr.length; i++) sum += arr[i];
  return sum;
}

export function cpuMin(input: NumericArray): number {
  const arr = toFloat32Array(input);
  let min = Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] < min) min = arr[i];
  return min;
}

export function cpuMax(input: NumericArray): number {
  const arr = toFloat32Array(input);
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
  return max;
}

export function cpuProduct(input: NumericArray): number {
  const arr = toFloat32Array(input);
  let prod = 1;
  for (let i = 0; i < arr.length; i++) prod *= arr[i];
  return prod;
}

export function cpuMatmul(
  a: NumericArray,
  b: NumericArray,
  opts: MatMulOpts
): Float32Array {
  const { rowsA, colsA, colsB } = opts;
  const arrA = toFloat32Array(a);
  const arrB = toFloat32Array(b);
  const result = new Float32Array(rowsA * colsB);

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
): Float32Array {
  const arr = toFloat32Array(input);
  const scanFn =
    typeof fn === "string"
      ? (new Function("a", "b", `return ${fn}`) as (a: number, b: number) => number)
      : fn;
  const result = new Float32Array(arr.length);
  let acc = identity;
  for (let i = 0; i < arr.length; i++) {
    acc = scanFn(acc, arr[i]);
    result[i] = acc;
  }
  return result;
}

export function cpuSort(input: NumericArray): Float32Array {
  const arr = toFloat32Array(input);
  const result = new Float32Array(arr);
  result.sort();
  return result;
}
