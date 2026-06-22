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
  fn: ((x: number, i: number, len: number) => number) | string,
  consts?: Record<string, NumericArray>
): TypedArray {
  const dtype = inferDataType(input);
  const arr = toTypedArray(input, dtype);
  const result = resultArray(dtype, arr.length);
  const constNames = consts ? Object.keys(consts) : [];
  const constValues = constNames.map((name) => consts![name]);
  const mapFn =
    typeof fn === "string"
      ? (new Function("x", "i", "len", ...constNames, `return ${fn}`) as (
          x: number,
          i: number,
          len: number,
          ...rest: NumericArray[]
        ) => number)
      : fn;
  for (let i = 0; i < arr.length; i++) {
    result[i] = mapFn(arr[i], i, arr.length, ...constValues);
  }
  return result;
}

export function cpuZip(
  a: NumericArray,
  b: NumericArray,
  fn: ((a: number, b: number) => number) | string
): TypedArray {
  const dtype = inferDataType(a);
  const arrA = toTypedArray(a, dtype);
  const arrB = toTypedArray(b, dtype);
  const result = resultArray(dtype, arrA.length);
  const zipFn =
    typeof fn === "string"
      ? (new Function("a", "b", `return ${fn}`) as (a: number, b: number) => number)
      : fn;
  for (let i = 0; i < arrA.length; i++) {
    result[i] = zipFn(arrA[i], arrB[i]);
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

// Strict `<` so the FIRST minimum wins on ties — matches the GPU first-occurrence tie-break.
export function cpuArgmin(input: NumericArray): number {
  const arr = toTypedArray(input, inferDataType(input));
  if (arr.length === 0) return -1;
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] < arr[best]) best = i;
  return best;
}

// Strict `>` so the FIRST maximum wins on ties — matches the GPU first-occurrence tie-break.
export function cpuArgmax(input: NumericArray): number {
  const arr = toTypedArray(input, inferDataType(input));
  if (arr.length === 0) return -1;
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

// output[k] = src[idx[k]]; an out-of-range index clamps to the last element, matching the
// GPU gather shader (which can't throw). Output dtype follows src; idx is read as u32.
export function cpuGather(src: NumericArray, idx: NumericArray): TypedArray {
  const dtype = inferDataType(src);
  const arrSrc = toTypedArray(src, dtype);
  const arrIdx = toTypedArray(idx, "u32");
  const result = resultArray(dtype, arrIdx.length);
  const last = arrSrc.length - 1;
  for (let k = 0; k < arrIdx.length; k++) {
    result[k] = arrSrc[Math.min(arrIdx[k], last)];
  }
  return result;
}

// out = copy of dst with vals scattered at idx. mode "set" overwrites (last write wins on
// duplicate idx); "add" accumulates. Out-of-range idx clamps to the last element, matching
// the GPU shader. Integer wrap on overflow matches the GPU atomics (TypedArray truncation).
export function cpuScatter(
  dst: NumericArray,
  idx: NumericArray,
  vals: NumericArray,
  mode: "set" | "add" = "set"
): TypedArray {
  const dtype = inferDataType(dst);
  const out = toTypedArray(dst, dtype).slice() as TypedArray;
  const arrIdx = toTypedArray(idx, "u32");
  const arrVals = toTypedArray(vals, dtype);
  const last = out.length - 1;
  for (let i = 0; i < arrIdx.length; i++) {
    const t = Math.min(arrIdx[i], last);
    if (mode === "add") out[t] += arrVals[i];
    else out[t] = arrVals[i];
  }
  return out;
}

// Mirror of the GPU histogram: equal-width bins over [min,max], out-of-range clamps to the
// edge bins, max==min puts everything in bin 0. Returns u32 counts.
export function cpuHistogram(
  input: NumericArray,
  bins: number,
  min: number,
  max: number
): Uint32Array {
  const arr = toTypedArray(input, inferDataType(input));
  const out = new Uint32Array(bins);
  const range = max - min;
  for (let i = 0; i < arr.length; i++) {
    let b = 0;
    if (range > 0) {
      const f = ((arr[i] - min) / range) * bins;
      if (f >= 0) b = Math.min(Math.floor(f), bins - 1);
    }
    out[b]++;
  }
  return out;
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

// Sort keys ascending and permute values to match. NOT stable for equal keys (matches the
// GPU bitonic sort, which is also unstable). values dtype is independent of keys dtype.
export function cpuSortByKey(
  keys: NumericArray,
  values: NumericArray
): [TypedArray, TypedArray] {
  const keyDtype = inferDataType(keys);
  const valDtype = inferDataType(values);
  const k = toTypedArray(keys, keyDtype);
  const v = toTypedArray(values, valDtype);
  const order = Array.from({ length: k.length }, (_, i) => i);
  order.sort((a, b) => k[a] - k[b]);
  const outK = resultArray(keyDtype, k.length);
  const outV = resultArray(valDtype, v.length);
  for (let i = 0; i < order.length; i++) {
    outK[i] = k[order[i]];
    outV[i] = v[order[i]];
  }
  return [outK, outV];
}
