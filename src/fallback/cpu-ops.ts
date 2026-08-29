import type { DataType, NumericArray, TypedArray, MatMulOpts } from "../core/types";
import { inferDataType } from "../core/types";
import { toTypedArray } from "../utils/data-conversion";

/** Allocate a zero-filled result array matching `dtype`. */
function resultArray(dtype: DataType, len: number): TypedArray {
  if (dtype === "i32") return new Int32Array(len);
  if (dtype === "u32") return new Uint32Array(len);
  return new Float32Array(len);
}

// The GPU expression language supports `Math.clamp` (WGSL `clamp`), which plain JS
// doesn't have — evaluate string expressions against a Math that includes it so the
// CPU fallback accepts everything the shader compiler does.
const MATH_WITH_CLAMP = Object.freeze(
  Object.assign(Object.create(Math), {
    clamp: (x: number, lo: number, hi: number) => Math.min(Math.max(x, lo), hi),
  })
);

/** Compile a string expression into a callable with `params` in scope. */
function compileExpr(params: string[], body: string): (...args: unknown[]) => number {
  const fn = new Function("Math", ...params, `return ${body}`);
  return (...args: unknown[]) => fn(MATH_WITH_CLAMP, ...args) as number;
}

// Elementwise binary over two flat CPU arrays with 1-D NumPy broadcasting: equal lengths pair
// up; otherwise one side must be length 1 and is broadcast against the other. The result takes
// the longer length and operand `a`'s dtype (matching the GPU path). The GPU path handles
// higher-rank broadcasting via reshaped GPUArrays; a bare CPU array is always 1-D here.
function broadcastBinary(
  a: NumericArray,
  b: NumericArray,
  op: (x: number, y: number) => number
): TypedArray {
  const dtype = inferDataType(a);
  const arrA = toTypedArray(a, dtype);
  const arrB = toTypedArray(b, dtype);
  if (arrA.length !== arrB.length && arrA.length !== 1 && arrB.length !== 1) {
    throw new Error(
      `Cannot broadcast lengths ${arrA.length} and ${arrB.length}: one operand must have length 1`
    );
  }
  const n = Math.max(arrA.length, arrB.length);
  const result = resultArray(dtype, n);
  for (let i = 0; i < n; i++) {
    result[i] = op(arrA[arrA.length === 1 ? 0 : i], arrB[arrB.length === 1 ? 0 : i]);
  }
  return result;
}

// Scalar fast path (b is a number); else elementwise with 1-D broadcasting.
function cpuBinary(
  a: NumericArray,
  b: NumericArray | number,
  op: (x: number, y: number) => number
): TypedArray {
  if (typeof b === "number") {
    const dtype = inferDataType(a);
    const arrA = toTypedArray(a, dtype);
    const result = resultArray(dtype, arrA.length);
    for (let i = 0; i < arrA.length; i++) result[i] = op(arrA[i], b);
    return result;
  }
  return broadcastBinary(a, b, op);
}

export function cpuAdd(a: NumericArray, b: NumericArray | number): TypedArray {
  return cpuBinary(a, b, (x, y) => x + y);
}

export function cpuSubtract(a: NumericArray, b: NumericArray | number): TypedArray {
  return cpuBinary(a, b, (x, y) => x - y);
}

export function cpuMultiply(a: NumericArray, b: NumericArray | number): TypedArray {
  return cpuBinary(a, b, (x, y) => x * y);
}

export function cpuDivide(a: NumericArray, b: NumericArray | number): TypedArray {
  return cpuBinary(a, b, (x, y) => x / y);
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
      ? (compileExpr(["x", "i", "len", ...constNames], fn) as (
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
  const zipFn =
    typeof fn === "string"
      ? (compileExpr(["a", "b"], fn) as (a: number, b: number) => number)
      : fn;
  // Same 1-D broadcasting as the elementwise ops: equal lengths pair up; otherwise one side
  // must be length 1 and is broadcast.
  return broadcastBinary(a, b, zipFn);
}

export function cpuReduce(
  input: NumericArray,
  fn: ((a: number, b: number) => number) | string,
  identity: number
): number {
  const arr = toTypedArray(input, inferDataType(input));
  const reduceFn =
    typeof fn === "string"
      ? (compileExpr(["a", "b"], fn) as (a: number, b: number) => number)
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

// Per-query binary search for the insertion index into an ascending `sorted` array.
// left (lower_bound) counts elements < q; right (upper_bound) counts elements <= q.
// `sorted` is assumed ascending; results are undefined otherwise. Output is always u32.
export function cpuSearchsorted(
  sorted: NumericArray,
  queries: NumericArray,
  side: "left" | "right" = "left"
): Uint32Array {
  const dtype = inferDataType(sorted);
  const s = toTypedArray(sorted, dtype);
  const q = toTypedArray(queries, dtype);
  const out = new Uint32Array(q.length);
  for (let i = 0; i < q.length; i++) {
    let lo = 0;
    let hi = s.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      const goRight = side === "right" ? s[mid] <= q[i] : s[mid] < q[i];
      if (goRight) lo = mid + 1;
      else hi = mid;
    }
    out[i] = lo;
  }
  return out;
}

// TypedArray assignment performs the numeric conversion (matches WGSL for in-range values).
export function cpuCast(input: NumericArray, toDtype: DataType): TypedArray {
  const arr = toTypedArray(input, inferDataType(input));
  const out = resultArray(toDtype, arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = arr[i];
  return out;
}

// out[c*rows+r] = input[r*cols+c]: transpose a row-major rows×cols array into cols×rows.
// A CPU array carries no shape, so { rows, cols } is required.
export function cpuTranspose(input: NumericArray, rows?: number, cols?: number): TypedArray {
  if (rows == null || cols == null) throw new Error("transpose of a CPU array requires { rows, cols }");
  const dtype = inferDataType(input);
  const arr = toTypedArray(input, dtype);
  const out = resultArray(dtype, rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out[c * rows + r] = arr[r * cols + c];
    }
  }
  return out;
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
      ? (compileExpr(["a", "b"], fn) as (a: number, b: number) => number)
      : fn;
  const result = resultArray(dtype, arr.length);
  let acc = identity;
  for (let i = 0; i < arr.length; i++) {
    acc = scanFn(acc, arr[i]);
    result[i] = acc;
  }
  return result;
}

// Keep elements where predicate(x, i, len) is truthy, preserving order. Output dtype follows
// input; length is the number of kept elements. Mirrors the GPU stream compaction.
export function cpuFilter(
  input: NumericArray,
  predicate: ((x: number, i: number, len: number) => boolean) | string
): TypedArray {
  const dtype = inferDataType(input);
  const arr = toTypedArray(input, dtype);
  const pred =
    typeof predicate === "string"
      ? (compileExpr(["x", "i", "len"], predicate) as (x: number, i: number, len: number) => unknown)
      : predicate;
  const kept: number[] = [];
  for (let i = 0; i < arr.length; i++) {
    if (pred(arr[i], i, arr.length)) kept.push(arr[i]);
  }
  const out = resultArray(dtype, kept.length);
  out.set(kept);
  return out;
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

export { cpuUnique } from "../ops/unique";
export { cpuSegmentedReduce } from "../ops/segmented-reduce";
export { cpuRandom } from "../ops/random";
export { cpuFft } from "../ops/fft";
export { cpuConvolve } from "../ops/convolve";
