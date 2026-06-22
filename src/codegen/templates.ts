import type { DataType } from "../core/types";
import { DEFAULT_WORKGROUP_SIZE, REDUCE_WORKGROUP_SIZE, MATMUL_TILE_SIZE, TRANSPOSE_TILE_SIZE } from "../core/types";
import { formatLiteral } from "./wgsl-emitter";

export function mapShader(
  expression: string,
  elemType: DataType = "f32",
  consts: { name: string; dtype: DataType }[] = [],
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  // One read-only storage binding per captured const array, after input/output.
  const constBindings = consts
    .map((c, i) => `@group(0) @binding(${i + 2}) var<storage, read> consts_${c.name}: array<${c.dtype}>;\n`)
    .join("");
  return `
@group(0) @binding(0) var<storage, read> input: array<${elemType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${elemType}>;
${constBindings}
@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= arrayLength(&input)) { return; }
  let x = input[idx];
  output[idx] = ${expression};
}
`;
}

export function zipShader(
  expression: string,
  elemType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  return `
@group(0) @binding(0) var<storage, read> a_in: array<${elemType}>;
@group(0) @binding(1) var<storage, read> b_in: array<${elemType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${elemType}>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= arrayLength(&a_in)) { return; }
  let a = a_in[idx];
  let b = b_in[idx];
  output[idx] = ${expression};
}
`;
}

export function elementwiseBinaryShader(
  op: string,
  elemType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  return `
@group(0) @binding(0) var<storage, read> a: array<${elemType}>;
@group(0) @binding(1) var<storage, read> b: array<${elemType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${elemType}>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= arrayLength(&a)) { return; }
  output[idx] = a[idx] ${op} b[idx];
}
`;
}

// output[k] = src[idx[k]]; one thread per idx element. An out-of-range index is clamped
// to the last src element — the GPU can't throw, so this avoids an out-of-bounds read.
export function gatherShader(
  elemType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  return `
@group(0) @binding(0) var<storage, read> src: array<${elemType}>;
@group(0) @binding(1) var<storage, read> idx_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<${elemType}>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= arrayLength(&idx_buf)) { return; }
  output[k] = src[min(idx_buf[k], arrayLength(&src) - 1u)];
}
`;
}

// Per-query binary search for the insertion index into an ascending `sorted` array.
// left (lower_bound): count of elements strictly < q. right (upper_bound): count of elements <= q.
export function searchsortedShader(
  elemType: DataType = "f32",
  side: "left" | "right" = "left",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  const cmp = side === "right" ? "<=" : "<";
  return `
@group(0) @binding(0) var<storage, read> sorted: array<${elemType}>;
@group(0) @binding(1) var<storage, read> queries: array<${elemType}>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&queries)) { return; }
  let q = queries[i];
  var lo = 0u;
  var hi = arrayLength(&sorted);
  loop {
    if (lo >= hi) { break; }
    let mid = lo + (hi - lo) / 2u;
    if (sorted[mid] ${cmp} q) {
      lo = mid + 1u;
    } else {
      hi = mid;
    }
  }
  out[i] = lo;
}
`;
}

// out[idx[i]] = vals[i]; one thread per idx element. Out-of-range indices clamp to the
// last element (the GPU can't throw). DUPLICATE indices race — last write wins,
// nondeterministically. Documented behaviour; use mode:"add" for deterministic accumulation.
export function scatterSetShader(
  elemType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  return `
@group(0) @binding(0) var<storage, read_write> out: array<${elemType}>;
@group(0) @binding(1) var<storage, read> idx_buf: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<${elemType}>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&idx_buf)) { return; }
  out[min(idx_buf[i], arrayLength(&out) - 1u)] = vals[i];
}
`;
}

// out[idx[i]] += vals[i], atomically (collision-safe for duplicate indices).
// WGSL atomics cover only i32/u32 → integers use atomicAdd. f32 has no native atomic add,
// so reinterpret the bits and CAS-loop with atomicCompareExchangeWeak (portable, core WGSL).
export function scatterAddShader(
  elemType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  if (elemType === "f32") {
    return `
@group(0) @binding(0) var<storage, read_write> out: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> idx_buf: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&idx_buf)) { return; }
  let t = min(idx_buf[i], arrayLength(&out) - 1u);
  let v = vals[i];
  var old = atomicLoad(&out[t]);
  loop {
    let res = atomicCompareExchangeWeak(&out[t], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}
`;
  }
  return `
@group(0) @binding(0) var<storage, read_write> out: array<atomic<${elemType}>>;
@group(0) @binding(1) var<storage, read> idx_buf: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<${elemType}>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&idx_buf)) { return; }
  let t = min(idx_buf[i], arrayLength(&out) - 1u);
  atomicAdd(&out[t], vals[i]);
}
`;
}

// Counts input values into `bins` equal-width buckets over [min, max], using atomic<u32>
// accumulators (collision-safe). Out-of-range values clamp to the edge bins. Input is cast
// to f32 for the bucket math, so i32/u32 inputs work too. Output is always u32 counts.
export function histogramShader(
  elemType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  return `
struct Params { bins: u32, minVal: f32, maxVal: f32 }

@group(0) @binding(0) var<storage, read> input: array<${elemType}>;
@group(0) @binding(1) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&input)) { return; }
  let x = f32(input[i]);
  let range = params.maxVal - params.minVal;
  var b = 0u;
  if (range > 0.0) {
    let f = (x - params.minVal) / range * f32(params.bins);
    if (f >= 0.0) {
      b = min(u32(floor(f)), params.bins - 1u);
    }
  }
  atomicAdd(&hist[b], 1u);
}
`;
}

export function scalarBroadcastShader(
  op: string,
  elemType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  return `
struct Params { scalar: ${elemType} }

@group(0) @binding(0) var<storage, read> input: array<${elemType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${elemType}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= arrayLength(&input)) { return; }
  output[idx] = input[idx] ${op} params.scalar;
}
`;
}

export function reduceShader(
  reduceExpression: string,
  identity: string,
  elemType: DataType = "f32",
  workgroupSize = REDUCE_WORKGROUP_SIZE
): string {
  return `
@group(0) @binding(0) var<storage, read> input: array<${elemType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${elemType}>;

var<workgroup> sdata: array<${elemType}, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u
) {
  let idx = gid.x;
  sdata[lid.x] = select(${identity}, input[idx], idx < arrayLength(&input));

  workgroupBarrier();

  for (var stride = ${workgroupSize}u / 2u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      let a = sdata[lid.x];
      let b = sdata[lid.x + stride];
      sdata[lid.x] = ${reduceExpression};
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    output[wid.x] = sdata[0];
  }
}
`;
}

// Block-wise reduction that carries (value, index) pairs through twin buffers so the
// surviving index of the min/max can be read back. Mirrors reduceShader's ping-pong
// loop: each pass reads inVals/inIdxs and writes the per-workgroup winner to
// outVals/outIdxs. On the first pass the carried index is the global index (params.firstPass);
// later passes read the index forwarded by the previous pass. `better` flips with the mode.
// Tie-break is first occurrence (NumPy): on equal values the smaller index wins, so the
// combine never replaces a pair with a later, equal one. The workgroup arrays are named
// sdata_* (NOT `shared`, a reserved WGSL keyword that would silently fail to compile).
export function argReduceShader(
  identity: string,
  better: ">" | "<",
  elemType: DataType = "f32",
  workgroupSize = REDUCE_WORKGROUP_SIZE
): string {
  return `
struct Params { firstPass: u32 }

@group(0) @binding(0) var<storage, read> inVals: array<${elemType}>;
@group(0) @binding(1) var<storage, read> inIdxs: array<u32>;
@group(0) @binding(2) var<storage, read_write> outVals: array<${elemType}>;
@group(0) @binding(3) var<storage, read_write> outIdxs: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> sdata_vals: array<${elemType}, ${workgroupSize}>;
var<workgroup> sdata_idxs: array<u32, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u
) {
  let idx = gid.x;
  let inBounds = idx < arrayLength(&inVals);
  sdata_vals[lid.x] = select(${identity}, inVals[idx], inBounds);
  sdata_idxs[lid.x] = select(0u, select(inIdxs[idx], idx, params.firstPass == 1u), inBounds);

  workgroupBarrier();

  for (var stride = ${workgroupSize}u / 2u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      let aVal = sdata_vals[lid.x];
      let aIdx = sdata_idxs[lid.x];
      let bVal = sdata_vals[lid.x + stride];
      let bIdx = sdata_idxs[lid.x + stride];
      // Keep pair b only when it is strictly better, or ties on value with a smaller index.
      if (bVal ${better} aVal || (bVal == aVal && bIdx < aIdx)) {
        sdata_vals[lid.x] = bVal;
        sdata_idxs[lid.x] = bIdx;
      }
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    outVals[wid.x] = sdata_vals[0];
    outIdxs[wid.x] = sdata_idxs[0];
  }
}
`;
}

// Per-block inclusive scan. Each workgroup scans its `workgroupSize` slice of `data`
// in place and writes the slice's total to `blockSums[workgroup_id]`. Padding lanes are
// initialised to `identity`, so lane `workgroupSize-1` always holds the true block total
// even for a partial final block. Block totals are stitched together by scanAddOffsetsShader.
export function blockScanShader(
  scanExpression: string,
  identity: string,
  elemType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  return `
struct Params { n: u32 }

@group(0) @binding(0) var<storage, read_write> data: array<${elemType}>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<${elemType}>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> sdata: array<${elemType}, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u
) {
  let idx = gid.x;
  sdata[lid.x] = select(${identity}, data[idx], idx < params.n);
  workgroupBarrier();

  // Hillis-Steele inclusive scan
  for (var offset = 1u; offset < ${workgroupSize}u; offset *= 2u) {
    var val = sdata[lid.x];
    if (lid.x >= offset) {
      let a = sdata[lid.x - offset];
      let b = val;
      val = ${scanExpression};
    }
    workgroupBarrier();
    sdata[lid.x] = val;
    workgroupBarrier();
  }

  if (idx < params.n) {
    data[idx] = sdata[lid.x];
  }
  if (lid.x == ${workgroupSize}u - 1u) {
    blockSums[wid.x] = sdata[lid.x];
  }
}
`;
}

// Adds each block's offset (the inclusive scan of all earlier blocks' totals) to every
// element of that block. `blockSums` must already be inclusively scanned; block 0 needs
// no offset. The offset is the left operand so non-commutative associative ops stay correct.
export function scanAddOffsetsShader(
  scanExpression: string,
  elemType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  return `
struct Params { n: u32 }

@group(0) @binding(0) var<storage, read_write> data: array<${elemType}>;
@group(0) @binding(1) var<storage, read> blockSums: array<${elemType}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(workgroup_id) wid: vec3u
) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  if (wid.x == 0u) { return; }
  let a = blockSums[wid.x - 1u];
  let b = data[idx];
  data[idx] = ${scanExpression};
}
`;
}

export function matmulShader(elemType: DataType = "f32", tileSize = MATMUL_TILE_SIZE): string {
  const zero = formatLiteral(0, elemType);
  return `
struct Dims {
  M: u32,
  K: u32,
  N: u32,
}

@group(0) @binding(0) var<storage, read> a: array<${elemType}>;
@group(0) @binding(1) var<storage, read> b: array<${elemType}>;
@group(0) @binding(2) var<storage, read_write> result: array<${elemType}>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tileA: array<array<${elemType}, ${tileSize}>, ${tileSize}>;
var<workgroup> tileB: array<array<${elemType}, ${tileSize}>, ${tileSize}>;

@compute @workgroup_size(${tileSize}, ${tileSize})
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u
) {
  let row = gid.y;
  let col = gid.x;
  let localRow = lid.y;
  let localCol = lid.x;

  var sum = ${zero};
  let numTiles = (dims.K + ${tileSize}u - 1u) / ${tileSize}u;

  for (var t = 0u; t < numTiles; t++) {
    let tiledCol = t * ${tileSize}u + localCol;
    let tiledRow = t * ${tileSize}u + localRow;

    if (row < dims.M && tiledCol < dims.K) {
      tileA[localRow][localCol] = a[row * dims.K + tiledCol];
    } else {
      tileA[localRow][localCol] = ${zero};
    }

    if (tiledRow < dims.K && col < dims.N) {
      tileB[localRow][localCol] = b[tiledRow * dims.N + col];
    } else {
      tileB[localRow][localCol] = ${zero};
    }

    workgroupBarrier();

    for (var k = 0u; k < ${tileSize}u; k++) {
      sum += tileA[localRow][k] * tileB[k][localCol];
    }

    workgroupBarrier();
  }

  if (row < dims.M && col < dims.N) {
    result[row * dims.N + col] = sum;
  }
}
`;
}

// output[idx] = toType(input[idx]); a map whose output dtype differs from its input dtype.
// WGSL value-conversion constructors (i32()/u32()/f32()) handle the numeric conversion;
// out-of-range / negative->unsigned results are implementation-defined (documented).
export function castShader(
  fromType: DataType,
  toType: DataType,
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  return `
@group(0) @binding(0) var<storage, read> input: array<${fromType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${toType}>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= arrayLength(&input)) { return; }
  output[idx] = ${toType}(input[idx]);
}
`;
}

// Tiled transpose. Input is row-major R×C (input[r*C+c]); output is row-major C×R
// (output[c*R+r] = input[r*C+c]). The shared tile is TILE+1 wide to avoid bank conflicts.
export function transposeShader(elemType: DataType = "f32", tileSize = TRANSPOSE_TILE_SIZE): string {
  return `
struct Dims { rows: u32, cols: u32 }

@group(0) @binding(0) var<storage, read> input: array<${elemType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${elemType}>;
@group(0) @binding(2) var<uniform> dims: Dims;

var<workgroup> tile: array<array<${elemType}, ${tileSize + 1}>, ${tileSize}>;

@compute @workgroup_size(${tileSize}, ${tileSize})
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u
) {
  let R = dims.rows;
  let C = dims.cols;

  // Read input[r][c] into tile[ly][lx] (coalesced over c = lid.x).
  let r = wid.y * ${tileSize}u + lid.y;
  let c = wid.x * ${tileSize}u + lid.x;
  if (r < R && c < C) {
    tile[lid.y][lid.x] = input[r * C + c];
  }
  workgroupBarrier();

  // Write output[c2][r2] = input[r2][c2], reading the tile transposed (coalesced over r2 = lid.x).
  let r2 = wid.y * ${tileSize}u + lid.x;
  let c2 = wid.x * ${tileSize}u + lid.y;
  if (r2 < R && c2 < C) {
    output[c2 * R + r2] = tile[lid.x][lid.y];
  }
}
`;
}

export function bitonicSortShader(
  elemType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  return `
struct Params {
  blockSize: u32,
  subBlockSize: u32,
  length: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<${elemType}>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let pairDistance = params.subBlockSize;
  let blockSize = params.blockSize;

  let leftIdx = (idx / pairDistance) * (pairDistance * 2u) + (idx % pairDistance);
  let rightIdx = leftIdx + pairDistance;

  if (rightIdx >= params.length) { return; }

  let sameDirection = ((leftIdx / blockSize) % 2u) == 0u;

  let leftVal = data[leftIdx];
  let rightVal = data[rightIdx];

  let shouldSwap = select((leftVal < rightVal), (leftVal > rightVal), sameDirection);

  if (shouldSwap) {
    data[leftIdx] = rightVal;
    data[rightIdx] = leftVal;
  }
}
`;
}

// Like bitonicSortShader but carries a values payload: keys drive the comparison, and when a
// pair is swapped the matching values are swapped too. keyType drives ordering; valType is
// independent (both 4 bytes).
export function bitonicSortByKeyShader(
  keyType: DataType = "f32",
  valType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  return `
struct Params {
  blockSize: u32,
  subBlockSize: u32,
  length: u32,
}

@group(0) @binding(0) var<storage, read_write> keys: array<${keyType}>;
@group(0) @binding(1) var<storage, read_write> values: array<${valType}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let pairDistance = params.subBlockSize;
  let blockSize = params.blockSize;

  let leftIdx = (idx / pairDistance) * (pairDistance * 2u) + (idx % pairDistance);
  let rightIdx = leftIdx + pairDistance;

  if (rightIdx >= params.length) { return; }

  let sameDirection = ((leftIdx / blockSize) % 2u) == 0u;

  let leftKey = keys[leftIdx];
  let rightKey = keys[rightIdx];

  let shouldSwap = select((leftKey < rightKey), (leftKey > rightKey), sameDirection);

  if (shouldSwap) {
    keys[leftIdx] = rightKey;
    keys[rightIdx] = leftKey;
    let tmp = values[leftIdx];
    values[leftIdx] = values[rightIdx];
    values[rightIdx] = tmp;
  }
}
`;
}

export function customKernelShader(
  shaderBody: string,
  inputTypes: DataType[],
  outputType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  let bindings = "";
  for (let i = 0; i < inputTypes.length; i++) {
    bindings += `@group(0) @binding(${i}) var<storage, read> input${i}: array<${inputTypes[i]}>;\n`;
  }
  bindings += `@group(0) @binding(${inputTypes.length}) var<storage, read_write> output: array<${outputType}>;\n`;

  return `
${bindings}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  ${shaderBody}
}
`;
}
