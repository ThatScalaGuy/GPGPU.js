import type { DataType } from "../core/types";
import { DEFAULT_WORKGROUP_SIZE, REDUCE_WORKGROUP_SIZE, MATMUL_TILE_SIZE } from "../core/types";
import { formatLiteral } from "./wgsl-emitter";

export function mapShader(
  expression: string,
  elemType: DataType = "f32",
  workgroupSize = DEFAULT_WORKGROUP_SIZE
): string {
  return `
@group(0) @binding(0) var<storage, read> input: array<${elemType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${elemType}>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= arrayLength(&input)) { return; }
  let x = input[idx];
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
