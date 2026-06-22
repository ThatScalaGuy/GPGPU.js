// NumPy-style broadcasting for the elementwise binary ops (add/sub/mul/div/zip).
// Only reached when the two operands have DIFFERENT flat lengths; equal-length inputs
// take the untouched fast path. Shapes come from a reshaped GPUArray; an input without a
// shape is treated as 1-D `[length]` (matching how NumPy reads a flat vector).

// Max logical rank the broadcast shader unrolls. 4 covers vectors, matrices, and 3-D/4-D
// tensors — well beyond the reshape ergonomics the library exposes today.
export const MAX_BROADCAST_RANK = 4;

/**
 * NumPy broadcast of two shapes: align from the trailing dim; each dim pair must be equal or
 * one must be 1; the result dim is their max. Throws if incompatible. Returns the result shape.
 */
export function broadcastShapes(
  shapeA: readonly number[],
  shapeB: readonly number[]
): number[] {
  const rank = Math.max(shapeA.length, shapeB.length);
  const out = new Array<number>(rank);
  for (let i = 0; i < rank; i++) {
    // Right-align: missing leading dims act as 1.
    const da = shapeA[shapeA.length - rank + i] ?? 1;
    const db = shapeB[shapeB.length - rank + i] ?? 1;
    if (da === db || da === 1 || db === 1) {
      out[i] = Math.max(da, db);
    } else {
      throw new Error(
        `Cannot broadcast shapes [${shapeA}] and [${shapeB}]: dimension ${i} mismatch (${da} vs ${db})`
      );
    }
  }
  return out;
}

/** Row-major strides for a shape: stride[k] = product of all dims to the right of k. */
function rowMajorStrides(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = acc;
    acc *= shape[i];
  }
  return strides;
}

/**
 * Per-input strides for indexing into `shape` while iterating over `outShape`, right-aligned
 * and padded to MAX_BROADCAST_RANK. A dim of size 1 that broadcasts gets stride 0, so every
 * output coordinate along it reads the same (single) input element.
 */
export function broadcastStrides(
  shape: readonly number[],
  outShape: readonly number[]
): number[] {
  const outRank = outShape.length;
  const base = rowMajorStrides(shape);
  const padded = new Array<number>(MAX_BROADCAST_RANK).fill(0);
  for (let i = 0; i < outRank; i++) {
    const dim = shape[shape.length - outRank + i];
    // Leading dims this input doesn't have, or a size-1 dim, contribute stride 0 (broadcast).
    if (dim === undefined || dim === 1) continue;
    padded[MAX_BROADCAST_RANK - outRank + i] = base[shape.length - outRank + i];
  }
  return padded;
}

/** Output shape padded to MAX_BROADCAST_RANK with leading 1s (so unused dims iterate once). */
export function paddedOutShape(outShape: readonly number[]): number[] {
  const padded = new Array<number>(MAX_BROADCAST_RANK).fill(1);
  for (let i = 0; i < outShape.length; i++) {
    padded[MAX_BROADCAST_RANK - outShape.length + i] = outShape[i];
  }
  return padded;
}
