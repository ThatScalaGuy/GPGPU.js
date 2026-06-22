# Broadcasting

The element-wise binary ops — `add`, `subtract`, `multiply`, `divide`, and the
custom-fn `zip` — apply **NumPy-style broadcasting** when their two operands have
different shapes. Operands of the same flat length take the plain element-wise
path unchanged; broadcasting only kicks in when the lengths differ.

## The rule

Shapes are aligned from the **trailing** dimension. For each aligned pair of
dimensions, one of the following must hold:

- the dimensions are equal, or
- one of them is `1` (it stretches to match the other), or
- one operand has no dimension there at all (a missing leading dimension acts as `1`).

The result shape is the element-wise maximum of the two. Anything else throws.

```
[2, 3]  +  [3]      →  [2, 3]   # row vector added to every row
[2, 3]  +  [2, 1]   →  [2, 3]   # column vector added to every column
[4]     +  [1]      →  [4]      # length-1 array broadcasts like a scalar
[2, 3]  +  [2]      →  error    # 3 vs 2, neither is 1
```

A stretched dimension is read with **stride 0** — every output coordinate along
that axis reads the same single input element, so no data is copied or expanded in
memory. One GPU thread runs per **output** element.

## Where shapes come from

A flat array carries no shape. The logical shape is the one a
[`reshape`](./shape-ops.md#reshape-a-zero-copy-non-owning-view) attaches to a
`GPUArray`:

```javascript
const mat = await gpu.reshape([1, 2, 3, 4, 5, 6], [2, 3]); // shape [2, 3]
const row = await gpu.reshape([10, 20, 30], [3]);          // shape [3]

const out = await gpu.add(mat, row, { keepOnGpu: true });  // shape [2, 3]
await out.toArray();  // [11, 22, 33, 14, 25, 36]
```

An operand **without** a shape (a plain CPU array, or a `GPUArray` straight from
`upload`) is treated as 1-D of its length. So a length-1 operand broadcasts like a
scalar even without a reshape:

```javascript
const mat = await gpu.reshape(new Int32Array([1, 2, 3, 4]), [2, 2]);
const scalar = await gpu.upload(new Int32Array([5]));     // no shape → read as [1]
await gpu.multiply(mat, scalar, { keepOnGpu: true });     // [5, 10, 15, 20]
```

Broadcasting is **symmetric** — `gpu.add(row, mat)` gives the same `[2, 3]` result
as `gpu.add(mat, row)`. The result `GPUArray` carries the broadcast result shape.

## CPU fallback

The CPU reference ops (`cpuAdd`/`cpuSubtract`/`cpuMultiply`/`cpuDivide`/`cpuZip`)
broadcast too, but a bare CPU array is always 1-D, so the CPU side only sees the
1-D case: equal lengths pair up, otherwise **one operand must have length 1** and
is broadcast against the other. Higher-rank broadcasting (`[m,n]` against `[n]` or
`[m,1]`) requires the shape a `reshape` puts on a `GPUArray`, so it runs on the
GPU path. Mismatched lengths where neither side is 1 throw.

## Numerics

Broadcasting changes only *which* element each thread reads — the per-element
arithmetic is the ordinary element-wise op. There is no reduction and no
reassociation, so results are exact for integers and match the CPU fallback for
floats just as the non-broadcast element-wise ops do (see
[docs/numerics.md](./numerics.md)).

## Limits

The broadcast kernel unrolls up to **4 logical dimensions**, which covers vectors,
matrices, and 3-D/4-D tensors — well beyond the 2-D shapes `reshape`/`transpose`
expose today.
