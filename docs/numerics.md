# Numerics: floating-point reduction order

GPGPU.js runs reductions (`sum`, `reduce`, `scan`) and the index reductions
(`argmin`, `argmax`) as a **parallel tree** across thousands of GPU threads. A
parallel tree adds the elements in a different order than a sequential CPU loop,
and floating-point addition is **not associative**: `(a + b) + c` and
`a + (b + c)` can round to different results.

Every individual IEEE-754 operation is still deterministic and correct. What
changes is the *grouping* of operations. So a GPU `sum` / `scan` / `reduce` can
differ from a sequential CPU result **in the last bits**, and the size of the
difference grows with the array length and the dynamic range of the values.

This is not a bug and it is not unique to this library — it is inherent to any
reassociated (parallel, blocked, or SIMD) floating-point reduction.

## What this means for you

- Do not assert exact equality between a GPU reduction and a sequential CPU
  reference. Compare with a tolerance.
- The CPU fallback runs a sequential loop, so switching backends (GPU vs CPU
  fallback) can itself shift the last bits of a float reduction.
- Integer reductions (`i32` / `u32`) are exact regardless of order, as long as
  the true result fits the type — integer addition *is* associative.

## Ties in `argmin` / `argmax`

`argmin` / `argmax` break ties toward the **first occurrence** (matching NumPy).
Because the values feeding the tie-break are themselves floats, a reassociated
reduction does not change *which* element is the extremum, but be aware that two
values that are "equal" only after rounding may compare differently than a CPU
reference that accumulated them in a different order.

## Tests

The test suite compares GPU float results against a CPU reference with an
absolute tolerance via the `expectClose` helper
(`tests/shared/tolerance.ts`) rather than strict equality. Tests over integer
inputs and over small, exactly-representable values still assert exact equality.
