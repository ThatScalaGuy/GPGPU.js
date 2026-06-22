# Random: a reproducible, counter-based generator

`gpu.random(n, opts?)` fills a fresh array of `n` pseudo-random values entirely
on the GPU. Unlike every other op it takes **no input array** — it is a
*generator*. It is the GPU counterpart to `numpy.random`, built for Monte-Carlo,
array initialisation, and sampling.

```javascript
await gpu.random(5);                          // Float32Array, 5 uniforms in [0, 1)
await gpu.random(5, { seed: 42 });            // a different, fixed stream
await gpu.random(5, { dtype: "u32" });        // raw 32-bit integers
await gpu.random(1_000_000, { keepOnGpu: true }); // stays resident for chaining
```

## Counter-based and reproducible

The generator is **stateless and counter-based**: each output is a pure hash of
`(seed, index)`. There is no running state to carry between elements, so every
value can be computed independently — which is exactly why it parallelises
perfectly (one GPU thread per index) and why it is fully reproducible.

The same `(n, seed, dtype)` always yields the **same** values, on every run and
on every device. The mixing is wrapping 32-bit integer arithmetic (the Murmur3
finalizer), which WebGPU and the JavaScript CPU reference compute identically, so
the GPU result and the `cpuRandom` fallback agree **bit-for-bit** — there is no
floating-point reassociation caveat here (unlike reductions; see
[numerics.md](./numerics.md)).

`seed` defaults to `0`. Pick a fixed seed for reproducible runs; vary it (e.g.
per experiment) for independent streams. Different seeds produce statistically
independent sequences.

## Output dtype

`dtype` (default `"f32"`) selects what the bits become:

| `dtype` | Output |
|---|---|
| `"f32"` | Uniform in `[0, 1)` — the top 24 bits of the hash divided by `2^24`. |
| `"u32"` | The raw 32-bit generator output, spanning the full `u32` range. |
| `"i32"` | Those same bits reinterpreted as signed, spanning the full `i32` range. |

The `f32` value is an integer `≤ 2^24` times a power-of-two reciprocal, so it is
exactly representable — the float you get back is the same on the GPU and the
CPU. The half-open interval `[0, 1)` means `0` can occur but `1` never does.

With `keepOnGpu: true` the result is a `GPUArray` of the requested `dtype` and
`length === n`, ready to feed into another op without a round-trip to the CPU.

## Generating other distributions

`random` is the uniform primitive; build the rest on top of it with `map`:

```javascript
// Uniform in [lo, hi):
const u = await gpu.random(n, { keepOnGpu: true });
const scaled = await gpu.map(u, `x * ${hi - lo} + ${lo}`);

// A coin flip / Bernoulli(p) as 0/1:
await gpu.map(await gpu.random(n, { keepOnGpu: true }), `select(0.0, 1.0, x < ${p})`);
```

For very large `n`, generating with `keepOnGpu: true` and consuming the
`GPUArray` directly avoids reading a big buffer back to the CPU.

## Empty input

`gpu.random(0)` returns an empty array of the requested dtype.
