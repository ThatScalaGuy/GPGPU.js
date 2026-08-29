# Benchmarks

GPU (WebGPU) vs plain JavaScript on the same machine, end-to-end wall clock
**including upload and readback** — the cost a one-shot call actually has.
Median of 5 runs after 2 warmups (device init, shader compile, pool warm).

```bash
npm run bench            # build + all ops
node bench/run.mjs sort matmul   # a subset (after npm run build)
```

Runs on Node via Google Dawn (the optional `webgpu` package); in CI-less
environments without a GPU, Mesa lavapipe works too (see `.github/workflows/ci.yml`).

## Results

Apple Silicon (Metal via Dawn), Node 22 — 2026-08:

| op | n | GPU (ms) | plain JS (ms) | speedup |
|---|---:|---:|---:|---:|
| map (sqrt(x)*2+1) | 100,000 | 0.54 | 0.13 | 0.2× |
| map (GPU-resident, no transfer) | 100,000 | 0.17 | 0.13 | 0.8× |
| map (sqrt(x)*2+1) | 1,000,000 | 1.31 | 1.16 | 0.9× |
| map (GPU-resident, no transfer) | 1,000,000 | 0.09 | 1.16 | 12.7× |
| map (sqrt(x)*2+1) | 4,000,000 | 6.07 | 5.95 | 1.0× |
| map (GPU-resident, no transfer) | 4,000,000 | 0.03 | 5.95 | 177.5× |
| sum | 100,000 | 0.44 | 0.10 | 0.2× |
| sum | 1,000,000 | 1.28 | 1.00 | 0.8× |
| sum | 4,000,000 | 3.89 | 3.84 | 1.0× |
| scan (prefix sum) | 100,000 | 0.61 | 0.13 | 0.2× |
| scan (prefix sum) | 1,000,000 | 1.58 | 1.17 | 0.7× |
| scan (prefix sum) | 4,000,000 | 6.18 | 5.00 | 0.8× |
| sort | 100,000 | 8.49 | 36.3 | 4.3× |
| sort | 1,000,000 | 11.8 | 382 | 32.2× |
| histogram (64 bins) | 100,000 | 0.41 | 0.21 | 0.5× |
| histogram (64 bins) | 1,000,000 | 2.00 | 2.03 | 1.0× |
| histogram (64 bins) | 4,000,000 | 5.63 | 7.96 | 1.4× |
| matmul 256x256 | 65,536 | 0.58 | 27.6 | 47.5× |
| matmul 512x512 | 262,144 | 1.99 | 218 | 109.5× |
| matmul 1024x1024 | 1,048,576 | 5.61 | 1742 | 310.7× |
| fft | 65,536 | 0.93 | — | — |
| fft | 1,048,576 | 4.04 | — | — |

## What the numbers mean

The table is honest about the transfer cost, and it tells a clear story:

- **One-shot elementwise ops (map/sum/scan/histogram) are transfer-bound.** The
  kernel itself is nearly free; upload + readback dominate, so a single call
  lands around 1× against plain JS — and *below* 1× for small arrays. If you
  call one elementwise op on CPU data and read it straight back, the GPU is not
  the win.
- **Residency is where elementwise wins.** The same map on a `GPUArray` with
  `keepOnGpu` (no transfers) is 12× at 1M and >100× at 4M. That is the entire
  point of `gpu.upload()` / `keepOnGpu` / `pipeline()`: pay the transfer once,
  chain the work.
- **Compute-heavy ops win outright**, transfers included: sort 32× at 1M,
  matmul up to ~300×.
- **Small arrays (≲100k) belong on the CPU** for simple ops. These numbers are
  the groundwork for a `minGpuSize` heuristic (issue #11).

Numbers vary by machine — run `npm run bench` on yours.
