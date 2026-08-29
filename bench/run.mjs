// GPGPU.js benchmark harness — GPU (WebGPU) vs plain JS on the same machine.
//
//   npm run bench                 # all ops, default sizes
//   node bench/run.mjs sum sort   # only these ops (after npm run build)
//
// Methodology: 2 warmup runs (device init, shader compile, buffer-pool warm),
// then the median of 5 timed runs. GPU timings are end-to-end wall clock
// INCLUDING upload and readback — the number a one-shot call actually costs.
// The "GPU-resident" map row skips both transfers (the chained-ops case).
// Results print as a markdown table on stdout.

import { writeSync } from "node:fs";
import { create, globals } from "webgpu";
import { GPU } from "../dist/index.js";

// Root a Dawn instance up front (before the library's lazy device init). This
// also works around a crash in the Node auto-detect path where the Dawn GPU
// instance could be garbage-collected mid-use on large allocations.
Object.assign(globalThis, globals);
Object.defineProperty(globalThis, "navigator", {
  value: { gpu: create([]) },
  configurable: true,
  writable: true,
});

const gpu = new GPU({ fallback: "throw" }); // never silently benchmark the CPU

function fill(n, seed = 1) {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s >>> 8) / 16777216;
  }
  return out;
}

async function median(runs, fn) {
  const times = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    await fn();
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

async function bench(fnGpu, fnCpu) {
  await fnGpu();
  await fnGpu();
  const gpuMs = await median(5, fnGpu);
  if (!fnCpu) return { gpuMs, cpuMs: 0 };
  fnCpu(); // JIT warmup
  const t0 = performance.now();
  fnCpu();
  const cpuMs = performance.now() - t0;
  return { gpuMs, cpuMs };
}

const fmt = (ms) => (ms >= 100 ? ms.toFixed(0) : ms >= 10 ? ms.toFixed(1) : ms.toFixed(2));

const SIZES = [100_000, 1_000_000, 4_000_000];

const OPS = {
  async map(rows) {
    for (const n of SIZES) {
      const data = fill(n);
      const r = await bench(
        () => gpu.map(data, "Math.sqrt(x) * 2 + 1"),
        () => {
          const out = new Float32Array(n);
          for (let i = 0; i < n; i++) out[i] = Math.sqrt(data[i]) * 2 + 1;
          return out;
        }
      );
      rows.push(["map (sqrt(x)*2+1)", n, r]);

      const g = await gpu.upload(data);
      const resident = await median(5, async () => {
        const x = await gpu.map(g, "x + 1", { keepOnGpu: true });
        x.destroy();
      });
      rows.push(["map (GPU-resident, no transfer)", n, { gpuMs: resident, cpuMs: r.cpuMs }]);
      g.destroy();
    }
  },
  async sum(rows) {
    for (const n of SIZES) {
      const data = fill(n);
      rows.push(["sum", n, await bench(
        () => gpu.sum(data),
        () => {
          let s = 0;
          for (let i = 0; i < n; i++) s += data[i];
          return s;
        }
      )]);
    }
  },
  async scan(rows) {
    for (const n of SIZES) {
      const data = fill(n);
      rows.push(["scan (prefix sum)", n, await bench(
        () => gpu.scan(data),
        () => {
          const out = new Float32Array(n);
          let s = 0;
          for (let i = 0; i < n; i++) {
            s += data[i];
            out[i] = s;
          }
          return out;
        }
      )]);
    }
  },
  async sort(rows) {
    for (const n of [100_000, 1_000_000]) {
      const data = fill(n);
      rows.push(["sort", n, await bench(
        () => gpu.sort(data),
        () => data.slice().sort((a, b) => a - b)
      )]);
    }
  },
  async histogram(rows) {
    for (const n of SIZES) {
      const data = fill(n);
      const opts = { bins: 64, min: 0, max: 1 };
      rows.push(["histogram (64 bins)", n, await bench(
        () => gpu.histogram(data, opts),
        () => {
          const out = new Uint32Array(64);
          for (let i = 0; i < n; i++) {
            let b = Math.floor(data[i] * 64);
            if (b > 63) b = 63;
            out[b]++;
          }
          return out;
        }
      )]);
    }
  },
  async matmul(rows) {
    for (const dim of [256, 512, 1024]) {
      const a = fill(dim * dim, 1);
      const b = fill(dim * dim, 2);
      const opts = { rowsA: dim, colsA: dim, colsB: dim };
      rows.push([`matmul ${dim}x${dim}`, dim * dim, await bench(
        () => gpu.matmul(a, b, opts),
        () => {
          const out = new Float32Array(dim * dim);
          for (let i = 0; i < dim; i++) {
            for (let k = 0; k < dim; k++) {
              const aik = a[i * dim + k];
              for (let j = 0; j < dim; j++) out[i * dim + j] += aik * b[k * dim + j];
            }
          }
          return out;
        }
      )]);
    }
  },
  async fft(rows) {
    for (const n of [65_536, 1_048_576]) {
      const data = fill(n);
      // no plain-JS FFT reference — the CPU column shows "—"
      rows.push(["fft", n, await bench(() => gpu.fft(data), null)]);
    }
  },
};

const only = process.argv.slice(2);
const rows = [];
for (const [name, fn] of Object.entries(OPS)) {
  if (only.length && !only.includes(name)) continue;
  process.stderr.write(`benchmarking ${name}...\n`);
  await fn(rows);
}

let table = "| op | n | GPU (ms) | plain JS (ms) | speedup |\n|---|---:|---:|---:|---:|\n";
for (const [name, n, { gpuMs, cpuMs }] of rows) {
  const speed = cpuMs ? `${(cpuMs / gpuMs).toFixed(1)}×` : "—";
  table += `| ${name} | ${n.toLocaleString("en-US")} | ${fmt(gpuMs)} | ${cpuMs ? fmt(cpuMs) : "—"} | ${speed} |\n`;
}
table += "\n_End-to-end wall clock incl. upload/readback, median of 5 after warmup._\n";

gpu.destroy();
// Write synchronously and exit explicitly: Dawn's native teardown at natural
// process exit can segfault (harmless but it eats buffered stdout).
writeSync(1, table);
process.exit(0);
