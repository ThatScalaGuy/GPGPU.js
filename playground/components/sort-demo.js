import { LitElement, html } from "lit";
import { gpu, fmt, timeit, runWithStats } from "../lib/shared.js";
import { controlStyles } from "./shared-styles.js";
import "./demo-card.js";

/**
 * Real use case: sorting a large dataset and seeing GPU throughput. Demos
 * gpu.sort (bitonic, ascending) on N random values and times it against a
 * plain JS Array.sort baseline.
 *
 * Note: both paths are correct; the library auto-selects GPU/CPU internally and
 * exposes no force-CPU switch, so this contrasts gpu.sort throughput with the
 * native engine sort — not a forced GPU-vs-CPU benchmark.
 */
export class SortDemo extends LitElement {
  static styles = controlStyles;
  static properties = {
    n: { state: true },
    output: { state: true },
    error: { state: true },
    busy: { state: true },
    stats: { state: true },
  };

  constructor() {
    super();
    this.n = "1024";
    this.output = "";
    this.error = "";
    this.busy = false;
    this.stats = null;
  }

  async run() {
    this.busy = true;
    this.error = "";
    this.output = "";
    this.stats = null;
    try {
      const n = Number(this.n);
      const data = Float32Array.from({ length: n }, () => Math.random() * 1000);

      const { result: { result: sorted, stats }, ms: gpuMs } = await timeit(() =>
        runWithStats(() => gpu.sort(data))
      );
      this.stats = stats;
      const { ms: jsMs } = await timeit(async () => data.slice().sort((a, b) => a - b));

      const preview = fmt(sorted.slice(0, 8));
      this.output =
        `sorted ${n} values — first 8: ${preview} …\n` +
        `gpu.sort:        ${gpuMs.toFixed(2)} ms\n` +
        `Array.sort (JS): ${jsMs.toFixed(2)} ms`;
    } catch (e) {
      this.error = String(e?.message ?? e);
    } finally {
      this.busy = false;
    }
  }

  render() {
    return html`
      <demo-card
        title="Sort + timing"
        description="GPU bitonic sort on N random values, timed against the native Array.sort."
        runLabel="Generate & sort"
        .output=${this.output}
        .error=${this.error}
        .busy=${this.busy}
        .stats=${this.stats}
        @run=${this.run}
      >
        <label>
          Element count (powers of two sort best)
          <input type="number" .value=${this.n} @input=${(e) => (this.n = e.target.value)} />
        </label>
      </demo-card>
    `;
  }
}

customElements.define("sort-demo", SortDemo);
