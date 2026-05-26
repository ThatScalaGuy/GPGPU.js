import { LitElement, html } from "lit";
import { gpu, parseNums, fmt, runWithStats } from "../lib/shared.js";
import { controlStyles } from "./shared-styles.js";
import "./demo-card.js";

/**
 * Real use case: running totals / cumulative metrics (e.g. cumulative revenue,
 * histogram offsets). Demos gpu.scan — inclusive prefix sum by default, plus a
 * prefix-product variant.
 */
export class ScanDemo extends LitElement {
  static styles = controlStyles;
  static properties = {
    input: { state: true },
    kind: { state: true },
    output: { state: true },
    error: { state: true },
    busy: { state: true },
    stats: { state: true },
  };

  constructor() {
    super();
    this.input = "1, 2, 3, 4, 5, 6";
    this.kind = "sum";
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
      const input = parseNums(this.input);
      const { result, stats } = await runWithStats(() =>
        this.kind === "product"
          ? gpu.scan(input, (a, b) => a * b, 1)
          : gpu.scan(input)
      );
      this.stats = stats;
      const op = this.kind === "product" ? "running product" : "running total";
      this.output = `input  ${fmt(input)}\n${op}  ${fmt(result)}`;
    } catch (e) {
      this.error = String(e?.message ?? e);
    } finally {
      this.busy = false;
    }
  }

  render() {
    return html`
      <demo-card
        title="Scan / running total"
        description="Inclusive prefix scan — each output is the accumulation of all elements up to it."
        .output=${this.output}
        .error=${this.error}
        .busy=${this.busy}
        .stats=${this.stats}
        @run=${this.run}
      >
        <label>
          Input
          <input .value=${this.input} @input=${(e) => (this.input = e.target.value)} />
        </label>
        <label>
          Accumulator
          <select @change=${(e) => (this.kind = e.target.value)}>
            <option value="sum">prefix sum (running total)</option>
            <option value="product">prefix product</option>
          </select>
        </label>
      </demo-card>
    `;
  }
}

customElements.define("scan-demo", ScanDemo);
