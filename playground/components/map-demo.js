import { LitElement, html } from "lit";
import { gpu, parseNums, fmt, runWithStats } from "../lib/shared.js";
import { controlStyles } from "./shared-styles.js";
import "./demo-card.js";

/**
 * Real use case: applying an activation function or transform to a tensor —
 * the bread-and-butter of ML preprocessing. Demos gpu.map with a live-editable
 * expression, showing off the JS→WGSL parser (Math.*, ternaries, etc.).
 */
// String functions are expression bodies in `x` (not arrow-function strings).
const PRESETS = [
  { label: "ReLU", fn: "Math.max(0, x)" },
  { label: "Sigmoid", fn: "1 / (1 + Math.exp(-x))" },
  { label: "Square", fn: "x * x" },
  { label: "Scale + bias", fn: "x * 0.5 + 1" },
];

export class MapDemo extends LitElement {
  static styles = controlStyles;
  static properties = {
    input: { state: true },
    fn: { state: true },
    output: { state: true },
    error: { state: true },
    busy: { state: true },
    stats: { state: true },
  };

  constructor() {
    super();
    this.input = "-2, -1, 0, 1, 2";
    this.fn = "Math.max(0, x)";
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
      const { result, stats } = await runWithStats(() => gpu.map(input, this.fn));
      this.stats = stats;
      this.output = `map(${this.fn})\n${fmt(input)} → ${fmt(result)}`;
    } catch (e) {
      this.error = String(e?.message ?? e);
    } finally {
      this.busy = false;
    }
  }

  render() {
    return html`
      <demo-card
        title="Map / activation"
        description="Apply a function to every element. The JS expression compiles to a WGSL shader."
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
          Expression in <code>x</code> (e.g. <code>Math.sqrt(x)</code>)
          <input .value=${this.fn} @input=${(e) => (this.fn = e.target.value)} />
        </label>
        <div class="chips">
          ${PRESETS.map(
            (p) => html`<button class="chip" @click=${() => (this.fn = p.fn)}>${p.label}</button>`
          )}
        </div>
      </demo-card>
    `;
  }
}

customElements.define("map-demo", MapDemo);
