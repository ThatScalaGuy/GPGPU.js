import { LitElement, html } from "lit";
import { gpu, parseNums, fmt, runWithStats } from "../lib/shared.js";
import { controlStyles } from "./shared-styles.js";
import "./demo-card.js";

/**
 * Real use case: dataset statistics. Demos the reduce convenience functions
 * (sum/min/max/product) plus a fully custom reduce with an identity element.
 */
export class ReduceDemo extends LitElement {
  static styles = controlStyles;
  static properties = {
    input: { state: true },
    kind: { state: true },
    fn: { state: true },
    identity: { state: true },
    output: { state: true },
    error: { state: true },
    busy: { state: true },
    stats: { state: true },
  };

  constructor() {
    super();
    this.input = "3, 1, 4, 1, 5, 9, 2, 6";
    this.kind = "sum";
    this.fn = "a + b";
    this.identity = "0";
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
        this.kind === "custom"
          ? gpu.reduce(input, this.fn, Number(this.identity))
          : gpu[this.kind](input)
      );
      this.stats = stats;
      const label = this.kind === "custom" ? `reduce(${this.fn}, ${this.identity})` : this.kind;
      this.output = `${label} of ${fmt(input)}\n= ${fmt(result)}`;
    } catch (e) {
      this.error = String(e?.message ?? e);
    } finally {
      this.busy = false;
    }
  }

  render() {
    return html`
      <demo-card
        title="Reduce / statistics"
        description="Collapse an array to one value: sum, min, max, product, or a custom reducer."
        .output=${this.output}
        .error=${this.error}
        .busy=${this.busy}
        .stats=${this.stats}
        @run=${this.run}
      >
        <label>
          Dataset
          <input .value=${this.input} @input=${(e) => (this.input = e.target.value)} />
        </label>
        <label>
          Reduction
          <select @change=${(e) => (this.kind = e.target.value)}>
            <option value="sum">sum</option>
            <option value="min">min</option>
            <option value="max">max</option>
            <option value="product">product</option>
            <option value="custom">custom</option>
          </select>
        </label>
        ${this.kind === "custom"
          ? html`
              <div class="row">
                <label>
                  Reducer — expression in a, b
                  <input .value=${this.fn} @input=${(e) => (this.fn = e.target.value)} />
                </label>
                <label>
                  Identity
                  <input .value=${this.identity} @input=${(e) => (this.identity = e.target.value)} />
                </label>
              </div>
            `
          : ""}
      </demo-card>
    `;
  }
}

customElements.define("reduce-demo", ReduceDemo);
