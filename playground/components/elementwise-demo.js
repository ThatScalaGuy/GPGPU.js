import { LitElement, html } from "lit";
import { gpu, parseNums, fmt } from "../lib/shared.js";
import { controlStyles } from "./shared-styles.js";
import "./demo-card.js";

/**
 * Real use case: mixing two signals or applying a gain.
 * Demos gpu.add / subtract / multiply / divide in both array⊕array and
 * array⊕scalar modes.
 */
export class ElementwiseDemo extends LitElement {
  static styles = controlStyles;
  static properties = {
    a: { state: true },
    b: { state: true },
    op: { state: true },
    mode: { state: true },
    output: { state: true },
    error: { state: true },
    busy: { state: true },
  };

  constructor() {
    super();
    this.a = "1, 2, 3, 4";
    this.b = "10, 20, 30, 40";
    this.op = "add";
    this.mode = "array";
    this.output = "";
    this.error = "";
    this.busy = false;
  }

  async run() {
    this.busy = true;
    this.error = "";
    this.output = "";
    try {
      const a = parseNums(this.a);
      const b = this.mode === "scalar" ? Number(this.b) : parseNums(this.b);
      const result = await gpu[this.op](a, b);
      this.output = `${fmt(a)} ${this._symbol()} ${this.mode === "scalar" ? this.b : fmt(b)}\n= ${fmt(result)}`;
    } catch (e) {
      this.error = String(e?.message ?? e);
    } finally {
      this.busy = false;
    }
  }

  _symbol() {
    return { add: "+", subtract: "−", multiply: "×", divide: "÷" }[this.op];
  }

  render() {
    return html`
      <demo-card
        title="Element-wise math"
        description="Mix two signals or apply a gain. Works array⊕array or array⊕scalar."
        .output=${this.output}
        .error=${this.error}
        .busy=${this.busy}
        @run=${this.run}
      >
        <label>
          Array A
          <input .value=${this.a} @input=${(e) => (this.a = e.target.value)} />
        </label>
        <div class="row">
          <label>
            Operation
            <select @change=${(e) => (this.op = e.target.value)}>
              <option value="add">add</option>
              <option value="subtract">subtract</option>
              <option value="multiply">multiply</option>
              <option value="divide">divide</option>
            </select>
          </label>
          <label>
            Second operand
            <select @change=${(e) => (this.mode = e.target.value)}>
              <option value="array">array B</option>
              <option value="scalar">scalar</option>
            </select>
          </label>
        </div>
        <label>
          ${this.mode === "scalar" ? "Scalar value" : "Array B"}
          <input .value=${this.b} @input=${(e) => (this.b = e.target.value)} />
        </label>
      </demo-card>
    `;
  }
}

customElements.define("elementwise-demo", ElementwiseDemo);
