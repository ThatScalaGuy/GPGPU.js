import { LitElement, html, css } from "lit";
import { gpu, parseNums, fmt } from "../lib/shared.js";
import { controlStyles } from "./shared-styles.js";
import "./demo-card.js";

/**
 * Real use case: a neural-net layer or 2D transform — the core of graphics and
 * ML. Demos gpu.matmul with flat arrays + explicit dimensions, rendering the
 * result as a grid.
 */
export class MatmulDemo extends LitElement {
  static styles = [
    controlStyles,
    css`
      .grid {
        display: grid;
        gap: var(--space-1);
        margin-top: var(--space-3);
        width: fit-content;
      }
      .cell {
        font-family: var(--font-mono);
        font-size: var(--font-size-sm);
        text-align: right;
        padding: var(--space-1) var(--space-3);
        background: var(--color-code-bg);
        border-radius: var(--radius-sm);
        min-width: 36px;
      }
      .result-label {
        font-size: var(--font-size-xs);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        color: var(--color-text-muted);
        margin-top: var(--space-4);
      }
    `,
  ];

  static properties = {
    a: { state: true },
    b: { state: true },
    rowsA: { state: true },
    colsA: { state: true },
    colsB: { state: true },
    result: { state: true },
    error: { state: true },
    busy: { state: true },
  };

  constructor() {
    super();
    this.a = "1, 2, 3, 4";
    this.b = "5, 6, 7, 8";
    this.rowsA = "2";
    this.colsA = "2";
    this.colsB = "2";
    this.result = null;
    this.error = "";
    this.busy = false;
  }

  async run() {
    this.busy = true;
    this.error = "";
    this.result = null;
    try {
      const a = parseNums(this.a);
      const b = parseNums(this.b);
      const opts = {
        rowsA: Number(this.rowsA),
        colsA: Number(this.colsA),
        colsB: Number(this.colsB),
      };
      const flat = await gpu.matmul(a, b, opts);
      this.result = { flat, rows: opts.rowsA, cols: opts.colsB };
    } catch (e) {
      this.error = String(e?.message ?? e);
    } finally {
      this.busy = false;
    }
  }

  _renderGrid() {
    if (!this.result) return "";
    const { flat, rows, cols } = this.result;
    return html`
      <div class="result-label">A · B = ${rows}×${cols}</div>
      <div class="grid" style="grid-template-columns: repeat(${cols}, auto);">
        ${Array.from(flat, (v) => html`<div class="cell">${fmt(v)}</div>`)}
      </div>
    `;
  }

  render() {
    return html`
      <demo-card
        title="Matrix multiply"
        description="A neural-net layer or 2D transform. Matrices are flat arrays plus explicit dimensions."
        .error=${this.error}
        .busy=${this.busy}
        @run=${this.run}
      >
        <label>
          Matrix A (flat, row-major)
          <input .value=${this.a} @input=${(e) => (this.a = e.target.value)} />
        </label>
        <label>
          Matrix B (flat, row-major)
          <input .value=${this.b} @input=${(e) => (this.b = e.target.value)} />
        </label>
        <div class="row">
          <label>
            rowsA
            <input .value=${this.rowsA} @input=${(e) => (this.rowsA = e.target.value)} />
          </label>
          <label>
            colsA / rowsB
            <input .value=${this.colsA} @input=${(e) => (this.colsA = e.target.value)} />
          </label>
          <label>
            colsB
            <input .value=${this.colsB} @input=${(e) => (this.colsB = e.target.value)} />
          </label>
        </div>
        <div slot="footer">${this._renderGrid()}</div>
      </demo-card>
    `;
  }
}

customElements.define("matmul-demo", MatmulDemo);
