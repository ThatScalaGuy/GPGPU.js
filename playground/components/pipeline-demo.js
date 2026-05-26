import { LitElement, html, css } from "lit";
import { gpu, parseNums, fmt, runWithStats } from "../lib/shared.js";
import { controlStyles } from "./shared-styles.js";
import "./demo-card.js";

/**
 * Real use case: chaining transforms (normalize → activate → reduce) while
 * keeping data on the GPU between steps, avoiding CPU round-trips. Demos
 * gpu.pipeline().map(...).map(...).reduce(...).run(...).
 */
export class PipelineDemo extends LitElement {
  static styles = [
    controlStyles,
    css`
      pre.code {
        margin: 0;
        padding: var(--space-3) var(--space-4);
        background: var(--color-code-bg);
        border-radius: var(--radius-md);
        font-family: var(--font-mono);
        font-size: var(--font-size-xs);
        white-space: pre;
        overflow-x: auto;
        color: var(--color-text);
      }
    `,
  ];

  static properties = {
    input: { state: true },
    output: { state: true },
    error: { state: true },
    busy: { state: true },
    stats: { state: true },
  };

  constructor() {
    super();
    this.input = "1, 2, 3, 4, 5";
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
        gpu
          .pipeline()
          .map((x) => x * 2)
          .map((x) => x + 1)
          .run(input)
      );
      this.stats = stats;
      this.output = `input  ${fmt(input)}\nresult ${fmt(result)}`;
    } catch (e) {
      this.error = String(e?.message ?? e);
    } finally {
      this.busy = false;
    }
  }

  render() {
    return html`
      <demo-card
        title="Pipeline / chaining"
        description="Chain operations with data staying on the GPU between steps — no CPU round-trips."
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
        <pre class="code">
gpu.pipeline()
   .map(x => x * 2)
   .map(x => x + 1)
   .run(input)</pre
        >
      </demo-card>
    `;
  }
}

customElements.define("pipeline-demo", PipelineDemo);
