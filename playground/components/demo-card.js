import { LitElement, html, css } from "lit";

/**
 * Reusable card shell shared by every feature demo. Provides consistent
 * layout: a title/description header, a slot for controls, a Run button with
 * busy state, and an output/error region. Feature components render a
 * <demo-card> and fill the slots — so the layout lives in exactly one place.
 *
 * Props:
 *   title, description — header text
 *   output             — string shown in the result region
 *   error              — string shown as an error (takes precedence)
 *   busy               — disables the Run button and shows "Running…"
 *   runLabel           — Run button text (default "Run")
 *   stats              — { backend, ms } shown as a badge after a run
 * Events:
 *   "run" — fired when the Run button is clicked
 */
export class DemoCard extends LitElement {
  static properties = {
    title: { type: String },
    description: { type: String },
    output: { type: String },
    error: { type: String },
    busy: { type: Boolean },
    runLabel: { type: String },
    stats: { type: Object },
  };

  constructor() {
    super();
    this.title = "";
    this.description = "";
    this.output = "";
    this.error = "";
    this.busy = false;
    this.runLabel = "Run";
    this.stats = null;
  }

  static styles = css`
    :host {
      display: block;
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-card);
      padding: var(--space-5);
    }
    h2 {
      margin: 0 0 var(--space-1);
      font-size: var(--font-size-lg);
      color: var(--color-text);
    }
    .desc {
      margin: 0 0 var(--space-4);
      font-size: var(--font-size-sm);
      color: var(--color-text-muted);
    }
    .controls {
      display: flex;
      flex-direction: column;
      gap: var(--space-3);
      margin-bottom: var(--space-4);
    }
    .footer {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: var(--space-3);
    }
    button.run {
      appearance: none;
      border: none;
      cursor: pointer;
      background: var(--color-accent);
      color: var(--color-accent-contrast);
      font-family: inherit;
      font-size: var(--font-size-sm);
      font-weight: 600;
      padding: var(--space-2) var(--space-5);
      border-radius: var(--radius-md);
      transition: background 0.15s ease;
    }
    button.run:hover:not(:disabled) {
      background: var(--color-accent-hover);
    }
    button.run:disabled {
      opacity: 0.55;
      cursor: default;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: var(--space-2);
      font-size: var(--font-size-xs);
      font-weight: 600;
      padding: var(--space-1) var(--space-3);
      border-radius: 999px;
    }
    .badge.gpu {
      background: var(--color-success-bg);
      color: var(--color-success);
    }
    .badge.cpu {
      background: var(--color-code-bg);
      color: var(--color-text-muted);
    }
    .output {
      margin: var(--space-4) 0 0;
      padding: var(--space-3) var(--space-4);
      background: var(--color-code-bg);
      border-radius: var(--radius-md);
      font-family: var(--font-mono);
      font-size: var(--font-size-sm);
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--color-text);
    }
    .output.error {
      background: var(--color-error-bg);
      color: var(--color-error);
    }
  `;

  _onRun() {
    this.dispatchEvent(new CustomEvent("run", { bubbles: true, composed: true }));
  }

  render() {
    return html`
      <h2>${this.title}</h2>
      <p class="desc">${this.description}</p>
      <div class="controls"><slot></slot></div>
      <div class="footer">
        <button class="run" ?disabled=${this.busy} @click=${this._onRun}>
          ${this.busy ? "Running…" : this.runLabel}
        </button>
        ${this.stats
          ? html`<span class="badge ${this.stats.backend}">
              ${this.stats.backend.toUpperCase()} · ${this.stats.ms.toFixed(2)} ms
            </span>`
          : ""}
        <slot name="footer"></slot>
      </div>
      ${this.error
        ? html`<pre class="output error">${this.error}</pre>`
        : this.output
        ? html`<pre class="output">${this.output}</pre>`
        : ""}
    `;
  }
}

customElements.define("demo-card", DemoCard);
