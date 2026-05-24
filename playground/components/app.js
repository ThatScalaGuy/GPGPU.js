import { LitElement, html, css } from "lit";
import { gpu } from "../lib/shared.js";

// Register every demo component (single <script> boots the whole playground).
import "./elementwise-demo.js";
import "./map-demo.js";
import "./reduce-demo.js";
import "./scan-demo.js";
import "./matmul-demo.js";
import "./sort-demo.js";
import "./pipeline-demo.js";
import "./kernel-demo.js";

/**
 * Root app: header with the GPU status banner and a light/dark theme toggle
 * (flipping data-theme on <html> re-themes everything via the design tokens),
 * over a responsive grid of feature demos.
 */
export class GpgpuApp extends LitElement {
  static properties = {
    available: { state: true },
    theme: { state: true },
  };

  constructor() {
    super();
    this.available = gpu.isAvailable();
    this.theme = document.documentElement.getAttribute("data-theme") || "light";
  }

  static styles = css`
    :host {
      display: block;
      max-width: 1200px;
      margin: 0 auto;
      padding: var(--space-6) var(--space-5) var(--space-6);
    }
    header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: var(--space-4);
      flex-wrap: wrap;
      margin-bottom: var(--space-5);
    }
    h1 {
      margin: 0;
      font-size: var(--font-size-xl);
      letter-spacing: -0.02em;
    }
    h1 .accent {
      color: var(--color-accent);
    }
    .tagline {
      margin: var(--space-1) 0 0;
      color: var(--color-text-muted);
      font-size: var(--font-size-sm);
    }
    .top-right {
      display: flex;
      align-items: center;
      gap: var(--space-3);
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: var(--space-2);
      font-size: var(--font-size-xs);
      font-weight: 600;
      padding: var(--space-1) var(--space-3);
      border-radius: 999px;
    }
    .chip.ok {
      background: var(--color-success-bg);
      color: var(--color-success);
    }
    .chip.cpu {
      background: var(--color-code-bg);
      color: var(--color-text-muted);
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
    }
    button.theme {
      appearance: none;
      cursor: pointer;
      font-family: inherit;
      font-size: var(--font-size-sm);
      color: var(--color-text);
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-md);
      padding: var(--space-2) var(--space-3);
    }
    button.theme:hover {
      border-color: var(--color-accent);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: var(--space-5);
    }
  `;

  _toggleTheme() {
    this.theme = this.theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", this.theme);
  }

  render() {
    return html`
      <header>
        <div>
          <h1>GPGPU<span class="accent">.js</span> playground</h1>
          <p class="tagline">
            GPU-accelerated array math in the browser via WebGPU — interactive demos of every feature.
          </p>
        </div>
        <div class="top-right">
          ${this.available
            ? html`<span class="chip ok"><span class="dot"></span>WebGPU active</span>`
            : html`<span class="chip cpu"><span class="dot"></span>CPU fallback</span>`}
          <button class="theme" @click=${this._toggleTheme}>
            ${this.theme === "dark" ? "☀ Light" : "☾ Dark"}
          </button>
        </div>
      </header>

      <div class="grid">
        <elementwise-demo></elementwise-demo>
        <map-demo></map-demo>
        <reduce-demo></reduce-demo>
        <scan-demo></scan-demo>
        <matmul-demo></matmul-demo>
        <sort-demo></sort-demo>
        <pipeline-demo></pipeline-demo>
        <kernel-demo></kernel-demo>
      </div>
    `;
  }
}

customElements.define("gpgpu-app", GpgpuApp);
