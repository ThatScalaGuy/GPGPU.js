import { LitElement, html, css } from "lit";
import { gpu, parseNums, fmt, runWithStats } from "../lib/shared.js";
import { controlStyles } from "./shared-styles.js";
import "./demo-card.js";

/**
 * Real use case: dropping to raw WGSL for an operation the high-level API
 * doesn't cover. Demos gpu.createKernel with an editable shader + input,
 * prefilled with a working "square each element" kernel.
 */
const DEFAULT_SHADER = `@group(0) @binding(0) var<storage, read> input0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&input0)) { return; }
  output[i] = input0[i] * input0[i];
}`;

export class KernelDemo extends LitElement {
  static styles = [
    controlStyles,
    css`
      textarea.shader {
        min-height: 190px;
        font-size: var(--font-size-xs);
        white-space: pre;
      }
    `,
  ];

  static properties = {
    input: { state: true },
    shader: { state: true },
    output: { state: true },
    error: { state: true },
    busy: { state: true },
    stats: { state: true },
  };

  constructor() {
    super();
    this.input = "1, 2, 3, 4, 5, 6, 7, 8";
    this.shader = DEFAULT_SHADER;
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
      const kernel = await gpu.createKernel({
        workgroupSize: 64,
        shader: this.shader,
        inputs: [{ type: "f32", size: input.length }],
        output: { type: "f32", size: input.length },
      });
      const { result, stats } = await runWithStats(() => kernel.run(input));
      this.stats = stats;
      this.output = `${fmt(input)} → ${fmt(result)}`;
    } catch (e) {
      this.error = String(e?.message ?? e);
    } finally {
      this.busy = false;
    }
  }

  render() {
    return html`
      <demo-card
        title="Custom WGSL kernel"
        description="Escape hatch: write your own compute shader for anything the high-level API can't express."
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
          WGSL shader
          <textarea
            class="shader"
            .value=${this.shader}
            @input=${(e) => (this.shader = e.target.value)}
          ></textarea>
        </label>
      </demo-card>
    `;
  }
}

customElements.define("kernel-demo", KernelDemo);
