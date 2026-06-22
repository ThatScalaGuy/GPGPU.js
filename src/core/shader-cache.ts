// Render numbered WGSL lines around the error with a caret under the offending column.
function codeFrame(code: string, lineNum: number, linePos: number, context = 2): string {
  const lines = code.split("\n");
  const start = Math.max(0, lineNum - 1 - context);
  const end = Math.min(lines.length, lineNum + context);
  const width = String(end).length;
  const rows: string[] = [];
  for (let i = start; i < end; i++) {
    const n = i + 1; // 1-based line number
    const marker = n === lineNum ? ">" : " ";
    rows.push(`  ${marker} ${String(n).padStart(width)} | ${lines[i]}`);
    if (n === lineNum) {
      const caret = " ".repeat(Math.max(0, linePos - 1)) + "^";
      rows.push(`    ${" ".repeat(width)} | ${caret}`);
    }
  }
  return rows.join("\n");
}

export class ShaderCache {
  private cache = new Map<string, GPUComputePipeline>();

  async getOrCreate(
    device: GPUDevice,
    shaderCode: string,
    label?: string,
    source?: string
  ): Promise<GPUComputePipeline> {
    const cached = this.cache.get(shaderCode);
    if (cached) return cached;

    const module = device.createShaderModule({
      code: shaderCode,
      label: label ?? "gpgpu-shader",
    });

    // Surface the real WGSL diagnostic. createComputePipelineAsync rejects with an
    // opaque GPUPipelineError on some backends (notably Safari/WebKit), so pull the
    // line:col messages out of getCompilationInfo before the pipeline call swallows them.
    if (typeof module.getCompilationInfo === "function") {
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === "error");
      if (errors.length > 0) {
        const detail = errors
          .map((m) => {
            const head = `  ${m.lineNum}:${m.linePos}: ${m.message}`;
            // lineNum is 1-based and >0 when the message is tied to a source line.
            return m.lineNum > 0 ? `${head}\n${codeFrame(shaderCode, m.lineNum, m.linePos)}` : head;
          })
          .join("\n\n");
        const srcNote = source ? `\n\n  from expression: ${source}` : "";
        throw new Error(
          `Shader compilation failed (${label ?? "gpgpu-shader"}):\n${detail}${srcNote}`
        );
      }
    }

    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });

    this.cache.set(shaderCode, pipeline);
    return pipeline;
  }

  clear(): void {
    this.cache.clear();
  }
}
