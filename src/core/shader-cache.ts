export class ShaderCache {
  private cache = new Map<string, GPUComputePipeline>();

  async getOrCreate(
    device: GPUDevice,
    shaderCode: string,
    label?: string
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
          .map((m) => `  ${m.lineNum}:${m.linePos}: ${m.message}`)
          .join("\n");
        throw new Error(
          `Shader compilation failed (${label ?? "gpgpu-shader"}):\n${detail}`
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
