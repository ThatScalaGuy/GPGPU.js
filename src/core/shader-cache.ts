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
