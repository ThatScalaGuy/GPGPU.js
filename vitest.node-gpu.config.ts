import { defineConfig } from "vitest/config";

// Runs the real WGSL in Node via the `webgpu` (Google Dawn) bindings injected in setup.
export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: ["tests/node-gpu/**/*.test.ts"],
    setupFiles: ["tests/setup/node-webgpu.ts"],
    testTimeout: 60000,
  },
});
