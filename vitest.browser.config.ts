import { defineConfig } from "vitest/config";

// Runs the real WGSL in headless Chromium (real browser WebGPU) via Playwright.
export default defineConfig({
  test: {
    include: ["tests/browser/**/*.test.ts"],
    testTimeout: 60000,
    browser: {
      enabled: true,
      provider: "playwright",
      name: "chromium",
      headless: true,
      providerOptions: {
        launch: {
          args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
        },
      },
    },
  },
});
