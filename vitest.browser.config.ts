import { defineConfig } from "vitest/config";
import { playwright } from "@vitest/browser-playwright";

// Runs the real WGSL in headless Chromium (real browser WebGPU) via Playwright.
export default defineConfig({
  test: {
    include: ["tests/browser/**/*.test.ts"],
    testTimeout: 60000,
    browser: {
      enabled: true,
      provider: playwright({
        launchOptions: {
          args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
        },
      }),
      instances: [{ browser: "chromium" }],
      headless: true,
    },
  },
});
