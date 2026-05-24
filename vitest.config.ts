import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    // CPU/unit tests only; the real-GPU suites live under tests/{node-gpu,browser}
    // and run via their own configs (test:node-gpu / test:browser).
    include: ["tests/unit/**/*.test.ts"],
  },
});
