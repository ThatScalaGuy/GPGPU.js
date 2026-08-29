import { describe, it, expect } from "vitest";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

// Regression for a SIGSEGV in the Node auto-detect runtime: DeviceManager dropped
// the lazily imported Dawn GPU instance after init, so V8 could collect it while
// the device was still in use — multi-MB allocations trigger GC and crashed the
// process. The vitest suites never saw it because tests/setup/node-webgpu.ts pins
// navigator.gpu globally, so this test exercises the packaged auto-detect path in
// a plain Node subprocess (dist/ exists in CI via the npm `prepare` build).
const root = process.cwd();
const dist = resolve(root, "dist/index.js");

describe("Node auto-detect runtime (plain subprocess)", () => {
  it.skipIf(!existsSync(dist))("survives multi-MB workloads under GC pressure", () => {
    const script = `
      const { GPU } = await import(${JSON.stringify(dist)});
      const gpu = new GPU({ fallback: "throw" });
      for (let i = 0; i < 5; i++) {
        const r = await gpu.sum(new Float32Array(4_000_000).fill(1));
        if (r !== 4_000_000) throw new Error("bad sum: " + r);
      }
      console.log("OK");
      process.exit(0);
    `;
    const out = execFileSync(process.execPath, ["--input-type=module", "-e", script], {
      cwd: root,
      timeout: 120_000,
      encoding: "utf8",
    });
    expect(out).toContain("OK");
  });
});
