import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
  },
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  sourcemap: true,
  splitting: true,
  treeshake: true,
  target: "es2022",
  // `webgpu` (Google Dawn) is an optional Node-only runtime, loaded via a lazy
  // dynamic import in src/core/device.ts. Keep it external so the browser bundle
  // never statically pulls the native package.
  external: ["webgpu"],
});
