// Gives the Node test runtime a real WebGPU implementation (Google Dawn via dawn.node),
// so gpu.scan() exercises the actual WGSL instead of the CPU fallback.
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);

Object.defineProperty(globalThis, "navigator", {
  value: { gpu: create([]) },
  configurable: true,
  writable: true,
});
