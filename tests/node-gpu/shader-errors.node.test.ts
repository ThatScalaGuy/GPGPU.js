import { describe, it, expect } from "vitest";
import { DeviceManager } from "../../src/core/device";
import { ShaderCache } from "../../src/core/shader-cache";

const dm = new DeviceManager();
const cache = new ShaderCache();

const BAD = `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let v = nonexistent_identifier;
}
`;

// A different broken shader so it is not served from the by-shaderCode cache.
const BAD2 = `
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let w = also_undefined_thing;
}
`;

describe("WGSL compile error formatting", () => {
  it("frames the error + includes the JS source", async () => {
    const device = await dm.getDevice();
    let err: Error | undefined;
    try {
      await cache.getOrCreate(device, BAD, "bad", "x => oops(x)");
    } catch (e) {
      err = e as Error;
    }
    expect(err).toBeDefined();
    const msg = err!.message;
    expect(msg).toContain("Shader compilation failed");
    expect(msg).toContain("nonexistent_identifier");
    expect(msg).toMatch(/\^/);
    expect(msg).toContain("from expression: x => oops(x)");
  });

  it("omits the expression note when no source is given", async () => {
    const device = await dm.getDevice();
    let err: Error | undefined;
    try {
      await cache.getOrCreate(device, BAD2, "bad2");
    } catch (e) {
      err = e as Error;
    }
    expect(err).toBeDefined();
    const msg = err!.message;
    expect(msg).toContain("Shader compilation failed");
    expect(msg).toContain("also_undefined_thing");
    expect(msg).toMatch(/\^/);
    expect(msg).not.toContain("from expression");
  });

  it("a valid shader still compiles", async () => {
    const device = await dm.getDevice();
    const pipeline = await cache.getOrCreate(
      device,
      "@compute @workgroup_size(1) fn main() {}",
      "ok"
    );
    expect(pipeline).toBeDefined();
  });
});
