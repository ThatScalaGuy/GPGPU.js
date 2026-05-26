import { describe, it, expect } from "vitest";
import { GPU } from "../../src/gpu";
import { GPUArray } from "../../src/pipeline/gpu-array";

// Exercise the public facade against the real GPU: GPUArray as a first-class
// input/output type, the auto / keepOnGpu rules, immutability, and errors.
const gpu = new GPU();

describe("GPUArray input/output (real GPU)", () => {
  it("upload + toArray round-trips", async () => {
    const g = await gpu.upload([1, 2, 3, 4]);
    expect(g).toBeInstanceOf(GPUArray);
    expect(Array.from(await g.toArray())).toEqual([1, 2, 3, 4]);
    g.destroy();
  });

  it("auto mode: GPUArray input -> GPUArray output, CPU input -> TypedArray", async () => {
    const g = await gpu.upload([1, 2, 3, 4]);
    const r = await gpu.map(g, (x) => x * 2);
    expect(r).toBeInstanceOf(GPUArray);
    expect(Array.from(await (r as GPUArray).toArray())).toEqual([2, 4, 6, 8]);
    (r as GPUArray).destroy();
    g.destroy();

    const cpu = await gpu.map([1, 2, 3, 4], (x) => x * 2);
    expect(cpu).toBeInstanceOf(Float32Array);
    expect(Array.from(cpu as Float32Array)).toEqual([2, 4, 6, 8]);
  });

  it("keepOnGpu: true keeps a CPU-array result on the GPU", async () => {
    const r = await gpu.map([1, 2, 3], (x) => x + 1, { keepOnGpu: true });
    expect(r).toBeInstanceOf(GPUArray);
    expect(Array.from(await r.toArray())).toEqual([2, 3, 4]);
    r.destroy();
  });

  it("keepOnGpu: false forces readback even for a GPUArray input", async () => {
    const g = await gpu.upload([1, 2, 3]);
    const r = await gpu.map(g, (x) => x + 1, { keepOnGpu: false });
    expect(r).toBeInstanceOf(Float32Array);
    expect(Array.from(r as Float32Array)).toEqual([2, 3, 4]);
    g.destroy();
  });

  it("chains several ops with one upload and one readback", async () => {
    const g = await gpu.upload([1, 2, 3, 4]);
    const a = (await gpu.multiply(g, 2)) as GPUArray; // [2,4,6,8]
    const b = (await gpu.add(a, 1)) as GPUArray; // [3,5,7,9]
    expect(a).toBeInstanceOf(GPUArray);
    expect(b).toBeInstanceOf(GPUArray);
    expect(Array.from(await b.toArray())).toEqual([3, 5, 7, 9]);
    g.destroy();
    a.destroy();
    b.destroy();
  });

  it("elementwise with two GPUArray inputs", async () => {
    const a = await gpu.upload([1, 2, 3]);
    const b = await gpu.upload([10, 20, 30]);
    const r = (await gpu.add(a, b)) as GPUArray;
    expect(Array.from(await r.toArray())).toEqual([11, 22, 33]);
    a.destroy();
    b.destroy();
    r.destroy();
  });

  it("reduce family accepts a GPUArray and returns a number", async () => {
    const g = await gpu.upload([3, 1, 4, 1, 5, 9, 2, 6]);
    expect(await gpu.sum(g)).toBe(31);
    expect(await gpu.min(g)).toBe(1);
    expect(await gpu.max(g)).toBe(9);
    g.destroy();
  });

  it("matmul accepts GPUArray inputs", async () => {
    const a = await gpu.upload([1, 2, 3, 4]);
    const b = await gpu.upload([5, 6, 7, 8]);
    const r = (await gpu.matmul(a, b, { rowsA: 2, colsA: 2, colsB: 2 })) as GPUArray;
    expect(r).toBeInstanceOf(GPUArray);
    expect(Array.from(await r.toArray())).toEqual([19, 22, 43, 50]);
    a.destroy();
    b.destroy();
    r.destroy();
  });

  it("scan does not mutate its GPUArray input", async () => {
    const g = await gpu.upload([1, 2, 3, 4, 5]);
    const r = (await gpu.scan(g)) as GPUArray;
    expect(Array.from(await r.toArray())).toEqual([1, 3, 6, 10, 15]);
    // input is untouched
    expect(Array.from(await g.toArray())).toEqual([1, 2, 3, 4, 5]);
    g.destroy();
    r.destroy();
  });

  it("sort does not mutate its GPUArray input (non-power-of-2 length)", async () => {
    const g = await gpu.upload([10, 2, 33, 4, 100]);
    const r = (await gpu.sort(g)) as GPUArray;
    expect(Array.from(await r.toArray())).toEqual([2, 4, 10, 33, 100]);
    expect(Array.from(await g.toArray())).toEqual([10, 2, 33, 4, 100]);
    g.destroy();
    r.destroy();
  });

  it("a sorted GPUArray feeds straight into another op", async () => {
    const g = await gpu.upload([3, 1, 2]);
    const sorted = (await gpu.sort(g)) as GPUArray; // [1,2,3]
    const plus10 = (await gpu.add(sorted, 10)) as GPUArray;
    expect(Array.from(await plus10.toArray())).toEqual([11, 12, 13]);
    g.destroy();
    sorted.destroy();
    plus10.destroy();
  });

  it("throws on mismatched length / dtype for two GPUArray inputs", async () => {
    const a = await gpu.upload([1, 2, 3]);
    const b = await gpu.upload([1, 2]);
    await expect(gpu.add(a, b)).rejects.toThrow(/same length/);

    const i = await gpu.upload(new Int32Array([1, 2, 3]));
    await expect(gpu.add(a, i)).rejects.toThrow(/data type/);
    a.destroy();
    b.destroy();
    i.destroy();
  });

  it("throws when a destroyed GPUArray is used as input", async () => {
    const g = await gpu.upload([1, 2, 3]);
    g.destroy();
    await expect(gpu.map(g, (x) => x + 1)).rejects.toThrow(/destroyed/);
    await expect(gpu.sum(g)).rejects.toThrow(/destroyed/);
    await expect(gpu.sort(g)).rejects.toThrow(/destroyed/);
  });

  it("pipeline accepts a GPUArray input and keeps the result on the GPU", async () => {
    const g = await gpu.upload([1, 2, 3, 4, 5]);
    const r = await gpu.pipeline().map((x) => x * 2).map((x) => x + 1).run(g, { keepOnGpu: true });
    expect(r).toBeInstanceOf(GPUArray);
    expect(Array.from(await (r as GPUArray).toArray())).toEqual([3, 5, 7, 9, 11]);
    // input untouched
    expect(Array.from(await g.toArray())).toEqual([1, 2, 3, 4, 5]);
    (r as GPUArray).destroy();
    g.destroy();
  });

  it("reduce-terminated pipeline still returns a number for a GPUArray input", async () => {
    const g = await gpu.upload([1, 2, 3, 4, 5]);
    const r = await gpu.pipeline().map((x) => x * 2).reduce((a, b) => a + b, 0).run(g);
    expect(r).toBe(30); // [2,4,6,8,10] -> 30
    g.destroy();
  });

  it("createKernel accepts a GPUArray input and keepOnGpu", async () => {
    const kernel = await gpu.createKernel({
      shader: `
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
          let i = gid.x;
          if (i < arrayLength(&output)) { output[i] = input[i] * 2.0; }
        }
      `,
      inputs: [{ type: "f32", size: 4 }],
      output: { type: "f32", size: 4 },
    });

    const g = await gpu.upload([1, 2, 3, 4]);
    const r = (await kernel.run(g, { keepOnGpu: true })) as GPUArray;
    expect(r).toBeInstanceOf(GPUArray);
    expect(Array.from(await r.toArray())).toEqual([2, 4, 6, 8]);
    g.destroy();
    r.destroy();

    // CPU input still reads back to a TypedArray
    const cpu = await kernel.run([5, 6, 7, 8]);
    expect(cpu).toBeInstanceOf(Float32Array);
    expect(Array.from(cpu as Float32Array)).toEqual([10, 12, 14, 16]);
  });
});
