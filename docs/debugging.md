# Debugging shader compile errors

When a shader fails to compile, GPGPU.js surfaces the real WGSL diagnostic as a
**code frame**: the generated WGSL around the failing line, numbered, with a caret
`^` under the offending column.

## Hand-written kernels (`createKernel`)

This is where the frame helps most — a typo in your own WGSL is pinpointed exactly.
Given a kernel whose shader references an undeclared `undefined_value`:

```
Shader compilation failed (custom-kernel):
  5:17: unresolved value 'undefined_value'
    3 | @compute @workgroup_size(64)
    4 | fn main(@builtin(global_invocation_id) gid: vec3u) {
  > 5 |   data[gid.x] = undefined_value;
      |                 ^
    6 | }
```

The `>` marks the failing line and the caret points at the column reported by the
WebGPU `getCompilationInfo` diagnostic.

## Codegen ops (`gpu.map`, `gpu.zip`, `gpu.scan`, `gpu.filter`)

Expressions are validated as they are compiled to WGSL, so most mistakes surface
*before* any shader is built, as a clear codegen error:

```js
await gpu.map(new Float32Array([1, 2, 3]), "x | 1");
// Error: Bitwise operator '|' requires an integer dtype (i32/u32), got f32

await gpu.map(new Float32Array([1, 2, 3]), "oops(x)");
// Error: Unknown identifier 'oops'. Expected a parameter name (x, i, len) or Math.fn()
```

If a generated shader ever does fail at the WGSL level, the error additionally
appends a **`from expression:`** note echoing your original JS, so you can connect
the WGSL diagnostic back to what you wrote:

```
Shader compilation failed (map-f32):
  <line>:<col>: <wgsl diagnostic>
  ... code frame ...

  from expression: x => /* your expression */
```

> Note: the line/column come from the WebGPU `getCompilationInfo` diagnostic and
> point into the **generated** WGSL, not your JS. Full per-sub-expression source
> maps (mapping a WGSL line back to the specific JS sub-expression) are out of scope.
