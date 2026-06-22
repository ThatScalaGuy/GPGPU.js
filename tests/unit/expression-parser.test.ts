import { describe, it, expect } from "vitest";
import type { DataType } from "../../src/core/types";
import { parseExpression } from "../../src/codegen/expression-parser";
import { emitWGSL } from "../../src/codegen/wgsl-emitter";

function parse(
  fn: ((...args: number[]) => number) | string,
  params?: string[],
  dtype: DataType = "f32"
): string {
  const ir = parseExpression(fn, params);
  return emitWGSL(ir, dtype);
}

// Mirrors how `gpuMap` parses index-aware expressions: `x`/`i`/`len` canonical params
// plus the captured const-array names.
function parseMap(
  fn: ((...args: number[]) => number) | string,
  consts: string[] = [],
  dtype: DataType = "f32"
): string {
  const ir = parseExpression(fn, ["x", "i", "len"], consts);
  return emitWGSL(ir, dtype);
}

describe("Expression Parser + WGSL Emitter", () => {
  describe("literals", () => {
    it("parses integer literals", () => {
      expect(parse("42", ["x"])).toBe("42.0");
    });

    it("parses float literals", () => {
      expect(parse("3.14", ["x"])).toBe("3.14");
    });

    it("parses scientific notation", () => {
      expect(parse("1e5", ["x"])).toBe("100000.0");
    });
  });

  describe("parameters", () => {
    it("parses single parameter", () => {
      expect(parse("x", ["x"])).toBe("x");
    });

    it("parses multiple parameters", () => {
      expect(parse("a", ["a", "b"])).toBe("a");
      expect(parse("b", ["a", "b"])).toBe("b");
    });
  });

  describe("arithmetic", () => {
    it("parses addition", () => {
      expect(parse("x + 1", ["x"])).toBe("(x + 1.0)");
    });

    it("parses multiplication", () => {
      expect(parse("x * 2", ["x"])).toBe("(x * 2.0)");
    });

    it("parses subtraction", () => {
      expect(parse("x - 3", ["x"])).toBe("(x - 3.0)");
    });

    it("parses division", () => {
      expect(parse("x / 4", ["x"])).toBe("(x / 4.0)");
    });

    it("parses modulo", () => {
      expect(parse("x % 2", ["x"])).toBe("(x % 2.0)");
    });

    it("respects operator precedence", () => {
      expect(parse("x + 2 * 3", ["x"])).toBe("(x + (2.0 * 3.0))");
    });

    it("handles parentheses", () => {
      expect(parse("(x + 2) * 3", ["x"])).toBe("((x + 2.0) * 3.0)");
    });

    it("handles complex expressions", () => {
      expect(parse("a + b", ["a", "b"])).toBe("(a + b)");
    });
  });

  describe("unary operators", () => {
    it("parses negation", () => {
      expect(parse("-x", ["x"])).toBe("(-x)");
    });

    it("parses logical not", () => {
      expect(parse("!x", ["x"])).toBe("(!x)");
    });
  });

  describe("comparison operators", () => {
    it("parses less than", () => {
      expect(parse("x < 5", ["x"])).toBe("(x < 5.0)");
    });

    it("parses greater equal", () => {
      expect(parse("x >= 10", ["x"])).toBe("(x >= 10.0)");
    });

    it("maps === to ==", () => {
      expect(parse("x === 0", ["x"])).toBe("(x == 0.0)");
    });

    it("maps !== to !=", () => {
      expect(parse("x !== 1", ["x"])).toBe("(x != 1.0)");
    });
  });

  describe("ternary", () => {
    it("parses ternary expression", () => {
      expect(parse("x > 0 ? x : -x", ["x"])).toBe(
        "select((-x), x, ((x > 0.0)))"
      );
    });
  });

  describe("Math functions", () => {
    it("parses Math.abs", () => {
      expect(parse("Math.abs(x)", ["x"])).toBe("abs(x)");
    });

    it("parses Math.sqrt", () => {
      expect(parse("Math.sqrt(x)", ["x"])).toBe("sqrt(x)");
    });

    it("parses Math.pow", () => {
      expect(parse("Math.pow(x, 2)", ["x"])).toBe("pow(x, 2.0)");
    });

    it("parses Math.min", () => {
      expect(parse("Math.min(a, b)", ["a", "b"])).toBe("min(a, b)");
    });

    it("parses Math.max", () => {
      expect(parse("Math.max(a, b)", ["a", "b"])).toBe("max(a, b)");
    });

    it("parses Math.sin", () => {
      expect(parse("Math.sin(x)", ["x"])).toBe("sin(x)");
    });

    it("parses Math.cos", () => {
      expect(parse("Math.cos(x)", ["x"])).toBe("cos(x)");
    });

    it("parses Math.floor", () => {
      expect(parse("Math.floor(x)", ["x"])).toBe("floor(x)");
    });

    it("parses Math.ceil", () => {
      expect(parse("Math.ceil(x)", ["x"])).toBe("ceil(x)");
    });

    it("parses Math.exp", () => {
      expect(parse("Math.exp(x)", ["x"])).toBe("exp(x)");
    });

    it("parses Math.log", () => {
      expect(parse("Math.log(x)", ["x"])).toBe("log(x)");
    });
  });

  describe("arrow function parsing", () => {
    it("parses simple arrow function", () => {
      expect(parse((x: number) => x * 2)).toBe("(x * 2.0)");
    });

    it("parses arrow function with addition", () => {
      expect(parse((x: number) => x + 1)).toBe("(x + 1.0)");
    });

    it("parses two-param arrow function", () => {
      expect(parse((a: number, b: number) => a + b)).toBe("(a + b)");
    });

    it("parses arrow function with Math", () => {
      expect(parse((x: number) => Math.sqrt(x))).toBe("sqrt(x)");
    });

    it("emits the caller's canonical param names, not the function's source names", () => {
      // A minifier (e.g. esm.sh) renames params: `(a,b)=>a+b` becomes `(n,t)=>n+t`.
      // The shader templates hardcode the canonical names, so emission must use those.
      expect(parse((n: number, t: number) => n + t, ["a", "b"])).toBe("(a + b)");
      expect(parse((s: number, e: number) => s * e, ["a", "b"])).toBe("(a * b)");
      expect(parse((q: number) => q * 2, ["x"])).toBe("(x * 2.0)");
    });
  });

  describe("complex expressions", () => {
    it("parses nested Math calls", () => {
      expect(parse("Math.abs(x - 0.5) * 2", ["x"])).toBe(
        "(abs((x - 0.5)) * 2.0)"
      );
    });

    it("parses chained arithmetic", () => {
      expect(parse("x * x + x", ["x"])).toBe("((x * x) + x)");
    });
  });

  describe("error handling", () => {
    it("rejects unknown identifiers", () => {
      expect(() => parse("foo", ["x"])).toThrow("Unknown identifier 'foo'");
    });

    it("rejects unsupported Math functions", () => {
      expect(() => parse("Math.random()", ["x"])).toThrow("Unsupported Math function");
    });
  });

  describe("typed literal formatting", () => {
    it("emits i32 integer literals without a decimal", () => {
      expect(parse("x + 1", ["x"], "i32")).toBe("(x + 1)");
    });

    it("emits u32 integer literals with a u suffix", () => {
      expect(parse("x + 1", ["x"], "u32")).toBe("(x + 1u)");
    });

    it("still emits f32 literals with a decimal", () => {
      expect(parse("x + 1", ["x"], "f32")).toBe("(x + 1.0)");
    });

    it("rejects non-integer literals for i32", () => {
      expect(() => parse("x + 1.5", ["x"], "i32")).toThrow("Non-integer literal");
    });

    it("rejects non-integer literals for u32", () => {
      expect(() => parse("x + 1.5", ["x"], "u32")).toThrow("Non-integer literal");
    });

    it("rejects negative literals for u32", () => {
      // -5 parses as unary minus over literal 5; u32 disallows unary negation
      expect(() => parse("x + -5", ["x"], "u32")).toThrow("Unary negation");
    });
  });

  describe("bitwise operators", () => {
    it("emits bitwise AND for integer dtypes", () => {
      expect(parse("x & 1", ["x"], "i32")).toBe("(x & 1)");
    });

    it("emits bitwise OR / XOR", () => {
      expect(parse("a | b", ["a", "b"], "u32")).toBe("(a | b)");
      expect(parse("a ^ b", ["a", "b"], "u32")).toBe("(a ^ b)");
    });

    it("emits shifts", () => {
      expect(parse("x << 2", ["x"], "i32")).toBe("(x << 2)");
      expect(parse("x >> 1", ["x"], "u32")).toBe("(x >> 1u)");
    });

    it("emits bitwise NOT", () => {
      expect(parse("~x", ["x"], "i32")).toBe("(~x)");
    });

    it("rejects bitwise operators on f32", () => {
      expect(() => parse("x & 1", ["x"], "f32")).toThrow("requires an integer dtype");
      expect(() => parse("~x", ["x"], "f32")).toThrow("requires an integer dtype");
    });

    it("respects bitwise precedence (& binds tighter than |)", () => {
      expect(parse("x | 1 & 2", ["x"], "i32")).toBe("(x | (1 & 2))");
    });

    it("places shift below additive", () => {
      expect(parse("x + 1 << 2", ["x"], "i32")).toBe("((x + 1) << 2)");
    });

    it("places relational below shift", () => {
      expect(parse("x << 1 < 4", ["x"], "i32")).toBe("((x << 1) < 4)");
    });
  });

  describe("index/length builtins", () => {
    it("emits `i` as a float-cast of idx in f32 arithmetic", () => {
      expect(parseMap("x + i")).toBe("(x + f32(idx))");
      expect(parseMap((x: number, i: number) => x + i)).toBe("(x + f32(idx))");
    });

    it("emits `i` raw inside u32 arithmetic", () => {
      expect(parseMap("x + i", [], "u32")).toBe("(x + idx)");
    });

    it("emits `i` as an i32-cast inside i32 arithmetic", () => {
      expect(parseMap("x + i", [], "i32")).toBe("(x + i32(idx))");
    });

    it("emits `len` as arrayLength, float-cast for f32", () => {
      expect(parseMap("x / len")).toBe("(x / f32(arrayLength(&input)))");
      expect(parseMap("len", [], "u32")).toBe("arrayLength(&input)");
    });

    it("resolves the builtin even when it arrives as a positional arrow param", () => {
      expect(parseMap((x: number, i: number, len: number) => x * len + i)).toBe(
        "((x * f32(arrayLength(&input))) + f32(idx))"
      );
    });
  });

  describe("const-array subscript indexing", () => {
    it("emits a const subscript with a u32 index", () => {
      expect(parseMap("x * hann[i]", ["hann"])).toBe("(x * consts_hann[u32(idx)])");
    });

    it("keeps the index expression in u32 (literal gets a u suffix)", () => {
      expect(parseMap("x * hann[i % 8]", ["hann"])).toBe(
        "(x * consts_hann[u32((idx % 8u))])"
      );
    });

    it("indexes by `len`", () => {
      expect(parseMap("taps[len - 1]", ["taps"])).toBe(
        "consts_taps[u32((arrayLength(&input) - 1u))]"
      );
    });

    it("rejects subscripting an unknown identifier", () => {
      expect(() => parseMap("x * foo[i]", [])).toThrow("Unknown identifier 'foo'");
    });
  });

  describe("bracket tokenization", () => {
    it("tokenizes `[` and `]` instead of throwing 'Unexpected character'", () => {
      // An unmatched bracket now reaches the parser (expecting an array name) rather than
      // dying in the tokenizer.
      expect(() => parseMap("x[", [])).not.toThrow("Unexpected character");
    });
  });
});
