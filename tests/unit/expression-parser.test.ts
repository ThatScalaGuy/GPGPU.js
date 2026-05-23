import { describe, it, expect } from "vitest";
import { parseExpression } from "../../src/codegen/expression-parser";
import { emitWGSL } from "../../src/codegen/wgsl-emitter";

function parse(fn: ((...args: number[]) => number) | string, params?: string[]): string {
  const ir = parseExpression(fn, params);
  return emitWGSL(ir);
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
});
