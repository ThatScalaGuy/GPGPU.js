import type { IRNode } from "./types";
import type { DataType } from "../core/types";

const MATH_FN_MAP: Record<string, string> = {
  abs: "abs",
  sqrt: "sqrt",
  pow: "pow",
  min: "min",
  max: "max",
  floor: "floor",
  ceil: "ceil",
  round: "round",
  sin: "sin",
  cos: "cos",
  tan: "tan",
  exp: "exp",
  log: "log",
  sign: "sign",
  clamp: "clamp",
};

const OP_MAP: Record<string, string> = {
  "+": "+",
  "-": "-",
  "*": "*",
  "/": "/",
  "%": "%",
  "<": "<",
  ">": ">",
  "<=": "<=",
  ">=": ">=",
  "==": "==",
  "!=": "!=",
  "===": "==",
  "!==": "!=",
  // JS logical operators are emitted as WGSL bitwise &/| (truthiness-correct only for
  // 0/1 operands). Pre-existing behaviour, kept for back-compat; distinct from the real
  // bitwise operators below, which the parser places at their own precedence levels.
  "&&": "&",
  "||": "|",
  "&": "&",
  "|": "|",
  "^": "^",
  "<<": "<<",
  ">>": ">>",
};

// Operators that are only legal on integer types in WGSL.
const BITWISE_OPS = new Set(["&", "|", "^", "<<", ">>"]);

export function emitWGSL(node: IRNode, dtype: DataType = "f32"): string {
  switch (node.kind) {
    case "literal":
      return formatLiteral(node.value, dtype);

    case "param":
      return node.name;

    case "binary": {
      const wgslOp = OP_MAP[node.op];
      if (!wgslOp) {
        throw new Error(`Unsupported operator '${node.op}' for WGSL emission`);
      }
      if (BITWISE_OPS.has(node.op) && dtype === "f32") {
        throw new Error(
          `Bitwise operator '${node.op}' requires an integer dtype (i32/u32), got f32`
        );
      }
      return `(${emitWGSL(node.left, dtype)} ${wgslOp} ${emitWGSL(node.right, dtype)})`;
    }

    case "unary": {
      if (node.op === "-") {
        if (dtype === "u32") {
          throw new Error("Unary negation '-' is not valid for u32");
        }
        return `(-${emitWGSL(node.operand, dtype)})`;
      }
      if (node.op === "!") return `(!${emitWGSL(node.operand, dtype)})`;
      if (node.op === "~") {
        if (dtype === "f32") {
          throw new Error("Bitwise operator '~' requires an integer dtype (i32/u32), got f32");
        }
        return `(~${emitWGSL(node.operand, dtype)})`;
      }
      throw new Error(`Unsupported unary operator '${node.op}'`);
    }

    case "call": {
      const wgslFn = MATH_FN_MAP[node.fn];
      if (!wgslFn) {
        throw new Error(`Unsupported function '${node.fn}' for WGSL emission`);
      }
      const args = node.args.map((arg) => emitWGSL(arg, dtype)).join(", ");
      return `${wgslFn}(${args})`;
    }

    case "ternary":
      return `select(${emitWGSL(node.alternate, dtype)}, ${emitWGSL(node.consequent, dtype)}, (${emitWGSL(node.test, dtype)}))`;
  }
}

export function formatLiteral(value: number, dtype: DataType = "f32"): string {
  if (dtype === "f32") {
    return Number.isInteger(value) ? value.toFixed(1) : String(value);
  }
  if (!Number.isInteger(value)) {
    throw new Error(`Non-integer literal ${value} is not valid for ${dtype}`);
  }
  if (dtype === "u32") {
    if (value < 0) {
      throw new Error(`Negative literal ${value} is not valid for u32`);
    }
    return `${value}u`;
  }
  return String(value); // i32
}
