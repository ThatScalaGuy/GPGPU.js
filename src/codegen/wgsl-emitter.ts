import type { IRNode } from "./types";

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
  "&&": "&",
  "||": "|",
};

export function emitWGSL(node: IRNode): string {
  switch (node.kind) {
    case "literal":
      return formatFloat(node.value);

    case "param":
      return node.name;

    case "binary": {
      const wgslOp = OP_MAP[node.op];
      if (!wgslOp) {
        throw new Error(`Unsupported operator '${node.op}' for WGSL emission`);
      }
      return `(${emitWGSL(node.left)} ${wgslOp} ${emitWGSL(node.right)})`;
    }

    case "unary": {
      if (node.op === "-") return `(-${emitWGSL(node.operand)})`;
      if (node.op === "!") return `(!${emitWGSL(node.operand)})`;
      throw new Error(`Unsupported unary operator '${node.op}'`);
    }

    case "call": {
      const wgslFn = MATH_FN_MAP[node.fn];
      if (!wgslFn) {
        throw new Error(`Unsupported function '${node.fn}' for WGSL emission`);
      }
      const args = node.args.map(emitWGSL).join(", ");
      return `${wgslFn}(${args})`;
    }

    case "ternary":
      return `select(${emitWGSL(node.alternate)}, ${emitWGSL(node.consequent)}, (${emitWGSL(node.test)}))`;
  }
}

function formatFloat(value: number): string {
  if (Number.isInteger(value)) {
    return value.toFixed(1);
  }
  return String(value);
}
