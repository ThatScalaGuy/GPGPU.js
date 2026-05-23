import type { IRNode, Token, TokenType } from "./types";

const MATH_FUNCTIONS = new Set([
  "abs", "sqrt", "pow", "min", "max",
  "floor", "ceil", "round",
  "sin", "cos", "tan",
  "exp", "log", "sign", "clamp",
]);

const OP_CHARS = new Set(["+", "-", "*", "/", "%", "<", ">", "=", "!", "&", "|"]);
const MULTI_CHAR_OPS = new Set(["<=", ">=", "==", "!=", "===", "!==", "&&", "||"]);

function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  while (i < source.length) {
    const ch = source[i];

    if (ch === " " || ch === "\t" || ch === "\n" || ch === "\r") {
      i++;
      continue;
    }

    // Numbers
    if (ch >= "0" && ch <= "9" || (ch === "." && i + 1 < source.length && source[i + 1] >= "0" && source[i + 1] <= "9")) {
      let num = "";
      while (i < source.length && ((source[i] >= "0" && source[i] <= "9") || source[i] === ".")) {
        num += source[i++];
      }
      // Handle scientific notation
      if (i < source.length && (source[i] === "e" || source[i] === "E")) {
        num += source[i++];
        if (i < source.length && (source[i] === "+" || source[i] === "-")) {
          num += source[i++];
        }
        while (i < source.length && source[i] >= "0" && source[i] <= "9") {
          num += source[i++];
        }
      }
      tokens.push({ type: "number", value: num });
      continue;
    }

    // Identifiers
    if ((ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z") || ch === "_" || ch === "$") {
      let ident = "";
      while (
        i < source.length &&
        ((source[i] >= "a" && source[i] <= "z") ||
          (source[i] >= "A" && source[i] <= "Z") ||
          (source[i] >= "0" && source[i] <= "9") ||
          source[i] === "_" || source[i] === "$")
      ) {
        ident += source[i++];
      }
      tokens.push({ type: "ident", value: ident });
      continue;
    }

    // Operators (multi-char first)
    if (OP_CHARS.has(ch)) {
      let op = ch;
      const next2 = source.slice(i, i + 3);
      const next1 = source.slice(i, i + 2);
      if (MULTI_CHAR_OPS.has(next2)) {
        op = next2;
        i += 3;
      } else if (MULTI_CHAR_OPS.has(next1)) {
        op = next1;
        i += 2;
      } else {
        i++;
      }
      tokens.push({ type: "op", value: op });
      continue;
    }

    if (ch === "(" || ch === ")") {
      tokens.push({ type: "paren", value: ch });
      i++;
      continue;
    }

    if (ch === ",") {
      tokens.push({ type: "comma", value: ch });
      i++;
      continue;
    }

    if (ch === ".") {
      tokens.push({ type: "dot", value: ch });
      i++;
      continue;
    }

    if (ch === "?") {
      tokens.push({ type: "question", value: ch });
      i++;
      continue;
    }

    if (ch === ":") {
      tokens.push({ type: "colon", value: ch });
      i++;
      continue;
    }

    throw new Error(`Unexpected character '${ch}' at position ${i}`);
  }

  tokens.push({ type: "eof", value: "" });
  return tokens;
}

class Parser {
  private tokens: Token[];
  private pos = 0;
  private params: string[];

  constructor(tokens: Token[], params: string[]) {
    this.tokens = tokens;
    this.params = params;
  }

  private peek(): Token {
    return this.tokens[this.pos];
  }

  private advance(): Token {
    return this.tokens[this.pos++];
  }

  private expect(type: TokenType, value?: string): Token {
    const tok = this.advance();
    if (tok.type !== type || (value !== undefined && tok.value !== value)) {
      throw new Error(
        `Expected ${type}${value ? ` '${value}'` : ""}, got ${tok.type} '${tok.value}'`
      );
    }
    return tok;
  }

  parse(): IRNode {
    const node = this.parseTernary();
    if (this.peek().type !== "eof") {
      throw new Error(`Unexpected token '${this.peek().value}' after expression`);
    }
    return node;
  }

  private parseTernary(): IRNode {
    let node = this.parseOr();
    if (this.peek().type === "question") {
      this.advance(); // consume ?
      const consequent = this.parseTernary();
      this.expect("colon");
      const alternate = this.parseTernary();
      return { kind: "ternary", test: node, consequent, alternate };
    }
    return node;
  }

  private parseOr(): IRNode {
    let left = this.parseAnd();
    while (this.peek().type === "op" && this.peek().value === "||") {
      const op = this.advance().value;
      const right = this.parseAnd();
      left = { kind: "binary", op, left, right };
    }
    return left;
  }

  private parseAnd(): IRNode {
    let left = this.parseComparison();
    while (this.peek().type === "op" && this.peek().value === "&&") {
      const op = this.advance().value;
      const right = this.parseComparison();
      left = { kind: "binary", op, left, right };
    }
    return left;
  }

  private parseComparison(): IRNode {
    let left = this.parseAdditive();
    const compOps = new Set(["<", ">", "<=", ">=", "==", "!=", "===", "!=="]);
    while (this.peek().type === "op" && compOps.has(this.peek().value)) {
      const op = this.advance().value;
      const right = this.parseAdditive();
      left = { kind: "binary", op, left, right };
    }
    return left;
  }

  private parseAdditive(): IRNode {
    let left = this.parseMultiplicative();
    while (
      this.peek().type === "op" &&
      (this.peek().value === "+" || this.peek().value === "-")
    ) {
      const op = this.advance().value;
      const right = this.parseMultiplicative();
      left = { kind: "binary", op, left, right };
    }
    return left;
  }

  private parseMultiplicative(): IRNode {
    let left = this.parseUnary();
    while (
      this.peek().type === "op" &&
      (this.peek().value === "*" || this.peek().value === "/" || this.peek().value === "%")
    ) {
      const op = this.advance().value;
      const right = this.parseUnary();
      left = { kind: "binary", op, left, right };
    }
    return left;
  }

  private parseUnary(): IRNode {
    if (
      this.peek().type === "op" &&
      (this.peek().value === "-" || this.peek().value === "!")
    ) {
      const op = this.advance().value;
      const operand = this.parseUnary();
      return { kind: "unary", op, operand };
    }
    return this.parsePrimary();
  }

  private parsePrimary(): IRNode {
    const tok = this.peek();

    // Number literal
    if (tok.type === "number") {
      this.advance();
      return { kind: "literal", value: parseFloat(tok.value) };
    }

    // Parenthesized expression
    if (tok.type === "paren" && tok.value === "(") {
      this.advance();
      const node = this.parseTernary();
      this.expect("paren", ")");
      return node;
    }

    // Identifier: parameter, Math.fn, or standalone function
    if (tok.type === "ident") {
      this.advance();

      // Math.fn(...)
      if (tok.value === "Math" && this.peek().type === "dot") {
        this.advance(); // consume dot
        const fnTok = this.expect("ident");
        if (!MATH_FUNCTIONS.has(fnTok.value)) {
          throw new Error(
            `Unsupported Math function 'Math.${fnTok.value}'. Supported: ${[...MATH_FUNCTIONS].join(", ")}`
          );
        }
        this.expect("paren", "(");
        const args = this.parseArgList();
        this.expect("paren", ")");
        return { kind: "call", fn: fnTok.value, args };
      }

      // Parameter reference
      const paramIndex = this.params.indexOf(tok.value);
      if (paramIndex !== -1) {
        return { kind: "param", name: tok.value, index: paramIndex };
      }

      throw new Error(
        `Unknown identifier '${tok.value}'. Expected a parameter name (${this.params.join(", ")}) or Math.fn()`
      );
    }

    throw new Error(`Unexpected token '${tok.value}' (${tok.type})`);
  }

  private parseArgList(): IRNode[] {
    const args: IRNode[] = [];
    if (this.peek().type === "paren" && this.peek().value === ")") {
      return args;
    }
    args.push(this.parseTernary());
    while (this.peek().type === "comma") {
      this.advance();
      args.push(this.parseTernary());
    }
    return args;
  }
}

function extractArrowParams(source: string): { params: string[]; body: string } {
  // Remove leading/trailing whitespace
  source = source.trim();

  // Match: (a, b) => expr  or  a => expr  or  (a, b) => { return expr; }
  const arrowIndex = source.indexOf("=>");
  if (arrowIndex === -1) {
    // Not an arrow function — treat as raw expression with implicit params
    throw new Error(
      "Expected an arrow function like 'x => x * 2' or '(a, b) => a + b'"
    );
  }

  const paramsPart = source.slice(0, arrowIndex).trim();
  let body = source.slice(arrowIndex + 2).trim();

  // Handle block body: { return expr; }
  if (body.startsWith("{")) {
    const returnMatch = body.match(/^\{\s*return\s+([\s\S]*?)\s*;?\s*\}$/);
    if (!returnMatch) {
      throw new Error(
        "Block body arrow functions must contain a single return statement"
      );
    }
    body = returnMatch[1];
  }

  // Parse params
  let paramsStr = paramsPart;
  if (paramsStr.startsWith("(") && paramsStr.endsWith(")")) {
    paramsStr = paramsStr.slice(1, -1);
  }
  // Also strip "function" keyword prefix if toString() gave us that
  paramsStr = paramsStr.replace(/^function\s*\(/, "").replace(/\)$/, "");

  const params = paramsStr
    .split(",")
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  return { params, body };
}

export function parseExpression(
  fn: ((...args: number[]) => number) | string,
  paramNames?: string[]
): IRNode {
  let body: string;
  let params: string[];

  if (typeof fn === "string") {
    body = fn;
    params = paramNames ?? ["x"];
  } else {
    const source = fn.toString();
    const extracted = extractArrowParams(source);
    body = extracted.body;
    params = extracted.params;
  }

  const tokens = tokenize(body);
  const parser = new Parser(tokens, params);
  return parser.parse();
}

export { tokenize, extractArrowParams };
