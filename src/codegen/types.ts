export type IRNode =
  | { kind: "literal"; value: number }
  | { kind: "param"; name: string; index: number }
  | { kind: "builtin"; name: "i" | "len" }
  | { kind: "index"; array: string; index: IRNode }
  | { kind: "binary"; op: string; left: IRNode; right: IRNode }
  | { kind: "unary"; op: string; operand: IRNode }
  | { kind: "call"; fn: string; args: IRNode[] }
  | { kind: "ternary"; test: IRNode; consequent: IRNode; alternate: IRNode };

export type TokenType =
  | "number"
  | "ident"
  | "op"
  | "paren"
  | "comma"
  | "dot"
  | "question"
  | "colon"
  | "lbracket"
  | "rbracket"
  | "eof";

export interface Token {
  type: TokenType;
  value: string;
}
