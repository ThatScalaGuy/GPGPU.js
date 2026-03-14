export type IRNode =
  | { kind: "literal"; value: number }
  | { kind: "param"; name: string; index: number }
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
  | "eof";

export interface Token {
  type: TokenType;
  value: string;
}
