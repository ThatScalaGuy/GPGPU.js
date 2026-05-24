// Shared library import + small helpers used across demo components.
import { gpu } from "@thatscalaguy/gpgpu.js";

export { gpu };

/** Parse a comma/space/newline-separated string into a number[]. */
export function parseNums(str) {
  return str
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number);
}

/** Format a numeric array (or single number) for display, rounding noise away. */
export function fmt(value, decimals = 4) {
  const round = (n) => {
    const r = Number(n.toFixed(decimals));
    return Object.is(r, -0) ? 0 : r;
  };
  if (typeof value === "number") return String(round(value));
  return `[${Array.from(value, round).join(", ")}]`;
}

/** Await an async fn, returning its result alongside elapsed milliseconds. */
export async function timeit(fn) {
  const start = performance.now();
  const result = await fn();
  return { result, ms: performance.now() - start };
}
