import { expect } from "vitest";

/**
 * Assert two numbers are equal within an absolute tolerance.
 *
 * GPU reductions (sum/scan/reduce) reassociate floating-point addition via a
 * parallel tree, so the last bits of a result can differ from a sequential CPU
 * reference even though every individual IEEE-754 op is deterministic. Compare
 * such GPU-vs-CPU floats with this helper instead of strict equality. See
 * docs/numerics.md.
 */
export function expectClose(actual: number, expected: number, opts: { eps?: number } = {}): void {
  const eps = opts.eps ?? 0.05;
  expect(Math.abs(actual - expected)).toBeLessThan(eps);
}
