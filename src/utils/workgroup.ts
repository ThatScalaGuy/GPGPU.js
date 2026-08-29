import { DEFAULT_WORKGROUP_SIZE } from "../core/types";

// WebGPU's default maxComputeWorkgroupsPerDimension. We stick to the spec default
// (not the adapter's possibly-higher limit) so dispatches stay portable.
export const MAX_WORKGROUPS_PER_DIM = 65535;

export function computeWorkgroupCount(
  totalItems: number,
  workgroupSize: number = DEFAULT_WORKGROUP_SIZE
): number {
  return Math.ceil(totalItems / workgroupSize);
}

/**
 * Workgroup grid for a per-element dispatch. A single dimension caps out at
 * 65535 workgroups (~4.19M elements at size 64) — beyond that the dispatch is a
 * WebGPU validation error — so larger totals split into a 2-D rectangle. The
 * matching shaders linearize with `gid.y * (nwg.x * workgroupSize) + gid.x` and
 * bounds-check away the rectangle's overshoot.
 */
export function computeWorkgroupGrid(
  totalItems: number,
  workgroupSize: number = DEFAULT_WORKGROUP_SIZE
): [number, number] {
  const total = computeWorkgroupCount(totalItems, workgroupSize);
  if (total <= MAX_WORKGROUPS_PER_DIM) return [total, 1];
  return [MAX_WORKGROUPS_PER_DIM, Math.ceil(total / MAX_WORKGROUPS_PER_DIM)];
}
