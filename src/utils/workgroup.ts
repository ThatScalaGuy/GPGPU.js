import { DEFAULT_WORKGROUP_SIZE } from "../core/types";

export function computeWorkgroupCount(
  totalItems: number,
  workgroupSize: number = DEFAULT_WORKGROUP_SIZE
): number {
  return Math.ceil(totalItems / workgroupSize);
}
