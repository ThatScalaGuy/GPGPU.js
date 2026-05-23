import { gpu } from "gpgpu.js";

async function main() {
  // 2x3 matrix × 3x2 matrix
  const matA = [1, 2, 3, 4, 5, 6]; // 2 rows, 3 cols
  const matB = [7, 8, 9, 10, 11, 12]; // 3 rows, 2 cols

  const result = await gpu.matmul(matA, matB, {
    rowsA: 2,
    colsA: 3,
    colsB: 2,
  });

  console.log("matmul result:", result);
  // Expected: [58, 64, 139, 154]
  // [1*7+2*9+3*11, 1*8+2*10+3*12]  = [58, 64]
  // [4*7+5*9+6*11, 4*8+5*10+6*12]  = [139, 154]

  gpu.destroy();
}

main();
