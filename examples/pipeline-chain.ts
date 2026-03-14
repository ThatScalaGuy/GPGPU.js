import { gpu } from "gpgpu.js";

async function main() {
  const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

  // Chain operations — data stays on GPU between steps
  const result = await gpu
    .pipeline()
    .map((x) => x * 2) // double each element
    .map((x) => x + 1) // add 1 to each
    .reduce((a, b) => a + b, 0); // sum everything

  const total = await result.run(data);
  console.log("pipeline result:", total);
  // Expected: sum of [3, 5, 7, 9, 11, 13, 15, 17, 19, 21] = 120

  gpu.destroy();
}

main();
