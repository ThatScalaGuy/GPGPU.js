import { gpu } from "gpgpu.js";

async function main() {
  // Add two arrays
  const a = [1, 2, 3, 4, 5];
  const b = [10, 20, 30, 40, 50];
  const sum = await gpu.add(a, b);
  console.log("add:", sum); // Float32Array [11, 22, 33, 44, 55]

  // Add scalar to every element
  const shifted = await gpu.add(a, 100);
  console.log("add scalar:", shifted); // Float32Array [101, 102, 103, 104, 105]

  // Multiply arrays
  const product = await gpu.multiply(a, b);
  console.log("multiply:", product); // Float32Array [10, 40, 90, 160, 250]

  // Scale by a constant
  const scaled = await gpu.multiply(a, 3);
  console.log("scale:", scaled); // Float32Array [3, 6, 9, 12, 15]

  gpu.destroy();
}

main();
