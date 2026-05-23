import { gpu } from "gpgpu.js";

async function main() {
  const data = [1, 2, 3, 4, 5, 6, 7, 8];

  // Map with arrow function — auto-compiled to WGSL
  const doubled = await gpu.map(data, (x) => x * 2);
  console.log("doubled:", doubled);

  // Map with more complex expression
  const transformed = await gpu.map(data, (x) => x * x + 1);
  console.log("x² + 1:", transformed);

  // Map with Math functions
  const roots = await gpu.map(data, (x) => Math.sqrt(x));
  console.log("sqrt:", roots);

  // Map with string expression (minifier-safe)
  const cubed = await gpu.map(data, "x * x * x");
  console.log("cubed:", cubed);

  // Reduce to sum
  const total = await gpu.sum(data);
  console.log("sum:", total); // 36

  // Reduce with custom function
  const maxVal = await gpu.max(data);
  console.log("max:", maxVal); // 8

  gpu.destroy();
}

main();
