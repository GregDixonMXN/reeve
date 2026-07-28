import { mkdir, readdir, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(scriptDir, "..", "dist");

await mkdir(distDir, { recursive: true });

for (const entry of await readdir(distDir, { withFileTypes: true })) {
  if (entry.name === ".gitkeep") {
    continue;
  }

  await rm(join(distDir, entry.name), { recursive: true, force: true });
}

await writeFile(join(distDir, ".gitkeep"), "");
