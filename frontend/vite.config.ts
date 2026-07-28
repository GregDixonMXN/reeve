import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";
import solidPlugin from "vite-plugin-solid";

export default defineConfig({
  plugins: [solidPlugin(), tailwindcss()],
  build: {
    target: "esnext",
    outDir: "dist",
    // Keep the tracked placeholder required by Go's frontend/dist embed.
    // The prebuild script removes generated output without removing .gitkeep.
    emptyOutDir: false,
  },
  server: {
    strictPort: true,
  },
});
