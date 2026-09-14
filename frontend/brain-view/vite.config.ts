import { defineConfig } from "vite";

export default defineConfig({
  build: {
    lib: {
      entry: "src/brain-view.ts",
      formats: ["es"],
      fileName: () => "brain-view.js",
    },
    outDir: "../../src/flybrain_interface/panel/static",
    emptyOutDir: false,
    sourcemap: false,
  },
});
