import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The bundle is served from /ui by the same process as the proxy and the API, and it
// is built straight into the Python package so the wheel can carry it (PLAN.md phase 5).
export default defineConfig({
  base: "/ui/",
  plugins: [react()],
  build: { outDir: "../src/agentvcr/ui_dist", emptyOutDir: true },
  server: {
    // `npm run dev` serves the UI itself and forwards /api to a running `agentvcr serve`.
    // changeOrigin stays off on purpose: the API compares Origin against the Host it was
    // dialed on, and rewriting one but not the other is exactly what it refuses.
    proxy: { "/api": { target: "http://127.0.0.1:8484", changeOrigin: false } },
  },
});
