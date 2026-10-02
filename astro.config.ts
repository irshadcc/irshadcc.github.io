import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import { unified } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { ptx, tablegen } from "./src/utils/compiler/shikiLangs";
import { mlir } from "./src/utils/mlir/shikiMlir";

// https://astro.build/config
export default defineConfig({
	site: "https://irshadcc.github.io",
	output: "static",
	integrations: [mdx(), sitemap()],
	markdown: {
		// remark/rehype pipeline so we can render $inline$ and $$block$$ math with KaTeX.
		processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] }),
		shikiConfig: {
			themes: { light: "github-light", dark: "github-dark" },
			wrap: false,
			// Grammars Shiki doesn't ship, for the LLVM post: ```tablegen and ```ptx, and for the
			// MLIR post: ```mlir.
			langs: [tablegen, ptx, mlir],
		},
	},
	vite: {
		// Pre-bundle client-only deps at startup; otherwise Vite discovers them on first
		// page load, re-optimizes, and force-reloads the page mid-session.
		optimizeDeps: { include: ["chart.js"] },
	},
});
