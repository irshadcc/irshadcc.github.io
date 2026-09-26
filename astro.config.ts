import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import { unified } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

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
		},
	},
});
