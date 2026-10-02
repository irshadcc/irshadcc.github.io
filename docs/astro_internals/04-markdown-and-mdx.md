# 04 — Markdown and MDX

Source: `core/config/schemas/base.js` (`markdown` options), `vite-plugin-markdown/`,
`@astrojs/markdown-remark`, `@astrojs/markdown-satteri`, `@astrojs/mdx/dist/`.

## 1. Markdown processors: satteri (default) vs unified (this repo)

Astro 7 made the Markdown engine pluggable through `markdown.processor`, an object with
`createRenderer()` (and `createMdxRenderer()` for MDX). Two ship with Astro:

| Processor | Package | Notes |
|---|---|---|
| `satteri()` | `@astrojs/markdown-satteri` | the **default**; a native (Rust) Markdown engine. GFM and smart punctuation on by default. **Does not run remark/rehype plugins.** |
| `unified({...})` | `@astrojs/markdown-remark` | the classic JS pipeline built on the unified ecosystem: remark (Markdown AST) → rehype (HTML AST). Takes `remarkPlugins` / `rehypePlugins`. |

This repo needs a remark and a rehype plugin for math, so `astro.config.mjs` switches to unified:

```js
markdown: {
	processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] }),
	shikiConfig: { themes: { light: "github-light", dark: "github-dark" }, wrap: false },
},
```

## 2. The unified pipeline, step by step

For each `.md` body (and, with extra steps, each `.mdx` body), in the order
`createMarkdownProcessor()` in `@astrojs/markdown-remark/dist/index.js` sets up the plugins:

```
Markdown text
  │  remark-parse               → mdast (Markdown syntax tree: heading, paragraph, code, …)
  │  remark-gfm                 → tables, strikethrough, task lists, autolinks
  │  remark-smartypants         → “curly quotes”, en/em dashes
  │  your remarkPlugins         → here: remark-math — $…$ / $$…$$ become math nodes, not text
  │  remarkCollectImages        → records local/remote image paths for the image pipeline
  ▼
  │  remark-rehype              → hast (HTML syntax tree); raw HTML kept as `raw` nodes
  │  rehypeShiki                → code blocks → <pre class="astro-code …"> with token spans
  │  your rehypePlugins         → here: rehype-katex — math nodes → <span class="katex">…
  │  rehypeImages               → local <img> → optimised-image placeholders
  │  rehypeHeadingIds           → heading ids (github-slugger) + the `headings` list
  │  rehype-raw                 → parses the raw HTML you wrote in the Markdown into real nodes
  ▼
  │  rehype-stringify           → HTML string
```

Your plugins always run in the middle: remark plugins after GFM/smartypants, and rehype plugins
after Shiki but before Astro's image and heading-id plugins.

Plugins act on **trees, not text**. A remark plugin sees Markdown structure (it can tell a `$` in
a code span from a `$` in prose), and a rehype plugin sees HTML elements. That is why `remark-math`
has to come before `remark-rehype` and `rehype-katex` after it.

**KaTeX.** `rehype-katex` renders the math to HTML at build time. The browser needs only the KaTeX
CSS and fonts (the `dist/_astro/KaTeX_*.woff2` files). No JavaScript runs to display math.
`dist/blog/strided-tensors/index.html` contains 24 pre-rendered `class="katex"` spans.

**Shiki.** Code blocks are highlighted at build time with TextMate grammars (the same ones VS Code
uses). With two `themes`, each token gets the light colour inline plus the dark one as a CSS
variable, e.g. `style="color:#24292e;--shiki-dark:#e1e4e8"`, on a
`<pre class="astro-code astro-code-themes github-light github-dark">`. Site CSS switches to the
`--shiki-dark*` values in dark mode. No highlighter ships to the browser.

**Heading ids.** Every heading gets an `id` from its text (`## Row-major order` →
`id="row-major-order"`), and `render(entry)` returns them as `headings`, e.g. for a table of
contents.

## 3. `.md` files

Rendered once, during content sync, to an HTML string stored in the data store (03 §4). When the
page renders, `<Content />` outputs that string unescaped. Markdown can contain raw HTML, but it
can't use components.

A `.md` file placed directly in `src/pages/` also works: it becomes a route, and its frontmatter
`layout:` names an `.astro` layout to wrap it. This repo keeps posts in a collection instead.

## 4. `.mdx` files

MDX = Markdown + JSX + ES modules. `@astrojs/mdx` registers:

- `.mdx` as a **page extension** and a **content entry type** (03 §4, rendered later);
- a JSX **renderer** (`astro:jsx`) so JSX in MDX can render Astro components;
- a Vite plugin, `vitePluginMdx`, whose `transform` hook handles every `*.mdx` id.

That `transform` hook:

1. Strips the frontmatter (`safeParseFrontmatter`).
2. Runs the processor's MDX renderer. With `extendMarkdownConfig: true` (the default) MDX uses the
   same `markdown.processor` as `.md`, so **remark-math, rehype-katex and Shiki apply to `.mdx`
   too**.
3. The MDX compiler parses the body into mdast *with* JSX and ESM nodes, runs the remark/rehype
   plugins, and then — unlike `.md` — does **not** stringify to HTML. It generates a **JS
   module**: `import` lines are kept, every Markdown element becomes a JSX call, and your JSX
   stays JSX. Roughly:

   ```mdx
   import Sidenote from "../../components/Sidenote.astro";
   Strides are cheap.<Sidenote>No data is copied.</Sidenote>
   ```
   becomes something like
   ```js
   import Sidenote from "../../components/Sidenote.astro";
   export default function MDXContent(props) {
     return jsx("p", { children: ["Strides are cheap.", jsx(Sidenote, { children: "No data is copied." })] });
   }
   ```
4. Returns the module to Vite, which bundles it like any other source file. The imported
   `Sidenote.astro` is compiled by the Astro plugin (02) as usual.

At render time Astro's JSX runtime walks those `jsx()` calls: HTML elements become HTML strings,
and Astro components are rendered with `renderComponent` exactly as in a `.astro` template. The
output is still static HTML. Any `<script>` inside an imported component is bundled for the client
and injected into the page head (02 §4, §6).

**Images in MDX.** `import rowMajor from "../../assets/strided-tensors/row-major-6x6.png"` gives
image *metadata* (src, width, height), not a URL string. Rendered through Astro's image pipeline,
the build then writes an optimised `.webp` via `sharp`. That's the "generating optimized images"
step in the build log, and why the built page contains
`<img src="/_astro/row-major-6x6.<hash>.webp" width="395" height="478" loading="lazy" decoding="async">`.
Local images referenced with Markdown syntax `![alt](./x.png)` go through the same pipeline.

## 5. Where each feature happens

| Feature | Stage | Runs in browser? |
|---|---|---|
| Frontmatter validation | content sync (Zod) | no |
| Markdown → HTML (`.md`) | content sync | no |
| MDX → JS module | Vite transform (build) | no |
| Math | rehype-katex at build | only CSS + fonts |
| Syntax highlighting | Shiki at build | only CSS |
| Components in MDX (`Sidenote`, charts) | server render at build | only if they have a `<script>` |
| Images | `sharp` at build | no |
