# Astro internals — working notes

Notes on how Astro turns this repo into the static site in `dist/`. Written against
**Astro 7.3.5** (Vite 8 + Rolldown, the Rust compiler `@astrojs/compiler-rs`) as installed in
`node_modules/`. Source paths below are relative to `node_modules/astro/dist/` unless they say
otherwise; they are compiled JS, so line numbers will drift between versions but the file names
are a good starting point for reading.

## The one-paragraph version

Astro is a build tool that wraps Vite. At build time it (1) syncs content collections into a
data store, (2) turns every file in `src/pages/` into a route, (3) compiles every `.astro` file
into a JS module whose default export is a function that renders HTML, (4) bundles those modules
with Vite into a throwaway "prerender" server bundle, then (5) imports that bundle in Node and,
for every URL it knows about, builds a `Request`, runs the page through the same renderer a
server would use, and writes the `Response` body to `dist/<path>/index.html`. Client-side
`<script>`s and CSS are bundled separately into `dist/_astro/`. Nothing Astro-specific runs in
the browser unless a component ships a `<script>`.

```
                 astro.config.mjs ──► integrations (mdx, sitemap) hook in
                          │
 src/content/** ──► content layer sync ──► .astro/data-store.json      (03)
 src/pages/**   ──► route list (file → pattern, priority)              (01)
 *.astro        ──► compiler-rs ──► JS module + CSS + scripts          (02)
 *.md / *.mdx   ──► markdown processor ──► HTML / JSX module           (04)
                          │
                          ▼
            Vite/Rolldown build, three environments                    (05)
              prerender  → server-side JS that can render every page
              ssr        → skipped here (static site, no server islands)
              client     → the <script>s and CSS that ship to browsers
                          │
                          ▼
    for each route × getStaticPaths(): Request → render → Response
                          │
                          ▼
         dist/**/index.html  +  dist/_astro/*  +  optimised images
```

## Files in this folder

| File | What it covers |
|---|---|
| [01-routing.md](01-routing.md) | How `src/pages/**` becomes routes; dynamic segments, `getStaticPaths`, route priority, output file names |
| [02-astro-components.md](02-astro-components.md) | What the compiler turns a `.astro` file into, and how the runtime renders it (escaping, slots, scoped CSS, scripts, `<head>`) |
| [03-content-collections.md](03-content-collections.md) | `content.config.ts`, the glob loader, schema validation, the data store in `.astro/`, `getCollection()` and `render()` |
| [04-markdown-and-mdx.md](04-markdown-and-mdx.md) | The Markdown processor (unified in this repo), remark/rehype, KaTeX, Shiki, images, and how MDX differs from `.md` |
| [05-build-pipeline.md](05-build-pipeline.md) | `astro build` end to end: config, hooks, Vite environments, page generation, images, what lands in `dist/` |
| [06-dev-server.md](06-dev-server.md) | `astro dev`: Vite dev server, how a request is matched and rendered on demand, HMR |

## How to check these notes yourself

- **Build log.** `npm run build` prints each phase: content sync, "Building static
  entrypoints", "generating static routes" (one line per output file), "generating optimized
  images". Don't pipe it into `head`: the build dies of SIGPIPE half way and leaves `dist/` empty.
- **Compile one component by hand.** This prints exactly what Vite gets for a `.astro` file:
  ```sh
  node --input-type=module -e '
  import { transform, preprocessStyles } from "@astrojs/compiler-rs";
  import fs from "node:fs";
  const f = process.argv[1], src = fs.readFileSync(f, "utf8");
  const pre = await preprocessStyles(src, (css) => ({ code: css }));
  const r = transform(src, { filename: "/" + f, normalizedFilename: "/" + f,
    internalURL: "astro/compiler-runtime", resultScopedSlot: true, preprocessedStyles: pre });
  console.log(r.code);' src/components/Sidenote.astro
  ```
- **Generated state.** `.astro/` (gitignored) holds the content data store, generated types and
  the lazy MDX module map. Deleting it is safe; the next `dev`/`build`/`astro sync` recreates it.
- **Output.** `dist/` is plain files. `npx astro preview` serves it the way GitHub Pages will.
