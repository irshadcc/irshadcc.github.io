# 05 — `astro build`, end to end

Source: `core/build/index.js` (`AstroBuilder`), `core/build/static-build.js`,
`core/build/generate.js`, `core/build/default-prerenderer.js`, `core/build/plugins/`,
`integrations/hooks.js`.

The build log of this repo maps onto the steps below; timestamps removed:

```
[content] Syncing content                       ← step 3
[types] Generated                               ← step 3
[build] output: "static" / mode: "static"       ← step 4
[build] Collecting build info...                ← step 4
[build] Building static entrypoints...          ← step 5
[vite] ✓ built in 636ms                         ← prerender environment
[vite] ✓ built in 36ms                          ← client environment
[build] Rearranging server assets...            ← step 6
 generating static routes                       ← step 7
  ├─ /404.html
  ├─ /blog/strided-tensors/index.html  …
 generating optimized images                    ← step 8
[@astrojs/sitemap] `sitemap-index.xml` created  ← step 9
[build] 17 page(s) built in 1.72s
```

## 1. Resolve config

`resolveConfig()` loads `astro.config.mjs` (itself bundled with Vite so it can be TypeScript and
use imports), validates it against a Zod schema (`core/config/schemas/`) and fills in defaults:
`base: "/"`, `trailingSlash: "ignore"`, `build.format: "directory"`,
`build.inlineStylesheets: "auto"`, `compressHTML: "jsx"`, `scopedStyleStrategy: "attribute"`,
`markdown.processor: satteri()` (overridden here), and so on.

## 2. Integrations: `astro:config:setup` / `astro:config:done`

Integrations are objects with named **hooks** that Astro calls at fixed points
(`integrations/hooks.js`). In this repo:

- `@astrojs/mdx` in `astro:config:setup` adds the `.mdx` page extension, content entry type, JSX
  renderer and Vite plugin (04 §4); in `astro:config:done` it picks the Markdown processor.
- `@astrojs/sitemap` waits for `astro:build:done`, then writes the sitemap from the list of built
  pages.

Other hooks: `astro:routes:resolved`, `astro:build:setup` (can modify the Vite config),
`astro:build:start`, `astro:build:generated`, `astro:server:setup` (dev).

After `config:setup`, Astro decides `buildOutput`. Every route here is prerendered (the default
with `output: "static"`), so it is `"static"`. If any route opted out with
`export const prerender = false`, the build would need an adapter (Node, Netlify, …) and would
also produce a server.

## 3. Routes and content sync

- `createRoutesList()` builds the sorted route list (01).
- `syncInternal()` runs the content layer (03) and writes the generated types in `.astro/`.

## 4. Collect page data

`collectPagesData()` makes one `PageBuildData` per route (component path, route data, and later
the CSS and scripts it needs). `viteBuild()` then **empties `dist/`** (keeping `.git`) unless
`vite.build.emptyOutDir` is `false`.

## 5. Vite build with three environments

Vite 8 builds several **environments** from one config in a single `createBuilder()` run. Each
environment has its own module graph, plugins and output. Astro's `buildApp()` in
`static-build.js` builds them in order:

1. **`prerender`** — a Node-targeted bundle of everything needed to render pages: every page
   module, the compiled `.astro` components, compiled MDX, the content runtime and the
   `astro:content` data. Its entry is a generated `prerender-entry.*.mjs` that exports an `app`
   (the same `App` class a production server would use). It goes to a temporary directory, not
   to `dist/`.
   While bundling, Astro's build plugins (`core/build/plugins/`) record for each page which CSS
   modules and which `<script>`s its module graph reaches.
2. **`ssr`** — only built when `buildOutput === "server"` or the site uses server islands. Skipped
   here.
3. **`client`** — the browser bundle. Its inputs are collected *from the prerender build*:
   hydrated framework components, renderer client entry points, and every discovered component
   `<script>` (`getClientInput()`). For this site that is just
   `LineChart.astro?astro&type=script&index=0&lang.ts` (plus Chart.js, which it imports), so
   `dist/_astro/` contains a single JS file. If nothing needs client JS, a no-op module is built.

CSS is emitted during these builds as hashed files in `dist/_astro/` (`Layout.<hash>.css`,
`logical-topology.<hash>.css`), or marked for inlining when small (02 §5).

Bundling uses **Rolldown** (Rust), which is why warnings print in Rolldown's format.

## 6. Post-processing chunks, moving assets

After the Vite builds (`astro:build-generate` plugin, `buildApp` post hook):

- **manifest injection**: some chunks contain placeholders — the serialized manifest (routes,
  per-page CSS/script lists, config) and content-asset link placeholders — that can only be
  filled once all environments are built. Astro patches those files on disk.
- **"Rearranging server assets"**: assets emitted by the prerender bundle (e.g. images imported
  by components) are moved from the temp directory into `dist/`.

## 7. Generate the pages

`generatePages()` in `generate.js`:

1. **Imports the prerender bundle**: the default prerenderer's `setup()` does
   `await import("<tmp>/prerender-entry.<hash>.mjs")` and takes its `app`. From here on, your
   page code is running in the same Node process as the build.
2. **Lists every path**: `StaticPaths.getAll()` (01 §4) — each fixed route's pathname, plus one
   pathname per `getStaticPaths()` entry. Duplicates are dropped (first wins).
3. **Renders each path** (`generatePathWithPrerenderer` → `renderPath`), one at a time
   (`build.concurrency` defaults to 1):
   - builds a URL (`https://irshadcc.github.io/blog/strided-tensors/`) and a `Request` with
     `isPrerendered: true`;
   - `app` matches the request to the route, runs middleware (none here), calls `getProps()` to
     get the matching `getStaticPaths` entry's props (01 §4), renders the page component tree to
     HTML (02 §3), and returns a `Response`;
   - a 3xx response becomes a small HTML redirect page (`<meta http-equiv="refresh">`), which is
     how `redirects` in config work on a static host;
   - otherwise the body is written to `getOutFile()` (01 §5), e.g.
     `dist/blog/strided-tensors/index.html`;
   - a render slower than 500 ms is shown in red in the log.
4. Deletes the temporary prerender directory. `dist/` contains no server code.

So "build-time rendering" is literally a server rendering requests, just once per URL and with
the responses saved to files.

`experimental.incrementalBuild` (off here) would hash each page's dependencies and reuse
unchanged output from a cache instead of re-rendering.

## 8. Optimise images

Rendering an `<Image>`, an imported image in MDX, or a Markdown image *records* a transform
(source file, format, size) in a global list instead of processing it on the spot. After all
pages are rendered, the unique transforms are processed in parallel (one per CPU) with `sharp`
and written to `dist/_astro/<name>.<hash>_<transformHash>.webp`. The log line
`gdb-crash-threads… (before: 564kB, after: 139kB)` is this step. Because the files are
content-hashed, they can be cached forever.

## 9. `astro:build:done`

Integrations get the list of pages and routes. `@astrojs/sitemap` writes `sitemap-index.xml` and
`sitemap-0.xml` from it using `site` from the config. Drafts were never rendered, so they are not
listed.

## 10. What ends up in `dist/`

```
dist/
  index.html, 404.html          ← pages (01 §5)
  about/index.html
  blog/<slug>/index.html        ← one per non-draft post
  demo/index.html, demo/<name>/index.html
  _astro/                       ← hashed build assets
    Layout.<hash>.css           ← site CSS (linked from every page)
    logical-topology.<hash>.css ← page-specific CSS
    LineChart.astro_astro_type_script_index_0_lang.<hash>.js   ← the only client JS
    KaTeX_*.woff2/.woff/.ttf    ← fonts referenced by the KaTeX CSS
    *.webp                      ← optimised images
  favicon.svg                   ← copied verbatim from public/
  sitemap-index.xml, sitemap-0.xml
```

Files in `public/` are copied as-is, with no processing or hashing.

## 11. Deployment

`.github/workflows/astro.yml` runs the build on push to `main` and publishes `dist/` to GitHub
Pages. The live site only changes when that workflow runs. Until then GitHub Pages keeps serving
the previous `dist/`.
