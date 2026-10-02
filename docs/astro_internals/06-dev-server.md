# 06 — `astro dev`

Source: `core/dev/`, `vite-plugin-astro-server/plugin.js`, `vite-plugin-app/createAstroServerApp.js`,
`vite-plugin-app/handle-request.js`, `vite-plugin-astro/hmr.js`, `content/watcher.js`.

## 1. Same renderer, different loop

The dev server uses **the same routing and rendering code** as the build (01–04). The
difference is *when* it runs:

| | `astro build` | `astro dev` |
|---|---|---|
| Which pages are rendered | every path, up front | only the URL you request, when you request it |
| How modules are loaded | bundled by Rolldown, then imported | transformed by Vite on demand and run through Vite's module runner — no bundle |
| `import.meta.env.DEV` | `false` | `true` (so drafts are included in this repo) |
| CSS / scripts | hashed files in `_astro/` | served individually by Vite, with HMR |
| Images | optimised to files after rendering | transformed on request by the `/_image` endpoint |

That's why the output is the same, but a broken page in dev only errors when you open it.

## 2. Startup

1. Config is resolved and integrations run `astro:config:setup` / `config:done` (05 §1–2), then
   `astro:server:setup`.
2. A Vite dev server is created with Astro's plugins. Astro's plugin `configureServer` hook
   (`vite-plugin-astro-server/plugin.js`):
   - loads the content config in a Vite module runner and starts a content sync (03), with the
     glob loader subscribed to the file watcher;
   - imports Astro's dev app (`createAstroServerApp`) **inside the `prerender` environment's
     module runner**, so page modules are loaded through Vite's transform pipeline (the `.astro`
     compiler, MDX, etc.);
   - installs a Connect middleware that sends page requests to that app.
3. The route list is built from `src/pages/` (01). It is rebuilt when files are added or removed
   under `pages/`, and pushed to the app with an `astro:routes-updated` event.

## 3. Handling a request

For `GET /blog/strided-tensors`:

1. Vite's own middleware handles requests for modules and assets first (`/src/...`,
   `/@vite/client`, `?astro&type=style…`, files in `public/`).
2. Anything else goes to Astro's handler. It applies base/trailing-slash rules, then **matches the
   pathname against the sorted route patterns** (01 §2–3): `/blog/[slug]` matches, with
   `params = { slug: "strided-tensors" }`.
3. It loads the route's module through the module runner. That is the moment
   `src/pages/blog/[slug].astro` and everything it imports get compiled, if they aren't cached.
4. `getProps()` calls `getStaticPaths()` (cached in the `RouteCache`), finds the entry for
   `strided-tensors` and uses its props (01 §4). No entry → 404 page.
5. The page renders exactly as in the build (02 §3). In dev the head also gets Vite's client
   script (`/@vite/client`) for HMR, and CSS is injected as `<style>`/module links so it can be
   hot-swapped.
6. Errors are shown in the Vite overlay with the `.astro` source location (from source maps and
   compiler diagnostics).

## 4. What happens when you edit a file

- **A `.astro` component** — the file is recompiled. If only its `<style>` changed, Vite hot-swaps
  the CSS with no reload (the `?astro&type=style` virtual modules are separate modules for this
  reason). Otherwise the page does a full reload. Astro components run only on the server, so
  there's no component state to preserve.
- **A client `<script>`** — handled by Vite's normal HMR → full reload.
- **A post in `src/content/`** — the content watcher re-syncs just that entry (03 §3), the route
  cache is cleared (`astro:content-changed`), and the page reloads with the new content.
  Frontmatter that fails the schema shows the validation error in the overlay.
- **`content.config.ts` or `astro.config.mjs`** — config digest changes → the data store is
  cleared and re-synced. For the Astro config the dev server restarts itself.
- **Adding/removing a page** — the route list is rebuilt (step 2.3).

## 5. Dev-only extras

- The **dev toolbar** (the bar at the bottom of the page) is injected only in dev. It is why the
  compiler gets `annotateSourceFile` in dev: elements carry source-location attributes so the
  toolbar can jump to the file.
- `astro preview` is **not** the dev server. It serves the already-built `dist/` as a plain static
  server, to check the production output locally.
