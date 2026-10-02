# 01 — Routing: from `src/pages/` to URLs

Source: `core/routing/create-manifest.js`, `core/routing/pattern.js`, `core/routing/priority.js`,
`core/render/route-cache.js`, `core/render/params-and-props.js`, `runtime/prerender/static-paths.js`.

## 1. Walking `src/pages/`

`createFileBasedRoutes()` in `create-manifest.js` walks the pages directory recursively. For each
entry it:

- **skips** anything whose name starts with `_` (e.g. `_helpers.ts`), and dotfiles except
  `.well-known`;
- accepts **pages** with extension `.astro`, the Markdown extensions (`.md`, `.markdown`, …) and
  any extension an integration registers with `addPageExtension()` — `@astrojs/mdx` registers
  `.mdx`;
- accepts **endpoints** with `.js` / `.ts` (a file that exports `GET` etc. and returns a
  `Response`, e.g. `src/pages/rss.xml.ts`);
- warns about `.tsx`, `.jsx`, `.vue`, `.svelte` in `pages/` — those can only be components.

Each path segment (a directory name, or a file name without extension) is split into **parts**
with the regex `/\[([^[\]()]+(?:\([^)]+\))?)\]/`. Text outside brackets is a static part, text
inside is a dynamic part, and `[...name]` is a *spread* (rest) part:

| File | Segments → parts | Route |
|---|---|---|
| `src/pages/index.astro` | `[]` (index files add no segment) | `/` |
| `src/pages/about.astro` | `[about]` | `/about` |
| `src/pages/demo/index.astro` | `[demo]` | `/demo` |
| `src/pages/demo/line-chart.astro` | `[demo] [line-chart]` | `/demo/line-chart` |
| `src/pages/blog/[slug].astro` | `[blog] [{dynamic: slug}]` | `/blog/[slug]` |
| `src/pages/404.astro` | `[404]` | `/404` (special-cased as the not-found page) |

Validation happens here too: parameter names must match `[\w$]+`, brackets must balance, two
params can't touch (`[a][b]`), and in `.astro` files a rest param must be a whole segment.

## 2. Each route gets a regex

`getPattern()` in `pattern.js` joins the segments into a `RegExp` that matches a pathname:

- static part → the literal text, regex-escaped;
- dynamic part → `([^/]+?)` — one path segment, no slashes;
- spread part → `(.*?)` (or `(?:\/(.*?))?` when it is the whole segment, so it can also match
  nothing).

The end of the pattern depends on `trailingSlash` (this site uses the default `"ignore"`, which
gives `\/?$`). So `blog/[slug].astro` becomes roughly `^\/blog\/([^/]+?)\/?$`. The capture groups
line up with the route's `params` array (`["slug"]`), which is how a pathname is turned back into
`{ slug: "strided-tensors" }` in `getParams()`.

Routes with no dynamic parts also get a fixed `pathname` (`/about`), which lets the build skip
`getStaticPaths` for them entirely.

## 3. Priority: which route wins

All routes (file-based, injected by integrations, and `redirects` from config) are sorted with
`routeComparator()` in `priority.js`, and matching takes the first route whose pattern matches.
Comparing segment by segment:

1. A **static** segment beats a dynamic one (`/blog/about` beats `/blog/[slug]`).
2. Between dynamic segments, one that **mixes** static text and params (`[id].json`) beats one
   that is all params.
3. A segment **without a spread** beats one with a spread.
4. If one route is a prefix of the other, the **longer** route wins (more specific), except a
   rest route that is one segment longer loses (`/a/[...rest]` vs `/a`).
5. Endpoints beat pages; then alphabetical, so the order is deterministic.

For a static site priority matters less than for a server, because the build doesn't match URLs
at all — it generates paths *from* each route (next section). It still matters in dev, and when
two routes generate the same path: the build keeps the first one it saw and, depending on
`prerenderConflictBehavior`, warns or errors about the other.

## 4. `getStaticPaths()` — how a dynamic route becomes many pages

A route with params has no fixed pathname, so for a static build the page must say which values
exist. That is what `getStaticPaths()` in `src/pages/blog/[slug].astro` does:

```ts
export async function getStaticPaths() {
	const posts = await getPosts({ includeDrafts: true });
	return posts.map((post) => ({ params: { slug: post.id }, props: { post } }));
}
```

What Astro does with it (`callGetStaticPaths()` in `route-cache.js`):

1. Calls it **once per route**, passing `{ paginate, routePattern }` (`paginate` is the helper for
   `/blog/[page]`-style listings).
2. Validates the result (`validateGetStaticPathsResult`): must be an array of
   `{ params, props? }`, and each param must be a string or number (or `undefined` for a rest
   param).
3. Builds a **key → entry** map: `stringifyParams(params, route)` turns
   `{ slug: "strided-tensors" }` into the concrete path `/blog/strided-tensors`, and the entry is
   stored under that key.
4. Caches the array (and the map) in a `RouteCache`, keyed by `route + component`. In dev the
   cache is cleared whenever content changes.

`StaticPaths.getAll()` (`runtime/prerender/static-paths.js`) then produces the list of
`{ pathname, route }` the build will render: fixed routes contribute their `pathname`, dynamic
routes contribute one pathname per `getStaticPaths` entry.

### How a page gets *its* post and nothing else

When the build (or the dev server) renders `/blog/strided-tensors`, `getProps()` in
`params-and-props.js`:

1. runs the route's regex on the pathname to get `params = { slug: "strided-tensors" }`;
2. calls `callGetStaticPaths()` — a cache hit after the first page, so `getPosts()` is not re-run
   for every post;
3. looks the params up in the key map (`findPathItemByKey`) to get that one entry;
4. returns a **copy of that entry's `props`**, which becomes `Astro.props` for the render.

If no entry matches, a static route throws `NoMatchingStaticPathFound` — in dev that is the 404
you see for `/blog/does-not-exist`.

So the template doesn't search for anything: each render receives exactly one entry's `props`,
and the page body only ever sees `Astro.props.post` for that one post.

## 5. From pathname to output file

`getOutFolder()` / `getOutFile()` (`core/build/common.js`) decide the file name from
`build.format` (default `"directory"`):

| Pathname | `"directory"` (this site) | `"file"` |
|---|---|---|
| `/` | `dist/index.html` | `dist/index.html` |
| `/about` | `dist/about/index.html` | `dist/about.html` |
| `/blog/strided-tensors` | `dist/blog/strided-tensors/index.html` | `dist/blog/strided-tensors.html` |
| `/404` | `dist/404.html` | `dist/404.html` |

`404` is special-cased to a flat `404.html` because that is the file GitHub Pages (and most static
hosts) serve for unknown URLs. The `"directory"` format is what makes URLs like
`/blog/strided-tensors/` work on GitHub Pages: the host maps a directory request to its
`index.html`. There is no router at request time.

A generated file that collides with a file in `public/` is skipped with a warning
(`checkPublicConflict` in `generate.js`).

## 6. `base` and links

`import.meta.env.BASE_URL` is `config.base` (here `/`). Pages in this repo build links as
`` `${base}/blog/${post.id}` `` after stripping the trailing slash, so the site would still work
if it were moved under a sub-path such as `/blog-site/`. Astro itself does not rewrite `href`s in
your HTML.
