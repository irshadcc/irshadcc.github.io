# 03 — Content collections

Source: `content/content-layer.js`, `content/loaders/glob.js`, `content/loaders/file.js`,
`content/utils.js`, `content/runtime.js`, `vite-plugin-markdown/content-entry-type.js`,
`@astrojs/mdx/dist/index.js`. Config in this repo: `src/content.config.ts`.

## 1. The model

A **collection** is a named set of **entries**. Each entry is:

```ts
{
  id: string,          // unique within the collection; becomes the URL slug here
  data: {...},         // validated against the collection's schema
  body?: string,       // raw Markdown/MDX source (for .md/.mdx)
  filePath?: string,   // e.g. "src/content/posts/strided-tensors.mdx"
  digest: string,      // hash of the file contents, for change detection
  rendered?: {...},    // pre-rendered HTML (.md only — see §4)
  deferredRender?: true // render later from a module (.mdx — see §4)
}
```

Collections are not read from disk when a page asks for them. They are **synced** into a data
store first, and `getCollection()` reads from that store.

## 2. `defineCollection({ loader, schema })`

```ts
const posts = defineCollection({
	loader: glob({ base: "src/content/posts", pattern: "**/*.{md,mdx}" }),
	schema: z.object({ title: z.string(), date: z.coerce.date(), draft: z.boolean().optional().default(false), ... }),
});
```

- The **loader** is an object `{ name, load(context) }`. Astro calls `load()` during sync and
  gives it a `context` with a `store` to write entries into, a `parseData()` function that runs
  the schema, `generateDigest()`, a `logger`, and in dev a file `watcher`.
- The **schema** is Zod (v4). Astro wraps it: `parseData()` → `getEntryData()` →
  `schema.safeParseAsync(data)`. A failure becomes an `InvalidContentEntryDataError` naming the
  file and the field. Successful parsing also *transforms*: `z.coerce.date()` turns the YAML
  string `2026-01-18` into a `Date`, and `.default(false)` fills in `draft`.
- Two built-in loaders are used here: `glob()` for `posts` and `file()` for `work.json` and
  `publications.json` (one JSON file → many entries, `id` taken from each object's `id`).

## 3. What `glob()` does (`content/loaders/glob.js`)

1. Resolves `base` against the project root and lists files with `tinyglobby` using `pattern`.
2. Picks an **entry type** by file extension. Entry types are registered by Astro (`.md`) and by
   integrations (`@astrojs/mdx` calls `addContentEntryType()` for `.mdx`). An entry type knows how
   to split a file into frontmatter and body (`getEntryInfo`) and, optionally, how to render it.
3. For each file (10 at a time):
   - reads it and splits it into `data` (parsed YAML frontmatter) and `body`;
   - computes the **id**: `data.slug` if the frontmatter sets one, otherwise the path relative to
     `base`, without extension and slugified (`strided-tensors.mdx` → `strided-tensors`,
     `a/b.md` → `a/b`);
   - computes a **digest** of the contents. If the store already has this id with the same digest,
     it is left alone — this is why re-syncs are cheap;
   - otherwise runs `parseData()` (schema validation), renders if appropriate (§4), and calls
     `store.set(entry)`.
4. Entries whose files disappeared are deleted from the store.
5. In dev it subscribes to the watcher, so adding, editing or deleting a post re-runs
   `syncData()` for just that file.

Two files producing the same id is a `DuplicateContentEntrySlugError` (warning or error depending
on `prerenderConflictBehavior`).

## 4. `.md` vs `.mdx` entries: rendered now vs rendered later

This is the most important internal difference between the two formats.

**`.md`** — the Markdown entry type (`content-entry-type.js`) has `getRenderFunction()`. During
sync, `glob()` calls it, which runs the configured Markdown processor (see 04) and stores the
**HTML string** in `entry.rendered.html`, plus metadata (headings, image paths). You can see it
in `.astro/data-store.json`: the `pytorch-dispatcher` entry contains the rendered
`<p>One of the core components…`.

**`.mdx`** — MDX can `import` components, so it cannot be turned into a static string without
bundling. The MDX entry type has `contentModuleTypes` instead of a render function, so `glob()`
stores it with `deferredRender: true` and registers the file as a module. Astro writes
`.astro/content-modules.mjs`, a map from file path to a lazy `import()`:

```js
export default new Map([
  ["src/content/posts/strided-tensors.mdx", () => import("astro:content-layer-deferred-module?...fileName=src%2Fcontent%2Fposts%2Fstrided-tensors.mdx...")],
  ...
]);
```

That virtual module goes through Vite like any other source file, so the MDX is compiled to a JS
component (04) and bundled with the page that uses it.

## 5. The data store and `.astro/`

The store is a map of collections → entries, kept in memory and persisted with `devalue` (a JSON
superset that keeps `Date`s, `Map`s, etc.) to `.astro/data-store.json`. It also stores digests of
`content.config.ts` and of the Astro config. If either changes, the whole store is cleared and
rebuilt. `astro build --force` clears it too.

Other files Astro generates in `.astro/` (all gitignored, all re-creatable):

| File | Purpose |
|---|---|
| `data-store.json` | the synced entries (above) |
| `content-modules.mjs` | lazy imports for deferred-render (MDX) entries |
| `content-assets.mjs` | image imports referenced from entries |
| `content.d.ts`, `types.d.ts` | generated types, so `getCollection("posts")` returns entries whose `data` is typed from the Zod schema |
| `collections/*.schema.json` | JSON Schema for each collection (editor autocomplete for frontmatter / JSON) |

`npm run build` runs this sync first ("[content] Syncing content" in the log). `astro dev` runs
it at startup and then incrementally.

## 6. Reading: `getCollection()` and `render()`

Both come from the virtual module `astro:content` (implemented in `content/runtime.js`).

**`getCollection(name, filter?)`** reads every entry of the collection from the store, resolves
image references in `data`, runs your `filter` if given, and returns an **array in store order**.
It does not sort. This repo's `getPosts()` in `src/posts.ts` sorts newest first and filters drafts:

```ts
getCollection("posts", (post) => !post.data.draft || (includeDrafts && import.meta.env.DEV))
```

`import.meta.env.DEV` is replaced by a constant (`false` in `astro build`), so in production a
draft never passes the filter. `blog/[slug].astro` therefore gets no `getStaticPaths` entry for
it, no page is generated, and it is not in the sitemap. `style-guide.mdx` (`draft: true`) is
missing from the build log and from `dist/blog/` for this reason.

**`render(entry)`** returns `{ Content, headings, remarkPluginFrontmatter }`:

- for a `.md` entry it wraps the stored `rendered.html` in a component that outputs it unescaped;
- for a `deferredRender` (MDX) entry it looks the file up in `astro:content-module-imports`
  (i.e. `.astro/content-modules.mjs`), `import()`s the compiled MDX module, and returns its
  default export as `Content`, plus a wrapper that collects the styles/scripts that MDX imports
  so they reach `<head>` (the "propagation" in 02 §4).

Either way `<Content />` is just a component that `[slug].astro` renders inside `Layout`.
