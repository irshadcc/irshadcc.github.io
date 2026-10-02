# 02 — `.astro` components: compile and render

Source: `core/compile/compile.js`, `vite-plugin-astro/index.js`, `runtime/server/render/*.js`,
`runtime/server/render/astro/render-template.js`, `@astrojs/compiler-rs` (Rust, shipped as a
native Node binding).

## 1. The compiler's job

A `.astro` file is not HTML with a JS header. It is compiled into an ordinary ES module. The Vite
plugin `astro:build` (`vite-plugin-astro/index.js`) has a `transform` hook for `*.astro` ids that
calls `compile()` → `@astrojs/compiler-rs` `transform(source, options)`. The result has:

| Field | Meaning |
|---|---|
| `code` | the JS module (below) |
| `map` | source map back to the `.astro` file |
| `css[]` | one string per `<style>` block, already scoped |
| `scripts[]` | one entry per processed `<script>` block |
| `scope` | the hash used for scoped CSS (e.g. `yk4hkwyg`) |
| `hydratedComponents`, `clientOnlyComponents` | framework components with `client:*` directives |
| `containsHead`, `propagation` | whether this component renders `<head>`, or needs its styles/scripts hoisted to its parent's head |
| `diagnostics` | compiler errors/warnings — an `error` here becomes the red overlay in dev |

Before `transform`, `preprocessStyles()` extracts each `<style>` and runs it through Vite's CSS
pipeline (PostCSS, `lang="scss"`, etc.), so the compiler only has to scope plain CSS.

## 2. What the compiled module looks like

Real output for this component (compiled with the command in the README):

```astro
---
import Badge from "./Badge.astro";
interface Props { title: string }
const { title } = Astro.props;
const items = ["a", "b"];
---
<div class="card">
	<h2>{title}</h2>
	<Badge label="new" />
	<ul>{items.map((i) => <li>{i}</li>)}</ul>
	<slot />
</div>
<style> .card { padding: 1rem; } </style>
<script> console.log("hello from the browser"); </script>
```

```js
import { render as $$render, createAstro as $$createAstro, createComponent as $$createComponent,
  renderComponent as $$renderComponent, maybeRenderHead as $$maybeRenderHead,
  renderSlot as $$renderSlot, renderScript as $$renderScript, createMetadata as $$createMetadata
} from "astro/compiler-runtime";
import Badge from "./Badge.astro";
import "/src/components/Card.astro?astro&type=style&index=0&lang.css";   // ← the <style>
// ... $$metadata (imported modules, hoisted scripts) ...

const $$Card = $$createComponent(($$result, $$props, $$slots) => {
	const Astro = $$result.createAstro($$props, $$slots);
	Astro.self = $$Card;
	const { title } = Astro.props;             // ← the frontmatter, verbatim
	const items = ["a", "b"];
	return $$render`${$$maybeRenderHead($$result)}<div class="card astro-yk4hkwyg">
	<h2 class="astro-yk4hkwyg">${title}</h2>
	${$$renderComponent($$result, "Badge", Badge, { "label": "new", "class": "astro-yk4hkwyg" })}
	<ul class="astro-yk4hkwyg">${items.map((i) => $$render`<li class="astro-yk4hkwyg">${i}</li>`)}</ul>
	${$$renderSlot($$result, $$slots["default"])}
</div>
${$$renderScript($$result, "/src/components/Card.astro?astro&type=script&index=0&lang.ts")}`;
}, "/src/components/Card.astro", undefined);
export default $$Card;
```

(The sample was compiled with the compiler's default class-based scoping. This project uses
Astro's default `scopedStyleStrategy: "attribute"`, which emits `data-astro-cid-yk4hkwyg`
attributes instead of `astro-yk4hkwyg` classes — same idea.)

Things to notice:

- **The frontmatter becomes the body of a function.** It runs once per render of that component,
  on the server (at build time for this site). Top-level `await` in the frontmatter works because
  the component function is awaited. `export async function getStaticPaths` is hoisted *out* of
  the function to module scope so Astro can call it without rendering.
- **The template becomes a tagged template literal** `` $$render`...` ``. Static HTML is the
  literal parts; every `{expression}` is an interpolation.
- **Child components** become `$$renderComponent(result, name, Component, props, slots)` calls.
- **`<slot />`** becomes `$$renderSlot(result, $$slots["default"])`. Children passed to a
  component are compiled into functions in `$$slots`, so they are rendered lazily, and only if the
  child actually renders the slot.
- **`<style>`** is removed from the template and turned into a side-effect import of a virtual
  module `Card.astro?astro&type=style&index=0&lang.css`. The plugin's `load` hook serves that id
  from the cached compile result. From there it is ordinary Vite CSS: bundled, hashed and
  linked from the page.
- **`<script>`** is removed too, recorded in `scripts[]`, and replaced by
  `$$renderScript(result, "Card.astro?astro&type=script&index=0&lang.ts")`. More below.

In the **client** Vite environment, the same `transform` hook replaces the whole module with a stub
that throws "Astro components cannot be used in the browser". `.astro` components only ever run
on the server side.

## 3. Rendering: from component function to HTML string

`$$render` creates a `RenderTemplateResult` (`render-template.js`) holding the HTML parts and the
expressions. Rendering is **streaming-by-design**: `render(destination)` writes each HTML part,
then renders each expression with `renderChild()` (`render/any.js`), which dispatches on type:

| Value | Output |
|---|---|
| `string` | **HTML-escaped** (`escapeHTML`) — `{"<b>"}` prints `&lt;b&gt;` |
| number, etc. | `String(value)` |
| `null`, `undefined`, `false`, `""` | nothing (but `0` prints `0`) |
| array / iterable | each item rendered in order — that's why `.map()` works |
| `Promise` | awaited, then rendered — you can put `{fetchSomething()}` in a template |
| another `RenderTemplateResult` or component instance | rendered recursively |
| `HTMLString` (from `set:html`, `<Fragment set:html>`, or `unescapeHTML`) | written as-is, **not** escaped |
| function | called, then rendered |

Promises don't block siblings from being *computed*: when an expression returns a promise, the
remaining expressions are started straight away into buffers (`createBufferedRenderer`) and
flushed in order when the promise resolves. Output order is always source order.

`renderPage()` (`render/page.js`) is the top: it renders the page component into a string (for a
static build; streaming is used by servers) and wraps it in a `Response` with `Content-Type:
text/html`. `404.astro` gets status 404. The build then writes `response.body` to disk.

`compressHTML` (default `"jsx"`) strips whitespace between tags at compile time. That's why the
generated `dist/*.html` is one long line.

## 4. `<head>`: where styles and scripts get injected

Components deep in the tree can add CSS and scripts, but those have to end up in `<head>`, which
has already been passed by the time the child renders. Astro handles this with render
*instructions*:

- The compiler puts `$$maybeRenderHead(result)` before the first element a component renders (and
  `renderHead` at an explicit `</head>`).
- Before rendering starts, the page's full set of styles, links and scripts is collected from the
  build's module graph (every CSS and script module reachable from the page), deduplicated, and
  put on the render `result`.
- The first `maybeRenderHead` that actually runs emits all of them (`renderAllHeadContent()` in
  `render/head.js`) and marks the head as rendered, so later calls do nothing.

In this repo `Layout.astro` writes the `<html><head>…</head>` itself, so every page's CSS appears
at the end of that `<head>`.

**Propagation.** Content rendered through `render(entry)` (MDX posts) may import components that
bring their own CSS. Those modules are marked with `"use astro:head-inject"` and
`?astroPropagatedAssets`, so their assets bubble up to the page head too. That is the source of
the harmless `MODULE_LEVEL_DIRECTIVE` warnings Rolldown prints during `npm run build`.

## 5. Scoped styles

Every `.astro` file gets a hash (`scope`). The compiler:

- rewrites each selector in a non-global `<style>` so it only matches elements from this
  component: with `scopedStyleStrategy: "attribute"` (default) `.card` becomes
  `.card[data-astro-cid-yk4hkwyg]`; with `"where"` or `"class"` it becomes
  `.card:where(.astro-yk4hkwyg)` / `.card.astro-yk4hkwyg`;
- adds the matching attribute (or class) to **every HTML element in the template**, and passes it
  to child components as a prop so their root element can be matched too.

`<style is:global>` and `:global(.x)` opt out. In the built `dist/demo/line-chart/index.html` you
can see `data-astro-cid-hx4b6n6i` on the elements `LineChart.astro` renders.

The CSS files themselves are bundled per page. With `build.inlineStylesheets: "auto"` (default)
a stylesheet smaller than Vite's `assetsInlineLimit` (4 kB) is inlined into a `<style>` tag, and
larger ones become `<link rel="stylesheet" href="/_astro/<name>.<hash>.css">`. In this build
`Layout.D8n6Ahka.css` (≈35 kB) is linked on every page, and the line-chart page also inlines one
small `<style>`.

## 6. Scripts

A plain `<script>` in a component is **processed**: it is TypeScript, can `import` npm packages,
and is bundled by Vite in the *client* environment. At render time `renderScript()` emits either:

- `<script type="module" src="/_astro/<name>.<hash>.js">`, or
- an inline `<script type="module">…</script>` when the bundled chunk is small (under the same
  4 kB limit), has no imports and no dynamic imports (`plugin-scripts.js`).

Both are deduplicated per page: a component used ten times on a page ships its script once. So a
component script must find its elements with `document.querySelectorAll(...)` and handle every
instance itself — the script does not know which instance "it" belongs to. That is why
`LineChart.astro` passes each chart's data through `data-*` attributes and initialises all charts
on the page from one script. It is also why Chart.js appears in
`dist/_astro/LineChart.astro_astro_type_script_index_0_lang.<hash>.js`.

`<script is:inline>` opts out: the tag is left exactly where it is, unbundled, and repeated for
every instance.

## 7. Framework components and islands (not used here)

A React/Svelte/… component with `client:load`, `client:visible`, etc. is rendered to HTML on the
server and then **hydrated** in the browser inside an `<astro-island>` element. That is the
"islands" architecture. The compiler records these in `hydratedComponents`, and the build adds
their entry points to the client bundle (`getClientInput()` in `static-build.js`). This site
only uses `.astro` components plus `<script>`, so it ships no framework runtime.
