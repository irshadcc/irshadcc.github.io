# DependencyGraph component

`DependencyGraph` draws a dependency graph in the style of a code-search "context" view: cards
with an icon, a monospace title and a muted subtitle, bare icons, faded skeleton placeholders and
small junction squares, joined by thin right-angled wires. Nodes are placed by
[dagre](https://github.com/dagrejs/dagre); the wires are routed by our own code.

| File | Role |
| --- | --- |
| `src/utils/graph/dependencyGraph.ts` | Types, node sizes, validation, dagre layout, wire routing |
| `src/utils/graph/DependencyGraph.astro` | Renders the SVG, styles, hover/click script, events |
| `src/utils/graph/graphFigures.ts` | Example graphs (code context graph, build pipeline) |
| `src/pages/demo/dependency-graph.astro` | Demo page at `/demo/dependency-graph/` |

## Usage

```mdx
import DependencyGraph from "../../utils/graph/DependencyGraph.astro";
import { codeNodes, codeEdges } from "../../utils/graph/graphFigures";

<DependencyGraph nodes={codeNodes} edges={codeEdges} theme="dark" directed active="compose"
  title="What composeMiddleware touches" />
```

### Props

| Prop | Type | Default | Meaning |
| --- | --- | --- | --- |
| `nodes` | `GraphNode[]` | required | The nodes (see below) |
| `edges` | `GraphEdge[]` | required | `{ from, to, directed? }` |
| `title` | `string` | none | Caption above the figure |
| `theme` | `"dark" \| "light"` | follows the site | Fixed palette on its own panel |
| `directed` | `boolean` | `false` | Arrowheads on every edge; an edge's own `directed` overrides it |
| `active` | `string` | none | Node id that is active on load (must be a card or icon node) |
| `id` | `string` | none | HTML id of the figure, so page scripts can listen for its events |
| `rankdir` | `"LR" \| "TB"` | `"LR"` | Left to right, or top to bottom |
| `nodesep` | `number` | `24` | Gap between nodes in the same column |
| `ranksep` | `number` | `56` | Gap between columns |
| `wide` | `boolean` | `true` | On wide screens, extend into the margin-note column |
| `hint` | `string` | "Hover a node…" | Text under the figure when no note is shown |

### Nodes

| `kind` | Draws | Needs |
| --- | --- | --- |
| `"card"` (default) | Box with icon, title, subtitle | `title` |
| `"icon"` | The icon alone, no box | `icon` |
| `"placeholder"` | Faded skeleton card (circle and two bars) | nothing |
| `"junction"` | Small filled square where wires meet | nothing |

`icon` is either a built-in name (`"fn"`, `"folder"`, `"file"`, `"commit"`, `"sparkle"`,
`"hexagon"`, `"dot"`) or a badge `{ text: "TS", bg: "#3178c6", fg: "#fff", outline?: boolean }`.
`note` (HTML) is shown under the figure while the node is hovered or active.

Invalid input fails the build with a message: duplicate ids, edges to unknown nodes, self-loops,
cards without a title, icon nodes without an icon, or an `active` id that is not a card or icon.

### Interaction

- **Hover or focus** a node: its wires and neighbours are highlighted, everything else fades, and
  its note appears.
- **Click, Enter or Space**: the node becomes the **active** node, which stays highlighted. Clicking
  the active node again clears it. Moving the pointer away returns to the active node.

## Reacting to the graph from other components

Astro components run at build time; their props are used to produce HTML and then discarded. A
function passed as a prop (`onHover={fn}`) would never reach the browser, and the component's
`<script>` cannot see props. So the graph communicates through **DOM custom events** fired on its
`<figure>`:

| Event | `e.detail` | When |
| --- | --- | --- |
| `depg:hover` | node id, or `""` when the pointer leaves | hover or focus changes (once per change) |
| `depg:active` | new active id, or `""` when cleared | click, Enter or Space |

Any script on the page listens and updates its own DOM:

```astro
<DependencyGraph id="deps" nodes={codeNodes} edges={codeEdges} directed active="compose" />
<div id="inspector"></div>

<script>
  import { codeNodes } from "../../utils/graph/graphFigures"; // data modules work client-side
  const byId = new Map(codeNodes.map((n) => [n.id, n]));
  const fig = document.getElementById("deps");
  let active = fig?.dataset.active ?? "";

  const render = (id: string) => {
    const el = document.getElementById("inspector");
    if (el) el.textContent = byId.get(id)?.title ?? "nothing selected";
  };
  fig?.addEventListener("depg:hover", (e) => render((e as CustomEvent<string>).detail || active));
  fig?.addEventListener("depg:active", (e) => render((active = (e as CustomEvent<string>).detail)));
  render(active);
</script>
```

The events bubble, so a single listener on a parent (or `document`) can serve several graphs;
`e.target.id` says which one fired.

Ways to "re-render" the other component:

- **Small updates**: set text or attributes directly, as above (the demo's inspector does this).
- **A fixed set of states**: render every variant at build time with `hidden` and toggle which one
  shows in the listener (how `BoxDiagram` handles its notes).
- **Heavy interactive UI**: add a UI framework island (React, Svelte, …) and put the listener
  inside it so state changes re-render it. It still connects to `DependencyGraph` through these
  events.

## How it works

```
build time (Astro)                                          browser
──────────────────────────────────────────────────────      ─────────────────────
nodes + edges ─► nodeSize() ─► dagre layout ─► wire routing ─► SVG string ─► <script>: hover,
(graphFigures)   (box sizes)   (x, y of each   (our code)     in the HTML   click, events
                               node)
```

Everything up to the SVG runs once, at build time; the page ships finished SVG and no layout code.
The browser script only toggles CSS classes and dispatches events.

### Where dagre is imported

The component never imports dagre itself. It goes through the layout module:

```
DependencyGraph.astro  ──imports layoutGraph──►  dependencyGraph.ts  ──imports──►  @dagrejs/dagre
                                                  import { graphlib, layout } from "@dagrejs/dagre";
```

`DependencyGraph.astro` imports `layoutGraph` and `validateGraph` from `./dependencyGraph`, and
that module is the only file that touches dagre. The split is deliberate:

1. **Repo convention.** `AGENTS.md` asks for "the computation in a sibling `.ts` file (as with
   `collectives.ts` and `pipeline.ts`) so the logic can be tested without rendering". `BoxDiagram`
   and `boxDiagram.ts`, and `IrCfg` and `irCfg.ts`, follow the same pattern.
2. **Testable on its own.** `layoutGraph` is plain TypeScript, so a script can import it with
   `node --experimental-strip-types` and check every wire without building the site or opening a
   browser. An `.astro` file can't be imported that way.
3. **dagre never reaches the browser.** The layout runs in the component's front matter, at build
   time. The browser `<script>` only toggles classes and fires events, and the demo's inspector
   script imports only *types* from `dependencyGraph.ts` (via `graphFigures.ts`), which are erased
   when compiled. No file under `dist/_astro/` contains dagre, so readers download the finished SVG
   and not the layout library.

### Why nodes aren't placed in a random order

dagre implements the **layered (Sugiyama) method** for drawing directed graphs. It runs in fixed
stages, each optimising one property of the picture. The stages below were checked against the
dagre 3.1.1 source (`runLayout` in `lib/layout.ts`), not recalled.

**0. Box sizes** (`nodeSize` in `dependencyGraph.ts`). dagre needs each node's width and height up
front. Card width is estimated from character counts of the monospace fonts (7.8 px per title
character, 6.6 px per subtitle character); icons are 32×32, placeholders 190×50, junctions 9×9.

**1. Break cycles** (`lib/acyclic.ts`). Layering needs a graph without cycles. dagre runs a
depth-first search and temporarily reverses every edge that points back to a node still on the
search stack, then flips those edges back at the end.

**2. Assign columns, called ranks** (`lib/rank/`). Each edge must point at least one column
"forward". The default ranker is **network simplex** (Gansner et al., *A Technique for Drawing
Directed Graphs*, the paper behind Graphviz `dot`), which picks ranks that minimise the total edge
length. That is why connected nodes end up close together and edges rarely skip columns. For the
demo's code graph:

```
col 0: tooltip  useExt  ph-a  ph-b
col 1: ts  hex  js  j1  j2
col 2: ph-c  components  auth  sparkle
col 3: compose  avatar  go
col 4: commit  ph-d  ph-e  ph-f
```

**3. Split edges with dummy nodes** (`lib/normalize.ts`). Every edge is cut into pieces that each
span one internal rank, joined by invisible **dummy nodes**. The dummies take part in ordering and
positioning like real nodes, which reserves a lane for the wire between real nodes.

dagre's internal ranks are finer than the columns you see. To leave room for edge labels,
`makeSpaceForEdgeLabels` (in `lib/layout.ts`) halves `ranksep` and doubles every edge's minimum
length, so there is an internal rank in the middle of every gap. As a result **every** edge gets
dummies, even one between neighbouring columns: one in the middle of each gap it crosses, plus one
in each column it skips. An edge from column 0 to column 3 has five (gap, column 1, gap, column 2,
gap). After layout, dagre returns each edge's `points` as the source border point, the dummy
positions, then the target border point.

**4. Order nodes within each column** (`lib/order/`). This is the stage that makes the picture look
deliberate: the number of crossing wires depends only on the vertical order inside each column.

- Initial order: a depth-first search from the first column, in input order.
- Then dagre sweeps across the columns, alternately left to right and right to left. In each
  column it sorts nodes by **barycenter**, the average position of their neighbours in the column it
  just came from, so a node connected to things near the top moves up.
- After each sweep it counts crossings and keeps the best order seen. It stops after four sweeps
  in a row bring no improvement.

Minimising crossings exactly is NP-hard, so this is a good heuristic, not a guaranteed optimum.

**5. Coordinates** (`lib/position/`). Columns are spaced by `ranksep`, and nodes within a column by
`nodesep`. Positions within a column come from **Brandes–Köpf** (*Fast and Simple Horizontal
Coordinate Assignment*): each node is aligned with its median neighbour so most wires run straight,
then the layout is packed as tightly as the gaps allow.

dagre uses no randomness, so the same input always gives the same picture. Ties are broken by input
order, which means reordering `nodes` can change the result.

### Edge path calculation (`layoutGraph`)

dagre's edge `points` form a diagonal polyline: they start and end wherever the straight line
towards the next point leaves the node's box. To draw right-angled wires that merge into shared
trunks, `layoutGraph` keeps dagre's node positions and its dummy points, and computes the wires
itself, in the seven steps below. The numbers come from the demo's code graph (`rankdir="LR"`,
`nodesep` 24, `ranksep` 56), traced with a script that replays the routing loop.

#### 1. Work in "main" and "cross" coordinates

```ts
const main = (p) => (lr ? p.x : p.y);   // along the ranks
const cross = (p) => (lr ? p.y : p.x);  // across the ranks
const pt = (m, c) => (lr ? { x: m, y: c } : { x: c, y: m });
```

Every rule below is written in terms of `main` and `cross`, so the same code routes both
`rankdir="LR"` (main = x) and `rankdir="TB"` (main = y). The rest of this section uses LR, where
main is x and cross is y.

#### 2. Measure the columns and the gaps between them

For every real node, take the centre of its column (rounded to a whole pixel) and half its width.
Each column keeps the half-width of its **widest** node, so the column's right edge is
`centre + half` and its left edge `centre - half`. The **gap** between two neighbouring columns
runs from one column's right edge to the next column's left edge, and every bend is placed at the
gap's midpoint:

```ts
// gapAfter(m): the middle of the gap after the column at or before main coordinate m.
return next ? (c + h + next[0] - next[1]) / 2 : c + h + ranksep / 2;
```

| Column | Centre | Widest half-width | Left edge | Right edge |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 131 | 119 | 12 | 250 |
| 1 | 322 | 16 | 306 | 338 |
| 2 | 517 | 122.5 | 394.5 | 639.5 |
| 3 | 821 | 125.5 | 695.5 | 946.5 |
| 4 | 1168 | 166 | 1002 | 1334 |

| Gap | From | To | Width | Bend at |
| ---: | ---: | ---: | ---: | ---: |
| 0 → 1 | 250 | 306 | 56 | 278 |
| 1 → 2 | 338 | 394.5 | 56.5 | 366.25 |
| 2 → 3 | 639.5 | 695.5 | 56 | 667.5 |
| 3 → 4 | 946.5 | 1002 | 55.5 | 974.25 |

Each gap is `ranksep` (56) wide, give or take the half-pixel rounding of column centres.
`gapAfter` searches from the last column backwards for the first centre `≤ m + 1`; the `+ 1`
absorbs that rounding. Dummy points sit at a column centre or a gap centre, so their main
coordinate always finds the right column.

#### 3. Pick the start and end points

dagre's border points depend on the angle of its diagonal line, so they land anywhere on the box.
Instead, a wire always leaves from the **middle of the source's right side** and enters at the
**middle of the target's left side** (bottom and top for TB):

```ts
const start = pt(main(ac) + aHalf, cross(ac)); // ac: source centre, aHalf: half its width
const end = pt(main(bc) - bHalf, cross(bc));   // bc: target centre
```

If the target were to the left of the source (`forward` is false, which can happen when dagre
reverses an edge to break a cycle), the sides swap. The demo graphs have no cycles, so that branch
is untested.

Example, `ts → components`: `ts` is the 32×32 icon at x 306–338 with centre y 284.5, so
`start = (338, 284.5)`. `components` is the 245×58 card at x 394–639 with centre y 249.5, so
`end = (394, 249.5)`.

#### 4. Bend between consecutive points

The edge must pass through `via = [start, ...dagre's dummy points, end]`. For each consecutive
pair `p → q`:

- If they are at the same height (cross coordinates within 0.5 px), go straight to `q`.
- Otherwise, find the gap after `p` with `m = gapAfter(min(main(p), main(q)))` and go
  **horizontal to `(m, p.y)`, vertical to `(m, q.y)`, then horizontal to `q`**.

```ts
if (Math.abs(cross(p) - cross(q)) > 0.5) {
	const m = gapAfter(Math.min(main(p), main(q)));
	route.push(pt(m, cross(p)), pt(m, cross(q)));
}
route.push(q);
```

Because the bend's x depends only on the gap and not on the edge, every wire crossing the same gap
bends on the same vertical line. That is what makes them merge into shared trunks, as in the
reference image.

Example, `ts → components`. dagre returns one dummy, `(366, 245.5)`, in the middle of gap 1 → 2:

```
via:   (338, 284.5)  (366, 245.5)  (394, 249.5)
route: (338, 284.5)  (366.25, 284.5)  (366.25, 245.5)  (366, 245.5)
                     (366.25, 245.5)  (366.25, 249.5)  (394, 249.5)
```

The first pair differs in y, so it bends at x = 366.25 (gap 1 → 2) from y 284.5 up to 245.5, the
dummy's height. The second pair also differs in y (245.5 vs 249.5), so it bends at the same
x = 366.25 back down to the target's centre line. The raw route goes up past the target and comes
back down, and step 5 cleans that up.

#### 5. Simplify

`simplify()` walks the route and removes two kinds of point:

- **Repeats**: a point within 0.5 px of the previous one in both x and y. This removes the dummy
  `(366, 245.5)` next to the bend `(366.25, 245.5)`; dagre's gap centre and ours differ by a
  quarter pixel because of rounding.
- **Middle points of a straight run**: if three points in a row share an x (or a y), the middle
  one goes. The test ignores direction, so an up-then-down overshoot on one vertical line collapses
  into a single segment.

`ts → components` ends up as four points, one bend each way:

```
(338, 284.5) → (366.25, 284.5) → (366.25, 249.5) → (394, 249.5)
```

A longer example, `n0 → n3` in a four-node chain (0 → 1 → 2 → 3 plus a shortcut 0 → 3). dagre
routes the shortcut above `n1` and `n2` at y = 11, with five dummies:

```
via:   (83, 32) (111, 11) (174.5, 11) (238, 11) (301.5, 11) (365, 11) (393, 32)
final: (83, 32) → (111.5, 32) → (111.5, 11) → (365.5, 11) → (365.5, 32) → (393, 32)
```

It bends up in the first gap, runs straight over the two middle columns through the dummies (which
reserve that lane, so it never crosses `n1` or `n2`), and bends down in the last gap.

#### 6. Arrowhead (directed edges)

For a directed edge, the last segment is shortened by `ARROW_LEN` (7 px) and a triangle fills the
space, with its tip exactly on the target's border:

```ts
const tip = points.at(-1), prev = points.at(-2);
const [ux, uy] = unit vector from prev to tip;
const base = tip - 7 * (ux, uy);
arrow = [tip, base + 3.5 * (-uy, ux), base - 3.5 * (-uy, ux)];
points[points.length - 1] = base;
```

`(-uy, ux)` is the direction perpendicular to the segment, so the two base corners sit 3.5 px on
each side of the line. For `ts → components` the last segment points right, `(ux, uy) = (1, 0)`:
`tip = (394, 249.5)`, `base = (387, 249.5)`, and the corners are `(387, 253)` and `(387, 246)`.

There is always room for the arrow. The last segment runs from a gap midpoint to the target's left
side, which is at or right of its column's left edge, so it is at least half a gap long, roughly
`ranksep / 2` = 28 px, well over 7.

#### 7. SVG path

The points become an SVG path, `M` for the first and `L` (line to) for the rest, rounded to one
decimal:

```
ts → components:  M338.0,284.5 L366.3,284.5 L366.3,249.5 L387.0,249.5
```

The component draws each edge as `<g class="dg-edge" data-from data-to>` holding that `<path>` and
the arrow `<polygon>`. The browser script uses `data-from` and `data-to` to find the wires to light
up when a node is hovered or active.

#### Checks and limits

A test script ran the routing on the demo graphs and on extra cases (a single node, an edge that
skips columns, LR and TB, directed and undirected) and checked that every wire:

- is made only of horizontal and vertical pieces;
- starts on the source's border and ends on the target's border (the arrow tip, for directed
  edges), with arrows exactly 7 px long and only on directed edges;
- never passes through another node's box.

Known limits:

- **Shared trunks hide individual wires.** Wires that bend in the same gap overlap on one vertical
  line, so you can't always tell by eye which source feeds which target. Hovering a node lights up
  its own wires.
- **Cycles are untested.** `validateGraph` rejects self-loops but not longer cycles. dagre lays
  them out by reversing an edge, and the routing has a branch for backward edges, but no demo or
  test covers it.
- **Text widths are estimates.** Box sizes come from character counts, not measured text, so an
  unusual font or very wide characters could overflow a card.

### Rendering

The component builds the SVG as a string at build time and inserts it with `set:html`. Colours are
CSS variables (`--dg-bg`, `--dg-card`, `--dg-line`, …) that follow the site theme by default;
`.depg-dark` and `.depg-light` override them with fixed palettes. The SVG keeps at least 75% of its
natural width so text stays readable, and scrolls sideways on narrow screens.

## Tuning a layout

- **Reorder nodes or edges** in the input. That changes the initial order in stage 4 and can
  untangle a layout.
- **Adjust `ranksep` and `nodesep`** for wider or tighter spacing.
- **Watch for junctions and placeholders in the middle of a chain.** Each one adds a column. Moving
  junctions into an existing column took the demo graph from 2586 px to 1346 px wide.
- dagre also supports edge `weight` (pull nodes closer together) and `minlen` (force an edge to
  span more columns). The component does not expose them yet; they could be passed through to
  `g.setEdge` in `layoutGraph`.

## Pitfalls

- **Astro compiler and nested template literals.** The Rust Astro compiler (0.5.1) failed to parse
  a `<text>` template literal nested inside another one in the component's front matter. The card
  markup is built from separate variables for that reason; keep it that way when editing.
