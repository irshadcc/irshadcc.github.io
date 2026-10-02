# AGENTS.md

Guidance for AI agents writing or editing posts on this blog. The blog is an Astro site of
technical explainers (GPU kernels, distributed training, ML), deployed to GitHub Pages from
`main`.

[README.md](README.md) is the reference for Markdown syntax, front matter, math, sidenotes and
publishing. Read it first. This file covers how to write a good post here, what visualisations
already exist, and how to check the result before handing it back.

## What a good post looks like

Every post should leave the reader able to explain the topic to someone else. In practice
that means four things.

### 1. A clear introduction

Open with a `## Background` section (no `# Title`; the page shows `title` already) that answers,
in two or three short paragraphs:

- **What problem is this about?** Start from something the reader already knows.
- **Why does it matter?** A cost, a failure mode or a design choice it explains.
- **What will this post cover?** Name the parts, in order, so the reader can find their way.

Link related posts on the blog (`[title](/blog/<slug>/)`) instead of re-explaining them.

### 2. A picture for every important idea

Readers understand a layout, a schedule or a data flow far faster from a figure than from
prose. Aim for one visual per concept, placed right after the paragraph that introduces it.

- **Reuse an existing component** where one fits (see the catalogue below). Prefer interactive
  ones (hover, step-through, controls) for anything that changes over time or has many parts.
- **The figure must show what the text says.** If the text says "thread 0 owns the top-left
  block", the figure must make that visible. Say in the text what to look at.
- **Captions name what is drawn.** `Matrix2D` and `LayoutGrid` caption a figure as
  `name = layout`, so `name` must be the name of the drawn layout, not of a function applied to it.
- **Keep figures small enough to read.** Scale a real configuration down (8×8 instead of
  64×512) and say that you did.
- **Tables** suit comparisons, cost formulas and per-step results. Right-align numbers.

### 3. Worked examples with real numbers

After stating a formula or an algorithm, work through a small concrete case: a 10-point
dataset, a 4-rank all-reduce, an 8×8 matrix. Show the intermediate steps, not just the answer,
and check one entry by hand in the text ("check one entry: $B(1, 0) = 3$, so ...").

- **Compute every number with a script**, never by hand. Keep scratch scripts outside the repo
  (in the session's scratchpad, not `src/`).
- **When the post has many figures, generate their data** from the verified computation into a
  TypeScript module (see `src/utils/matrix/cuteDslFigures.ts`) instead of hand-writing formulas
  in the MDX.
- **State the assumptions** behind any estimate (for example $\alpha = 10\ \mu\text{s}$,
  $50$ GB/s), and round consistently.

### 4. Simple to understand

- **Define every term on first use**, in bold: "a **process group** is ...". Spell out
  abbreviations the first time (tensor parallel, TP).
- **One idea per paragraph.** Short sentences, active voice, plain words.
- **Explain why, not just what.** "TP stays inside a server because it all-reduces in every
  layer" teaches more than "TP stays inside a server".
- **Build up gradually**: simple version first, then the refinement (naive kernel, then
  vectorized, then thread-value layout).
- **Use sidenotes** for detail that would break the flow: proofs, caveats, citations,
  verification notes.
- **End with a `## Summary`** of three to five bullets: the ideas a reader should keep.

## Structure and length

- `##` for sections, `###` for subsections. The sidebar table of contents is built
  automatically from `##` and `###` when a post has at least 3 of them; `####` is left out,
  so use it for fine-grained steps that would crowd the sidebar.
- Group related sections under one `##` (all six collectives under "The collectives") rather
  than many sibling sections. Keep the sidebar to roughly 25 entries.
- If a section grows into its own topic, offer to split it into a separate post and link the
  two.
- A typical order: Background → core concepts (each with a figure) → worked example →
  practical considerations → Summary → References.

## Accuracy

These posts teach, so a wrong number or claim is worse than a missing one.

- **Read the source, don't recall it.** For library internals (Megatron-LM, CUTLASS, vLLM,
  NCCL), fetch the code and quote what it does. Link to a pinned commit, not `main`.
- **Cross-check computed results** against an independent source when one exists (a
  notebook's recorded output, a docstring example, a paper's table).
- **Label anything illustrative or unverified**: "scaled-down version of ...", "follows the
  pattern described in the code's comments". If you could not run something (for example
  kernels that need a GPU), say so in your report to the user.
- **Prefer precise, checkable statements** ("vLLM uses one-shot all-reduce below 256 KiB on 8
  GPUs") over vague ones ("serving stacks favour few-step algorithms").
- Cite sources in a `## References` section or in sidenotes.

## Visualisation components

All are `.astro` components used from `.mdx` posts. Each file starts with a comment describing
its props and a `Usage in .mdx:` example; read it before using the component. Every component
has a demo page under `src/pages/demo/`, listed at `/demo`.

| Component | Path | Use for |
| --- | --- | --- |
| `Sidenote`, `MarginNote` | `src/components/` | Numbered and unnumbered margin notes |
| `LineChart` | `src/utils/charts/LineChart.astro` | Functions $y = f(x)$, several series, zoom |
| `ConfidenceChart` | `src/utils/charts/ConfidenceChart.astro` | A curve with a shaded confidence band and data points |
| `SurfacePlot` | `src/utils/charts/3DSurfacePlot.astro` | Rotatable 3D surface $z = f(x, y)$ |
| `MatrixInput` | `src/utils/matrix/MatrixInput.astro` | An editable matrix that drives other figures |
| `LayoutGrid` | `src/utils/matrix/LayoutGrid.astro` | A flat rank-2 CuTe layout as a grid of offsets |
| `Matrix2D` | `src/utils/matrix/Matrix2D.astro` | Nested rank-2 layouts, with per-cell values and coloured group labels (tiles, threads) |
| `GpuGemmDataflow` | `src/utils/gpu/GpuGemmDataflow.astro` | Step-through tiled GEMM on one SM |
| `FeedForward`, `NeuralNetworkGraph`, `SelfAttention` | `src/utils/neuralnetwork/` | Networks, weights and attention |
| `LogicalDistributedTopology` | `src/utils/distributed/LogicalDistributedTopology.astro` | A training job's TP/CP/EP/DP/PP groups, with hop counts |
| `PhysicalClusterTopology` | `src/utils/distributed/PhysicalClusterTopology.astro` | Servers, switches and links of a cluster |
| `CollectiveSteps` | `src/utils/distributed/CollectiveSteps.astro` | Step-through broadcast, reduce, reduce-scatter, all-gather, all-reduce, all-to-all with α–β cost |
| `PipelineSchedule` | `src/utils/distributed/PipelineSchedule.astro` | Pipeline-parallel schedules as timelines |

### Building a new component

When no component fits, build one rather than settling for a static image.

- Put it in `src/utils/<area>/`, with the computation in a sibling `.ts` file (as with
  `collectives.ts` and `pipeline.ts`) so the logic can be tested without rendering.
- Start the file with a comment explaining what it draws and a `Usage in .mdx:` example.
- Add a demo page in `src/pages/demo/` that exercises its main props.
- Match the site's look: colours from the CSS variables in `src/styles/site.css` (`--text`,
  `--muted`, `--faint`, `--surface`, `--accent`), with a dark-mode variant for any colour you
  add, and the play/step controls used by `CollectiveSteps` and `GpuGemmDataflow` for
  step-through figures.
- Test the logic with a script (run `.ts` files with `node --experimental-strip-types`) for
  several sizes, including edge cases.

## Workflow

1. **Research** the topic from primary sources; collect the code, docs and numbers you will cite.
2. **Outline** the sections and decide which figure goes with each.
3. **Compute** examples and figure data with scripts; cross-check them.
4. **Write** the post in `src/content/posts/<slug>.mdx`. The filename is the URL. Set `date` to
   today. Use `draft: true` only if the user asks for a draft.
5. **Build and check** (below).
6. **Report back** with what the post covers, what was verified and how, and anything you could
   not verify. Do not commit or push unless the user asks.

### Checks before handing a post back

- `npm run build` succeeds.
- No math errors: `grep -c katex-error dist/blog/<slug>/index.html` prints `0`.
- Every figure renders: count them in the built HTML, and look at the page (a headless Chrome
  screenshot of the built page works) to confirm figures are readable and not cut off.
- Figure data matches the computation it came from.
- New `.ts` and `.astro` files pass `npx biome check <files>`.

## Pitfalls seen in this repo

- **Stale dev content cache.** After changing the post schema in `src/content.config.ts`, a new
  field can work in `npm run build` but be missing in `npm run dev`. Delete
  `.astro/data-store.json` and restart the dev server.
- **One dev server per project.** Astro 7 refuses to start a second `astro dev` and may run it in
  the background; stop it with `npx astro dev stop`.
- **MDX syntax.** `{`, `}` and `<` are code in `.mdx`. Put them in backticks or code blocks, or
  escape them.
- **SVG `hidden`.** SVG elements ignore the `hidden` attribute; add a CSS rule such as
  `.step[hidden] { display: none; }`.
- **Global class names.** `site.css` defines layout classes such as `.frame`; don't reuse those
  names inside components.
- **Headless screenshots** of a long page can come out blank when scrolled to an anchor; capture
  the full page or load it in a shifted iframe instead.
