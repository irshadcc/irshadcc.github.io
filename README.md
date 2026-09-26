# irshadcc.github.io

A minimal technical blog built with [Astro](https://astro.build), loosely inspired by
[siboehm.com](https://siboehm.com). Deployed to GitHub Pages by `.github/workflows/astro.yml`
on every push to `main`.

## Commands

```bash
npm install      # install dependencies
npm run dev      # local dev server at http://localhost:4321
npm run build    # production build into dist/ (drafts excluded)
npm run preview  # serve the production build locally
```

## Writing guide

### 1. Create the post

Add a file to `src/content/posts/`. The filename becomes the URL:

```
src/content/posts/cuda-matmul.md   →   https://irshadcc.github.io/blog/cuda-matmul
```

Use `.md` for plain Markdown. Use `.mdx` if you want sidenotes or margin notes (see below).

### 2. Fill in the frontmatter

Every post starts with a frontmatter block:

```yaml
---
title: "How the PyTorch Dispatcher Works"
description: "One or two sentences. Shown under the title on the home page and in link previews."
date: 2026-01-15
updated: 2026-02-01   # optional: shown next to the date on the post
draft: true           # optional: see "Drafts" below
---
```

| Field         | Required | Notes                                                        |
| ------------- | :------: | ------------------------------------------------------------ |
| `title`       |   yes    | Post heading and browser tab title.                          |
| `description` |   yes    | Plain text, no Markdown. Keep it to one or two sentences.    |
| `date`        |   yes    | `YYYY-MM-DD`. Sets the order and year group on the home page. |
| `updated`     |    no    | Adds "updated …" to the post's date line.                    |
| `draft`       |    no    | `true` hides the post from the site. Defaults to `false`.    |

The reading time on each post is calculated automatically.

### 3. Write the body

Don't add a `# Title` heading: the page already shows `title`. Start sections at `##`.

**Headings.** `##` for sections, `###` for subsections, `####` sparingly.

**Text.** Standard Markdown: `**bold**`, `*italic*`, `` `inline code` ``, `[links](https://…)`,
lists, `> blockquotes` and `---` for a horizontal rule. Straight quotes and `--` are turned
into typographic quotes and dashes automatically.

**Code blocks.** Fence them with the language name for syntax highlighting. Colors switch
automatically in dark mode.

````md
```python
x = torch.randn(1024, 1024, device="cuda")
y = x @ x.T
```
````

Common languages: `python`, `cpp`, `bash`, `yaml`, `json`, `rust`, `go`, `sql`. There is no
CUDA grammar, so use `cpp` for CUDA kernels.

**Math.** Rendered with KaTeX. Use `$…$` inline and `$$…$$` on their own lines for display math:

```md
The layer computes $y = Wx + b$, followed by

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$
```

To write a literal dollar sign, escape it: `\$5`.

**Tables.** GitHub-style tables. Use `:` in the separator row to align columns (right-align numbers):

```md
| Kernel        | GFLOPs/s | % of cuBLAS |
| ------------- | -------: | ----------: |
| Naive         |    309.0 |        1.3% |
| cuBLAS        |  23249.6 |        100% |
```

**Images.** Put the image next to the post (or in a folder beside it) and use a relative path.
Astro optimizes it at build time:

```md
![Dispatcher call flow](./images/dispatcher-flow.png)
```

For a caption, use a `<figure>`. Files in `public/` are served as-is from the site root:

```html
<figure>
  <img src="/images/dispatcher-flow.png" alt="Dispatcher call flow" />
  <figcaption>How a call to torch.add reaches its kernel.</figcaption>
</figure>
```

### 4. Sidenotes and margin notes (`.mdx` only)

Notes sit in the right margin on wide screens. On narrower screens they fold into the text
and open when the reader taps the marker. Import the components once, below the frontmatter:

```mdx
import Sidenote from "../../components/Sidenote.astro";
import MarginNote from "../../components/MarginNote.astro";

The dispatcher looks up a kernel for every op.<Sidenote>Numbered automatically: 1, 2, 3…</Sidenote>
It uses a dispatch key set.<MarginNote>Unnumbered, shown with ⊕ on mobile.</MarginNote>
```

Put the component directly after the word it annotates, with no space before it, and keep it
inside the same paragraph. Notes can contain links, `code` and math.

MDX treats `{`, `}` and `<` as code. In `.mdx` files, write them inside backticks or code
blocks, or escape them as `\{` and `&lt;`.

### 5. Drafts and previewing

- Run `npm run dev` and open http://localhost:4321. The page reloads as you save.
- With `draft: true`, the post never appears on the home page and isn't published.
  While the dev server is running, you can still preview it by URL at `/blog/<filename>`.
- `src/content/posts/style-guide.mdx` is a draft that uses every feature above. Open
  http://localhost:4321/blog/style-guide to see how each one renders.

### 6. Publish

Remove `draft: true` (or set it to `false`), check it with `npm run build`, then commit and
push to `main`. GitHub Actions builds and deploys the site in a couple of minutes.

## Customising

- Name, socials: `src/site.ts`
- About page: `src/pages/about.astro`, with data in `src/content/work.json` and `src/content/publications.json`
- Styles: `src/styles/site.css` (colors are CSS variables at the top; dark mode follows the OS setting)
- Margin note components: `src/components/Sidenote.astro` and `src/components/MarginNote.astro`
