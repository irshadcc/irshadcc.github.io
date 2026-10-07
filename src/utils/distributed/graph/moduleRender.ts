// Draws a laid-out module graph (moduleLayout.ts) as an SVG string: module and function boxes,
// wires with arrowheads, box labels, cards, and the collapse / expand buttons. Pure, so
// ModuleGraph.astro uses it at build time and again in the browser after a box is toggled.
import { iconSvg } from "./icons";
import {
	CARD,
	type Layout,
	type PlacedBlock,
	type PlacedNode,
	TOGGLE,
} from "./moduleLayout";

const MARGIN = 2;
const f = (v: number) => (Math.round(v * 10) / 10).toString();
const esc = (t: string) =>
	t
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;");

/** A small square button with − (collapse) or + (expand), its top-left corner at (x, y). */
function toggle(
	box: string,
	name: string,
	expanded: boolean,
	x: number,
	y: number,
): string {
	const c = TOGGLE / 2;
	const bar = `M${f(x + 4)},${f(y + c)}H${f(x + TOGGLE - 4)}`;
	const stem = expanded ? "" : `M${f(x + c)},${f(y + 4)}V${f(y + TOGGLE - 4)}`;
	const action = expanded ? "Collapse" : "Expand";
	return `<g class="mg-toggle" data-box="${esc(box)}" role="button" tabindex="0" aria-expanded="${expanded}" aria-label="${action} ${esc(name)}"><title>${action} ${esc(name)}</title><rect x="${f(x)}" y="${f(y)}" width="${TOGGLE}" height="${TOGGLE}" rx="3"/><path d="${bar}${stem}"/></g>`;
}

/** A card: box, icon at the left, title and subtitle; a collapsed box also gets its + button. */
function card(n: PlacedNode): string {
	const ix = n.x + CARD.pad;
	const tx = ix + CARD.icon + CARD.iconGap;
	const ty = n.subtitle ? n.y + 24 : n.y + n.h / 2 + 4.5;
	const icon = iconSvg(n.icon, ix, n.y + (n.h - CARD.icon) / 2, CARD.icon);
	const sub = n.subtitle
		? `<text class="mg-s" x="${f(tx)}" y="${f(ty + 19)}">${esc(n.subtitle)}</text>`
		: "";
	const button = n.collapsed
		? toggle(
				n.id,
				n.id || n.label,
				false,
				n.x + n.w - TOGGLE - 8,
				n.y + (n.h - TOGGLE) / 2,
			)
		: "";
	return `<g class="mg-node k-${n.kind}${n.collapsed ? " collapsed" : ""}" data-id="${esc(n.id)}" tabindex="0"><rect class="mg-cardbox" x="${f(n.x)}" y="${f(n.y)}" width="${f(n.w)}" height="${f(n.h)}" rx="6"/>${icon}<text class="mg-t" x="${f(tx)}" y="${f(ty)}">${esc(n.label)}</text>${sub}</g>${button}`;
}

/** A box's label; a box with weights gets a hover target over it, as wide as the text. */
function boxLabel(b: PlacedBlock): string {
	const text = `<text class="mg-box-t${b.card ? " has-card" : ""}" x="${f(b.x + 10)}" y="${f(b.y + 18)}">${esc(b.label)}</text>`;
	if (!b.card) return text;
	const w = Math.min([...b.label].length * 6.6 + 12, b.w - TOGGLE - 14);
	return `<rect class="mg-box-hit" data-box="${esc(b.id)}" tabindex="0" x="${f(b.x + 4)}" y="${f(b.y + 4)}" width="${f(w)}" height="22"/>${text}`;
}

export function renderSvg(layout: Layout, title?: string): string {
	const W = layout.width + 2 * MARGIN;
	const H = layout.height + 2 * MARGIN;
	const boxes = layout.blocks
		.map(
			(b) =>
				`<rect class="mg-box${b.dashed ? " dashed" : ""}" data-depth="${b.depth}" x="${f(b.x)}" y="${f(b.y)}" width="${f(b.w)}" height="${f(b.h)}" rx="8"/>`,
		)
		.join("");
	const wires = layout.edges
		.map(
			(e) =>
				`<g class="mg-edge" data-i="${e.index}" data-from="${esc(e.from)}" data-to="${esc(e.to)}"><path class="mg-wire" d="${e.d}"/><polygon class="mg-arrow" points="${e.arrow.map((p) => `${f(p.x)},${f(p.y)}`).join(" ")}"/><path class="mg-hit" d="${e.d}"/></g>`,
		)
		.join("");
	// Box labels and buttons go above the wires; the labels have a halo in the page colour, so a
	// wire entering a box cuts behind its label rather than through it.
	const labels = layout.blocks
		.map(
			(b) =>
				`${boxLabel(b)}${toggle(b.id, b.id || b.label, true, b.x + b.w - TOGGLE - 6, b.y + 6)}`,
		)
		.join("");
	return `<svg viewBox="${-MARGIN} ${-MARGIN} ${W} ${H}" width="${W}" height="${H}" style="min-width:${Math.round(W * 0.75)}px" role="img" aria-label="${esc(title ?? "Module dataflow graph")}">${boxes}<g class="mg-edges">${wires}</g>${labels}${layout.nodes.map(card).join("")}</svg>`;
}
