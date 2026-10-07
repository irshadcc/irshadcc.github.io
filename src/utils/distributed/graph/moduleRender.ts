// Draws a laid-out module graph (moduleLayout.ts) as an SVG string: module and function boxes,
// wires with rounded corners, arrowheads and dots where bundled wires split or join, box labels
// as chips, cards (an accent bar and a tinted icon tile in their kind's colour), and the
// collapse / expand buttons. Pure, so
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

/** Corner radius of the wires. */
const WIRE_R = 6;
type Pt = { x: number; y: number };

/** The corners of a right-angled path made of M, H and V commands, without repeats or straight-through points. */
export function pathPoints(d: string): Pt[] {
	const pts: Pt[] = [];
	for (const [, cmd, arg] of d.matchAll(/([MHV])([^MHV]+)/g)) {
		const last = pts[pts.length - 1];
		if (cmd === "M") {
			const [x, y] = arg.split(",").map(Number);
			pts.push({ x, y });
		} else if (cmd === "H") pts.push({ x: Number(arg), y: last.y });
		else pts.push({ x: last.x, y: Number(arg) });
	}
	const out: Pt[] = [];
	for (const p of pts) {
		const a = out[out.length - 1];
		if (a && Math.abs(a.x - p.x) < 0.05 && Math.abs(a.y - p.y) < 0.05) continue;
		const b = out[out.length - 2];
		// Drop the middle of three points on one line.
		if (
			a &&
			b &&
			((Math.abs(b.x - a.x) < 0.05 && Math.abs(a.x - p.x) < 0.05) ||
				(Math.abs(b.y - a.y) < 0.05 && Math.abs(a.y - p.y) < 0.05))
		)
			out.pop();
		out.push(p);
	}
	return out;
}

/** The path through the points with each corner rounded (at most half of either segment). */
export function roundedPath(pts: Pt[], r = WIRE_R): string {
	if (!pts.length) return "";
	let d = `M${f(pts[0].x)},${f(pts[0].y)}`;
	for (let i = 1; i < pts.length - 1; i++) {
		const [a, p, b] = [pts[i - 1], pts[i], pts[i + 1]];
		const lin = Math.hypot(p.x - a.x, p.y - a.y);
		const lout = Math.hypot(b.x - p.x, b.y - p.y);
		const k = Math.min(r, lin / 2, lout / 2);
		const s0 = {
			x: p.x - ((p.x - a.x) / lin) * k,
			y: p.y - ((p.y - a.y) / lin) * k,
		};
		const s1 = {
			x: p.x + ((b.x - p.x) / lout) * k,
			y: p.y + ((b.y - p.y) / lout) * k,
		};
		d += `L${f(s0.x)},${f(s0.y)}Q${f(p.x)},${f(p.y)} ${f(s1.x)},${f(s1.y)}`;
	}
	const z = pts[pts.length - 1];
	return `${d}L${f(z.x)},${f(z.y)}`;
}

/**
 * Points where three or more wire arms meet: a bundled wire splitting or two joining. Arms are
 * the directions (up, down, left, right) that wire segments leave a point in.
 */
export function junctions(paths: Pt[][]): Pt[] {
	const near = (u: number, v: number) => Math.abs(u - v) < 0.6;
	const segs = paths.flatMap((ps) =>
		ps.slice(1).map((b, i) => [ps[i], b] as const),
	);
	const seen: Pt[] = [];
	const out: Pt[] = [];
	for (const p of paths.flatMap((ps) => ps.slice(1, -1))) {
		if (seen.some((q) => near(q.x, p.x) && near(q.y, p.y))) continue;
		seen.push(p);
		const arms = new Set<string>();
		const arm = (q: Pt) =>
			arms.add(
				near(q.x, p.x) ? (q.y < p.y ? "u" : "d") : q.x < p.x ? "l" : "r",
			);
		for (const [a, b] of segs) {
			if (near(a.x, p.x) && near(a.y, p.y)) arm(b);
			else if (near(b.x, p.x) && near(b.y, p.y)) arm(a);
			else if (
				(near(a.x, b.x) && near(a.x, p.x) && (p.y - a.y) * (p.y - b.y) < 0) ||
				(near(a.y, b.y) && near(a.y, p.y) && (p.x - a.x) * (p.x - b.x) < 0)
			) {
				arm(a);
				arm(b);
			}
		}
		if (arms.size >= 3) out.push(p);
	}
	return out;
}

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
	const iy = n.y + (n.h - CARD.icon) / 2;
	const tile = `<rect class="mg-tile" x="${f(ix - 5)}" y="${f(iy - 5)}" width="${CARD.icon + 10}" height="${CARD.icon + 10}" rx="7"/>`;
	const bar = `<rect class="mg-bar" x="${f(n.x + 3)}" y="${f(n.y + 8)}" width="2.5" height="${f(n.h - 16)}" rx="1.25"/>`;
	return `<g class="mg-node k-${n.kind}${n.collapsed ? " collapsed" : ""}" data-id="${esc(n.id)}" tabindex="0"><rect class="mg-cardbox" x="${f(n.x)}" y="${f(n.y)}" width="${f(n.w)}" height="${f(n.h)}" rx="9"/>${bar}${tile}${icon}<text class="mg-t" x="${f(tx)}" y="${f(ty)}">${esc(n.label)}</text>${sub}</g>${button}`;
}

/** A box's label; a box with weights gets a hover target over it, as wide as the text. */
function boxLabel(b: PlacedBlock): string {
	const w = Math.min([...b.label].length * 6.6 + 14, b.w - TOGGLE - 16);
	const chip = `<rect class="mg-chip${b.dashed ? " fn" : ""}" x="${f(b.x + 6)}" y="${f(b.y + 6)}" width="${f(w)}" height="18" rx="9"/>`;
	const text = `<text class="mg-box-t${b.card ? " has-card" : ""}" x="${f(b.x + 13)}" y="${f(b.y + 18.5)}">${esc(b.label)}</text>`;
	if (!b.card) return chip + text;
	return `${chip}<rect class="mg-box-hit" data-box="${esc(b.id)}" tabindex="0" x="${f(b.x + 4)}" y="${f(b.y + 4)}" width="${f(w + 4)}" height="22"/>${text}`;
}

export function renderSvg(layout: Layout, title?: string): string {
	const W = layout.width + 2 * MARGIN;
	const H = layout.height + 2 * MARGIN;
	const boxes = layout.blocks
		.map(
			(b) =>
				`<rect class="mg-box${b.dashed ? " dashed" : ""}" data-depth="${b.depth}" x="${f(b.x)}" y="${f(b.y)}" width="${f(b.w)}" height="${f(b.h)}" rx="12"/>`,
		)
		.join("");
	const paths = layout.edges.map((e) => pathPoints(e.d));
	const wires = layout.edges
		.map((e, i) => {
			const d = roundedPath(paths[i]);
			return `<g class="mg-edge" data-i="${e.index}" data-from="${esc(e.from)}" data-to="${esc(e.to)}"><path class="mg-wire" d="${d}"/><path class="mg-flow" d="${d}"/><polygon class="mg-arrow" points="${e.arrow.map((p) => `${f(p.x)},${f(p.y)}`).join(" ")}"/><path class="mg-hit" d="${d}"/></g>`;
		})
		.join("");
	const dots = junctions(paths)
		.map(
			(p) =>
				`<circle class="mg-junction" cx="${f(p.x)}" cy="${f(p.y)}" r="2.6"/>`,
		)
		.join("");
	// Box labels and buttons go above the wires; the labels sit on opaque chips, so a wire
	// entering a box passes behind its label rather than through it.
	const labels = layout.blocks
		.map(
			(b) =>
				`${boxLabel(b)}${toggle(b.id, b.id || b.label, true, b.x + b.w - TOGGLE - 6, b.y + 6)}`,
		)
		.join("");
	return `<svg viewBox="${-MARGIN} ${-MARGIN} ${W} ${H}" width="${W}" height="${H}" style="min-width:${Math.round(W * 0.75)}px" role="img" aria-label="${esc(title ?? "Module dataflow graph")}">${boxes}<g class="mg-edges">${wires}</g><g class="mg-junctions">${dots}</g>${labels}${layout.nodes.map(card).join("")}</svg>`;
}
