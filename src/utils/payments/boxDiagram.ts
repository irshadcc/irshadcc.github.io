// Boxes and arrows on a grid, for BoxDiagram.astro: an architecture or deployment diagram where
// each box can carry a note. Positions are in grid cells (col, row), so diagrams stay aligned
// without pixel arithmetic. Edges run between box borders, along the line joining the centres.

export interface BoxNode {
	id: string;
	label: string;
	sub?: string;
	col: number;
	row: number;
	/** Width and height in cells (default 1). */
	w?: number;
	h?: number;
	/** Colour group 0..5; -1 draws a dashed container behind other boxes. */
	group?: number;
	/** Shown under the figure when the box is selected. HTML allowed. */
	note?: string;
}

export interface BoxEdge {
	from: string;
	to: string;
	label?: string;
	/** Draw arrowheads at both ends. */
	both?: boolean;
	dashed?: boolean;
}

export interface BoxGrid {
	cellW?: number;
	cellH?: number;
	gapX?: number;
	gapY?: number;
}

const esc = (s: string) =>
	s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

export function validateBoxes(
	nodes: readonly BoxNode[],
	edges: readonly BoxEdge[],
): string | undefined {
	const ids = new Set<string>();
	for (const n of nodes) {
		if (ids.has(n.id)) return `duplicate node "${n.id}"`;
		ids.add(n.id);
	}
	for (const e of edges) {
		if (!ids.has(e.from)) return `edge from unknown node "${e.from}"`;
		if (!ids.has(e.to)) return `edge to unknown node "${e.to}"`;
	}
	return undefined;
}

interface Rect {
	x: number;
	y: number;
	w: number;
	h: number;
}

/** Where the segment from the centre of r towards (tx, ty) leaves r. */
function exit(r: Rect, tx: number, ty: number): [number, number] {
	const cx = r.x + r.w / 2;
	const cy = r.y + r.h / 2;
	const dx = tx - cx;
	const dy = ty - cy;
	if (dx === 0 && dy === 0) return [cx, cy];
	const s = Math.min(
		dx === 0 ? Number.POSITIVE_INFINITY : r.w / 2 / Math.abs(dx),
		dy === 0 ? Number.POSITIVE_INFINITY : r.h / 2 / Math.abs(dy),
	);
	return [cx + dx * s, cy + dy * s];
}

export function drawBoxes(
	nodes: readonly BoxNode[],
	edges: readonly BoxEdge[],
	grid: BoxGrid = {},
	id = "bx",
) {
	const cw = grid.cellW ?? 110;
	const ch = grid.cellH ?? 44;
	const gx = grid.gapX ?? 26;
	const gy = grid.gapY ?? 30;
	const m = 6;
	// A container's label sits 17px above its first row: leave room for it.
	const top = nodes.some((n) => n.group === -1 && n.row === 0) ? m + 17 : m;
	const rect = new Map<string, Rect>();
	for (const n of nodes) {
		const w = n.w ?? 1;
		const h = n.h ?? 1;
		rect.set(n.id, {
			x: m + n.col * (cw + gx),
			y: top + n.row * (ch + gy),
			w: w * cw + (w - 1) * gx,
			h: h * ch + (h - 1) * gy,
		});
	}
	let width = Math.max(...[...rect.values()].map((r) => r.x + r.w)) + m;
	const height = Math.max(...[...rect.values()].map((r) => r.y + r.h)) + m;

	const containers = nodes
		.filter((n) => n.group === -1)
		.map((n) => {
			const r = rect.get(n.id) as Rect;
			return `<g class="box ctr" data-id="${n.id}"><rect x="${r.x - 4}" y="${r.y - 17}" width="${r.w + 8}" height="${r.h + 21}" rx="10"/><text class="cl" x="${r.x + 2}" y="${r.y - 6}">${esc(n.label)}</text></g>`;
		})
		.join("");

	const lines = edges
		.map((e) => {
			const a = rect.get(e.from) as Rect;
			const b = rect.get(e.to) as Rect;
			const [x1, y1] = exit(a, b.x + b.w / 2, b.y + b.h / 2);
			const [x2, y2] = exit(b, a.x + a.w / 2, a.y + a.h / 2);
			const len = Math.hypot(x2 - x1, y2 - y1) || 1;
			const ux = (x2 - x1) / len;
			const uy = (y2 - y1) / len;
			const vertical = Math.abs(uy) > 0.5;
			// Labels beside vertical edges can stick out past the last column: widen to fit.
			if (e.label && vertical)
				width = Math.max(width, (x1 + x2) / 2 + 5 + e.label.length * 5.2 + m);
			const lab = e.label
				? `<text class="el" x="${(x1 + x2) / 2 + (vertical ? 5 : 0)}" y="${(y1 + y2) / 2 - (vertical ? 0 : 4)}" text-anchor="${vertical ? "start" : "middle"}">${esc(e.label)}</text>`
				: "";
			return `<g class="edge${e.dashed ? " dashed" : ""}" data-from="${e.from}" data-to="${e.to}"><line x1="${x1 + ux * 2}" y1="${y1 + uy * 2}" x2="${x2 - ux * 3}" y2="${y2 - uy * 3}" marker-end="url(#${id}-ah)"${e.both ? ` marker-start="url(#${id}-ah)"` : ""}/>${lab}</g>`;
		})
		.join("");

	const boxes = nodes
		.filter((n) => n.group !== -1)
		.map((n) => {
			const r = rect.get(n.id) as Rect;
			const cx = r.x + r.w / 2;
			const cy = r.y + r.h / 2;
			return `<g class="box g${n.group ?? 0}${n.note ? " has-note" : ""}" data-id="${n.id}" tabindex="${n.note ? 0 : -1}"><rect x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" rx="6"/><text class="bl" x="${cx}" y="${n.sub ? cy - 2 : cy + 4}">${esc(n.label)}</text>${n.sub ? `<text class="bs" x="${cx}" y="${cy + 11}">${esc(n.sub)}</text>` : ""}</g>`;
		})
		.join("");

	const defs = `<defs><marker id="${id}-ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="ah" d="M0 0 L8 4 L0 8 z"/></marker></defs>`;
	return {
		width,
		height,
		svg: `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img">${defs}${containers}${lines}${boxes}</svg>`,
	};
}
