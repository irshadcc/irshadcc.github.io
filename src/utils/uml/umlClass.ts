// A UML class diagram drawn as SVG, for UmlClassDiagram.astro. Each class is a box with three
// compartments (name with an optional «stereotype», attributes, methods) placed on a grid by
// (col, row). A row is as tall as its tallest class, so boxes of different lengths stay aligned.
// Relations run between box borders along the line joining the centres and end in the UML
// marker for their kind: a filled diamond on the whole (composition), a hollow diamond
// (aggregation), an open arrow (association, dependency) or a hollow triangle on the parent
// (inheritance, realization). Dependencies and realizations are dashed.

export type RelationKind =
	| "composition"
	| "aggregation"
	| "association"
	| "dependency"
	| "inheritance"
	| "realization";

export interface UmlClass {
	id: string;
	name: string;
	/** Shown above the name in guillemets, e.g. "process" or "interface". */
	stereotype?: string;
	attrs?: string[];
	methods?: string[];
	col: number;
	row: number;
	/** Width in grid cells (default 1). */
	w?: number;
	/** Colour group 0..5, e.g. one per process. */
	group?: number;
	/** Shown under the figure when the class is selected. HTML allowed. */
	note?: string;
}

export interface UmlRelation {
	/** For composition and aggregation: the whole. For inheritance: the child. */
	from: string;
	/** For composition and aggregation: the part. For inheritance: the parent. */
	to: string;
	kind: RelationKind;
	label?: string;
}

export interface UmlGrid {
	/** Column width in px; by default wide enough for the longest member line. */
	cellW?: number;
	gapX?: number;
	gapY?: number;
}

export interface UmlLayout {
	width: number;
	height: number;
	svg: string;
}

const esc = (s: string) =>
	s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** Approximate advance of one character of the 9px monospace member font. */
export const MEMBER_CHAR_W = 5.45;
const NAME_CHAR_W = 6.3;
const LINE_H = 12;
const PAD_X = 7;
const NAME_H = 20;
const STEREO_H = 11;
const EMPTY_H = 6;

export function validateUml(
	classes: readonly UmlClass[],
	relations: readonly UmlRelation[],
): string | undefined {
	const ids = new Set<string>();
	const cells = new Set<string>();
	for (const c of classes) {
		if (ids.has(c.id)) return `duplicate class "${c.id}"`;
		ids.add(c.id);
		for (let k = 0; k < (c.w ?? 1); k++) {
			const cell = `${c.col + k},${c.row}`;
			if (cells.has(cell)) return `class "${c.id}" overlaps another at ${cell}`;
			cells.add(cell);
		}
	}
	for (const r of relations) {
		if (!ids.has(r.from)) return `relation from unknown class "${r.from}"`;
		if (!ids.has(r.to)) return `relation to unknown class "${r.to}"`;
		if (r.from === r.to) return `self relation on "${r.from}" is not supported`;
	}
	return undefined;
}

/** Height of a class box: name compartment, then attributes, then methods. */
export function classHeight(c: UmlClass): number {
	const head = NAME_H + (c.stereotype ? STEREO_H : 0);
	const block = (n: number) => (n > 0 ? n * LINE_H + 6 : EMPTY_H);
	return head + block(c.attrs?.length ?? 0) + block(c.methods?.length ?? 0);
}

/** Narrowest column that fits every class's longest line. */
export function fitCellWidth(
	classes: readonly UmlClass[],
	gapX: number,
): number {
	let cw = 120;
	for (const c of classes) {
		const lines = [...(c.attrs ?? []), ...(c.methods ?? [])];
		const need = Math.max(
			c.name.length * NAME_CHAR_W,
			...lines.map((l) => l.length * MEMBER_CHAR_W),
		);
		const w = c.w ?? 1;
		// Spread the need over the cells the class spans, minus the gaps it also covers.
		cw = Math.max(cw, Math.ceil((need + 2 * PAD_X - (w - 1) * gapX) / w));
	}
	return cw;
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

/** Box rectangles keyed by class id, plus the overall size. */
export function layoutClasses(
	classes: readonly UmlClass[],
	grid: UmlGrid = {},
): { rects: Map<string, Rect>; width: number; height: number } {
	const gx = grid.gapX ?? 64;
	const gy = grid.gapY ?? 38;
	const cw = grid.cellW ?? fitCellWidth(classes, gx);
	const m = 8;
	const rows = Math.max(...classes.map((c) => c.row)) + 1;
	const rowH = new Array<number>(rows).fill(0);
	for (const c of classes) rowH[c.row] = Math.max(rowH[c.row], classHeight(c));
	const rowY: number[] = [];
	let y = m;
	for (let r = 0; r < rows; r++) {
		rowY.push(y);
		y += rowH[r] + (rowH[r] > 0 ? gy : 0);
	}
	const rects = new Map<string, Rect>();
	for (const c of classes) {
		const w = c.w ?? 1;
		// Centre each box vertically in its row so short boxes don't hug the top.
		const h = classHeight(c);
		rects.set(c.id, {
			x: m + c.col * (cw + gx),
			y: rowY[c.row] + (rowH[c.row] - h) / 2,
			w: w * cw + (w - 1) * gx,
			h,
		});
	}
	const width = Math.max(...[...rects.values()].map((r) => r.x + r.w)) + m;
	const height = Math.max(...[...rects.values()].map((r) => r.y + r.h)) + m;
	return { rects, width, height };
}

const startMarker: Partial<Record<RelationKind, string>> = {
	composition: "dia-f",
	aggregation: "dia-h",
};
const endMarker: Partial<Record<RelationKind, string>> = {
	association: "arr",
	dependency: "arr",
	inheritance: "tri",
	realization: "tri",
};

export function drawUml(
	classes: readonly UmlClass[],
	relations: readonly UmlRelation[],
	grid: UmlGrid = {},
	id = "uml",
): UmlLayout {
	const { rects, width: w0, height } = layoutClasses(classes, grid);
	let width = w0;

	const lines = relations
		.map((r) => {
			const a = rects.get(r.from) as Rect;
			const b = rects.get(r.to) as Rect;
			const [x1, y1] = exit(a, b.x + b.w / 2, b.y + b.h / 2);
			const [x2, y2] = exit(b, a.x + a.w / 2, a.y + a.h / 2);
			const s = startMarker[r.kind];
			const e = endMarker[r.kind];
			const dashed = r.kind === "dependency" || r.kind === "realization";
			let lab = "";
			if (r.label) {
				const vertical = Math.abs(y2 - y1) > Math.abs(x2 - x1);
				const mx = (x1 + x2) / 2 + (vertical ? 5 : 0);
				const my = (y1 + y2) / 2 - (vertical ? 0 : 4);
				if (vertical) width = Math.max(width, mx + r.label.length * 5 + 8);
				lab = `<text class="ul" x="${mx}" y="${my}" text-anchor="${vertical ? "start" : "middle"}">${esc(r.label)}</text>`;
			}
			return `<g class="rel ${r.kind}${dashed ? " dashed" : ""}" data-from="${r.from}" data-to="${r.to}"><line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"${s ? ` marker-start="url(#${id}-${s})"` : ""}${e ? ` marker-end="url(#${id}-${e})"` : ""}/>${lab}</g>`;
		})
		.join("");

	const boxes = classes
		.map((c) => {
			const r = rects.get(c.id) as Rect;
			const cx = r.x + r.w / 2;
			const parts: string[] = [
				`<rect class="body" x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" rx="3"/>`,
			];
			let y = r.y;
			const headH = NAME_H + (c.stereotype ? STEREO_H : 0);
			parts.push(
				`<rect class="head" x="${r.x}" y="${r.y}" width="${r.w}" height="${headH}" rx="3"/>`,
			);
			if (c.stereotype) {
				parts.push(
					`<text class="st" x="${cx}" y="${y + 10}">«${esc(c.stereotype)}»</text>`,
				);
				y += STEREO_H;
			}
			parts.push(
				`<text class="cn" x="${cx}" y="${y + 14}">${esc(c.name)}</text>`,
			);
			y = r.y + headH;
			for (const block of [c.attrs ?? [], c.methods ?? []]) {
				parts.push(
					`<line class="sep" x1="${r.x}" x2="${r.x + r.w}" y1="${y}" y2="${y}"/>`,
				);
				if (block.length === 0) {
					y += EMPTY_H;
					continue;
				}
				y += 3;
				for (const line of block) {
					parts.push(
						`<text class="mb" x="${r.x + PAD_X}" y="${y + 9.5}">${esc(line)}</text>`,
					);
					y += LINE_H;
				}
				y += 3;
			}
			return `<g class="cls g${c.group ?? 0}${c.note ? " has-note" : ""}" data-id="${c.id}" tabindex="${c.note ? 0 : -1}">${parts.join("")}</g>`;
		})
		.join("");

	// Markers don't inherit colour from the line, so each one is styled by class in CSS.
	const defs = `<defs>
<marker id="${id}-dia-f" viewBox="0 0 14 8" refX="14" refY="4" markerWidth="14" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path class="mk fill" d="M0 4 L7 0 L14 4 L7 8 z"/></marker>
<marker id="${id}-dia-h" viewBox="0 0 14 8" refX="14" refY="4" markerWidth="14" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path class="mk hollow" d="M0.6 4 L7 0.6 L13.4 4 L7 7.4 z"/></marker>
<marker id="${id}-arr" viewBox="0 0 10 10" refX="9.5" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto"><path class="mk open" d="M0.5 0.8 L9.5 5 L0.5 9.2"/></marker>
<marker id="${id}-tri" viewBox="0 0 12 12" refX="11.4" refY="6" markerWidth="12" markerHeight="12" markerUnits="userSpaceOnUse" orient="auto"><path class="mk hollow" d="M0.6 0.6 L11.4 6 L0.6 11.4 z"/></marker>
</defs>`;
	return {
		width,
		height,
		svg: `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img">${defs}${lines}${boxes}</svg>`,
	};
}

/** Kinds that appear in a diagram, in a fixed order, for the legend. */
export function usedKinds(relations: readonly UmlRelation[]): RelationKind[] {
	const order: RelationKind[] = [
		"composition",
		"aggregation",
		"association",
		"dependency",
		"inheritance",
		"realization",
	];
	const used = new Set(relations.map((r) => r.kind));
	return order.filter((k) => used.has(k));
}
