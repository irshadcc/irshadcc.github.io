// Lays out a lowered module graph (moduleGraph.ts) top to bottom. The boxes come from the nodes'
// dotted module names: "experts.0" sits in "experts", which sits in the root "". One box at a
// time, innermost first. Inside a box, its children (nodes and child boxes, each a fixed-size item) are ranked
// by their longest path from the box's sources, ordered within each row to reduce crossings
// (barycentre sweeps; `layout.order` pins named children left to right), and placed as close
// as possible above or below what they connect to. An edge that skips rows gets a lane (a
// dummy cell) in every row it crosses; an edge that leaves or enters a box gets a lane down to
// the box's bottom (from its top) and a port there, so every edge is routed through free space
// and each box is exactly as big as its contents. Lanes are bundled: edges into one node share a
// trunk that the others join, and the remaining edges from one source share one until they branch.
import { type LoweredGraph, type Node, moduleLabel } from "./moduleGraph";

/** The invisible outermost box: the root's input nodes, the root box, its output nodes. */
const FRAME = "$frame";
/** The box enclosing the module box at `path`. */
const parentPath = (path: string) =>
	path === ""
		? FRAME
		: path.includes(".")
			? path.slice(0, path.lastIndexOf("."))
			: "";

/**
 * Cards: an icon at the left, a title and an optional subtitle on a second line, in monospace.
 * Text widths are estimated from character counts (SVG text can't be measured at build time):
 * titles at 13px, subtitles and box labels at 11px.
 */
export const CARD = {
	pad: 14,
	icon: 22,
	iconGap: 12,
	titleChar: 7.8,
	subChar: 6.6,
	h: 40,
	hWithSub: 58,
};
const BOX_CHAR_W = 6.6;
/** The collapse/expand button: its side, and the room it takes beside a label. */
export const TOGGLE = 16;
const TOGGLE_ROOM = TOGGLE + 10;
export const ARROW_LEN = 7;
const ARROW_HALF = 3.5;
/** Box padding: the top strip holds the box's label. */
const BOX = { top: 30, side: 14, bottom: 14 };
const ROW_GAP = 38;
const ITEM_GAP = 22;
/** Gaps next to a lane: to another lane, to a node, to a box (wider, so a lane never reads as a border). */
const LANE_GAP = 8;
const LANE_NODE_GAP = 12;
const LANE_BOX_GAP = 22;

export function nodeSize(n: Node): { w: number; h: number } {
	const text = Math.max(
		[...n.label].length * CARD.titleChar,
		[...(n.subtitle ?? "")].length * CARD.subChar,
	);
	return {
		w: Math.ceil(
			2 * CARD.pad +
				CARD.icon +
				CARD.iconGap +
				text +
				4 +
				(n.collapsed ? TOGGLE_ROOM : 0),
		),
		h: n.subtitle ? CARD.hWithSub : CARD.h,
	};
}

export interface Rect {
	x: number;
	y: number;
	w: number;
	h: number;
}
export interface PlacedNode extends Node, Rect {}
export interface PlacedBlock extends Rect {
	/** Dotted module name ("" for the root). */
	id: string;
	label: string;
	dashed: boolean;
	depth: number;
	/** The module lists weights: its label opens a hover card. */
	card: boolean;
}
export interface PlacedEdge {
	index: number;
	from: string;
	to: string;
	/** SVG path, right angles only; it stops at the arrowhead's base. */
	d: string;
	/** Arrowhead: tip (on the target's top edge), then the two base corners. */
	arrow: { x: number; y: number }[];
}
export interface Layout {
	nodes: PlacedNode[];
	/** Outermost first, so inner boxes draw on top. */
	blocks: PlacedBlock[];
	edges: PlacedEdge[];
	width: number;
	height: number;
}

interface Cell {
	id: string;
	dummy: boolean;
	/** A child box rather than a node or a lane. */
	box: boolean;
	w: number;
	h: number;
	rank: number;
	/** Centre, in the box's own coordinates. */
	x: number;
	y: number;
	preds: Cell[];
	succs: Cell[];
}

interface BoxLayout {
	w: number;
	h: number;
	cells: Map<string, Cell>;
	rows: Cell[][];
	rowTop: number[];
	rowH: number[];
	/** Each edge's lane cells by key: `i|src|to` inside, `in|src|to` from the top, `out|src` to the bottom. */
	routes: Map<string, Cell[]>;
	/** Port x (box coordinates) on the top edge, per edge: key `targetNode|src`. */
	inPort: Map<string, number>;
	/** Port x on the bottom edge, one per producing node. */
	outPort: Map<string, number>;
}

export function layoutModule(g: LoweredGraph): Layout {
	// Parents: a node's box is its module (FRAME for the root's inputs and outputs); a box's is
	// its dotted name minus the last part. Every prefix of a node's module is a box.
	const parentOf = new Map<string, string>();
	const boxIds: string[] = [];
	const seq = new Map<string, number>();
	g.nodes.forEach((n, i) => {
		parentOf.set(n.id, n.module ?? FRAME);
		seq.set(n.id, i);
		for (let m = n.module; m !== null && m !== FRAME; m = parentPath(m)) {
			if (!seq.has(m)) {
				boxIds.push(m);
				seq.set(m, i);
				parentOf.set(m, parentPath(m));
			}
		}
	});
	const nodeById = new Map(g.nodes.map((n) => [n.id, n]));
	const children = new Map<string, string[]>([[FRAME, []]]);
	for (const b of boxIds) children.set(b, []);
	for (const id of [...g.nodes.map((n) => n.id), ...boxIds])
		children.get(parentOf.get(id) ?? FRAME)?.push(id);
	for (const list of children.values())
		list.sort((a, b) => (seq.get(a) ?? 0) - (seq.get(b) ?? 0));
	const labelOf = (box: string) => moduleLabel(box, g.modules[box]);

	/** The boxes around a node, innermost first, ending with FRAME. */
	const ancestors = (id: string) => {
		const out: string[] = [];
		for (let p = parentOf.get(id); p !== undefined; p = parentOf.get(p))
			out.push(p);
		return out;
	};
	/** The child of `box` that holds `id` (id itself when it is a direct child), or undefined. */
	const itemOf = (box: string, id: string): string | undefined => {
		let cur = id;
		for (let p = parentOf.get(cur); p !== undefined; p = parentOf.get(cur)) {
			if (p === box) return cur;
			cur = p;
		}
		return undefined;
	};
	const depthOf = (box: string): number =>
		box === FRAME ? -1 : 1 + depthOf(parentOf.get(box) ?? FRAME);

	const boxes = new Map<string, BoxLayout>();
	const order = [...boxIds].sort((a, b) => depthOf(b) - depthOf(a));
	for (const id of [...order, FRAME]) boxes.set(id, layoutBox(id));

	function layoutBox(box: string): BoxLayout {
		const frame = box === FRAME;
		const pad = frame ? { top: 0, side: 0, bottom: 0 } : BOX;
		const items = children.get(box) ?? [];
		const size = (id: string) => {
			const n = nodeById.get(id);
			if (n) return nodeSize(n);
			const l = boxes.get(id) as BoxLayout;
			return { w: l.w, h: l.h };
		};

		// Classify every edge by where it runs relative to this box.
		type Inner = { a: string; b: string; src: string; to: string };
		const inner: Inner[] = [];
		const incoming: { b: string; src: string; to: string }[] = [];
		const outgoing: { a: string; src: string }[] = [];
		for (const e of g.edges) {
			const a = itemOf(box, e.from);
			const b = itemOf(box, e.to);
			if (a !== undefined && b !== undefined && a !== b)
				inner.push({ a, b, src: e.from, to: e.to });
			else if (a !== undefined && b === undefined)
				outgoing.push({ a, src: e.from });
			else if (a === undefined && b !== undefined)
				incoming.push({ b, src: e.from, to: e.to });
		}

		// Ranks: longest path from the box's sources.
		const succ = new Map(items.map((i) => [i, new Set<string>()]));
		for (const { a, b } of inner) succ.get(a)?.add(b);
		const indeg = new Map(items.map((i) => [i, 0]));
		for (const s of succ.values())
			for (const b of s) indeg.set(b, (indeg.get(b) ?? 0) + 1);
		const rank = new Map(items.map((i) => [i, 0]));
		const queue = items.filter((i) => indeg.get(i) === 0);
		let done = 0;
		while (queue.length) {
			const v = queue.shift() as string;
			done++;
			for (const w of succ.get(v) ?? []) {
				rank.set(w, Math.max(rank.get(w) ?? 0, (rank.get(v) ?? 0) + 1));
				indeg.set(w, (indeg.get(w) ?? 0) - 1);
				if (indeg.get(w) === 0) queue.push(w);
			}
		}
		if (done < items.length)
			throw new Error(
				`ModuleGraph: the children of ${box} depend on each other in a cycle`,
			);
		const maxRank = Math.max(0, ...rank.values());

		const cells = new Map<string, Cell>();
		const rows: Cell[][] = Array.from({ length: maxRank + 1 }, () => []);
		const addCell = (
			id: string,
			dummy: boolean,
			r: number,
			w: number,
			h: number,
		) => {
			const c: Cell = {
				id,
				dummy,
				box: boxes.has(id),
				w,
				h,
				rank: r,
				x: 0,
				y: 0,
				preds: [],
				succs: [],
			};
			cells.set(id, c);
			rows[r].push(c);
			return c;
		};
		for (const i of items)
			addCell(i, false, rank.get(i) ?? 0, size(i).w, size(i).h);
		const link = (u: Cell, v: Cell) => {
			if (u.succs.includes(v)) return;
			u.succs.push(v);
			v.preds.push(u);
		};
		// Lanes: an edge that skips rows runs down a column of dummy cells, one per row it crosses.
		// Lanes are shared so that wires bundle instead of running side by side: the edges into
		// one node join one trunk (each enters it in the gap above its first row, |__ __|), and
		// the other edges from one source share the source's trunk, each leaving it in the gap
		// above its target. `routes` holds each edge's slice of a trunk, by key: `i|src|to`
		// inside the box, `in|src|to` from its top edge, `out|src` to its bottom edge.
		const routes = new Map<string, Cell[]>();
		const trunk = (key: string, from: number, to: number) => {
			const out: Cell[] = [];
			for (let r = from; r <= to; r++)
				out.push(addCell(`${key}#${r}`, true, r, 0, 0));
			for (let k = 1; k < out.length; k++) link(out[k - 1], out[k]);
			return out;
		};
		type Route = {
			key: string;
			src: string;
			to?: string;
			a?: string;
			b?: string;
			/** The rows its lane crosses; empty when lo > hi. */
			lo: number;
			hi: number;
		};
		const routeOf = new Map<string, Route>();
		const rankOf = (id: string) => rank.get(id) ?? 0;
		for (const { a, b, src, to } of inner) {
			const key = `i|${src}|${to}`;
			if (!routeOf.has(key))
				routeOf.set(key, {
					key,
					src,
					to,
					a,
					b,
					lo: rankOf(a) + 1,
					hi: rankOf(b) - 1,
				});
		}
		for (const { b, src, to } of incoming) {
			const key = `in|${src}|${to}`;
			if (!routeOf.has(key))
				routeOf.set(key, { key, src, to, b, lo: 0, hi: rankOf(b) - 1 });
		}
		for (const { a, src } of outgoing) {
			const key = `out|${src}`;
			if (!routeOf.has(key))
				routeOf.set(key, { key, src, a, lo: rankOf(a) + 1, hi: maxRank });
		}
		const groups = (rs: Route[], by: (r: Route) => string) => {
			const m = new Map<string, Route[]>();
			for (const r of rs) m.set(by(r), [...(m.get(by(r)) ?? []), r]);
			return m;
		};
		const lanes = [...routeOf.values()].filter((r) => r.lo <= r.hi);
		const merged = new Set<Route>();
		for (const [to, rs] of groups(
			lanes.filter((r) => r.to !== undefined),
			(r) => r.to as string,
		)) {
			if (rs.length < 2) continue;
			const lo = Math.min(...rs.map((r) => r.lo));
			const t = trunk(`fi|${to}`, lo, rs[0].hi);
			for (const r of rs) {
				routes.set(r.key, t.slice(r.lo - lo));
				merged.add(r);
			}
		}
		for (const [src, rs] of groups(
			lanes.filter((r) => !merged.has(r)),
			(r) => r.src,
		)) {
			const lo = Math.min(...rs.map((r) => r.lo));
			const t = trunk(`fo|${src}`, lo, Math.max(...rs.map((r) => r.hi)));
			for (const r of rs) routes.set(r.key, t.slice(r.lo - lo, r.hi - lo + 1));
		}
		// Ordering links: the source item to its route's first cell, the last cell to the target.
		for (const r of routeOf.values()) {
			const cs = routes.get(r.key) ?? [];
			const ca = r.a === undefined ? undefined : cells.get(r.a);
			const cb = r.b === undefined ? undefined : cells.get(r.b);
			if (!cs.length) {
				if (ca && cb) link(ca, cb);
				continue;
			}
			if (ca) link(ca, cs[0]);
			if (cb) link(cs[cs.length - 1], cb);
		}

		orderRows(
			rows,
			(g.modules[box]?.order ?? []).filter((id) => cells.has(id)),
		);
		placeRows(rows);

		// Vertical placement: rows as tall as their tallest item, items centred in their row.
		const rowH = rows.map((r) => Math.max(0, ...r.map((c) => c.h)));
		const rowTop: number[] = [];
		let y = pad.top;
		for (const h of rowH) {
			rowTop.push(y);
			y += h + ROW_GAP;
		}
		const height = y - ROW_GAP + pad.bottom;
		const all = [...cells.values()];
		const minX = Math.min(...all.map((c) => c.x - c.w / 2));
		const maxX = Math.max(...all.map((c) => c.x + c.w / 2));
		const label = frame
			? 0
			: [...labelOf(box)].length * BOX_CHAR_W + 8 + TOGGLE_ROOM;
		const inner_w = Math.max(maxX - minX, label);
		const shift = pad.side - minX + (inner_w - (maxX - minX)) / 2;
		for (const c of all) {
			c.x += shift;
			c.y = rowTop[c.rank] + rowH[c.rank] / 2;
		}
		const layout: BoxLayout = {
			w: inner_w + 2 * pad.side,
			h: height,
			cells,
			rows,
			rowTop,
			rowH,
			routes,
			inPort: new Map(),
			outPort: new Map(),
		};

		// Ports: where each edge crosses the top edge (straight down its lane, or straight into
		// the item below), and each tensor the bottom edge.
		for (const { b, src, to } of incoming) {
			const ds = routes.get(`in|${src}|${to}`) ?? [];
			layout.inPort.set(
				`${to}|${src}`,
				ds.length ? ds[0].x : entryX(cells.get(b) as Cell, src, to),
			);
		}
		for (const { a, src } of outgoing) {
			if (layout.outPort.has(src)) continue;
			const ds = routes.get(`out|${src}`) ?? [];
			layout.outPort.set(
				src,
				ds.length ? ds[ds.length - 1].x : exitX(cells.get(a) as Cell, src),
			);
		}
		return layout;
	}

	/** Where an edge from `src` to `to` enters an item, in its parent box's coordinates. */
	function entryX(c: Cell, src: string, to: string): number {
		const l = boxes.get(c.id);
		if (!l) return c.x;
		const port = l.inPort.get(`${to}|${src}`);
		return c.x - c.w / 2 + (port ?? c.w / 2);
	}
	function exitX(c: Cell, src: string): number {
		const l = boxes.get(c.id);
		return l ? c.x - c.w / 2 + (l.outPort.get(src) ?? c.w / 2) : c.x;
	}

	// Absolute positions, from the frame inwards.
	const abs = new Map<string, Rect>([
		[
			FRAME,
			{ x: 0, y: 0, w: boxes.get(FRAME)?.w ?? 0, h: boxes.get(FRAME)?.h ?? 0 },
		],
	]);
	const place = (box: string) => {
		const at = abs.get(box) as Rect;
		const l = boxes.get(box) as BoxLayout;
		for (const c of l.cells.values()) {
			if (c.dummy) continue;
			abs.set(c.id, {
				x: at.x + c.x - c.w / 2,
				y: at.y + c.y - c.h / 2,
				w: c.w,
				h: c.h,
			});
			if (boxes.has(c.id)) place(c.id);
		}
	};
	place(FRAME);

	// Edges: a list of waypoints (lanes and ports) joined by right-angled steps. Each waypoint
	// spans a whole row band of its box (the space above and below a short item in its row is
	// free), so every horizontal step falls in the gap between two rows, never across an item.
	type Way = { x: number; top: number; bottom: number };
	const cellOf = new Map<string, { box: string; cell: Cell }>();
	for (const [box, l] of boxes)
		for (const c of l.cells.values())
			if (!c.dummy) cellOf.set(c.id, { box, cell: c });
	/** The top and bottom of the row an item sits in, in absolute coordinates. */
	const band = (id: string) => {
		const { box, cell } = cellOf.get(id) as { box: string; cell: Cell };
		const l = boxes.get(box) as BoxLayout;
		const top = (abs.get(box) as Rect).y + l.rowTop[cell.rank];
		return { top, bottom: top + l.rowH[cell.rank] };
	};
	const laneWays = (box: string, key: string): Way[] => {
		const l = boxes.get(box) as BoxLayout;
		const at = abs.get(box) as Rect;
		return (l.routes.get(key) ?? []).map((c) => ({
			x: at.x + c.x,
			top: at.y + l.rowTop[c.rank],
			bottom: at.y + l.rowTop[c.rank] + l.rowH[c.rank],
		}));
	};
	const edges = g.edges.map((e, index): PlacedEdge => {
		const s = abs.get(e.from) as Rect;
		const t = abs.get(e.to) as Rect;
		const up = ancestors(e.from);
		const downSet = new Set(ancestors(e.to));
		const lca = up.find((b) => downSet.has(b)) ?? FRAME;
		const ways: Way[] = [];
		for (const b of up) {
			if (b === lca) break;
			ways.push(...laneWays(b, `out|${e.from}`));
			const at = abs.get(b) as Rect;
			const x =
				at.x + ((boxes.get(b) as BoxLayout).outPort.get(e.from) ?? at.w / 2);
			ways.push({ x, top: at.y + at.h, bottom: band(b).bottom });
		}
		ways.push(...laneWays(lca, `i|${e.from}|${e.to}`));
		const down = ancestors(e.to);
		for (const b of down.slice(0, down.indexOf(lca)).reverse()) {
			const at = abs.get(b) as Rect;
			const port = (boxes.get(b) as BoxLayout).inPort.get(`${e.to}|${e.from}`);
			ways.push({
				x: at.x + (port ?? at.w / 2),
				top: band(b).top,
				bottom: at.y,
			});
			ways.push(...laneWays(b, `in|${e.from}|${e.to}`));
		}
		ways.push({ x: t.x + t.w / 2, top: band(e.to).top, bottom: t.y });
		// The last waypoint ends on the target's top edge; stop short of it for the arrowhead.
		const last = ways[ways.length - 1];
		last.bottom -= ARROW_LEN;
		last.top = Math.min(last.top, last.bottom);
		let [px, py] = [s.x + s.w / 2, band(e.from).bottom];
		let d = `M${r1(px)},${r1(s.y + s.h)}V${r1(py)}`;
		for (const w of ways) {
			if (Math.abs(w.x - px) < 0.5) d += `V${r1(w.top)}`;
			else {
				const mid = (py + w.top) / 2;
				d += `V${r1(mid)}H${r1(w.x)}V${r1(w.top)}`;
			}
			if (w.bottom > w.top) d += `V${r1(w.bottom)}`;
			[px, py] = [w.x, w.bottom];
		}
		const tip = { x: t.x + t.w / 2, y: t.y };
		const arrow = [
			tip,
			{ x: tip.x - ARROW_HALF, y: tip.y - ARROW_LEN },
			{ x: tip.x + ARROW_HALF, y: tip.y - ARROW_LEN },
		];
		return { index: e.index ?? index, from: e.from, to: e.to, d, arrow };
	});

	const nodes = g.nodes.map((n) => ({ ...n, ...(abs.get(n.id) as Rect) }));
	const blocks = boxIds
		.map((id) => ({
			id,
			label: labelOf(id),
			dashed: g.modules[id]?.function !== undefined,
			card: g.modules[id]?.weights !== undefined,
			...(abs.get(id) as Rect),
			depth: depthOf(id),
		}))
		.sort((a, b) => a.depth - b.depth);
	const f = boxes.get(FRAME) as BoxLayout;
	return {
		nodes,
		blocks,
		edges,
		width: Math.ceil(f.w),
		height: Math.ceil(f.h),
	};
}

const r1 = (v: number) => Math.round(v * 10) / 10;

/** Barycentre sweeps, keeping the order with the fewest crossings; pinned cells keep their order. */
function orderRows(rows: Cell[][], pinned: string[]) {
	const pin = new Map(pinned.map((id, i) => [id, i]));
	const index = new Map<Cell, number>();
	const reindex = () => {
		for (const r of rows) r.forEach((c, i) => index.set(c, i));
	};
	const applyPins = (row: Cell[]) => {
		const slots = row
			.map((c, i) => (pin.has(c.id) ? i : -1))
			.filter((i) => i >= 0);
		const pinnedCells = slots
			.map((i) => row[i])
			.sort((a, b) => (pin.get(a.id) ?? 0) - (pin.get(b.id) ?? 0));
		slots.forEach((slot, k) => {
			row[slot] = pinnedCells[k];
		});
	};
	for (const r of rows) applyPins(r);
	reindex();
	const sortRow = (r: number, neighbours: (c: Cell) => Cell[]) => {
		const row = rows[r];
		const key = new Map(
			row.map((c) => {
				const ns = neighbours(c);
				return [
					c,
					ns.length
						? ns.reduce((s, n) => s + (index.get(n) ?? 0), 0) / ns.length
						: (index.get(c) ?? 0),
				];
			}),
		);
		row.sort((a, b) => (key.get(a) ?? 0) - (key.get(b) ?? 0));
		applyPins(row);
		row.forEach((c, i) => index.set(c, i));
	};
	const crossings = () => {
		let n = 0;
		for (let r = 0; r + 1 < rows.length; r++) {
			const segs = rows[r].flatMap((u) =>
				u.succs.map((v) => [index.get(u) ?? 0, index.get(v) ?? 0]),
			);
			for (let i = 0; i < segs.length; i++)
				for (let j = i + 1; j < segs.length; j++)
					if ((segs[i][0] - segs[j][0]) * (segs[i][1] - segs[j][1]) < 0) n++;
		}
		return n;
	};
	let best = rows.map((r) => [...r]);
	let bestN = crossings();
	for (let it = 0; it < 8; it++) {
		if (it % 2 === 0)
			for (let r = 1; r < rows.length; r++) sortRow(r, (c) => c.preds);
		else for (let r = rows.length - 2; r >= 0; r--) sortRow(r, (c) => c.succs);
		const n = crossings();
		if (n < bestN) {
			bestN = n;
			best = rows.map((r) => [...r]);
		}
	}
	best.forEach((r, i) => {
		rows[i] = r;
	});
}

/** Free space between two neighbours in a row. */
function spacing(a: Cell, b: Cell): number {
	if (a.dummy && b.dummy) return LANE_GAP;
	if (a.dummy || b.dummy) return a.box || b.box ? LANE_BOX_GAP : LANE_NODE_GAP;
	return ITEM_GAP;
}

/**
 * Horizontal placement: each row packed with fixed gaps, then moved, row by row, as close as the
 * gaps allow to the mean x of what each cell connects to above (down sweeps) or below (up sweeps).
 */
function placeRows(rows: Cell[][]) {
	const gap = (a: Cell, b: Cell) => a.w / 2 + b.w / 2 + spacing(a, b);
	for (const row of rows) {
		let x = 0;
		row.forEach((c, i) => {
			if (i > 0) x += gap(row[i - 1], c);
			c.x = x;
		});
		const mid = x / 2;
		for (const c of row) c.x -= mid;
	}
	const sweep = (r: number, neighbours: (c: Cell) => Cell[]) => {
		const row = rows[r];
		const want = row.map((c) => {
			const ns = neighbours(c);
			return ns.length ? ns.reduce((s, n) => s + n.x, 0) / ns.length : c.x;
		});
		const xs = fit(
			want,
			row.slice(1).map((c, i) => gap(row[i], c)),
		);
		row.forEach((c, i) => {
			c.x = xs[i];
		});
	};
	for (let it = 0; it < 6; it++) {
		for (let r = 1; r < rows.length; r++) sweep(r, (c) => c.preds);
		for (let r = rows.length - 2; r >= 0; r--) sweep(r, (c) => c.succs);
	}
	for (let r = 1; r < rows.length; r++) sweep(r, (c) => c.preds);
}

/**
 * The positions closest (least squares) to `want` with x[i+1] - x[i] >= gaps[i]: shift out the
 * gaps, then pool adjacent violators so the remainder is non-decreasing.
 */
export function fit(want: number[], gaps: number[]): number[] {
	const offset = [0];
	for (const g of gaps) offset.push(offset[offset.length - 1] + g);
	const t = want.map((w, i) => w - offset[i]);
	const blocks: { sum: number; n: number }[] = [];
	for (const v of t) {
		blocks.push({ sum: v, n: 1 });
		while (blocks.length > 1) {
			const b = blocks[blocks.length - 1];
			const a = blocks[blocks.length - 2];
			if (a.sum / a.n <= b.sum / b.n) break;
			blocks.splice(-2, 2, { sum: a.sum + b.sum, n: a.n + b.n });
		}
	}
	const y = blocks.flatMap((b) => new Array(b.n).fill(b.sum / b.n));
	return y.map((v, i) => v + offset[i]);
}
