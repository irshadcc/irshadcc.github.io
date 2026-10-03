// Lays out a dependency graph for DependencyGraph.astro: file/function cards, bare icons,
// skeleton placeholders and small junction squares, joined by thin orthogonal (right-angled)
// wires. dagre places the nodes in ranks; the wires are then routed here so every bend sits in
// the middle of the gap between two ranks, which makes wires between the same ranks share
// trunks. Each edge passes through dagre's dummy points: dagre halves ranksep and doubles every
// edge's length internally, so it has one in the middle of every gap it crosses and one in every
// column it skips. A directed edge stops short of its target and gets an arrowhead whose tip
// touches the target's border. docs/depgraph-component.md walks through the routing.
import { graphlib, layout } from "@dagrejs/dagre";

/** A small square with text, e.g. a "TS" or "Go" language badge. */
export interface BadgeIcon {
	text: string;
	bg?: string;
	fg?: string;
	/** Draw an outlined square instead of a filled one. */
	outline?: boolean;
}

export type BuiltinIcon =
	| "fn"
	| "folder"
	| "file"
	| "commit"
	| "sparkle"
	| "hexagon"
	| "dot";

export type GraphIcon = BuiltinIcon | BadgeIcon;

export interface GraphNode {
	id: string;
	/**
	 * card: icon, title and subtitle in a box (default).
	 * icon: the icon alone, no box.
	 * placeholder: a faded skeleton card (circle and two bars), for context.
	 * junction: a small filled square where wires meet.
	 */
	kind?: "card" | "icon" | "placeholder" | "junction";
	title?: string;
	subtitle?: string;
	icon?: GraphIcon;
	/** Shown under the figure when the node is hovered or focused. */
	note?: string;
}

export interface GraphEdge {
	from: string;
	to: string;
	/** Draw an arrowhead at `to`; overrides the graph's `directed`. */
	directed?: boolean;
}

export interface GraphOptions {
	rankdir?: "LR" | "TB";
	nodesep?: number;
	ranksep?: number;
	/** Default for edges that don't set `directed`. */
	directed?: boolean;
}

export interface PlacedNode extends GraphNode {
	kind: NonNullable<GraphNode["kind"]>;
	/** Top-left corner and size. */
	x: number;
	y: number;
	w: number;
	h: number;
}

export interface Point {
	x: number;
	y: number;
}

export interface PlacedEdge extends GraphEdge {
	directed: boolean;
	/** The wire; for a directed edge it ends at the arrowhead's base. */
	points: Point[];
	d: string;
	/** Arrowhead polygon (tip, then the two base corners), for directed edges. */
	arrow?: Point[];
}

export interface GraphLayout {
	nodes: PlacedNode[];
	edges: PlacedEdge[];
	width: number;
	height: number;
}

export const ARROW_LEN = 7;
const ARROW_HALF = 3.5;

// Text widths are estimated from character counts of the monospace fonts the component uses.
export const TITLE_CHAR = 7.8; // 13px
export const SUB_CHAR = 6.6; // 11px
const PAD = 14;
const ICON = 22;
const ICON_GAP = 12;

export function nodeSize(n: GraphNode): { w: number; h: number } {
	switch (n.kind ?? "card") {
		case "junction":
			return { w: 9, h: 9 };
		case "icon":
			return { w: 32, h: 32 };
		case "placeholder":
			return { w: 190, h: 50 };
		default: {
			const text = Math.max(
				(n.title ?? "").length * TITLE_CHAR,
				(n.subtitle ?? "").length * SUB_CHAR,
			);
			return {
				w: Math.ceil(PAD + (n.icon ? ICON + ICON_GAP : 0) + text + PAD + 4),
				h: n.subtitle ? 58 : 40,
			};
		}
	}
}

export function validateGraph(
	nodes: GraphNode[],
	edges: GraphEdge[],
): string | null {
	const ids = new Set<string>();
	for (const n of nodes) {
		if (ids.has(n.id)) return `duplicate node id "${n.id}"`;
		ids.add(n.id);
		if ((n.kind ?? "card") === "card" && !n.title)
			return `card "${n.id}" needs a title`;
		if (n.kind === "icon" && !n.icon)
			return `icon node "${n.id}" needs an icon`;
	}
	for (const e of edges) {
		if (!ids.has(e.from)) return `edge from unknown node "${e.from}"`;
		if (!ids.has(e.to)) return `edge to unknown node "${e.to}"`;
		if (e.from === e.to) return `self-loop on "${e.from}"`;
	}
	return null;
}

/** Drop repeated points and middle points of straight runs. */
function simplify(pts: Point[]): Point[] {
	const out: Point[] = [];
	for (const p of pts) {
		const q = out[out.length - 1];
		if (q && Math.abs(q.x - p.x) < 0.5 && Math.abs(q.y - p.y) < 0.5) continue;
		const r = out[out.length - 2];
		if (
			q &&
			r &&
			((Math.abs(r.x - q.x) < 0.5 && Math.abs(q.x - p.x) < 0.5) ||
				(Math.abs(r.y - q.y) < 0.5 && Math.abs(q.y - p.y) < 0.5))
		)
			out.pop();
		out.push(p);
	}
	return out;
}

export function layoutGraph(
	nodes: GraphNode[],
	edges: GraphEdge[],
	opts: GraphOptions = {},
): GraphLayout {
	const { rankdir = "LR", nodesep = 24, ranksep = 56, directed = false } = opts;
	const g = new graphlib.Graph({ multigraph: true });
	g.setGraph({ rankdir, nodesep, ranksep, marginx: 12, marginy: 12 });
	g.setDefaultEdgeLabel(() => ({}));
	for (const n of nodes) {
		const { w, h } = nodeSize(n);
		g.setNode(n.id, { width: w, height: h });
	}
	edges.forEach((e, i) => g.setEdge(e.from, e.to, {}, `e${i}`));
	layout(g);

	const placed: PlacedNode[] = nodes.map((n) => {
		const p = g.node(n.id);
		return {
			...n,
			kind: n.kind ?? "card",
			x: p.x - p.width / 2,
			y: p.y - p.height / 2,
			w: p.width,
			h: p.height,
		};
	});
	const byId = new Map(placed.map((n) => [n.id, n]));

	// Work in (main, cross) coordinates: main runs along the ranks (x for LR, y for TB).
	const lr = rankdir === "LR";
	const main = (p: Point) => (lr ? p.x : p.y);
	const cross = (p: Point) => (lr ? p.y : p.x);
	const pt = (m: number, c: number): Point =>
		lr ? { x: m, y: c } : { x: c, y: m };

	// Each rank's centre and half-extent along the main axis, from the real nodes.
	const ranks = new Map<number, number>();
	for (const n of placed) {
		const c = Math.round(lr ? n.x + n.w / 2 : n.y + n.h / 2);
		const half = (lr ? n.w : n.h) / 2;
		ranks.set(c, Math.max(ranks.get(c) ?? 0, half));
	}
	const rankList = [...ranks.entries()].sort((a, b) => a[0] - b[0]);
	/** Middle of the gap after the rank at or before main coordinate m. */
	const gapAfter = (m: number) => {
		let i = rankList.length - 1;
		while (i > 0 && rankList[i][0] > m + 1) i--;
		const [c, h] = rankList[i];
		const next = rankList[i + 1];
		return next ? (c + h + next[0] - next[1]) / 2 : c + h + ranksep / 2;
	};

	const placedEdges: PlacedEdge[] = edges.map((e, i) => {
		const a = byId.get(e.from) as PlacedNode;
		const b = byId.get(e.to) as PlacedNode;
		const ac = { x: a.x + a.w / 2, y: a.y + a.h / 2 };
		const bc = { x: b.x + b.w / 2, y: b.y + b.h / 2 };
		const forward = main(bc) >= main(ac);
		const aHalf = (lr ? a.w : a.h) / 2;
		const bHalf = (lr ? b.w : b.h) / 2;
		const start = pt(main(ac) + (forward ? aHalf : -aHalf), cross(ac));
		const end = pt(main(bc) + (forward ? -bHalf : bHalf), cross(bc));
		// dagre's points start and end on the node borders; the inner ones are its dummy nodes
		// (one per gap crossed, one per column skipped), which the route passes through.
		const dummies = (
			g.edge({ v: e.from, w: e.to, name: `e${i}` }).points ?? []
		).slice(1, -1);
		const via = [start, ...dummies, end];
		const route: Point[] = [start];
		for (let k = 1; k < via.length; k++) {
			const p = via[k - 1];
			const q = via[k];
			if (Math.abs(cross(p) - cross(q)) > 0.5) {
				const m = gapAfter(Math.min(main(p), main(q)));
				route.push(pt(m, cross(p)), pt(m, cross(q)));
			}
			route.push(q);
		}
		const points = simplify(route);
		const dir = e.directed ?? directed;
		let arrow: Point[] | undefined;
		if (dir) {
			// The last segment is axis-aligned and at least ranksep / 2 long, so it holds the head.
			const tip = points[points.length - 1];
			const prev = points[points.length - 2];
			const len = Math.hypot(tip.x - prev.x, tip.y - prev.y) || 1;
			const ux = (tip.x - prev.x) / len;
			const uy = (tip.y - prev.y) / len;
			const base = { x: tip.x - ux * ARROW_LEN, y: tip.y - uy * ARROW_LEN };
			arrow = [
				tip,
				{ x: base.x - uy * ARROW_HALF, y: base.y + ux * ARROW_HALF },
				{ x: base.x + uy * ARROW_HALF, y: base.y - ux * ARROW_HALF },
			];
			points[points.length - 1] = base;
		}
		const d = points
			.map((p, k) => `${k ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
			.join(" ");
		return { ...e, directed: dir, points, d, arrow };
	});

	const gr = g.graph();
	return {
		nodes: placed,
		edges: placedEdges,
		width: Math.ceil(gr.width ?? 0),
		height: Math.ceil(gr.height ?? 0),
	};
}
