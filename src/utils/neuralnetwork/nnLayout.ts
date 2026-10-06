// Lays out an NNGraph for NeuralNetworkGraph: dagre ranks the nodes along the flow direction and
// routes the edges around them. Returns plain geometry (node boxes, text baselines, edge paths,
// group boxes, the overall extent) and no markup, so the component only has to draw it.
import { type Point, graphlib, layout } from "@dagrejs/dagre";
import type { GraphEdge, GraphGroup, GraphNode, NNGraph } from "./NNGraph";

/** Top-to-bottom, bottom-to-top, left-to-right or right-to-left. */
export type Direction = "TB" | "BT" | "LR" | "RL";

type LineClass = "label" | "detail" | "shape";

export interface PlacedNode extends GraphNode {
	/** Centre and size. */
	x: number;
	y: number;
	w: number;
	h: number;
	/** Drawn as a circle: a 1–2 character op such as "+". */
	circle: boolean;
	/** Text lines with their baselines, stacked around the centre (above the tray, if any). */
	lines: { cls: LineClass; text: string; y: number; x?: number }[];
	/** The node's tray (see GraphNode.tray): top-left corner and size. */
	trayBox?: { x: number; y: number; w: number; h: number };
	/** Top-left corner and size of the icon (see GraphNode.icon). */
	iconAt?: { x: number; y: number; size: number };
}

export interface PlacedEdge extends GraphEdge {
	/** SVG path through dagre's bend points. */
	d: string;
	/** Where the edge's label goes, if it has one. */
	labelAt?: { x: number; y: number; anchor: "start" | "middle" };
}

export interface PlacedGroup extends GraphGroup {
	/** Top-left corner and size of the box, label strip included. */
	x: number;
	y: number;
	w: number;
	h: number;
}

export interface GraphLayout {
	nodes: PlacedNode[];
	edges: PlacedEdge[];
	groups: PlacedGroup[];
	/** Extent of everything drawn, for the SVG's viewBox. */
	box: { x: number; y: number; w: number; h: number };
}

// ---- Node sizes. SVG text can't be measured at build time, so widths are estimated from
// character counts at the font sizes set in NeuralNetworkGraph's styles (generous, so text
// never overflows).
const LINE: Record<LineClass, number> = { label: 16, detail: 14, shape: 14 };
const CHAR: Record<LineClass, number> = { label: 7.3, detail: 6.1, shape: 6.4 };
const CIRCLE = 30;
const GROUP_PAD = { side: 8, top: 20, bottom: 8 };
const TRAY_PAD = 5;
/** Icon size and the gap between icon and label, in px. */
export const ICON_SIZE = 15;
const ICON_GAP = 5;
/** The "diagram" look: a large icon at the left, the text centred in the rest of the box. */
const DIAGRAM = {
	icon: 28,
	pad: 8,
	gap: 8,
	minH: 48,
	char: { label: 6.6, detail: 5.0, shape: 5.4 },
};

/** How nodes are sized and edges routed: see NeuralNetworkGraph's `look` prop. */
export type Look = "flat" | "glass" | "diagram";

const isCircle = (n: GraphNode) => n.kind === "op" && [...n.label].length <= 2;

function textLines(n: GraphNode) {
	return (
		[
			{ cls: "label", text: n.label },
			{ cls: "detail", text: n.detail },
			{ cls: "shape", text: n.shape },
		] as { cls: LineClass; text?: string }[]
	).filter((l): l is { cls: LineClass; text: string } => !!l.text);
}

/** Width of the text block of a "diagram" node. */
function diagramTextW(n: GraphNode) {
	return Math.max(
		...textLines(n).map((l) => [...l.text].length * DIAGRAM.char[l.cls]),
	);
}

function size(n: GraphNode, look: Look) {
	if (isCircle(n)) return { width: CIRCLE, height: CIRCLE };
	if (look === "diagram") {
		const iconW = n.icon ? DIAGRAM.icon + DIAGRAM.gap : 0;
		const height = textLines(n).reduce((h, l) => h + LINE[l.cls], 20);
		return {
			width: Math.round(
				Math.max(90, 2 * DIAGRAM.pad + iconW + diagramTextW(n) + 4),
			),
			height: Math.max(DIAGRAM.minH, height),
		};
	}
	const ls = textLines(n);
	const width = Math.max(
		72,
		...ls.map((l) => [...l.text].length * CHAR[l.cls] + 24),
	);
	const height = ls.reduce((h, l) => h + LINE[l.cls], 14);
	const labelW =
		[...n.label].length * CHAR.label + 24 + (n.icon ? ICON_SIZE + ICON_GAP : 0);
	const textW = Math.max(width, labelW);
	if (!n.tray) return { width: Math.round(textW), height };
	return {
		width: Math.round(Math.max(textW, n.tray.width + 2 * TRAY_PAD)),
		height: height + n.tray.height + TRAY_PAD,
	};
}

export interface LayoutOptions {
	/** Also keep each node's outputs in the order they were added (see outputOrder). */
	keepOutputOrder?: boolean;
	/** Gap between ranks (rows, for top-to-bottom flows); more room lets fan-out edges arrive from above. */
	rankSep?: number;
	/** "diagram" also routes edges at right angles between rows (top-to-bottom flows). */
	look?: Look;
}

export function layoutGraph(
	graph: NNGraph,
	direction: Direction,
	options: LayoutOptions = {},
): GraphLayout {
	// Groups become compound (parent) nodes so dagre keeps their members together and reports
	// a bounding box for each.
	const g = new graphlib.Graph({ compound: true, multigraph: true });
	g.setGraph({
		rankdir: direction,
		nodesep: 22,
		ranksep: options.rankSep ?? 34,
		edgesep: 14,
		marginx: 12,
		marginy: 12,
	});
	g.setDefaultEdgeLabel(() => ({}));
	for (const grp of graph.groups) g.setNode(grp.id, { width: 0, height: 0 });
	for (const n of graph.nodes) {
		g.setNode(n.id, size(n, options.look ?? "flat"));
		if (n.group) g.setParent(n.id, n.group);
	}
	// Edge labels get no room in the layout (reserving it bends the edge around a phantom box);
	// they are drawn beside the middle of the finished curve instead, see placeEdge.
	graph.edges.forEach((e, i) => g.setEdge(e.from, e.to, {}, `e${i}`));
	const constraints = [
		...inputOrder(graph),
		...(options.keepOutputOrder ? outputOrder(graph) : []),
	];
	layout(g, { constraints });

	const groups = graph.groups.map((grp): PlacedGroup => {
		const b = g.node(grp.id);
		// dagre's box plus room for the label along the top edge.
		return {
			...grp,
			x: b.x! - b.width / 2 - GROUP_PAD.side,
			y: b.y! - b.height / 2 - GROUP_PAD.top,
			w: b.width + 2 * GROUP_PAD.side,
			h: b.height + GROUP_PAD.top + GROUP_PAD.bottom,
		};
	});

	const nodes = graph.nodes.map((n): PlacedNode => {
		const b = g.node(n.id);
		const [x, y] = [b.x!, b.y!];
		// With a tray, the text keeps its usual height at the top and the tray fills the rest.
		const trayH = n.tray ? n.tray.height + TRAY_PAD : 0;
		const trayBox = n.tray && {
			x: x - n.tray.width / 2,
			y: y + b.height / 2 - trayH,
			w: n.tray.width,
			h: n.tray.height,
		};
		const lines = stack(n, y - trayH / 2);
		// The icon sits left of the label; the label shifts right by half the icon's room.
		let iconAt: PlacedNode["iconAt"];
		if (options.look === "diagram" && !isCircle(n)) {
			// Icon at the left edge, the text centred in the space to its right.
			const left = x - b.width / 2 + DIAGRAM.pad;
			const textLeft = n.icon ? left + DIAGRAM.icon + DIAGRAM.gap : left;
			const textX = (textLeft + x + b.width / 2 - DIAGRAM.pad) / 2;
			for (const l of lines) l.x = textX;
			if (n.icon)
				iconAt = { x: left, y: y - DIAGRAM.icon / 2, size: DIAGRAM.icon };
		} else if (n.icon && !isCircle(n)) {
			const shift = (ICON_SIZE + ICON_GAP) / 2;
			const labelW = [...n.label].length * CHAR.label * 0.93;
			const label = lines.find((l) => l.cls === "label");
			if (label) {
				label.x = x + shift;
				iconAt = {
					x: x + shift - labelW / 2 - ICON_GAP - ICON_SIZE,
					y: label.y - ICON_SIZE + 3,
					size: ICON_SIZE,
				};
			}
		}
		return {
			...n,
			x,
			y,
			w: b.width,
			h: b.height,
			circle: isCircle(n),
			lines,
			trayBox,
			iconAt,
		};
	});

	const vertical = direction === "TB" || direction === "BT";
	const byId = new Map(nodes.map((n) => [n.id, n]));
	const edges = graph.edges.map((e, i) => {
		const placed = placeEdge(
			e,
			g.edge({ v: e.from, w: e.to, name: `e${i}` }).points ?? [],
			vertical,
		);
		const [a, b] = [byId.get(e.from), byId.get(e.to)];
		if (options.look !== "diagram" || direction !== "TB" || !a || !b)
			return placed;
		// Right angles: down from the source, along a bus halfway between the rows, down into the
		// target. Edges that fan out from (or into) one node share their bus.
		const sy = a.y + a.h / 2;
		const ty = b.y - b.h / 2;
		const mid = Math.round((sy + ty) / 2);
		return { ...placed, d: `M${a.x},${sy}V${mid}H${b.x}V${ty}` };
	});

	// The extent: dagre's size, widened for the group label padding added above.
	const gl = g.graph();
	const minX = Math.min(0, ...groups.map((b) => b.x));
	const minY = Math.min(0, ...groups.map((b) => b.y));
	const maxX = Math.max(gl.width ?? 0, ...groups.map((b) => b.x + b.w));
	const maxY = Math.max(gl.height ?? 0, ...groups.map((b) => b.y + b.h));
	const box = {
		x: minX,
		y: minY,
		w: Math.ceil(maxX - minX),
		h: Math.ceil(maxY - minY),
	};

	return { nodes, edges, groups, box };
}

/**
 * Layout constraints that keep the inputs of each node in the order they were listed in `from`
 * (left to right, or top to bottom for LR), instead of whatever order dagre's crossing
 * minimisation settles on.
 */
function inputOrder(graph: NNGraph) {
	return graph.nodes.flatMap((n) => {
		const ins = graph.edges
			.filter((e) => e.to === n.id && !e.label)
			.map((e) => e.from);
		return ins.slice(1).map((right, i) => ({ left: ins[i], right }));
	});
}

/**
 * The same for outputs: the nodes a node feeds keep the order they were added in. Useful when one
 * node fans out to many parallel branches (a collective feeding one column per rank).
 */
function outputOrder(graph: NNGraph) {
	return graph.nodes.flatMap((n) => {
		const outs = graph.edges
			.filter((e) => e.from === n.id && !e.label)
			.map((e) => e.to);
		return outs.slice(1).map((right, i) => ({ left: outs[i], right }));
	});
}

/** A node's text lines with their baselines, stacked around its centre `cy`. */
function stack(n: GraphNode, cy: number): PlacedNode["lines"] {
	const ls = textLines(n);
	let y = cy - ls.reduce((h, l) => h + LINE[l.cls], 0) / 2;
	return ls.map((l) => {
		y += LINE[l.cls];
		return { ...l, y: y - 4 };
	});
}

function placeEdge(e: GraphEdge, pts: Point[], vertical: boolean): PlacedEdge {
	// Label anchor: the middle bend point, nudged off the line (right of it for vertical flows,
	// above it for horizontal ones) so the text doesn't sit on the stroke.
	const mid = pts[Math.floor(pts.length / 2)];
	const labelAt =
		e.label && mid
			? vertical
				? { x: mid.x + 6, y: mid.y + 4, anchor: "start" as const }
				: { x: mid.x, y: mid.y - 6, anchor: "middle" as const }
			: undefined;
	return { ...e, d: curve(pts), labelAt };
}

const r1 = (v: number) => Math.round(v * 10) / 10;

/**
 * Smooth path through dagre's bend points: a uniform B-spline (d3's curveBasis). Unlike a
 * Catmull-Rom spline it never overshoots, so closely spaced bend points can't make loops.
 * It passes through the first and last points, which is where the edge meets the nodes.
 */
function curve(p: Point[]): string {
	const pt = (x: number, y: number) => `${r1(x)},${r1(y)}`;
	if (p.length < 3) return `M${p.map((q) => pt(q.x, q.y)).join(" L")}`;
	let d = `M${pt(p[0].x, p[0].y)} L${pt((5 * p[0].x + p[1].x) / 6, (5 * p[0].y + p[1].y) / 6)}`;
	// Each segment is a cubic Bézier built from three consecutive points a, b, c.
	const seg = (a: Point, b: Point, c: Point) =>
		` C${pt((2 * a.x + b.x) / 3, (2 * a.y + b.y) / 3)} ${pt((a.x + 2 * b.x) / 3, (a.y + 2 * b.y) / 3)} ${pt(
			(a.x + 4 * b.x + c.x) / 6,
			(a.y + 4 * b.y + c.y) / 6,
		)}`;
	for (let i = 2; i < p.length; i++) d += seg(p[i - 2], p[i - 1], p[i]);
	const [a, b] = [p[p.length - 2], p[p.length - 1]];
	d += seg(a, b, b);
	return `${d} L${pt(b.x, b.y)}`;
}
