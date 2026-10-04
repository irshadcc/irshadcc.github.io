// Max-flow / min-cut on the kind of network AOTAutograd's partitioner builds, for MinCutSteps.astro.
//
// The network is given at the level of FX nodes. Every node has a capacity (the cost of saving its
// output, or null for "cannot be saved"); every edge has infinite capacity:
//
//   source -> X   X cannot be recomputed in the backward (a "banned" node)
//   X -> Y        Y reads X's output (a data dependency)
//   X -> sink     X's output must be available to the backward
//
// As in torch/_functorch/partitioners.py (solve_min_cut), each node X is split into X_in -> X_out
// with the node's capacity on that edge, so that a minimum s-t cut cuts nodes, not data edges.
// The flow itself is found with Edmonds-Karp (breadth-first augmenting paths), recording one step
// per augmenting path. The final cut follows networkx.minimum_cut, which the partitioner calls: the
// sink side is every vertex that can still reach the sink in the residual graph, so among several
// minimum cuts the one nearest the sink wins. That side is the same for every maximum flow, so
// using Edmonds-Karp here instead of networkx's preflow-push does not change which nodes are saved.

import { graphlib, layout } from "@dagrejs/dagre";

export interface FlowNode {
	id: string;
	/** Name drawn on the box, usually the FX node name. */
	label: string;
	/** Second line: the op, e.g. "mm" or "rsqrt". */
	op?: string;
	/** Cost of saving this node's output; null means it can never be saved. */
	cap: number | null;
	/** Colour hint: a graph input, a forward op, or an op from the backward. */
	kind?: "input" | "fwd" | "bwd";
	/** Shown under the figure when the node is hovered. */
	note?: string;
}

export interface FlowEdge {
	/** A node id or "source". */
	from: string;
	/** A node id or "sink". */
	to: string;
	/** Why the edge exists (shown on hover). */
	note?: string;
}

export type NodeSide = "fwd" | "saved" | "bwd";

export interface FlowStep {
	/** Short timeline label. */
	label: string;
	/** One or two sentences describing the step. */
	text: string;
	/** Node ids on this step's augmenting path, in order. */
	path: string[];
	/** Nodes the path crosses against an earlier flow (undoing it). */
	reversed: string[];
	/** Edge keys (`from>to`) on the augmenting path. */
	pathEdges: string[];
	/** Flow pushed by this step, and the running total. */
	add: number;
	total: number;
	/** Flow through each node and along each edge after this step. */
	nodeFlow: Record<string, number>;
	edgeFlow: Record<string, number>;
	/** Set on the final step: which side of the cut each node ends on. */
	sides?: Record<string, NodeSide>;
	saved?: string[];
}

export const edgeKey = (e: { from: string; to: string }) => `${e.from}>${e.to}`;

export function validateFlow(
	nodes: FlowNode[],
	edges: FlowEdge[],
): string | undefined {
	const ids = new Set<string>();
	for (const n of nodes) {
		if (n.id === "source" || n.id === "sink")
			return `"${n.id}" is reserved for the terminals`;
		if (ids.has(n.id)) return `duplicate node "${n.id}"`;
		if (n.cap !== null && !(n.cap >= 0)) return `bad capacity on "${n.id}"`;
		ids.add(n.id);
	}
	const seen = new Set<string>();
	for (const e of edges) {
		if (e.from === "sink" || e.to === "source")
			return `edge ${edgeKey(e)} points the wrong way`;
		if (e.from !== "source" && !ids.has(e.from))
			return `edge from unknown node "${e.from}"`;
		if (e.to !== "sink" && !ids.has(e.to))
			return `edge to unknown node "${e.to}"`;
		if (seen.has(edgeKey(e))) return `duplicate edge ${edgeKey(e)}`;
		seen.add(edgeKey(e));
	}
	return undefined;
}

interface Arc {
	to: number;
	cap: number;
	flow: number;
	/** Index of the paired arc in adj[to]. */
	rev: number;
	/** What the arc stands for: a node's internal edge, a data edge, or a reverse arc (""). */
	node?: string;
	edge?: string;
}

const fmt = (v: number) => v.toLocaleString("en-US");

/** Runs Edmonds-Karp and returns the initial state, one step per augmenting path, and the cut. */
export function minCutSteps(nodes: FlowNode[], edges: FlowEdge[]): FlowStep[] {
	const err = validateFlow(nodes, edges);
	if (err) throw new Error(`minCutSteps: ${err}`);

	// Vertices: 0 = source, 1 = sink, then (in, out) per node.
	const index = new Map<string, number>();
	nodes.forEach((n, i) => index.set(n.id, 2 + 2 * i));
	const vin = (id: string) => (id === "source" ? 0 : (index.get(id) as number));
	const vout = (id: string) =>
		id === "sink" ? 1 : (index.get(id) as number) + 1;
	const nv = 2 + 2 * nodes.length;
	const adj: Arc[][] = Array.from({ length: nv }, () => []);
	const addArc = (u: number, v: number, cap: number, tag: Partial<Arc>) => {
		adj[u].push({ to: v, cap, flow: 0, rev: adj[v].length, ...tag });
		adj[v].push({ to: u, cap: 0, flow: 0, rev: adj[u].length - 1 });
	};
	for (const n of nodes)
		addArc(vin(n.id), vout(n.id), n.cap ?? Number.POSITIVE_INFINITY, {
			node: n.id,
		});
	for (const e of edges) {
		const u = e.from === "source" ? 0 : vout(e.from);
		const v = e.to === "sink" ? 1 : vin(e.to);
		addArc(u, v, Number.POSITIVE_INFINITY, { edge: edgeKey(e) });
	}
	const byId = new Map(nodes.map((n) => [n.id, n]));
	const label = (id: string) => byId.get(id)?.label ?? id;

	const snapshot = () => {
		const nodeFlow: Record<string, number> = {};
		const edgeFlow: Record<string, number> = {};
		for (const arcs of adj)
			for (const a of arcs) {
				if (a.node) nodeFlow[a.node] = a.flow;
				if (a.edge) edgeFlow[a.edge] = a.flow;
			}
		return { nodeFlow, edgeFlow };
	};

	const steps: FlowStep[] = [
		{
			label: "network",
			text: "Every box is an FX node and its capacity is the cost of saving its output. Edges have unlimited capacity. Find the cheapest set of boxes whose removal disconnects source from sink.",
			path: [],
			reversed: [],
			pathEdges: [],
			add: 0,
			total: 0,
			...snapshot(),
		},
	];

	let total = 0;
	for (let round = 0; round < 10_000; round++) {
		// Breadth-first search for the shortest augmenting path in the residual graph.
		const prev: ([number, number] | undefined)[] = new Array(nv);
		const seen = new Array<boolean>(nv).fill(false);
		seen[0] = true;
		const queue = [0];
		while (queue.length && !seen[1]) {
			const u = queue.shift() as number;
			adj[u].forEach((a, k) => {
				if (!seen[a.to] && a.cap - a.flow > 0) {
					seen[a.to] = true;
					prev[a.to] = [u, k];
					queue.push(a.to);
				}
			});
		}
		if (!seen[1]) break;

		const arcs: Arc[] = [];
		for (let v = 1; v !== 0; ) {
			const [u, k] = prev[v] as [number, number];
			arcs.unshift(adj[u][k]);
			v = u;
		}
		const add = Math.min(...arcs.map((a) => a.cap - a.flow));
		if (!Number.isFinite(add))
			throw new Error(
				"minCutSteps: a path of unlimited capacity joins source and sink, so some value must be saved but cannot be",
			);
		const path: string[] = [];
		const reversed: string[] = [];
		const pathEdges: string[] = [];
		const filled: string[] = [];
		for (const a of arcs) {
			a.flow += add;
			adj[a.to][a.rev].flow -= add;
			const back = adj[a.to][a.rev];
			if (a.node) {
				path.push(a.node);
				if (a.flow === a.cap) filled.push(a.node);
			} else if (back.node) {
				// Crossing a node from out to in: this path undoes earlier flow through it.
				path.push(back.node);
				reversed.push(back.node);
			}
			if (a.edge) pathEdges.push(a.edge);
			if (back.edge) pathEdges.push(back.edge);
		}
		total += add;
		const route = ["source", ...path.map(label), "sink"].join(" → ");
		const undo = reversed.length
			? ` It runs backwards through ${reversed.map(label).join(", ")}, moving flow that an earlier path sent there onto a new route.`
			: "";
		const full = filled.length
			? ` ${filled.map(label).join(" and ")} ${filled.length > 1 ? "are" : "is"} now full.`
			: "";
		steps.push({
			label: `path ${steps.length}`,
			text: `Augmenting path ${route} carries ${fmt(add)} (its narrowest box), so the flow is now ${fmt(total)}.${undo}${full}`,
			path,
			reversed,
			pathEdges,
			add,
			total,
			...snapshot(),
		});
	}

	// Sink side: every vertex that can still reach the sink through an arc with spare capacity.
	const reach = new Array<boolean>(nv).fill(false);
	reach[1] = true;
	const queue = [1];
	while (queue.length) {
		const v = queue.shift() as number;
		for (let u = 0; u < nv; u++)
			if (!reach[u] && adj[u].some((a) => a.to === v && a.cap - a.flow > 0)) {
				reach[u] = true;
				queue.push(u);
			}
	}
	if (reach[0]) throw new Error("minCutSteps: flow is not maximal");
	const sides: Record<string, NodeSide> = {};
	const saved: string[] = [];
	for (const n of nodes) {
		const i = reach[vin(n.id)];
		const o = reach[vout(n.id)];
		sides[n.id] = !o ? "fwd" : !i ? "saved" : "bwd";
		if (!i && o) saved.push(n.id);
	}
	const cutCost = nodes
		.filter((n) => saved.includes(n.id))
		.reduce((s, n) => s + (n.cap ?? 0), 0);
	if (cutCost !== total)
		throw new Error(`minCutSteps: cut ${cutCost} differs from flow ${total}`);
	steps.push({
		label: "cut",
		text: `No augmenting path is left, so the flow of ${fmt(total)} is maximal. The full boxes that separate what the source can still reach from what can reach the sink form the minimum cut: save ${saved.map(label).join(", ")} (total ${fmt(cutCost)}). Boxes on the sink side are recomputed in the backward.`,
		path: [],
		reversed: [],
		pathEdges: [],
		add: 0,
		total,
		...snapshot(),
		sides,
		saved,
	});
	return steps;
}

export interface PlacedFlowNode {
	id: string;
	x: number;
	y: number;
	w: number;
	h: number;
}

export interface FlowLayout {
	nodes: PlacedFlowNode[];
	/** SVG path per edge key. */
	edges: { key: string; d: string; from: string; to: string }[];
	width: number;
	height: number;
}

export const NODE_W = 92;
export const NODE_H = 50;
const TERMINAL = 34;

/**
 * Lays the network out with dagre, source first and sink last: left to right when that fits in
 * `maxWidth`, otherwise top to bottom.
 */
export function layoutFlow(
	nodes: FlowNode[],
	edges: FlowEdge[],
	maxWidth = 760,
): FlowLayout {
	const lr = layoutIn(nodes, edges, "LR");
	return lr.width <= maxWidth ? lr : layoutIn(nodes, edges, "TB");
}

function layoutIn(
	nodes: FlowNode[],
	edges: FlowEdge[],
	rankdir: "LR" | "TB",
): FlowLayout {
	const g = new graphlib.Graph({ multigraph: false });
	g.setGraph({
		rankdir,
		nodesep: rankdir === "LR" ? 16 : 22,
		ranksep: rankdir === "LR" ? 34 : 20,
		marginx: 6,
		marginy: 14,
	});
	g.setDefaultEdgeLabel(() => ({}));
	g.setNode("source", { width: TERMINAL, height: TERMINAL });
	g.setNode("sink", { width: TERMINAL, height: TERMINAL });
	for (const n of nodes) g.setNode(n.id, { width: NODE_W, height: NODE_H });
	for (const e of edges)
		g.setEdge(e.from, e.to, {
			// Keep the terminals' edges from dragging nodes into long detours.
			weight: e.from === "source" || e.to === "sink" ? 1 : 3,
		});
	layout(g);
	const ids = ["source", "sink", ...nodes.map((n) => n.id)];
	const raw = ids.map((id) => {
		const p = g.node(id);
		return { id, x: p.x, y: p.y, w: p.width, h: p.height };
	});
	const pts = edges.map((e) => g.edge(e.from, e.to).points);
	// dagre's graph size ignores edge bends that stick out, so measure everything drawn.
	const PAD = 8;
	const xs = [
		...raw.flatMap((n) => [n.x - n.w / 2, n.x + n.w / 2]),
		...pts.flat().map((p) => p.x),
	];
	const ys = [
		...raw.flatMap((n) => [n.y - n.h / 2, n.y + n.h / 2 + 14]),
		...pts.flat().map((p) => p.y),
	];
	const dx = PAD - Math.min(...xs);
	const dy = PAD - Math.min(...ys);
	const placed = raw.map((n) => ({ ...n, x: n.x + dx, y: n.y + dy }));
	const placedEdges = edges.map((e, i) => {
		const d = pts[i]
			.map(
				(p, k) =>
					`${k ? "L" : "M"}${(p.x + dx).toFixed(1)},${(p.y + dy).toFixed(1)}`,
			)
			.join(" ");
		return { key: edgeKey(e), d, from: e.from, to: e.to };
	});
	return {
		nodes: placed,
		edges: placedEdges,
		width: Math.ceil(Math.max(...xs) - Math.min(...xs) + 2 * PAD),
		height: Math.ceil(Math.max(...ys) - Math.min(...ys) + 2 * PAD),
	};
}
