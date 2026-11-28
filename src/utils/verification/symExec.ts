// Symbolic-execution trees for SymExecTree.astro. A tree is a list of nodes, each with a parent,
// the branch condition it adds, the path condition so far and what the solver said about it.
// Steps reveal the nodes in depth-first order, the order an explorer such as CrossHair or KLEE
// visits paths. Layout: leaves get consecutive columns, a parent sits over the middle of its
// children, rows are depths.

export type SymStatus = "open" | "sat" | "unsat" | "bug" | "done";

export interface SymNode {
	id: string;
	parent?: string;
	/** Branch condition added on the edge into this node, e.g. "x > y". */
	label: string;
	/** Effect of the straight-line code after the branch, e.g. "z = x − y". */
	effect?: string;
	/** Path condition at this node, as the solver sees it. */
	pc: string;
	status: SymStatus;
	/** Solver model (a concrete input that reaches the node), if satisfiable. */
	model?: string;
	/** Index of the source line the node corresponds to. */
	line: number;
	/** Sentence shown when the node is the current step. */
	text: string;
}

export interface PlacedNode extends SymNode {
	x: number;
	y: number;
	depth: number;
}

export interface SymLayout {
	nodes: PlacedNode[];
	edges: { from: string; to: string; d: string }[];
	width: number;
	height: number;
}

export function validateTree(nodes: readonly SymNode[]): string | undefined {
	const ids = new Set<string>();
	for (const n of nodes) {
		if (ids.has(n.id)) return `duplicate node "${n.id}"`;
		if (n.parent !== undefined && !ids.has(n.parent))
			return `node "${n.id}" listed before its parent "${n.parent}"`;
		ids.add(n.id);
	}
	if (nodes.filter((n) => n.parent === undefined).length !== 1)
		return "tree needs exactly one root";
	return undefined;
}

export function layoutTree(
	nodes: readonly SymNode[],
	colW = 86,
	rowH = 66,
	nodeW = 76,
	nodeH = 38,
): SymLayout {
	const kids = new Map<string, string[]>();
	for (const n of nodes)
		if (n.parent !== undefined)
			kids.set(n.parent, [...(kids.get(n.parent) ?? []), n.id]);
	const xs = new Map<string, number>();
	const depth = new Map<string, number>();
	let leaf = 0;
	const visit = (id: string, d: number): number => {
		depth.set(id, d);
		const ch = kids.get(id) ?? [];
		const x =
			ch.length === 0
				? leaf++ * colW
				: ch.map((c) => visit(c, d + 1)).reduce((a, b) => a + b, 0) / ch.length;
		xs.set(id, x);
		return x;
	};
	const root = nodes.find((n) => n.parent === undefined);
	if (!root) throw new Error("layoutTree: no root");
	visit(root.id, 0);
	const pad = 6;
	const placed: PlacedNode[] = nodes.map((n) => ({
		...n,
		x: (xs.get(n.id) ?? 0) + nodeW / 2 + pad,
		y: (depth.get(n.id) ?? 0) * rowH + nodeH / 2 + pad,
		depth: depth.get(n.id) ?? 0,
	}));
	const pos = new Map(placed.map((p) => [p.id, p]));
	const edges = placed
		.filter((p) => p.parent !== undefined)
		.map((p) => {
			const q = pos.get(p.parent as string) as PlacedNode;
			const y0 = q.y + nodeH / 2;
			const y1 = p.y - nodeH / 2;
			const ym = (y0 + y1) / 2;
			return {
				from: q.id,
				to: p.id,
				d: `M${q.x},${y0} C${q.x},${ym} ${p.x},${ym} ${p.x},${y1}`,
			};
		});
	const maxDepth = Math.max(...placed.map((p) => p.depth));
	return {
		nodes: placed,
		edges,
		width: Math.max(leaf, 1) * colW - (colW - nodeW) + 2 * pad,
		height: maxDepth * rowH + nodeH + 2 * pad,
	};
}

/** Counts shown on the last step: paths explored, infeasible, failing. */
export function summarize(nodes: readonly SymNode[]): {
	leaves: number;
	unsat: number;
	bugs: number;
} {
	const parents = new Set(nodes.map((n) => n.parent));
	const leaves = nodes.filter((n) => !parents.has(n.id));
	return {
		leaves: leaves.length,
		unsat: leaves.filter((n) => n.status === "unsat").length,
		bugs: leaves.filter((n) => n.status === "bug").length,
	};
}
