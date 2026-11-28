// Layout of a nested CuTe tuple (an IntTuple, or any tuple of leaves) as a tree, for TupleTree.astro.
// Leaves sit on the bottom row, spaced evenly in order; a tuple node sits above the middle of its
// children, one row per nesting level. Edges carry the child's index inside its parent, so the path
// from the root to a leaf reads as the hierarchical coordinate `find` and `get(mode=[...])` use.

export type Tup = number | string | null | Tup[];

export interface TreeNode {
	/** Child indices from the root, e.g. [2, 1, 0]. */
	path: number[];
	x: number;
	y: number;
	/** Leaf value as text; undefined for a tuple node. */
	label?: string;
	/** Position of a leaf in the flattened tuple; -1 for a tuple node. */
	leafIndex: number;
}

export interface TreeEdge {
	from: TreeNode;
	to: TreeNode;
	/** Index of `to` inside `from`. */
	index: number;
}

export interface TreeLayout {
	nodes: TreeNode[];
	edges: TreeEdge[];
	width: number;
	height: number;
	depth: number;
}

export const leafText = (v: Exclude<Tup, Tup[]>): string =>
	v === null ? "_" : String(v);

/** Nesting depth with CuTe's convention: a leaf has depth 0, (1, 2) has depth 1. */
export function tupleDepth(t: Tup): number {
	return Array.isArray(t) ? 1 + Math.max(0, ...t.map(tupleDepth)) : 0;
}

/** Leaves in order, as text. */
export function leaves(t: Tup): string[] {
	return Array.isArray(t) ? t.flatMap(leaves) : [leafText(t)];
}

/** CuTe-style text of a tuple, e.g. ((2,3),4). */
export function formatTup(t: Tup): string {
	return Array.isArray(t) ? `(${t.map(formatTup).join(",")})` : leafText(t);
}

export function samePath(a: readonly number[], b: readonly number[]): boolean {
	return a.length === b.length && a.every((v, i) => v === b[i]);
}

/** True when `prefix` is a (non-strict) prefix of `path`. */
export function isPrefix(
	prefix: readonly number[],
	path: readonly number[],
): boolean {
	return prefix.length <= path.length && prefix.every((v, i) => v === path[i]);
}

export function layoutTree(t: Tup, dx = 40, dy = 44, pad = 22): TreeLayout {
	const depth = tupleDepth(t);
	const nodes: TreeNode[] = [];
	const edges: TreeEdge[] = [];
	let nextLeaf = 0;
	// Leaves all share the bottom row so that equal leaves line up across trees of equal depth.
	const leafY = pad + depth * dy;

	const visit = (node: Tup, path: number[], level: number): TreeNode => {
		if (!Array.isArray(node)) {
			const n: TreeNode = {
				path,
				x: pad + nextLeaf * dx,
				y: leafY,
				label: leafText(node),
				leafIndex: nextLeaf,
			};
			nextLeaf++;
			nodes.push(n);
			return n;
		}
		if (node.length === 0) {
			// An empty tuple takes a leaf slot so it stays visible.
			const n: TreeNode = {
				path,
				x: pad + nextLeaf * dx,
				y: pad + level * dy,
				leafIndex: -1,
			};
			nextLeaf++;
			nodes.push(n);
			return n;
		}
		const kids = node.map((c, i) => visit(c, [...path, i], level + 1));
		const n: TreeNode = {
			path,
			x: (kids[0].x + kids[kids.length - 1].x) / 2,
			y: pad + level * dy,
			leafIndex: -1,
		};
		nodes.push(n);
		kids.forEach((k, i) => {
			edges.push({ from: n, to: k, index: i });
		});
		return n;
	};

	visit(t, [], 0);
	const width = 2 * pad + Math.max(0, nextLeaf - 1) * dx;
	const height = 2 * pad + depth * dy;
	return { nodes, edges, width, height, depth };
}
