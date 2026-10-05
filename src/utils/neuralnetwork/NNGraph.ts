// Describes a neural network as a graph of layers, for NeuralNetworkGraph to draw.
// Build it by adding nodes in order; each node names the nodes it reads from:
//
//   const g = new NNGraph();
//   const x = g.node("input", "Image", { detail: "224×224×3" });
//   const c = g.node("conv", "Conv 11×11", { from: x, detail: "96 filters, stride 4", shape: "55×55×96" });
//   g.node("output", "Logits", { from: c });
//
// A node can also carry the formula it computes (LaTeX) and its weights; both are shown in a
// card when the node is hovered or focused:
//
//   g.node("linear", "FC", { from: c, formula: "y = Wx + b", weights: { W: w, b: bias } });

/** What a node does; picks its colour and its legend entry. */
export type NodeKind =
	| "input"
	| "output"
	| "embedding"
	| "linear"
	| "conv"
	| "pool"
	| "attention"
	| "norm"
	| "activation"
	| "op"
	| "other";

/**
 * A weight tensor: nested arrays (number[] for a vector, number[][] for a matrix, and so on),
 * or flat row-major data with its shape, e.g. { shape: [96, 3, 11, 11], data: Float32Array }.
 */
export type Tensor = NestedArray | { shape: number[]; data: ArrayLike<number> };
export type NestedArray = number[] | NestedArray[];

/** One tensor (shown as "W"), or several by name, e.g. { W_1: ..., b_1: ... }; names are LaTeX. */
export type NodeWeights = Tensor | Record<string, Tensor>;

export interface GraphNode {
	id: string;
	kind: NodeKind;
	/** Main text, e.g. "Multi-head attention". A 1–2 character op ("+", "×") is drawn as a circle. */
	label: string;
	/** Second line in grey, e.g. hyperparameters: "96 filters, 11×11, stride 4". */
	detail?: string;
	/** Output tensor shape, shown last in monospace, e.g. "55×55×96". */
	shape?: string;
	/** Group (see NNGraph.group) this node is drawn inside. */
	group?: string;
	/** What the node computes, in LaTeX (rendered with KaTeX), e.g. "y = \\sigma(Wx + b)". */
	formula?: string;
	/** The node's parameters; the hover card shows their distribution and rank. */
	weights?: NodeWeights;
	/** Empty space reserved below the text, for figures that draw on top of the graph. */
	tray?: { width: number; height: number };
}

export interface GraphEdge {
	from: string;
	to: string;
	/** Small text drawn on the edge, e.g. "residual". */
	label?: string;
}

export interface GraphGroup {
	id: string;
	/** Drawn in the group's top-left corner, e.g. "Encoder layer × 6". */
	label: string;
}

export interface NodeOptions {
	/** Node id(s) this node takes its input from; an edge is drawn from each. */
	from?: string | string[];
	detail?: string;
	shape?: string;
	group?: string;
	formula?: string;
	weights?: NodeWeights;
	tray?: { width: number; height: number };
	/** Explicit id; defaults to an auto-generated one. Ids must be unique. */
	id?: string;
}

export class NNGraph {
	readonly nodes: GraphNode[] = [];
	readonly edges: GraphEdge[] = [];
	readonly groups: GraphGroup[] = [];

	/** Adds a node and returns its id, to pass as `from` to later nodes. */
	node(kind: NodeKind, label: string, opts: NodeOptions = {}): string {
		const id = opts.id ?? `n${this.nodes.length}`;
		if (this.has(id)) throw new Error(`NNGraph: duplicate node id "${id}"`);
		if (opts.group && !this.groups.some((g) => g.id === opts.group)) {
			throw new Error(
				`NNGraph: node "${label}" is in unknown group "${opts.group}"`,
			);
		}
		const { detail, shape, group, formula, weights, tray } = opts;
		this.nodes.push({
			id,
			kind,
			label,
			detail,
			shape,
			group,
			formula,
			weights,
			tray,
		});
		for (const from of [opts.from ?? []].flat()) this.edge(from, id);
		return id;
	}

	/** Adds an edge between two existing nodes, with an optional label. */
	edge(from: string, to: string, label?: string): void {
		for (const id of [from, to]) {
			if (!this.has(id))
				throw new Error(
					`NNGraph: edge ${from} -> ${to} uses unknown node "${id}"`,
				);
		}
		this.edges.push({ from, to, label });
	}

	/** Declares a group (drawn as a box around its nodes) and returns its id. */
	group(label: string): string {
		const id = `g${this.groups.length}`;
		this.groups.push({ id, label });
		return id;
	}

	private has(id: string): boolean {
		return this.nodes.some((n) => n.id === id);
	}
}
