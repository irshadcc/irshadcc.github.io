// The input format of ParallelGraph: a distributed computation as JSON-compatible data.
//
//   {
//     "parallel_config": { "dp_size": 4, "tp_size": 1, "pp_size": 1, "ep_size": 4 },
//     // Nodes: PyTorch operators, communication collectives, inputs and outputs.
//     "operations": {
//       "router0": { "type": "operator", "label": "Router", "rank": 0, "icon": "network",
//                    "equation": { "title": "...", "latex": ["p_t = ..."], "note": "..." } },
//       "a2a":     { "type": "collective", "label": "All-to-all: dispatch", "ranks": [0, 1, 2, 3] },
//       ...
//     },
//     // Edges: the tensor flowing from one operation to the next.
//     "tensors": {
//       "routing_map@0": { "from": "router0", "to": "a2a", "name": "routing_map",
//                          "symbolic_shape": "(batch_size·seq_len, num_experts)", "shape": [2, 8],
//                          "axes": ["Tokens", "Expert"], "row_labels": ["0", "1"],
//                          "col_labels": ["0", ..., "7"], "values": [["1", "0", ...], ...] },
//       ...
//     }
//   }
//
// Operations are drawn in the order they are listed; an operation's inputs in the order their
// tensors are listed. A per-rank operation's second line is its coordinates in the process grid,
// computed from `rank` and `parallel_config` by rankCoords.
import { NNGraph, type NodeKind } from "../neuralnetwork/NNGraph";
import type { NNIconName } from "../neuralnetwork/nnIcons";

export interface ParallelConfig {
	dp_size: number;
	tp_size: number;
	pp_size: number;
	ep_size: number;
}

export interface Equation {
	title: string;
	/** LaTeX, one display line each. */
	latex: string[];
	note?: string;
}

export interface Operation {
	/** Picks the box colour: input/output red, operator blue, collective orange. */
	type: "input" | "output" | "operator" | "collective";
	label: string;
	/** Global rank the operation runs on; omitted for collectives, which span `ranks`. */
	rank?: number;
	ranks?: number[];
	icon?: NNIconName;
	/** Shown when the node is hovered. */
	equation?: Equation;
}

export interface Tensor {
	from: string;
	to: string;
	/** Name in the code. */
	name: string;
	/** Shape in words, e.g. "(batch_size·seq_len, num_experts)". */
	symbolic_shape: string;
	/** Shape in this example. */
	shape: number[];
	/** Titles of the row and column axes, e.g. ["Tokens", "Expert"]. */
	axes: [string, string];
	row_labels: string[];
	col_labels: string[];
	/** The values, as display strings, one array per row. */
	values: string[][];
	/** Optional tint per row: the rank a row belongs to (e.g. that holds its expert), or null. */
	row_ranks?: (number | null)[];
	note?: string;
}

export interface ParallelGraphSpec {
	parallel_config: ParallelConfig;
	operations: Record<string, Operation>;
	tensors: Record<string, Tensor>;
}

/**
 * A rank's coordinates in the process grid, (DP, TP, PP, EP). Ranks are numbered TP fastest,
 * then DP, then PP (Megatron's default order with CP = 1). Expert parallelism reuses the ranks
 * of one pipeline stage: EP groups are blocks of `ep_size` consecutive ranks of the stage.
 */
export function rankCoords(
	rank: number,
	c: ParallelConfig,
): [number, number, number, number] {
	const stage = c.tp_size * c.dp_size;
	const tp = rank % c.tp_size;
	const dp = Math.floor(rank / c.tp_size) % c.dp_size;
	const pp = Math.floor(rank / stage);
	const ep = (rank % stage) % c.ep_size;
	return [dp, tp, pp, ep];
}

export const coordsLine = (rank: number, c: ParallelConfig) =>
	`(DP, TP, PP, EP) = (${rankCoords(rank, c).join(", ")})`;

const KIND: Record<Operation["type"], NodeKind> = {
	input: "input",
	output: "output",
	operator: "linear",
	collective: "other",
};

/** Checks that every tensor joins two listed operations; returns the problems found. */
export function validateSpec(spec: ParallelGraphSpec): string[] {
	const ops = new Set(Object.keys(spec.operations));
	const problems: string[] = [];
	for (const [id, t] of Object.entries(spec.tensors)) {
		if (!ops.has(t.from))
			problems.push(`tensor ${id}: unknown operation ${t.from}`);
		if (!ops.has(t.to))
			problems.push(`tensor ${id}: unknown operation ${t.to}`);
		if (t.values.length !== t.row_labels.length)
			problems.push(
				`tensor ${id}: ${t.values.length} rows, ${t.row_labels.length} labels`,
			);
		if (t.values.some((r) => r.length !== t.col_labels.length))
			problems.push(`tensor ${id}: a row is not ${t.col_labels.length} wide`);
	}
	return problems;
}

/** The graph NeuralNetworkGraph draws: one node per operation, one edge per tensor. */
export function specToGraph(spec: ParallelGraphSpec): NNGraph {
	const problems = validateSpec(spec);
	if (problems.length) throw new Error(`ParallelGraph: ${problems.join("; ")}`);
	const tensors = Object.values(spec.tensors);
	const g = new NNGraph();
	for (const [id, op] of Object.entries(spec.operations))
		g.node(KIND[op.type], op.label, {
			id,
			detail:
				op.rank === undefined
					? undefined
					: coordsLine(op.rank, spec.parallel_config),
			from: tensors.filter((t) => t.to === id).map((t) => t.from),
			icon: op.icon,
		});
	return g;
}

/** The key a tensor's hover card is found by: its edge. */
export const edgeKey = (t: Pick<Tensor, "from" | "to">) => `${t.from}->${t.to}`;
