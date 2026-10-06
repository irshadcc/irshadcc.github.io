// The five expert-parallel flows of MoeEpSteps (epFlows.ts) as ParallelGraph specs: operations
// from the flow's nodes, tensors from epAnnotations' edge matrices, equations from its node
// equations. The process grid: attention is data-parallel (DP = 4) in every flow but SGLang's,
// where it is tensor-parallel (TP = 4); EP = 4 reuses the attention ranks.
import type { NNIconName } from "../neuralnetwork/nnIcons";
import { annotateFlow } from "./epAnnotations";
import { EP } from "./epDispatch";
import { type FlowName, type FlowNode, buildFlow } from "./epFlows";
import type {
	Operation,
	ParallelConfig,
	ParallelGraphSpec,
	Tensor,
} from "./parallelGraph";

/** A node's role: its id without the flow prefix and rank number. */
const role = (id: string) =>
	id.replace(/^(v|g|ds|dk)-/, "").replace(/\d+$/, "");
const rankOf = (id: string) => Number(id.match(/(\d+)$/)?.[1] ?? 0);

const ICON_BY_ROLE: [RegExp, NNIconName][] = [
	[/^tok$/, "tokens"],
	[/^(router|gate|exp)$/, "network"],
	[/^gather$/, "count"],
	[/^(perm|recv|unsort|all)$/, "permute"],
	[/^buf$/, "buffer"],
	[/^a2a(-dispatch|-combine)?$/, "ring"],
	[/^allgather$/, "allgather"],
	[/^reducescatter$/, "reducescatter"],
	[/^allreduce$/, "allreduce"],
	[/^(dispatch|combine)$/, "rdma"],
	[/^partial$/, "partial"],
	[/^out$/, "output"],
];
/** Roles that run on one rank; the rest are collectives over all ranks. */
const PER_RANK = /^(tok|router|gate|perm|recv|exp|unsort|out|all|partial|buf)$/;

const typeOf = (k: FlowNode["kind"]): Operation["type"] =>
	k === "input"
		? "input"
		: k === "output"
			? "output"
			: k === "other"
				? "collective"
				: "operator";

function label(name: FlowName, n: FlowNode): string {
	const r = rankOf(n.id);
	switch (role(n.id)) {
		case "tok":
			return name === "deepseek"
				? `Input · node ${Math.floor(r / 2)}`
				: "Input";
		case "exp":
			return `Experts ${2 * r}, ${2 * r + 1}`;
		case "gate":
			return "Gate: top-2, C = 1";
		default:
			return n.label;
	}
}

export function parallelConfig(name: FlowName): ParallelConfig {
	return name === "sglang"
		? { dp_size: 1, tp_size: EP, pp_size: 1, ep_size: EP }
		: { dp_size: EP, tp_size: 1, pp_size: 1, ep_size: EP };
}

export function flowSpec(name: FlowName): ParallelGraphSpec {
	const flow = buildFlow(name);
	const ann = annotateFlow(name);
	const ranks = [...Array(EP).keys()];
	const operations: Record<string, Operation> = {};
	for (const n of flow.nodes) {
		const perRank = PER_RANK.test(role(n.id));
		operations[n.id] = {
			type: typeOf(n.kind),
			label: label(name, n),
			...(perRank ? { rank: rankOf(n.id) } : { ranks }),
			icon: ICON_BY_ROLE.find(([re]) => re.test(role(n.id)))?.[1],
			equation: ann.equations[n.id],
		};
	}
	// One tensor per edge, listed in the order of each node's inputs.
	const tensors: Record<string, Tensor> = {};
	for (const n of flow.nodes)
		for (const from of n.from ?? []) {
			const e = ann.edges[`${from}->${n.id}`];
			if (!e) throw new Error(`epSpecs: no tensor on ${from} -> ${n.id}`);
			tensors[`${e.name}@${from}->${n.id}`] = {
				from,
				to: n.id,
				name: e.name,
				symbolic_shape: e.sym,
				shape: e.shape.replace(/[()]/g, "").split(",").map(Number),
				axes: [e.rowAxis, e.colAxis],
				row_labels: e.rowLabels,
				col_labels: e.colLabels,
				values: e.values,
				row_ranks: e.tones,
				note: e.note,
			};
		}
	return { parallel_config: parallelConfig(name), operations, tensors };
}
