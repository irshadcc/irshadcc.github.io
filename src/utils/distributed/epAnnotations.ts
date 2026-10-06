// Hover annotations for MoeEpSteps' graphs. Every edge carries a tensor: its name, symbolic and
// concrete shape, and its values as a matrix with axis titles (rows: tokens or token copies,
// columns: experts or hidden dimensions). Every node gets the equation it computes. Values come
// from the numeric model in epNumeric.ts; layouts follow each flow (epFlows.ts).
import {
	COPIES,
	type Copy,
	EP,
	EXPERTS,
	LOCAL,
	ROUTING,
	TOKENS,
	TOKENS_PER_RANK,
	TOP_K,
	node,
	simulate,
} from "./epDispatch";
import {
	DS_CAPACITY,
	type FlowName,
	GPUS_PER_NODE,
	deepseekIbSends,
	deepspeedSlots,
} from "./epFlows";
import {
	H,
	XS,
	YS,
	copyOutput,
	partialOutput,
	routingMapRow,
} from "./epNumeric";

export interface EdgeView {
	name: string;
	/** Symbolic shape, e.g. "(batch_size·seq_len, num_experts)". */
	sym: string;
	/** Concrete shape in this example. */
	shape: string;
	rowAxis: string;
	colAxis: string;
	rowLabels: string[];
	colLabels: string[];
	values: string[][];
	/** Row tint: the rank that holds the row's expert, or null. */
	tones: (number | null)[];
	note: string;
}

export interface NodeEquation {
	title: string;
	/** LaTeX, one display line each. */
	latex: string[];
	note: string;
}

export interface Annotations {
	edges: Record<string, EdgeView>;
	equations: Record<string, NodeEquation>;
}

const range = (n: number) => [...Array(n).keys()];
const f2 = (v: number) => (Math.abs(v) < 0.005 ? "0.00" : v.toFixed(2));
const hCols = range(H).map((j) => `${j}`);
const eCols = range(EXPERTS).map((e) => `${e}`);
const key = (from: string, to: string) => `${from}->${to}`;
const homeTokens = (r: number) =>
	range(TOKENS_PER_RANK).map((i) => r * TOKENS_PER_RANK + i);
const copyLabel = (c: Copy) => `t${c.t}→E${c.expert}`;
const zero = () => new Array<number>(H).fill(0);

/** A [rows, hidden] tensor from row vectors. */
function hiddenEdge(
	name: string,
	sym: string,
	rowAxis: string,
	rows: { label: string; v: number[]; tone?: number | null }[],
	note: string,
): EdgeView {
	return {
		name,
		sym,
		shape: `(${rows.length}, ${H})`,
		rowAxis,
		colAxis: "hidden",
		rowLabels: rows.map((r) => r.label),
		colLabels: hCols,
		values: rows.map((r) => r.v.map(f2)),
		tones: rows.map((r) => r.tone ?? null),
		note,
	};
}
const tokenRows = (ts: number[]) =>
	ts.map((t) => ({ label: `t${t}`, v: XS[t] }));
const inputRows = (cs: Copy[]) =>
	cs.map((c) => ({ label: copyLabel(c), v: XS[c.t], tone: c.dest }));
const outputRows = (cs: Copy[]) =>
	cs.map((c) => ({ label: copyLabel(c), v: copyOutput(c), tone: c.dest }));

function routingMapEdge(ts: number[], note: string): EdgeView {
	return {
		name: "routing_map",
		sym: "(batch_size·seq_len, num_experts)",
		shape: `(${ts.length}, ${EXPERTS})`,
		rowAxis: "Tokens",
		colAxis: "Expert",
		rowLabels: ts.map((t) => `${t}`),
		colLabels: eCols,
		values: ts.map((t) => routingMapRow(t).map(String)),
		tones: ts.map(() => null),
		note,
	};
}
function topkEdge(
	ts: number[],
	name: string,
	map: (e: number) => number,
	note: string,
): EdgeView {
	return {
		name,
		sym: "(batch_size·seq_len, top_k)",
		shape: `(${ts.length}, ${TOP_K})`,
		rowAxis: "Tokens",
		colAxis: "k",
		rowLabels: ts.map((t) => `${t}`),
		colLabels: range(TOP_K).map(String),
		values: ts.map((t) => ROUTING[t].map((e) => String(map(e)))),
		tones: ts.map(() => null),
		note,
	};
}
function countsEdge(
	name: string,
	sym: string,
	rowAxis: string,
	rowLabels: string[],
	colAxis: string,
	colLabels: string[],
	values: number[][],
	note: string,
): EdgeView {
	return {
		name,
		sym,
		shape: `(${values.length}, ${values[0].length})`,
		rowAxis,
		colAxis,
		rowLabels,
		colLabels,
		values: values.map((r) => r.map(String)),
		tones: values.map(() => null),
		note,
	};
}

// ---------------------------------------------------------------- equations shared by flows

const EQ = {
	tokens: (n: string): NodeEquation => ({
		title: "Input tokens",
		latex: [`X \\in \\mathbb{R}^{${n} \\times h}, \\qquad x_t = X[t, :]`],
		note: "The hidden states the attention layer produced, one row per token.",
	}),
	router: {
		title: "Router: softmax, top-2",
		latex: [
			"p_t = \\operatorname{softmax}(W_r\\, x_t) \\in \\mathbb{R}^{E}",
			"\\mathcal{T}_t = \\operatorname{top}_2(p_t), \\qquad g_{t,e} = \\frac{p_{t,e}}{\\sum_{j \\in \\mathcal{T}_t} p_{t,j}}",
			"\\text{routing\\_map}[t, e] = \\mathbb{1}[e \\in \\mathcal{T}_t]",
		],
		note: "Each token keeps its two best experts; their gates sum to 1.",
	} as NodeEquation,
	experts: (local: string): NodeEquation => ({
		title: `Experts ${local}`,
		latex: [
			"z = W_1^{(e)} x_t = \\begin{bmatrix} z_{\\text{gate}} \\\\ z_{\\text{up}} \\end{bmatrix}",
			"y_{t,e} = W_2^{(e)} \\big( g_{t,e} \\cdot \\operatorname{silu}(z_{\\text{gate}}) \\odot z_{\\text{up}} \\big)",
		],
		note: "A SwiGLU MLP per expert, run as one grouped GEMM over its rows; the gate weight is applied inside the activation.",
	}),
	unpermute: {
		title: "Unpermute: sum each token's copies",
		latex: ["y_t = \\sum_{k=1}^{2} y_{t, e_k}"],
		note: "A scatter-add of the returned rows into their tokens' positions.",
	} as NodeEquation,
};

// ---------------------------------------------------------------- Megatron-LM

function megatron(): Annotations {
	const S = simulate();
	const c = (id: number) => COPIES[id];
	const edges: Record<string, EdgeView> = {};
	const equations: Record<string, NodeEquation> = {};
	const counts = S.map((s) => s.perExpert);
	for (let r = 0; r < EP; r++) {
		const ts = homeTokens(r);
		edges[key(node.tok(r), node.router(r))] = hiddenEdge(
			"hidden_states",
			"(batch_size·seq_len, hidden)",
			"Tokens",
			tokenRows(ts),
			`Rank ${r}'s own tokens.`,
		);
		edges[key(node.router(r), node.gather)] = routingMapEdge(
			ts,
			"1 where the token chose the expert. probs has the same pattern, holding the gate weights.",
		);
		edges[key(node.gather, node.perm(r))] = countsEdge(
			"num_global_tokens_per_expert",
			"(ep_size, num_experts)",
			"Source rank",
			range(EP).map(String),
			"Expert",
			eCols,
			counts,
			"Column sums of every rank's routing_map, all-gathered: every rank sees the whole table.",
		);
		edges[key(node.perm(r), node.dispatch)] = hiddenEdge(
			"permutated_local_input_tokens",
			"(batch_size·seq_len·top_k, hidden)",
			"Copies",
			inputRows(S[r].permuted.map(c)),
			`Sorted by expert, so each destination rank's chunk is contiguous. input_splits = [${S[r].inputSplits.join(", ")}].`,
		);
		edges[key(node.dispatch, node.recv(r))] = hiddenEdge(
			"global_input_tokens",
			"(sum(output_splits), hidden)",
			"Copies",
			inputRows(S[r].received.map(c)),
			`Grouped by source rank; output_splits = [${S[r].outputSplits.join(", ")}].`,
		);
		edges[key(node.recv(r), node.experts(r))] = hiddenEdge(
			"dispatched_input",
			"(sum(tokens_per_expert), hidden)",
			"Copies",
			inputRows(S[r].sorted.map(c)),
			`Regrouped by local expert: tokens_per_expert = [${S[r].tokensPerExpert.join(", ")}].`,
		);
		edges[key(node.experts(r), node.unsort(r))] = hiddenEdge(
			"expert_output",
			"(sum(tokens_per_expert), hidden)",
			"Copies",
			outputRows(S[r].sorted.map(c)),
			"Each row is g · W2 · SwiGLU(W1 x) for its copy.",
		);
		edges[key(node.unsort(r), node.combine)] = hiddenEdge(
			"hidden_states",
			"(sum(output_splits), hidden)",
			"Copies",
			outputRows(S[r].unsorted.map(c)),
			"Back in (source rank, expert) order, ready to return.",
		);
		edges[key(node.combine, node.out(r))] = hiddenEdge(
			"permutated_local_input_tokens",
			"(batch_size·seq_len·top_k, hidden)",
			"Copies",
			outputRows(S[r].returned.map(c)),
			`Rank ${r}'s copies, back in the order it sent them. Unpermute adds each token's rows: ${ts
				.map((t) => `y${t} = [${YS[t].map(f2).join(", ")}]`)
				.join("; ")}.`,
		);
		equations[node.tok(r)] = EQ.tokens("T_r");
		equations[node.router(r)] = EQ.router;
		equations[node.perm(r)] = {
			title: "Permute",
			latex: [
				"X^{\\text{perm}} = X[\\text{sorted\\_indices}]",
				"\\text{sorted\\_indices} = \\operatorname{argsort}_{\\text{stable}}(\\text{routing\\_map}^{\\top}\\text{ by expert})",
			],
			note: "Copies each token once per chosen expert and lines the copies up by expert.",
		};
		equations[node.recv(r)] = {
			title: "sort_chunks_by_idxs: by local expert",
			latex: ["\\text{(source, expert)} \\;\\to\\; \\text{(expert, source)}"],
			note: "The all-to-all delivers rows grouped by source rank; this reorders the chunks so each local expert's rows are contiguous for the grouped GEMM.",
		};
		equations[node.experts(r)] = EQ.experts(`E${r * LOCAL}, E${r * LOCAL + 1}`);
		equations[node.unsort(r)] = {
			title: "sort_chunks_by_idxs: restore",
			latex: ["\\text{(expert, source)} \\;\\to\\; \\text{(source, expert)}"],
			note: "The inverse of the regrouping, so each source's rows are contiguous again.",
		};
		equations[node.out(r)] = EQ.unpermute;
	}
	equations[node.gather] = {
		title: "Count copies, all-gather counts",
		latex: [
			"c_{r,e} = \\sum_{t} \\text{routing\\_map}_r[t, e]",
			"\\text{input\\_splits}_r[j] = \\sum_{e \\in \\mathcal{E}_j} c_{r,e}, \\quad \\text{output\\_splits}_j[r] = \\text{input\\_splits}_r[j]",
		],
		note: "𝓔_j is the set of experts on rank j. The counts reach the CPU here.",
	};
	equations[node.dispatch] = {
		title: "All-to-all: dispatch",
		latex: [
			"\\text{rank } i \\text{ sends } X^{\\text{perm}}_i[\\text{chunk}_{i \\to j}] \\text{ to rank } j",
		],
		note: "Uneven chunks: their sizes are the input_splits.",
	};
	equations[node.combine] = {
		title: "All-to-all: combine",
		latex: [
			"\\text{rank } j \\text{ returns the rows of chunk}_{i \\to j} \\text{ to rank } i",
		],
		note: "The dispatch with the splits swapped.",
	};
	return { edges, equations };
}

// ---------------------------------------------------------------- vLLM

function vllm(): Annotations {
	const edges: Record<string, EdgeView> = {};
	const equations: Record<string, NodeEquation> = {};
	const local = (r: number) =>
		COPIES.filter((c) => c.dest === r).sort(
			(a, b) => a.expert - b.expert || a.t - b.t,
		);
	for (let r = 0; r < EP; r++) {
		const ts = homeTokens(r);
		edges[key(`v-tok${r}`, `v-router${r}`)] = hiddenEdge(
			"x",
			"(num_tokens, hidden)",
			"Tokens",
			tokenRows(ts),
			`Rank ${r}'s own requests.`,
		);
		edges[key(`v-router${r}`, "v-allgather")] = topkEdge(
			ts,
			"topk_ids",
			(e) => e,
			"Expert ids of each token's two choices; topk_weights holds their gates.",
		);
		edges[key("v-allgather", `v-all${r}`)] = hiddenEdge(
			"x (gathered)",
			"(dp_size·num_tokens, hidden)",
			"Tokens",
			tokenRows(range(TOKENS)),
			"Every token of every rank, with its topk_ids and topk_weights.",
		);
		edges[key(`v-all${r}`, `v-exp${r}`)] = hiddenEdge(
			"rows read via sorted_token_ids",
			"(local copies, hidden)",
			"Copies",
			inputRows(local(r)),
			"Only copies for this rank's experts; expert_map turns the rest into -1 and they are never read.",
		);
		edges[key(`v-exp${r}`, `v-partial${r}`)] = hiddenEdge(
			"intermediate_cache3",
			"(num_tokens, top_k, hidden)",
			"Copies",
			outputRows(local(r)),
			"Each local copy's gate-weighted expert output; the other (token, k) rows stay zero.",
		);
		edges[key(`v-partial${r}`, "v-reducescatter")] = hiddenEdge(
			"out (partial)",
			"(dp_size·num_tokens, hidden)",
			"Tokens",
			range(TOKENS).map((t) => ({ label: `t${t}`, v: partialOutput(r, t) })),
			"moe_sum of each token's rows: this rank's experts' share only.",
		);
		edges[key("v-reducescatter", `v-out${r}`)] = hiddenEdge(
			"output",
			"(num_tokens, hidden)",
			"Tokens",
			ts.map((t) => ({ label: `t${t}`, v: YS[t] })),
			"The sum of every rank's partial rows for this rank's tokens.",
		);
		equations[`v-tok${r}`] = EQ.tokens("T_r");
		equations[`v-router${r}`] = EQ.router;
		equations[`v-all${r}`] = {
			title: "moe_align_block_size",
			latex: [
				"\\text{expert\\_map}[e] = \\begin{cases} e - rE/N & e \\in \\mathcal{E}_r \\\\ -1 & \\text{otherwise} \\end{cases}",
			],
			note: "moe_align_block_size sorts the copies by expert and pads each group to BLOCK_M; -1 groups are skipped.",
		};
		equations[`v-exp${r}`] = EQ.experts(`E${r * LOCAL}, E${r * LOCAL + 1}`);
		equations[`v-partial${r}`] = {
			title: "moe_sum",
			latex: [
				"o^{(r)}_t = \\sum_{k \\,:\\, e_k \\in \\mathcal{E}_r} y_{t, e_k}",
			],
			note: "Zero for tokens none of this rank's experts saw.",
		};
		equations[`v-out${r}`] = {
			title: "Output",
			latex: ["y_t = \\sum_{r} o^{(r)}_t"],
			note: "What the reduce-scatter delivered.",
		};
	}
	equations["v-allgather"] = {
		title: "All-gather",
		latex: [
			"X_{\\text{all}} = \\begin{bmatrix} X_0 \\\\ \\vdots \\\\ X_{N-1} \\end{bmatrix}",
		],
		note: "Also gathers topk_ids and topk_weights.",
	};
	equations["v-reducescatter"] = {
		title: "Reduce-scatter",
		latex: [
			"y_t = \\sum_{r=0}^{N-1} o^{(r)}_t \\quad \\text{for the tokens of the receiving rank}",
		],
		note: "Sums the partial outputs and hands each rank its own rows.",
	};
	return { edges, equations };
}

// ---------------------------------------------------------------- SGLang

function sglang(): Annotations {
	const edges: Record<string, EdgeView> = {};
	const equations: Record<string, NodeEquation> = {};
	const all = range(TOKENS);
	for (let r = 0; r < EP; r++) {
		const local = COPIES.filter((c) => c.dest === r).sort(
			(a, b) => a.expert - b.expert || a.t - b.t,
		);
		edges[key(`g-tok${r}`, `g-router${r}`)] = hiddenEdge(
			"hidden_states",
			"(num_tokens, hidden)",
			"Tokens",
			tokenRows(all),
			"The same batch on every rank of the TP group.",
		);
		edges[key(`g-router${r}`, `g-exp${r}`)] = topkEdge(
			all,
			"topk_ids (after local_expert_mapping)",
			(e) => (Math.floor(e / LOCAL) === r ? e - r * LOCAL : -1),
			`Local expert ids on rank ${r}; -1 for experts on other ranks, which the kernel skips.`,
		);
		edges[key(`g-exp${r}`, "g-allreduce")] = hiddenEdge(
			"out (partial)",
			"(num_tokens, hidden)",
			"Tokens",
			all.map((t) => ({ label: `t${t}`, v: partialOutput(r, t) })),
			"Each token's gate-weighted rows summed over this rank's experts; zero for tokens none of them saw.",
		);
		edges[key("g-allreduce", `g-out${r}`)] = hiddenEdge(
			"final_hidden_states",
			"(num_tokens, hidden)",
			"Tokens",
			all.map((t) => ({ label: `t${t}`, v: YS[t] })),
			"Identical on every rank.",
		);
		equations[`g-tok${r}`] = EQ.tokens("T");
		equations[`g-router${r}`] = EQ.router;
		equations[`g-exp${r}`] = {
			title: `Fused MoE: experts E${r * LOCAL}, E${r * LOCAL + 1}`,
			latex: [
				...EQ.experts("").latex,
				"o^{(r)}_t = \\sum_{k \\,:\\, e_k \\in \\mathcal{E}_r} y_{t, e_k}",
			],
			note: "The Triton kernel runs both GEMMs and, at the end, sums each token's rows: a partial output per token.",
		};
		equations[`g-out${r}`] = {
			title: "Output",
			latex: ["y_t = \\sum_{r} o^{(r)}_t"],
			note: "The same on every rank.",
		};
	}
	equations["g-allreduce"] = {
		title: "All-reduce",
		latex: ["Y = \\sum_{r=0}^{N-1} O^{(r)}"],
		note: "The same all-reduce that ends a tensor-parallel MLP.",
	};
	return { edges, equations };
}

// ---------------------------------------------------------------- DeepSpeed

function deepspeed(): Annotations {
	const C = DS_CAPACITY;
	const { cells, dropped } = deepspeedSlots();
	const edges: Record<string, EdgeView> = {};
	const equations: Record<string, NodeEquation> = {};
	const slotRow = (r: number, e: number, j: number, out: boolean) => {
		const id = cells[r][e][j];
		if (id < 0) return { label: `E${e}·s${j}: pad`, v: zero(), tone: null };
		const cp = COPIES[id];
		return {
			label: `E${e}·s${j}: t${cp.t}`,
			v: out ? copyOutput(cp) : XS[cp.t],
			tone: cp.dest,
		};
	};
	const bufRows = (r: number, out: boolean) =>
		range(EXPERTS).flatMap((e) => range(C).map((j) => slotRow(r, e, j, out)));
	const recvRows = (d: number, out: boolean) =>
		range(LOCAL).flatMap((le) =>
			range(EP).flatMap((s) =>
				range(C).map((j) => ({
					...slotRow(s, d * LOCAL + le, j, out),
					label: `from ${s} · ${slotRow(s, d * LOCAL + le, j, out).label}`,
				})),
			),
		);
	for (let r = 0; r < EP; r++) {
		const ts = homeTokens(r);
		edges[key(`ds-tok${r}`, `ds-gate${r}`)] = hiddenEdge(
			"x",
			"(S, hidden)",
			"Tokens",
			tokenRows(ts),
			"S = this rank's tokens.",
		);
		edges[key(`ds-gate${r}`, `ds-buf${r}`)] = {
			name: "dispatch mask",
			sym: "(S, num_experts · C)",
			shape: `(${ts.length}, ${EXPERTS * C})`,
			rowAxis: "Tokens",
			colAxis: "Expert · slot",
			rowLabels: ts.map(String),
			colLabels: range(EXPERTS).flatMap((e) =>
				range(C).map((j) => (C === 1 ? `${e}` : `${e}.${j}`)),
			),
			values: ts.map((t) =>
				range(EXPERTS).flatMap((e) =>
					range(C).map((j) =>
						cells[r][e][j] >= 0 && COPIES[cells[r][e][j]].t === t ? "1" : "0",
					),
				),
			),
			tones: ts.map(() => null),
			note: `Like routing_map, but only copies that got a slot.${dropped.some((id) => COPIES[id].home === r) ? ` t${COPIES[dropped[0]].t}'s choice of E${COPIES[dropped[0]].expert} was dropped.` : ""}`,
		};
		edges[key(`ds-buf${r}`, "ds-a2a1")] = hiddenEdge(
			"dispatched_input",
			"(num_experts, C, hidden)",
			"Expert · slot",
			bufRows(r, false),
			"Empty slots are zero padding, and they are sent too.",
		);
		edges[key("ds-a2a1", `ds-exp${r}`)] = hiddenEdge(
			"dispatched_input",
			"(ep_size, num_local_experts, C, hidden)",
			"Source · slot",
			recvRows(r, false),
			"C rows from every rank for each local expert, reshaped per local expert for a batched GEMM.",
		);
		edges[key(`ds-exp${r}`, "ds-a2a2")] = hiddenEdge(
			"expert_output",
			"(ep_size, num_local_experts, C, hidden)",
			"Source · slot",
			recvRows(r, true),
			"Padding rows stay zero: SwiGLU(0) = 0.",
		);
		edges[key("ds-a2a2", `ds-out${r}`)] = hiddenEdge(
			"expert_output",
			"(num_experts, C, hidden)",
			"Expert · slot",
			bufRows(r, true),
			"This rank's slots, now holding expert outputs.",
		);
		equations[`ds-tok${r}`] = EQ.tokens("S");
		equations[`ds-gate${r}`] = {
			title: "Top-2 gate with capacity",
			latex: [
				"C = \\left\\lceil \\frac{2S}{E} \\cdot \\text{cf} \\right\\rceil",
				"\\text{loc}_{t,e} = \\sum_{t' < t} \\text{mask}_1[t', e] \\;(\\text{1st choices first, then 2nd})",
				"\\text{keep } (t,e) \\iff \\text{loc}_{t,e} < C",
			],
			note: "Copies are numbered per expert with a cumulative sum; numbers past C are dropped.",
		};
		equations[`ds-buf${r}`] = {
			title: "_sparse_encode",
			latex: [
				"D[e, s] = \\begin{cases} x_t & \\text{slot } s \\text{ of } e \\text{ holds } t \\\\ 0 & \\text{otherwise} \\end{cases}",
			],
			note: "Gathers each slot's token into an [E, C, h] buffer; empty slots stay zero.",
		};
		equations[`ds-exp${r}`] = EQ.experts(`E${r * LOCAL}, E${r * LOCAL + 1}`);
		equations[`ds-out${r}`] = {
			title: "_sparse_decode",
			latex: ["y_t = \\sum_{(e,s) \\text{ holding } t} g_{t,e}\\, D'[e, s]"],
			note: "Gather, weight, sum (here the weight is already inside D').",
		};
	}
	equations["ds-a2a1"] = {
		title: "All-to-all, equal chunks",
		latex: [
			"\\text{rank } i \\to j: \\; D_i[jL : (j+1)L, :, :] \\quad (L \\cdot C \\text{ rows})",
		],
		note: "Every chunk has the same size, so no counts are exchanged.",
	};
	equations["ds-a2a2"] = {
		title: "All-to-all back",
		latex: ["\\text{rank } j \\to i: \\; R'_j[i, :, :, :]"],
		note: "The reverse of the first exchange.",
	};
	return { edges, equations };
}

// ---------------------------------------------------------------- DeepSeek / DeepEP

function deepseek(): Annotations {
	const sends = deepseekIbSends();
	const nodeOf = (r: number) => Math.floor(r / GPUS_PER_NODE);
	const crossNode = (c: Copy) => nodeOf(c.dest) !== nodeOf(c.home);
	const edges: Record<string, EdgeView> = {};
	const equations: Record<string, NodeEquation> = {};
	for (let r = 0; r < EP; r++) {
		const ts = homeTokens(r);
		const local = COPIES.filter((c) => c.dest === r).sort(
			(a, b) => a.expert - b.expert || a.t - b.t,
		);
		edges[key(`dk-tok${r}`, `dk-router${r}`)] = hiddenEdge(
			"x",
			"(num_tokens, hidden)",
			"Tokens",
			tokenRows(ts),
			`GPU ${r}'s own tokens.`,
		);
		edges[key(`dk-router${r}`, "dk-dispatch")] = topkEdge(
			ts,
			"topk_idx",
			(e) => e,
			"Experts on at most M nodes per token; here both nodes are allowed. x and topk_weights go into buffer.dispatch with it.",
		);
		edges[key("dk-dispatch", `dk-exp${r}`)] = hiddenEdge(
			"recv_x",
			"(num_recv_rows, hidden)",
			"Copies",
			inputRows(local),
			"The expanded layout: rows grouped by local expert, ready for the grouped GEMM.",
		);
		edges[key(`dk-exp${r}`, "dk-combine")] = hiddenEdge(
			"expert_output",
			"(num_recv_rows, hidden)",
			"Copies",
			outputRows(local),
			"Gate-weighted outputs, sent back along the path they came.",
		);
		edges[key("dk-combine", `dk-out${r}`)] = hiddenEdge(
			"combined_x",
			"(num_tokens, hidden)",
			"Tokens",
			ts.map((t) => ({ label: `t${t}`, v: YS[t] })),
			"Local and remote results added together.",
		);
		equations[`dk-tok${r}`] = EQ.tokens("T_r");
		equations[`dk-router${r}`] = {
			title: "Node-limited router",
			latex: [
				"s_{t,e} = \\sigma(u_t^{\\top} c_e), \\quad \\text{score}(n) = \\sum \\operatorname{top}_{K/M}\\{ s_{t,e} + b_e : e \\in n \\}",
				"\\mathcal{T}_t = \\operatorname{top}_K \\{ s_{t,e} + b_e : e \\in \\operatorname{top}_M(\\text{nodes}) \\}",
			],
			note: "DeepSeek-V3: K = 8 experts from M = 4 of 8 nodes, with the balancing bias b.",
		};
		equations[`dk-exp${r}`] = EQ.experts(`E${r * LOCAL}, E${r * LOCAL + 1}`);
		equations[`dk-out${r}`] = {
			title: "Output",
			latex: ["y_t = \\sum_{n} s_{t,n}"],
			note: "One partial sum per node the token visited.",
		};
	}
	const cross = COPIES.filter(crossNode);
	equations["dk-dispatch"] = {
		title: "buffer.dispatch (DeepEP)",
		latex: [
			"\\text{RDMA: } x_t \\;\\to\\; \\text{GPU}(n, \\operatorname{idx}(\\text{home}_t)) \\quad \\forall n \\in \\text{nodes}(\\mathcal{T}_t)",
			"\\text{NVLink: } x_t \\;\\to\\; \\text{GPU}(e) \\quad \\forall e \\in \\mathcal{T}_t \\cap n",
		],
		note: `One kernel, two hops: a token crosses InfiniBand once per target node (${sends.length} transfers for ${cross.length} cross-node copies here), then NVLink forwards it to each GPU of that node holding one of its experts.`,
	};
	equations["dk-combine"] = {
		title: "buffer.combine (DeepEP)",
		latex: [
			"\\text{NVLink: } s_{t,n} = \\sum_{k \\,:\\, \\text{node}(e_k) = n} y_{t, e_k} \\text{ on the forwarding GPU}",
			"\\text{RDMA: } y_t = \\sum_{n} s_{t,n} \\text{ on } \\text{home}_t",
		],
		note: "The dispatch in reverse: a node's results for a token are summed before they cross InfiniBand, one row per (token, node).",
	};
	return { edges, equations };
}

export function annotateFlow(name: FlowName): Annotations {
	if (name === "vllm") return vllm();
	if (name === "sglang") return sglang();
	if (name === "deepspeed") return deepspeed();
	if (name === "deepseek") return deepseek();
	return megatron();
}
