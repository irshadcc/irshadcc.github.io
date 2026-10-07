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
	simulate as megatronState,
} from "./epDispatch";
import {
	DS_CAPACITY,
	GPUS_PER_NODE,
	deepseekIbSends,
	deepspeedSlots,
} from "./epFlows";
import {
	H,
	I,
	W1S,
	W2S,
	XS,
	YS,
	partialOutput,
	probsRow,
	routingMapRow,
} from "./epNumeric";
// One expert-parallel MoE layer as each framework's code runs it on one rank, as ModuleGraph specs
// for the expert-parallelism post. All five use the shared example of epDispatch.ts (EP = 4, 8
// experts, 2 per rank, top-2, 8 tokens, 2 per rank, the same routing) and the numbers of
// epNumeric.ts (hidden size 4, SwiGLU experts of width 2), so every tensor on a wire holds real
// values, and the outputs match the single-device MoE (YS). Each spec follows the condensed code
// the post shows for that framework; rows are tinted by the rank that holds their expert.
//   megatron  – rank 1: MoEAlltoAllTokenDispatcher (dispatch_preprocess, token_dispatch, ...).
//   deepspeed – rank 0: MOELayer with capacity 1, where t0's second choice is dropped.
//   deepseek  – rank 1 (node 0, GPU 1): DeepEP's two hops, where t6 arrives once for two experts.
//   vllm      – rank 1: all-gather, fused MoE on local experts, reduce-scatter.
//   sglang    – rank 1: no dispatch, fused MoE on local experts, all-reduce.
import type { Equation, ModuleSpec, TensorValue } from "./graph/moduleGraph";

export type EpModuleFlow =
	| "megatron"
	| "deepspeed"
	| "deepseek"
	| "vllm"
	| "sglang";

const range = (n: number) => [...Array(n).keys()];
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const f2 = (v: number) => (Math.abs(v) < 0.005 ? "0.00" : v.toFixed(2));
const cols = (n: number) => range(n).map(String);
const eCols = range(EXPERTS).map((e) => `E${e}`);
const silu = (v: number) => v / (1 + Math.exp(-v));
const matvec = (m: number[][], v: number[]) =>
	m.map((row) => row.reduce((a, w, j) => a + w * v[j], 0));
const homeTokens = (r: number) =>
	range(TOKENS_PER_RANK).map((i) => r * TOKENS_PER_RANK + i);
const localExperts = (r: number) => range(LOCAL).map((le) => r * LOCAL + le);

/** fc1 output [gate ‖ up], the activation (gated or not) and the expert output of one copy. */
const fc1 = (c: Copy) => matvec(W1S[c.expert], XS[c.t]);
const act = (c: Copy, gated: boolean) => {
	const z = fc1(c);
	return range(I).map((i) => silu(z[i]) * z[I + i] * (gated ? c.gate : 1));
};
const out = (c: Copy, gated = true) => matvec(W2S[c.expert], act(c, gated));
const label = (c: Copy) => `t${c.t}→E${c.expert}`;

const SYMBOLS: Record<string, string> = {
	tokens: `tokens on this rank (${TOKENS_PER_RANK} here)`,
	all_tokens: `tokens on all ranks (${TOKENS} here)`,
	hidden: `model width h (${H} here)`,
	ffn: `inner width of an expert (${I} here)`,
	num_experts: `experts across all ranks (${EXPERTS} here)`,
	num_local_experts: `experts on this rank (${LOCAL} here)`,
	ep_size: `expert-parallel ranks (${EP} here)`,
	top_k: `experts per token (${TOP_K} here)`,
};

interface Row {
	label: string;
	v: (number | string)[] | null;
	tone?: number | null;
}
/** A matrix tensor; a row with v = null is drawn as "·" (uninitialized or padding). */
function mat(
	name: string,
	symbolic_shape: string,
	axes: [string, string],
	colLabels: string[],
	rows: Row[],
	note?: string,
	shape?: number[],
): TensorValue {
	return {
		name,
		symbolic_shape,
		shape: shape ?? [rows.length, colLabels.length],
		axes,
		row_labels: rows.map((r) => r.label),
		col_labels: colLabels,
		values: rows.map((r) =>
			r.v
				? r.v.map((x) => (typeof x === "number" ? f2(x) : x))
				: colLabels.map(() => "·"),
		),
		row_tones: rows.some((r) => r.tone != null)
			? rows.map((r) => r.tone ?? null)
			: undefined,
		note,
	};
}
const vec = (
	name: string,
	symbolic_shape: string,
	colAxis: string,
	colLabels: string[],
	v: (number | string)[],
	note?: string,
): TensorValue => ({
	...mat(
		name,
		symbolic_shape,
		["", colAxis],
		colLabels,
		[{ label: "", v }],
		note,
	),
	shape: [v.length],
});
const tokenRows = (ts: number[], vs: (t: number) => number[] = (t) => XS[t]) =>
	ts.map((t) => ({ label: `t${t}`, v: vs(t) }));
const hiddenOf = (
	name: string,
	sym: string,
	axis: string,
	rows: Row[],
	note?: string,
) => mat(name, sym, [axis, "hidden"], cols(H), rows, note);
const copyIn = (cs: Copy[]) =>
	cs.map((c) => ({ label: label(c), v: XS[c.t], tone: c.dest }));
const copyOut = (cs: Copy[], gated = true) =>
	cs.map((c) => ({ label: label(c), v: out(c, gated), tone: c.dest }));

const EQ: Record<string, Equation> = {
	router: {
		title: "Router: softmax, top-2",
		latex: [
			"p_t = \\operatorname{softmax}(W_r\\, x_t), \\qquad \\mathcal{T}_t = \\operatorname{top}_2(p_t)",
			"g_{t,e} = \\frac{p_{t,e}}{\\sum_{j \\in \\mathcal{T}_t} p_{t,j}}",
		],
		note: "The routing comes from a seeded PyTorch run; only the chosen experts' gates are shown.",
	},
	fc1: {
		title: "First linear layer",
		latex: [
			"z = W_1^{(e)} x_t = \\begin{bmatrix} z_{\\text{gate}} \\\\ z_{\\text{up}} \\end{bmatrix}",
		],
	},
	swiglu: {
		title: "SwiGLU",
		latex: ["a = \\operatorname{silu}(z_{\\text{gate}}) \\odot z_{\\text{up}}"],
	},
	swigluGated: {
		title: "SwiGLU, scaled by the gate",
		latex: [
			"a = g_{t,e} \\cdot \\operatorname{silu}(z_{\\text{gate}}) \\odot z_{\\text{up}}",
		],
		note: "Applying the gate here, before the second linear layer, gives the same result as after it, and leaves the combine only additions.",
	},
	fc2: {
		title: "Second linear layer",
		latex: ["y_{t,e} = W_2^{(e)} a"],
	},
};

// ---------------------------------------------------------------- Megatron-LM

function megatron(r: number): ModuleSpec {
	const S = megatronState()[r];
	const all = megatronState();
	const c = (id: number) => COPIES[id];
	const ts = homeTokens(r);
	const sorted = S.sorted.map(c);
	const mine = localExperts(r);
	const splitsNote = (xs: number[]) => `[${xs.join(", ")}]`;
	return {
		name: "",
		type: "module",
		class: "MoELayer",
		symbols: {
			...SYMBOLS,
			num_out_tokens: "token copies this rank sends: tokens · top_k",
			num_recv: `copies this rank receives: sum(output_splits) (${sum(S.outputSplits)} here)`,
		},
		inputs: {
			hidden_states: hiddenOf(
				"hidden_states",
				"(tokens, hidden)",
				"Tokens",
				tokenRows(ts),
				`Rank ${r}'s own tokens. Rank r holds experts 2r and 2r + 1.`,
			),
		},
		operations: {
			router: {
				type: "module",
				class: "TopKRouter",
				inputs: { input: "hidden_states" },
				outputs: {
					probs: mat(
						"probs",
						"(tokens, num_experts)",
						["Tokens", "Expert"],
						eCols,
						tokenRows(ts, probsRow),
						"The renormalized gates of the chosen experts, 0 elsewhere.",
					),
					routing_map: mat(
						"routing_map",
						"(tokens, num_experts)",
						["Tokens", "Expert"],
						eCols,
						ts.map((t) => ({
							label: `t${t}`,
							v: routingMapRow(t).map(String),
						})),
						"1 where the token chose the expert.",
					),
				},
				equation: EQ.router,
			},
			"token_dispatcher:dispatch_preprocess": {
				type: "function",
				inputs: {
					hidden_states: "hidden_states",
					probs: "router.probs",
					routing_map: "router.routing_map",
				},
				operations: {
					count: {
						type: "op",
						op: "Tensor.sum",
						label: "routing_map.sum(dim=0)",
						inputs: { input: "routing_map" },
						outputs: {
							out: vec(
								"num_local_tokens_per_expert",
								"(num_experts)",
								"Expert",
								eCols,
								S.perExpert.map(String),
								"Copies this rank sends to each expert.",
							),
						},
						equation: {
							title: "Count copies per expert",
							latex: ["c_e = \\sum_t \\text{routing\\_map}[t, e]"],
						},
					},
					input_splits: {
						type: "op",
						op: "Tensor.reshape, Tensor.sum",
						label: "input_splits",
						inputs: { input: "count.out" },
						outputs: {
							out: vec(
								"input_splits",
								"(ep_size)",
								"Rank",
								cols(EP),
								S.inputSplits.map(String),
								"Rows this rank will send to each rank: the counts summed per rank's experts.",
							),
						},
						equation: {
							title: "Rows per destination",
							latex: [
								"\\text{input\\_splits}[j] = \\sum_{e \\in \\mathcal{E}_j} c_e",
							],
							note: "𝓔_j is the set of experts on rank j.",
						},
					},
					gather: {
						type: "op",
						op: "gather_from_sequence_parallel_region",
						kind: "collective",
						label: "all-gather counts",
						inputs: { input: "count.out" },
						outputs: {
							out: mat(
								"num_global_tokens_per_expert",
								"(ep_size, num_experts)",
								["Source rank", "Expert"],
								eCols,
								all.map((o, s) => ({
									label: `rank ${s}`,
									v: o.perExpert.map(String),
								})),
								"Every rank's counts. Column 2r and 2r + 1 tell rank r how much it will receive.",
							),
						},
						equation: {
							title: "All-gather over the EP group",
							latex: ["C[s, :] = c^{(s)}"],
						},
					},
					output_splits: {
						type: "op",
						op: "Tensor.__getitem__, Tensor.sum, .cpu()",
						label: "output_splits",
						inputs: { input: "gather.out" },
						outputs: {
							per_local: mat(
								"num_global_tokens_per_local_expert",
								"(ep_size, num_local_experts)",
								["Source rank", "Local expert"],
								mine.map((e) => `E${e}`),
								S.globalPerLocal.map((row, s) => ({
									label: `rank ${s}`,
									v: row.map(String),
								})),
							),
							splits: vec(
								"output_splits",
								"(ep_size)",
								"Rank",
								cols(EP),
								S.outputSplits.map(String),
								"Rows this rank receives from each rank. The split sizes are copied to the CPU for all_to_all: a synchronization point.",
							),
							tokens_per_expert: vec(
								"tokens_per_expert",
								"(num_local_experts)",
								"Local expert",
								mine.map((e) => `E${e}`),
								S.tokensPerExpert.map(String),
							),
						},
						equation: {
							title: "This rank's columns",
							latex: [
								"\\text{output\\_splits}[s] = \\sum_{e \\in \\mathcal{E}_r} C[s, e]",
							],
						},
					},
					permute: {
						type: "op",
						op: "permute",
						label: "permute",
						inputs: {
							tokens: "hidden_states",
							routing_map: "routing_map",
							probs: "probs",
						},
						outputs: {
							tokens: hiddenOf(
								"permutated_local_input_tokens",
								"(num_out_tokens, hidden)",
								"Copies",
								copyIn(S.permuted.map(c)),
								`One row per copy, sorted by expert, so each destination's chunk is contiguous. input_splits = ${splitsNote(S.inputSplits)}.`,
							),
							probs: mat(
								"permuted_probs",
								"(num_out_tokens)",
								["Copies", "gate"],
								["g"],
								S.permuted.map(c).map((cp) => ({
									label: label(cp),
									v: [cp.gate],
									tone: cp.dest,
								})),
								undefined,
								[S.permuted.length],
							),
						},
						equation: {
							title: "Permute",
							latex: [
								"\\text{sorted\\_indices} = \\operatorname{argsort}_{\\text{stable}}(\\text{routing\\_map}^{\\top})",
								"X^{\\text{perm}} = X[\\text{sorted\\_indices}]",
							],
							note: "Copies each token once per chosen expert and lines the copies up by expert.",
						},
					},
				},
				outputs: {
					tokens: "permute.tokens",
					probs: "permute.probs",
					input_splits: "input_splits.out",
					output_splits: "output_splits.splits",
					per_local: "output_splits.per_local",
					tokens_per_expert: "output_splits.tokens_per_expert",
				},
			},
			"token_dispatcher:token_dispatch": {
				type: "function",
				inputs: {
					tokens: "token_dispatcher:dispatch_preprocess.tokens",
					probs: "token_dispatcher:dispatch_preprocess.probs",
					input_splits: "token_dispatcher:dispatch_preprocess.input_splits",
					output_splits: "token_dispatcher:dispatch_preprocess.output_splits",
				},
				operations: {
					a2a_tokens: {
						type: "op",
						op: "all_to_all",
						kind: "collective",
						label: "all-to-all tokens",
						inputs: {
							input: "tokens",
							output_splits: "output_splits",
							input_splits: "input_splits",
						},
						outputs: {
							out: hiddenOf(
								"global_input_tokens",
								"(num_recv, hidden)",
								"Copies",
								copyIn(S.received.map(c)),
								`Grouped by source rank, then expert: output_splits = ${splitsNote(S.outputSplits)}.`,
							),
						},
						equation: {
							title: "All-to-all: dispatch",
							latex: [
								"\\text{rank } i \\text{ sends } X^{\\text{perm}}_i[\\text{chunk}_{i \\to j}] \\text{ to rank } j",
							],
							note: "Uneven chunks: their sizes are the input_splits.",
						},
					},
					a2a_probs: {
						type: "op",
						op: "all_to_all",
						kind: "collective",
						label: "all-to-all probs",
						inputs: {
							input: "probs",
							output_splits: "output_splits",
							input_splits: "input_splits",
						},
						outputs: {
							out: mat(
								"global_probs",
								"(num_recv)",
								["Copies", "gate"],
								["g"],
								S.received.map(c).map((cp) => ({
									label: label(cp),
									v: [cp.gate],
									tone: cp.dest,
								})),
								"The gates travel with their tokens.",
								[S.received.length],
							),
						},
					},
				},
				outputs: { tokens: "a2a_tokens.out", probs: "a2a_probs.out" },
			},
			"token_dispatcher:dispatch_postprocess": {
				type: "function",
				inputs: {
					tokens: "token_dispatcher:token_dispatch.tokens",
					probs: "token_dispatcher:token_dispatch.probs",
					per_local: "token_dispatcher:dispatch_preprocess.per_local",
				},
				operations: {
					sort: {
						type: "op",
						op: "sort_chunks_by_idxs",
						label: "sort chunks by local expert",
						inputs: {
							input: "tokens",
							probs: "probs",
							split_sizes: "per_local",
						},
						outputs: {
							tokens: hiddenOf(
								"dispatched_input",
								"(num_recv, hidden)",
								"Copies",
								copyIn(sorted),
								`Regrouped by local expert: tokens_per_expert = ${splitsNote(S.tokensPerExpert)}.`,
							),
							probs: mat(
								"probs",
								"(num_recv)",
								["Copies", "gate"],
								["g"],
								sorted.map((cp) => ({
									label: label(cp),
									v: [cp.gate],
									tone: cp.dest,
								})),
								undefined,
								[sorted.length],
							),
						},
						equation: {
							title: "sort_chunks_by_idxs",
							latex: [
								"\\text{(source, expert)} \\;\\to\\; \\text{(expert, source)}",
							],
							note: "Each local expert's rows become contiguous for the grouped GEMM.",
						},
					},
				},
				outputs: { tokens: "sort.tokens", probs: "sort.probs" },
			},
			experts: {
				type: "module",
				class: "TEGroupedMLP",
				inputs: {
					tokens: "token_dispatcher:dispatch_postprocess.tokens",
					tokens_per_expert:
						"token_dispatcher:dispatch_preprocess.tokens_per_expert",
					probs: "token_dispatcher:dispatch_postprocess.probs",
				},
				operations: {
					linear_fc1: {
						type: "module",
						class: "TEColumnParallelGroupedLinear",
						inputs: { input: "tokens", m_splits: "tokens_per_expert" },
						outputs: {
							out: mat(
								"fc1_output",
								"(num_recv, 2 · ffn)",
								["Copies", "gate ‖ up"],
								cols(2 * I),
								sorted.map((cp) => ({
									label: label(cp),
									v: fc1(cp),
									tone: cp.dest,
								})),
								"One grouped GEMM: each group of rows uses its own expert's W1.",
							),
						},
						equation: EQ.fc1,
					},
					activation: {
						type: "op",
						op: "weighted_bias_swiglu_impl",
						label: "SwiGLU × probs",
						inputs: { input: "linear_fc1.out", probs: "probs" },
						outputs: {
							out: mat(
								"intermediate",
								"(num_recv, ffn)",
								["Copies", "ffn"],
								cols(I),
								sorted.map((cp) => ({
									label: label(cp),
									v: act(cp, true),
									tone: cp.dest,
								})),
							),
						},
						equation: EQ.swigluGated,
					},
					linear_fc2: {
						type: "module",
						class: "TERowParallelGroupedLinear",
						inputs: { input: "activation.out", m_splits: "tokens_per_expert" },
						outputs: {
							out: hiddenOf(
								"expert_output",
								"(num_recv, hidden)",
								"Copies",
								copyOut(sorted),
							),
						},
						equation: EQ.fc2,
					},
				},
				outputs: { out: "linear_fc2.out" },
				equation: {
					title: `Experts E${mine[0]} and E${mine[1]}`,
					latex: [
						"y_{t,e} = W_2^{(e)} \\big( g_{t,e} \\cdot \\operatorname{silu}(z_{\\text{gate}}) \\odot z_{\\text{up}} \\big)",
					],
				},
			},
			"token_dispatcher:combine_preprocess": {
				type: "function",
				inputs: {
					input: "experts.out",
					per_local: "token_dispatcher:dispatch_preprocess.per_local",
				},
				operations: {
					unsort: {
						type: "op",
						op: "sort_chunks_by_idxs",
						label: "restore chunk order",
						inputs: { input: "input", split_sizes: "per_local" },
						outputs: {
							out: hiddenOf(
								"hidden_states",
								"(num_recv, hidden)",
								"Copies",
								copyOut(S.unsorted.map(c)),
								"Back in (source rank, expert) order, so each source's rows are contiguous.",
							),
						},
						equation: {
							title: "sort_chunks_by_idxs, inverse",
							latex: [
								"\\text{(expert, source)} \\;\\to\\; \\text{(source, expert)}",
							],
						},
					},
				},
				outputs: { out: "unsort.out" },
			},
			"token_dispatcher:token_combine": {
				type: "function",
				inputs: {
					input: "token_dispatcher:combine_preprocess.out",
					input_splits: "token_dispatcher:dispatch_preprocess.input_splits",
					output_splits: "token_dispatcher:dispatch_preprocess.output_splits",
				},
				operations: {
					a2a: {
						type: "op",
						op: "all_to_all",
						kind: "collective",
						label: "all-to-all back",
						inputs: {
							input: "input",
							output_splits: "input_splits",
							input_splits: "output_splits",
						},
						outputs: {
							out: hiddenOf(
								"permutated_local_input_tokens",
								"(num_out_tokens, hidden)",
								"Copies",
								copyOut(S.returned.map(c)),
								"This rank's copies, in the order it sent them, now holding expert outputs.",
							),
						},
						equation: {
							title: "All-to-all: combine",
							latex: [
								"\\text{rank } j \\text{ returns the rows of chunk}_{i \\to j} \\text{ to rank } i",
							],
							note: "The dispatch with the splits swapped.",
						},
					},
				},
				outputs: { out: "a2a.out" },
			},
			"token_dispatcher:combine_postprocess": {
				type: "function",
				inputs: { input: "token_dispatcher:token_combine.out" },
				operations: {
					unpermute: {
						type: "op",
						op: "unpermute",
						label: "unpermute",
						inputs: { input: "input" },
						outputs: {
							out: hiddenOf(
								"output",
								"(tokens, hidden)",
								"Tokens",
								tokenRows(ts, (t) => YS[t]),
								"Each token's two rows added: the same as the MoE layer on one device.",
							),
						},
						equation: {
							title: "Unpermute: add each token's copies",
							latex: ["y_t = \\sum_{k=1}^{2} y_{t, e_k}"],
							note: "A scatter-add into the tokens' positions; the gates were applied inside the experts.",
						},
					},
				},
				outputs: { out: "unpermute.out" },
			},
		},
		outputs: { output: "token_dispatcher:combine_postprocess.out" },
	};
}

// ---------------------------------------------------------------- DeepSpeed

function deepspeed(r: number): ModuleSpec {
	const C = DS_CAPACITY;
	const { cells, dropped } = deepspeedSlots();
	const ts = homeTokens(r);
	const mine = localExperts(r);
	const copyAt = (s: number, e: number, j: number) =>
		cells[s][e][j] >= 0 ? COPIES[cells[s][e][j]] : null;
	const slotLabel = (e: number, j: number) =>
		C === 1 ? `E${e}` : `E${e}·s${j}`;
	const bufRows = (s: number, f: (c: Copy) => number[]) =>
		range(EXPERTS).flatMap((e) =>
			range(C).map((j) => {
				const cp = copyAt(s, e, j);
				return {
					label: `${slotLabel(e, j)}: ${cp ? `t${cp.t}` : "pad"}`,
					v: cp ? f(cp) : new Array<number>(H).fill(0),
					tone: Math.floor(e / LOCAL),
				};
			}),
		);
	const recvRows = (f: (c: Copy) => number[]) =>
		mine.flatMap((e) =>
			range(EP).flatMap((s) =>
				range(C).map((j) => {
					const cp = copyAt(s, e, j);
					return {
						label: `${slotLabel(e, j)} from ${s}: ${cp ? `t${cp.t}` : "pad"}`,
						v: cp ? f(cp) : new Array<number>(H).fill(0),
						tone: r,
					};
				}),
			),
		);
	const myDropped = dropped
		.map((id) => COPIES[id])
		.filter((cp) => cp.home === r);
	const kept = (t: number) =>
		COPIES.filter((cp) => cp.t === t && !dropped.includes(cp.id));
	const y = (t: number) =>
		kept(t)
			.map((cp) => out(cp, true))
			.reduce(
				(a, o) => a.map((v, j) => v + o[j]),
				new Array<number>(H).fill(0),
			);
	const dropNote = myDropped
		.map(
			(cp) =>
				`t${cp.t}'s choice of E${cp.expert} is dropped: the slot is taken.`,
		)
		.join(" ");
	return {
		name: "",
		type: "module",
		class: "MOELayer",
		symbols: {
			...SYMBOLS,
			S: `tokens on this rank (${TOKENS_PER_RANK} here)`,
			C: `capacity: slots per expert per rank, ⌈2S / E · capacity factor⌉ (${C} here)`,
			E: `experts across all ranks (${EXPERTS} here)`,
		},
		inputs: {
			x: hiddenOf(
				"x",
				"(S, hidden)",
				"Tokens",
				tokenRows(ts),
				`Rank ${r}'s own tokens.`,
			),
		},
		operations: {
			gate: {
				type: "module",
				class: "TopKGate",
				inputs: { input: "x" },
				outputs: {
					indices: mat(
						"indices",
						"(S, top_k)",
						["Tokens", "k"],
						cols(TOP_K),
						ts.map((t) => ({ label: `t${t}`, v: ROUTING[t].map(String) })),
						"Each token's two experts, first choice first.",
					),
					locations: mat(
						"locations",
						"(S, top_k)",
						["Tokens", "k"],
						cols(TOP_K),
						ts.map((t) => ({
							label: `t${t}`,
							v: range(TOP_K).map((k) => {
								const cp = COPIES[t * TOP_K + k];
								const j = cells[r][cp.expert].indexOf(cp.id);
								return j >= 0 ? String(j) : `${C}+`;
							}),
						})),
						`The slot each copy gets in its expert's buffer; first choices are numbered before second choices. ${C}+ means past the capacity.`,
					),
					gates: mat(
						"gates",
						"(S, top_k)",
						["Tokens", "k"],
						cols(TOP_K),
						ts.map((t) => ({
							label: `t${t}`,
							v: range(TOP_K).map((k) => COPIES[t * TOP_K + k].gate),
						})),
					),
				},
				equation: {
					title: "Top-2 gate with capacity",
					latex: [
						"C = \\left\\lceil \\frac{2S}{E} \\cdot \\text{cf} \\right\\rceil",
						"\\text{keep } (t,e) \\iff \\text{loc}_{t,e} < C",
					],
					note: "Copies are numbered per expert with a cumulative sum, all first choices before any second choice; numbers past C are dropped.",
				},
			},
			encode: {
				type: "op",
				op: "_sparse_encode",
				label: "scatter into [E, C, h]",
				inputs: {
					x: "x",
					indices: "gate.indices",
					locations: "gate.locations",
				},
				outputs: {
					out: hiddenOf(
						"dispatched_input",
						"(E, C, hidden)",
						"Expert slot",
						bufRows(r, (cp) => XS[cp.t]),
						`One slot per expert. Empty slots are zero padding, and they are sent too. ${dropNote}`,
					),
				},
				equation: {
					title: "_sparse_encode",
					latex: [
						"D[e, s] = \\begin{cases} x_t & \\text{slot } s \\text{ of } e \\text{ holds } t \\\\ 0 & \\text{otherwise} \\end{cases}",
					],
				},
			},
			a2a1: {
				type: "function",
				function: "_AllToAll.apply",
				inputs: { input: "encode.out" },
				operations: {
					a2a: {
						type: "op",
						op: "dist.all_to_all_single",
						kind: "collective",
						label: "all-to-all, equal chunks",
						inputs: { input: "input" },
						outputs: {
							out: hiddenOf(
								"dispatched_input",
								"(ep_size, num_local_experts, C, hidden)",
								"Expert slot, source",
								recvRows((cp) => XS[cp.t]),
								`C rows from every rank for each of this rank's experts, padding included: ${EP * LOCAL * C} rows whatever the routing.`,
							),
						},
						equation: {
							title: "All-to-all, equal chunks",
							latex: [
								"\\text{rank } i \\to j: \\; D_i[jL : (j+1)L, :, :] \\quad (L \\cdot C \\text{ rows})",
							],
							note: "Every chunk has the same size, so no counts are exchanged.",
						},
					},
				},
				outputs: { out: "a2a.out" },
			},
			experts: {
				type: "module",
				class: "Experts",
				inputs: { input: "a2a1.out" },
				outputs: {
					out: hiddenOf(
						"expert_output",
						"(ep_size, num_local_experts, C, hidden)",
						"Expert slot, source",
						recvRows((cp) => out(cp, false)),
						"Not yet scaled by the gates. Padding rows stay zero: SwiGLU(0) = 0.",
					),
				},
				equation: {
					title: `Experts E${mine[0]} and E${mine[1]}`,
					latex: [
						"y_{t,e} = W_2^{(e)} \\big( \\operatorname{silu}(z_{\\text{gate}}) \\odot z_{\\text{up}} \\big), \\quad z = W_1^{(e)} x_t",
					],
					note: "Each local expert runs on its ep_size · C rows, padding included.",
				},
			},
			a2a2: {
				type: "function",
				function: "_AllToAll.apply",
				inputs: { input: "experts.out" },
				operations: {
					a2a: {
						type: "op",
						op: "dist.all_to_all_single",
						kind: "collective",
						label: "all-to-all back",
						inputs: { input: "input" },
						outputs: {
							out: hiddenOf(
								"expert_output",
								"(E, C, hidden)",
								"Expert slot",
								bufRows(r, (cp) => out(cp, false)),
								"This rank's slots, now holding its copies' expert outputs.",
							),
						},
						equation: {
							title: "All-to-all back",
							latex: ["\\text{rank } j \\to i: \\; R'_j[i, :, :, :]"],
							note: "The reverse of the first exchange.",
						},
					},
				},
				outputs: { out: "a2a.out" },
			},
			decode: {
				type: "op",
				op: "_sparse_decode",
				label: "gather, weight, sum",
				inputs: {
					input: "a2a2.out",
					indices: "gate.indices",
					locations: "gate.locations",
					gates: "gate.gates",
				},
				outputs: {
					out: hiddenOf(
						"output",
						"(S, hidden)",
						"Tokens",
						tokenRows(ts, y),
						myDropped.length
							? `t${myDropped[0].t} gets only its kept expert's output, still scaled by that expert's gate.`
							: "Each token's two outputs, gate-weighted and added.",
					),
				},
				equation: {
					title: "_sparse_decode",
					latex: [
						"y_t = \\sum_{(e,s) \\text{ holding } t} g_{t,e}\\, D'[e, s]",
					],
					note: "Dropped copies contribute nothing.",
				},
			},
		},
		outputs: { output: "decode.out" },
	};
}

// ---------------------------------------------------------------- DeepSeek / DeepEP

function deepseek(r: number): ModuleSpec {
	const nodeOf = (g: number) => Math.floor(g / GPUS_PER_NODE);
	const idxOf = (g: number) => g % GPUS_PER_NODE;
	const sends = deepseekIbSends();
	const ts = homeTokens(r);
	const mine = localExperts(r);
	const local = COPIES.filter((cp) => cp.dest === r).sort(
		(a, b) => a.expert - b.expert || a.t - b.t,
	);
	const ibIn = sends.filter((s) => s.to === r);
	const ibOut = sends.filter((s) => s.from === r);
	const gpu = (g: number) => `node ${nodeOf(g)} · GPU ${idxOf(g)}`;
	// Combine: this GPU first sums, over NVLink, its node's results for every token it is
	// responsible for (its own, and those it received over IB); those for other nodes go back over IB.
	const node = nodeOf(r);
	const respTokens = range(TOKENS).filter(
		(t) =>
			idxOf(Math.floor(t / TOKENS_PER_RANK)) === idxOf(r) &&
			COPIES.some((cp) => cp.t === t && nodeOf(cp.dest) === node),
	);
	const nodePartial = (t: number) =>
		COPIES.filter((cp) => cp.t === t && nodeOf(cp.dest) === node)
			.map((cp) => out(cp))
			.reduce(
				(a, o) => a.map((v, j) => v + o[j]),
				new Array<number>(H).fill(0),
			);
	const via = (cp: Copy) => {
		if (cp.home === r) return "own token";
		if (nodeOf(cp.home) === node) return `NVLink from GPU ${idxOf(cp.home)}`;
		const hop = nodeOf(cp.dest) * GPUS_PER_NODE + idxOf(cp.home);
		return hop === r
			? "RDMA, landed here"
			: `RDMA to GPU ${idxOf(hop)}, then NVLink`;
	};
	return {
		name: "",
		type: "module",
		class: "MoE",
		symbols: {
			...SYMBOLS,
			num_recv: `token copies for this GPU's experts (${local.length} here)`,
			num_ib:
				"tokens that reach this GPU over InfiniBand: one row per token and node",
			num_resp:
				"tokens whose results this GPU collects for its node: its own and those it received over InfiniBand",
		},
		inputs: {
			x: hiddenOf(
				"x",
				"(tokens, hidden)",
				"Tokens",
				tokenRows(ts),
				`The tokens of ${gpu(r)}. 2 nodes × 2 GPUs; GPU g holds experts 2g and 2g + 1.`,
			),
		},
		operations: {
			gate: {
				type: "module",
				class: "Gate",
				inputs: { input: "x" },
				outputs: {
					topk_idx: mat(
						"topk_idx",
						"(tokens, top_k)",
						["Tokens", "k"],
						cols(TOP_K),
						ts.map((t) => ({ label: `t${t}`, v: ROUTING[t].map(String) })),
					),
					topk_weights: mat(
						"topk_weights",
						"(tokens, top_k)",
						["Tokens", "k"],
						cols(TOP_K),
						ts.map((t) => ({
							label: `t${t}`,
							v: range(TOP_K).map((k) => COPIES[t * TOP_K + k].gate),
						})),
					),
				},
				equation: {
					title: "Node-limited router",
					latex: [
						"\\text{score}(n) = \\sum \\operatorname{top}_{K/M}\\{ s_{t,e} + b_e : e \\in n \\}",
						"\\mathcal{T}_t = \\operatorname{top}_K \\{ s_{t,e} + b_e : e \\in \\operatorname{top}_M(\\text{nodes}) \\}",
					],
					note: "DeepSeek-V3: K = 8 experts from M = 4 of 8 nodes. Here both nodes are allowed and the routing is the shared example's.",
				},
			},
			dispatch: {
				type: "function",
				function: "buffer.dispatch",
				inputs: {
					x: "x",
					topk_idx: "gate.topk_idx",
					topk_weights: "gate.topk_weights",
				},
				operations: {
					rdma: {
						type: "op",
						op: "RDMA over InfiniBand",
						kind: "collective",
						label: "RDMA: one row per token and node",
						inputs: { x: "x", topk_idx: "topk_idx" },
						outputs: {
							out: hiddenOf(
								"rdma_recv_x",
								"(num_ib, hidden)",
								"Tokens",
								ibIn.map((s) => ({
									label: `t${s.t} from ${gpu(s.from)}`,
									v: XS[s.t],
									tone: null,
								})),
								`Tokens from the other node, sent to the GPU with their home GPU's local index. This GPU sends ${ibOut.map((s) => `t${s.t}`).join(", ")} the other way. ${ibIn
									.filter((s) => s.copies.length > 1)
									.map(
										(s) =>
											`t${s.t} crosses once for ${s.copies.length} experts on this node.`,
									)
									.join(" ")}`,
							),
						},
						equation: {
							title: "Hop 1: InfiniBand",
							latex: [
								"x_t \\;\\to\\; \\text{GPU}(n, \\operatorname{idx}(\\text{home}_t)) \\quad \\forall n \\in \\text{nodes}(\\mathcal{T}_t),\\; n \\ne \\text{node}(\\text{home}_t)",
							],
						},
					},
					nvlink: {
						type: "op",
						op: "NVLink forward",
						kind: "collective",
						label: "NVLink: forward to expert GPUs",
						inputs: { rdma: "rdma.out", x: "x", topk_weights: "topk_weights" },
						outputs: {
							recv_x: hiddenOf(
								"recv_x",
								"(num_recv, hidden)",
								"Copies",
								local.map((cp) => ({
									label: `${label(cp)} (${via(cp)})`,
									v: XS[cp.t],
									tone: cp.dest,
								})),
								"Every copy for this GPU's experts, grouped by local expert for the grouped GEMM.",
							),
							recv_topk_weights: mat(
								"recv_topk_weights",
								"(num_recv)",
								["Copies", "gate"],
								["g"],
								local.map((cp) => ({
									label: label(cp),
									v: [cp.gate],
									tone: cp.dest,
								})),
								undefined,
								[local.length],
							),
						},
						equation: {
							title: "Hop 2: NVLink",
							latex: [
								"x_t \\;\\to\\; \\text{GPU}(e) \\quad \\forall e \\in \\mathcal{T}_t \\cap \\text{this node}",
							],
							note: "The GPU a token landed on forwards it to every GPU of the node that holds one of its experts. DeepEP runs both hops in one kernel.",
						},
					},
				},
				outputs: {
					recv_x: "nvlink.recv_x",
					recv_topk_weights: "nvlink.recv_topk_weights",
				},
			},
			experts: {
				type: "op",
				op: "grouped_gemm, Tensor.mul",
				label: "grouped GEMM × weights",
				inputs: {
					input: "dispatch.recv_x",
					weights: "dispatch.recv_topk_weights",
				},
				outputs: {
					out: hiddenOf(
						"expert_output",
						"(num_recv, hidden)",
						"Copies",
						copyOut(local),
					),
				},
				equation: {
					title: `Experts E${mine[0]} and E${mine[1]}`,
					latex: [
						"y_{t,e} = g_{t,e} \\cdot W_2^{(e)} \\big( \\operatorname{silu}(z_{\\text{gate}}) \\odot z_{\\text{up}} \\big)",
					],
				},
			},
			combine: {
				type: "function",
				function: "buffer.combine",
				inputs: { input: "experts.out" },
				operations: {
					nvlink: {
						type: "op",
						op: "NVLink, with accumulation",
						kind: "collective",
						label: "NVLink: sum this node's results",
						inputs: { input: "input" },
						outputs: {
							out: hiddenOf(
								"node_partial",
								"(num_resp, hidden)",
								"Tokens",
								respTokens.map((t) => ({
									label: `t${t}`,
									v: nodePartial(t),
									tone: null,
								})),
								`For each token this GPU collects, the sum of node ${node}'s expert outputs, from this GPU and the other GPU of the node.`,
							),
						},
						equation: {
							title: "Hop 1 back: NVLink",
							latex: [
								"s_{t,n} = \\sum_{k \\,:\\, \\text{node}(e_k) = n} y_{t, e_k}",
							],
						},
					},
					rdma: {
						type: "op",
						op: "RDMA, with accumulation",
						kind: "collective",
						label: "RDMA: send back, add",
						inputs: { input: "nvlink.out" },
						outputs: {
							out: hiddenOf(
								"combined_x",
								"(tokens, hidden)",
								"Tokens",
								tokenRows(ts, (t) => YS[t]),
								`Partial sums for ${ibIn.map((s) => `t${s.t}`).join(", ")} go back over InfiniBand; the other node's partial sums for this GPU's tokens arrive and are added.`,
							),
						},
						equation: {
							title: "Hop 2 back: InfiniBand",
							latex: ["y_t = \\sum_{n} s_{t,n} \\text{ on } \\text{home}_t"],
						},
					},
				},
				outputs: { out: "rdma.out" },
			},
		},
		outputs: { combined_x: "combine.out" },
	};
}

// ---------------------------------------------------------------- vLLM

const VLLM_BLOCK_M = 4;

function vllm(r: number): ModuleSpec {
	const ts = homeTokens(r);
	const all = range(TOKENS);
	const mine = localExperts(r);
	const isMine = (cp: Copy) => cp.dest === r;
	// moe_align_block_size with ignore_invalid_experts=True: only this rank's copies, grouped by
	// local expert, each group padded to BLOCK_M with the sentinel topk_ids.numel().
	const pad = TOKENS * TOP_K;
	const blocks = mine.flatMap((e, le) => {
		const ids = COPIES.filter((cp) => cp.expert === e).map((cp) => cp.id);
		const n = Math.max(1, Math.ceil(ids.length / VLLM_BLOCK_M));
		return range(n).map((b) => ({
			le,
			ids: range(VLLM_BLOCK_M).map((j) => ids[b * VLLM_BLOCK_M + j] ?? pad),
		}));
	});
	const flatRows = (f: (cp: Copy) => number[] | null, width: number) =>
		COPIES.map((cp) => ({
			label: `t${cp.t}·k${cp.k}→E${cp.expert}`,
			v: f(cp) ?? null,
			tone: cp.dest,
			width,
		}));
	return {
		name: "",
		type: "module",
		class: "FusedMoE",
		symbols: {
			...SYMBOLS,
			num_blocks: `blocks of BLOCK_M = ${VLLM_BLOCK_M} rows, one or more per local expert`,
		},
		inputs: {
			x: hiddenOf(
				"x",
				"(tokens, hidden)",
				"Tokens",
				tokenRows(ts),
				`Rank ${r}'s own requests.`,
			),
			router_logits: {
				name: "router_logits",
				symbolic_shape: "(tokens, num_experts)",
				shape: [TOKENS_PER_RANK, EXPERTS],
				note: "From the model's gate. Values not shown: the routing comes from a seeded run.",
			},
		},
		operations: {
			select_experts: {
				type: "op",
				op: "select_experts",
				label: "select_experts",
				inputs: { x: "x", router_logits: "router_logits" },
				outputs: {
					topk_weights: mat(
						"topk_weights",
						"(tokens, top_k)",
						["Tokens", "k"],
						cols(TOP_K),
						ts.map((t) => ({
							label: `t${t}`,
							v: range(TOP_K).map((k) => COPIES[t * TOP_K + k].gate),
						})),
					),
					topk_ids: mat(
						"topk_ids",
						"(tokens, top_k)",
						["Tokens", "k"],
						cols(TOP_K),
						ts.map((t) => ({ label: `t${t}`, v: ROUTING[t].map(String) })),
					),
				},
				equation: EQ.router,
			},
			all_gather: {
				type: "op",
				op: "all_gatherv",
				kind: "collective",
				label: "all-gather tokens and routing",
				inputs: {
					x: "x",
					topk_weights: "select_experts.topk_weights",
					topk_ids: "select_experts.topk_ids",
				},
				outputs: {
					x: hiddenOf(
						"x",
						"(all_tokens, hidden)",
						"Tokens",
						tokenRows(all),
						"Every rank's tokens, whatever experts they chose.",
					),
					topk_weights: mat(
						"topk_weights",
						"(all_tokens, top_k)",
						["Tokens", "k"],
						cols(TOP_K),
						all.map((t) => ({
							label: `t${t}`,
							v: range(TOP_K).map((k) => COPIES[t * TOP_K + k].gate),
						})),
					),
					topk_ids: mat(
						"topk_ids",
						"(all_tokens, top_k)",
						["Tokens", "k"],
						cols(TOP_K),
						all.map((t) => ({ label: `t${t}`, v: ROUTING[t].map(String) })),
					),
				},
				equation: {
					title: "All-gather over the EP group",
					latex: [
						"X_{\\text{all}} = \\begin{bmatrix} X_0 \\\\ \\vdots \\\\ X_{N-1} \\end{bmatrix}",
					],
				},
			},
			fused_experts: {
				type: "function",
				function: "fused_experts_impl",
				inputs: {
					x: "all_gather.x",
					topk_weights: "all_gather.topk_weights",
					topk_ids: "all_gather.topk_ids",
				},
				operations: {
					align: {
						type: "op",
						op: "moe_align_block_size",
						label: "moe_align_block_size",
						inputs: { topk_ids: "topk_ids" },
						outputs: {
							sorted_token_ids: mat(
								"sorted_token_ids",
								"(num_blocks, BLOCK_M)",
								["Block", "slot"],
								cols(VLLM_BLOCK_M),
								blocks.map((b, i) => ({
									label: `block ${i}`,
									v: b.ids.map(String),
									tone: r,
								})),
								`Flat copy indices t · top_k + k of this rank's copies, by local expert; ${pad} marks padding. With the expert_map and ignore_invalid_experts, other ranks' copies are not even counted.`,
								[blocks.length * VLLM_BLOCK_M],
							),
							expert_ids: vec(
								"expert_ids",
								"(num_blocks)",
								"Block",
								blocks.map((_, i) => `block ${i}`),
								blocks.map((b) => String(b.le)),
								`The local expert of each block: 0 = E${mine[0]}, 1 = E${mine[1]}.`,
							),
						},
						equation: {
							title: "Align copies to blocks",
							latex: [
								"\\text{expert\\_map}[e] = \\begin{cases} e - rE/N & e \\in \\mathcal{E}_r \\\\ -1 & \\text{otherwise} \\end{cases}",
							],
							note: `BLOCK_M = ${VLLM_BLOCK_M} for this example; the tuned configs use 16 or more.`,
						},
					},
					gemm1: {
						type: "op",
						op: "fused_moe_kernel",
						label: "fused_moe_kernel (w1)",
						inputs: {
							x: "x",
							sorted_token_ids: "align.sorted_token_ids",
							expert_ids: "align.expert_ids",
						},
						outputs: {
							out: mat(
								"intermediate_cache1",
								"(all_tokens, top_k, 2 · ffn)",
								["(token, k)", "gate ‖ up"],
								cols(2 * I),
								flatRows((cp) => (isMine(cp) ? fc1(cp) : null), 2 * I),
								"Only rows listed in sorted_token_ids are written; the others (·) are left uninitialized and never read.",
								[TOKENS, TOP_K, 2 * I],
							),
						},
						equation: EQ.fc1,
					},
					act: {
						type: "op",
						op: "silu_and_mul",
						label: "silu_and_mul",
						inputs: { input: "gemm1.out" },
						outputs: {
							out: mat(
								"intermediate_cache2",
								"(all_tokens · top_k, ffn)",
								["(token, k)", "ffn"],
								cols(I),
								flatRows((cp) => (isMine(cp) ? act(cp, false) : null), I),
							),
						},
						equation: EQ.swiglu,
					},
					gemm2: {
						type: "op",
						op: "fused_moe_kernel",
						label: "fused_moe_kernel (w2, × weight)",
						inputs: {
							input: "act.out",
							sorted_token_ids: "align.sorted_token_ids",
							expert_ids: "align.expert_ids",
							topk_weights: "topk_weights",
						},
						outputs: {
							out: mat(
								"intermediate_cache3",
								"(all_tokens, top_k, hidden)",
								["(token, k)", "hidden"],
								cols(H),
								flatRows(
									(cp) => (isMine(cp) ? out(cp) : new Array<number>(H).fill(0)),
									H,
								),
								"Zeroed first because an expert_map is set, so other ranks' copies add nothing.",
								[TOKENS, TOP_K, H],
							),
						},
						equation: {
							title: "Second linear layer, × gate",
							latex: ["y_{t,e} = g_{t,e} \\cdot W_2^{(e)} a"],
							note: "MUL_ROUTED_WEIGHT: the kernel multiplies by topk_weights in its epilogue.",
						},
					},
					moe_sum: {
						type: "op",
						op: "ops.moe_sum",
						label: "moe_sum",
						inputs: { input: "gemm2.out" },
						outputs: {
							out: hiddenOf(
								"out_hidden_states",
								"(all_tokens, hidden)",
								"Tokens",
								tokenRows(all, (t) => partialOutput(r, t)),
								"This rank's share of every token: zero for tokens none of its experts saw.",
							),
						},
						equation: {
							title: "moe_sum",
							latex: ["o^{(r)}_t = \\sum_{k} \\text{cache3}[t, k]"],
						},
					},
				},
				outputs: { out: "moe_sum.out" },
			},
			reduce_scatter: {
				type: "op",
				op: "reduce_scatterv",
				kind: "collective",
				label: "reduce-scatter",
				inputs: { input: "fused_experts.out" },
				outputs: {
					out: hiddenOf(
						"output",
						"(tokens, hidden)",
						"Tokens",
						tokenRows(ts, (t) => YS[t]),
						"The sum of every rank's partial rows, for this rank's tokens only.",
					),
				},
				equation: {
					title: "Reduce-scatter",
					latex: [
						"y_t = \\sum_{r=0}^{N-1} o^{(r)}_t \\quad \\text{for this rank's tokens}",
					],
				},
			},
		},
		outputs: { output: "reduce_scatter.out" },
	};
}

// ---------------------------------------------------------------- SGLang

function sglang(r: number): ModuleSpec {
	const all = range(TOKENS);
	const mine = localExperts(r);
	const localId = (e: number) =>
		Math.floor(e / LOCAL) === r ? e - r * LOCAL : -1;
	return {
		name: "",
		type: "module",
		class: "DeepseekV2MoE",
		symbols: SYMBOLS,
		inputs: {
			hidden_states: hiddenOf(
				"hidden_states",
				"(all_tokens, hidden)",
				"Tokens",
				tokenRows(all),
				"The same batch on every rank: tensor-parallel attention ended with an all-reduce.",
			),
		},
		operations: {
			gate: {
				type: "module",
				class: "MoEGate",
				inputs: { input: "hidden_states" },
				outputs: {
					out: {
						name: "router_logits",
						symbolic_shape: "(all_tokens, num_experts)",
						shape: [TOKENS, EXPERTS],
						note: "Values not shown: the routing comes from a seeded run.",
					},
				},
				equation: {
					title: "Gate",
					latex: ["\\ell_t = W_r\\, x_t"],
					note: "Every rank routes every token, the same work on all ranks.",
				},
			},
			topk: {
				type: "module",
				class: "TopK",
				inputs: { hidden_states: "hidden_states", router_logits: "gate.out" },
				outputs: {
					topk_ids: mat(
						"topk_ids",
						"(all_tokens, top_k)",
						["Tokens", "k"],
						cols(TOP_K),
						all.map((t) => ({ label: `t${t}`, v: ROUTING[t].map(String) })),
					),
					topk_weights: mat(
						"topk_weights",
						"(all_tokens, top_k)",
						["Tokens", "k"],
						cols(TOP_K),
						all.map((t) => ({
							label: `t${t}`,
							v: range(TOP_K).map((k) => COPIES[t * TOP_K + k].gate),
						})),
					),
				},
				equation: EQ.router,
			},
			dispatch: {
				type: "op",
				op: "StandardDispatcher.dispatch",
				label: "local_expert_mapping[topk_ids]",
				inputs: { topk_ids: "topk.topk_ids" },
				outputs: {
					out: mat(
						"topk_ids",
						"(all_tokens, top_k)",
						["Tokens", "k"],
						cols(TOP_K),
						all.map((t) => ({
							label: `t${t}`,
							v: ROUTING[t].map((e) => String(localId(e))),
						})),
						`Local ids on rank ${r}: 0 = E${mine[0]}, 1 = E${mine[1]}, −1 for experts on other ranks, which the kernel skips. No communication.`,
					),
				},
				equation: {
					title: "Rename experts",
					latex: [
						"\\text{local}[e] = \\begin{cases} e - rE/N & e \\in \\mathcal{E}_r \\\\ -1 & \\text{otherwise} \\end{cases}",
					],
				},
			},
			experts: {
				type: "module",
				class: "FusedMoE",
				inputs: {
					hidden_states: "hidden_states",
					topk_ids: "dispatch.out",
					topk_weights: "topk.topk_weights",
				},
				outputs: {
					out: hiddenOf(
						"out",
						"(all_tokens, hidden)",
						"Tokens",
						tokenRows(all, (t) => partialOutput(r, t)),
						"Each token's gate-weighted rows summed over this rank's experts; zero for tokens none of them saw.",
					),
				},
				equation: {
					title: `Fused MoE: experts E${mine[0]}, E${mine[1]}`,
					latex: [
						"y_{t,e} = g_{t,e} \\cdot W_2^{(e)} \\big( \\operatorname{silu}(z_{\\text{gate}}) \\odot z_{\\text{up}} \\big)",
						"o^{(r)}_t = \\sum_{k \\,:\\, e_k \\in \\mathcal{E}_r} y_{t, e_k}",
					],
					note: "The Triton runner computes both GEMMs and sums each token's rows.",
				},
			},
			all_reduce: {
				type: "op",
				op: "tensor_model_parallel_all_reduce",
				kind: "collective",
				label: "all-reduce",
				inputs: { input: "experts.out" },
				outputs: {
					out: hiddenOf(
						"final_hidden_states",
						"(all_tokens, hidden)",
						"Tokens",
						tokenRows(all, (t) => YS[t]),
						"Identical on every rank.",
					),
				},
				equation: {
					title: "All-reduce",
					latex: ["Y = \\sum_{r=0}^{N-1} O^{(r)}"],
					note: "The same all-reduce that ends a tensor-parallel MLP.",
				},
			},
		},
		outputs: { final_hidden_states: "all_reduce.out" },
	};
}

/** The rank each figure shows. */
export const EP_MODULE_RANK: Record<EpModuleFlow, number> = {
	megatron: 1,
	deepspeed: 0,
	deepseek: 1,
	vllm: 1,
	sglang: 1,
};

export function epModuleSpec(
	flow: EpModuleFlow,
	rank = EP_MODULE_RANK[flow],
): ModuleSpec {
	if (flow === "deepspeed") return deepspeed(rank);
	if (flow === "deepseek") return deepseek(rank);
	if (flow === "vllm") return vllm(rank);
	if (flow === "sglang") return sglang(rank);
	return megatron(rank);
}
