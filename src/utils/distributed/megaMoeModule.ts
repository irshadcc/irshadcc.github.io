// The Mega MoE figure in the expert-parallelism post: one launch of DeepGEMM's fused MoE kernel
// (deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh) as a ModuleGraph spec, seen from
// rank 0 of megaKernel.ts's example (2 ranks, 4 experts, top-2, 2 tokens per rank, H = 4, I = 2,
// BLOCK_M = 4). Rank 0 is both a source rank (it owns tokens t0, t1) and an expert rank (it holds
// experts 0 and 1). Every value comes from megaKernel.ts's simulate(), in FP32 like
// MegaKernelSteps, so the two figures agree.
import type { ModuleSpec, TensorValue } from "./graph/moduleGraph";
import {
	EXPERTS,
	H,
	I,
	LOCAL,
	RANKS,
	TOKENS,
	TOPK,
	TOPK_IDX,
	TOPK_W,
	W1,
	W2,
	X,
	simulate,
} from "./megaKernel";

const R = 0;
const sim = simulate();
const range = (n: number) => [...Array(n).keys()];
const f2 = (v: number) => (Math.abs(v) < 0.005 ? "0.00" : v.toFixed(2));
const row = (v: number[]) => v.map(f2);
const cols = (n: number) => range(n).map(String);
const tok = (rank: number, token: number) => `t${rank * TOKENS + token}`;
const myTokens = range(TOKENS).map((t) => tok(R, t));

const rows = sim.l1Rows[R];
const rowLabels = rows.map((c) =>
	c ? `${tok(c.rank, c.token)}·k${c.k}→E${c.expert}` : "pad",
);
const rowTones = rows.map((c) => (c ? c.expert % LOCAL : null));
const padRow = (n: number) => new Array<string>(n).fill("·");
const ringTensor = (
	name: string,
	values: (number[] | null)[],
	width: number,
	symbolic_shape: string,
	axes: [string, string],
	note?: string,
): TensorValue => ({
	name,
	symbolic_shape,
	shape: [rows.length, width],
	axes,
	row_labels: rowLabels,
	col_labels: cols(width),
	values: values.map((v) => (v ? row(v) : padRow(width))),
	row_tones: rowTones,
	note,
});
const valid = rows.filter(Boolean).length;
const ringNote = `${valid} token copies and ${rows.length - valid} padding rows (·), in blocks of BLOCK_M = 4 rows per expert.`;

const localExperts = range(LOCAL).map((le) => R * LOCAL + le);
const w1: TensorValue = {
	name: "l1_weights",
	symbolic_shape: "(num_local_experts, 2 · intermediate, hidden)",
	shape: [LOCAL, 2 * I, H],
	axes: ["(expert, output)", "hidden"],
	row_labels: localExperts.flatMap((e) => [
		...range(I).map((i) => `E${e} gate${i}`),
		...range(I).map((i) => `E${e} up${i}`),
	]),
	col_labels: cols(H),
	values: localExperts.flatMap((e) => W1[e].map(row)),
	row_tones: localExperts.flatMap((_, le) => new Array(2 * I).fill(le)),
	note: "Shown with gate rows before up rows; transform_weights_for_mega_moe interleaves them in groups of 8.",
};
const w2: TensorValue = {
	name: "l2_weights",
	symbolic_shape: "(num_local_experts, hidden, intermediate)",
	shape: [LOCAL, H, I],
	axes: ["(expert, output)", "intermediate"],
	row_labels: localExperts.flatMap((e) => range(H).map((h) => `E${e} out${h}`)),
	col_labels: cols(I),
	values: localExperts.flatMap((e) => W2[e].map(row)),
	row_tones: localExperts.flatMap((_, le) => new Array(H).fill(le)),
};

// Rank 0's view of the counts: how many copies each source rank sends to each of its experts.
const recv = sim.recvCount[R];

export const megaMoeSpec: ModuleSpec = {
	name: "",
	type: "function",
	function: "deep_gemm.fp8_fp4_mega_moe",
	symbols: {
		tokens: `tokens on this rank (${TOKENS} here)`,
		hidden: `model width h (${H} here)`,
		top_k: `experts per token (${TOPK} here)`,
		num_experts: `experts across all ranks (${EXPERTS} here)`,
		num_ranks: `ranks in the NVLink domain (${RANKS} here)`,
		num_local_experts: `experts on this rank (${LOCAL} here)`,
		intermediate: `inner width of an expert (${I} here)`,
		ring_rows:
			"rows of the L1 and L2 rings: each local expert's copies, padded to a multiple of BLOCK_M (4)",
	},
	inputs: {
		x: {
			name: "buffer.x",
			symbolic_shape: "(tokens, hidden)",
			shape: [TOKENS, H],
			axes: ["Tokens", "hidden"],
			row_labels: myTokens,
			col_labels: cols(H),
			values: X[R].map(row),
			note: "In the symmetric buffer, so peers can pull these rows over NVLink. Rank 1 holds t2, t3.",
		},
		topk_idx: {
			name: "buffer.topk_idx",
			symbolic_shape: "(tokens, top_k)",
			shape: [TOKENS, TOPK],
			axes: ["Tokens", "k"],
			row_labels: myTokens,
			col_labels: cols(TOPK),
			values: TOPK_IDX[R].map((r) => r.map(String)),
			note: "Experts 0 and 1 live on rank 0, experts 2 and 3 on rank 1.",
		},
		topk_weights: {
			name: "buffer.topk_weights",
			symbolic_shape: "(tokens, top_k)",
			shape: [TOKENS, TOPK],
			axes: ["Tokens", "k"],
			row_labels: myTokens,
			col_labels: cols(TOPK),
			values: TOPK_W[R].map(row),
		},
		l1: w1,
		l2: w2,
	},
	operations: {
		dispatch: {
			type: "function",
			function: "dispatch warps",
			inputs: { topk_idx: "topk_idx", x: "x", topk_weights: "topk_weights" },
			operations: {
				count: {
					type: "op",
					op: "atomicAdd_block, ptx::atomic_add",
					label: "count copies per expert",
					inputs: { topk_idx: "topk_idx" },
					outputs: {
						out: {
							name: "expert_send_count",
							symbolic_shape: "(num_experts)",
							shape: [EXPERTS],
							axes: ["", "Expert"],
							row_labels: ["rank 0"],
							col_labels: range(EXPERTS).map((e) => `E${e}`),
							values: [sim.counts[R].map(String)],
							note: "Copies rank 0 sends to each expert. Each SM reserves a range of slots with one global atomic per expert.",
						},
					},
					equation: {
						title: "Count and reserve slots",
						latex: [
							"c_e = \\left|\\{(t, k) : \\text{topk\\_idx}[t, k] = e\\}\\right|",
						],
						note: "Counted in shared memory first, then one global atomic per expert per SM; the value returned is the SM's first slot.",
					},
				},
				publish: {
					type: "op",
					op: "stores into peers, NVLink barrier",
					kind: "collective",
					label: "publish counts",
					inputs: { counts: "count.out" },
					outputs: {
						out: {
							name: "recv_count",
							symbolic_shape: "(num_ranks, num_local_experts)",
							shape: [RANKS, LOCAL],
							axes: ["Source rank", "Local expert"],
							row_labels: range(RANKS).map((s) => `rank ${s}`),
							col_labels: localExperts.map((e) => `E${e}`),
							values: recv.map((r) => r.map(String)),
							note: "Row s: copies rank s sends to each of rank 0's experts. They stay on the GPU; no copy to the CPU.",
						},
					},
					equation: {
						title: "Publish the totals",
						latex: [
							"\\text{recv}[s, j] = c^{(s)}_{\\,r E_{\\text{local}} + j}",
						],
						note: "Each SM also stores each copy's token_topk_idx into the destination rank's slot list. SM 0 writes the totals to every rank, and an NVLink barrier waits for all of them.",
					},
				},
				pull: {
					type: "op",
					op: "ptx::tma_load_1d over NVLink",
					kind: "collective",
					label: "pull token rows",
					inputs: {
						recv_count: "publish.out",
						x: "x",
						topk_weights: "topk_weights",
					},
					outputs: {
						tokens: ringTensor(
							"l1_token_buffer",
							rows.map((c) => (c ? X[c.rank][c.token] : null)),
							H,
							"(ring_rows, hidden)",
							["Token copies", "hidden"],
							`The L1 ring. ${ringNote} Rows from rank 1 are read from its buffer.x.`,
						),
						weights: ringTensor(
							"l1_topk_weights_buffer",
							rows.map((c) => (c ? [TOPK_W[c.rank][c.token][c.k]] : null)),
							1,
							"(ring_rows, 1)",
							["Token copies", ""],
							"Each copy's gate weight, loaded from the source rank before its row.",
						),
					},
					equation: {
						title: "Pull each copy's row",
						latex: [
							"\\text{L1}[i] = x^{(s_i)}[t_i], \\qquad w[i] = \\text{topk\\_weights}^{(s_i)}[t_i, k_i]",
						],
						note: "Slots within an expert go to source ranks round-robin. Each row moves remote HBM → shared memory → L1 ring, and l1_full_count signals the GEMM when a block is complete.",
					},
				},
			},
			outputs: { tokens: "pull.tokens", weights: "pull.weights" },
		},
		linear1: {
			type: "function",
			function: "GEMM 1 (gate ‖ up)",
			inputs: { a: "dispatch.tokens", b: "l1" },
			operations: {
				tma_a: {
					type: "op",
					op: "TMA load, multicast to the 2-SM cluster",
					label: "load token tile",
					inputs: { src: "a" },
					outputs: {
						out: ringTensor(
							"smem A",
							rows.map((c) => (c ? X[c.rank][c.token] : null)),
							H,
							"(ring_rows, hidden)",
							["Token copies", "hidden"],
							"Staged in shared memory, BLOCK_K columns per pipeline stage (one stage here).",
						),
					},
				},
				tma_b: {
					type: "op",
					op: "TMA load",
					label: "load weight tile",
					inputs: { src: "b" },
					outputs: {
						out: {
							...w1,
							name: "smem B",
							note: "Streamed from local HBM, BLOCK_N × BLOCK_K per stage.",
						},
					},
				},
				umma: {
					type: "op",
					op: "tcgen05 2-SM UMMA",
					label: "MMA into tensor memory",
					inputs: { a: "tma_a.out", b: "tma_b.out" },
					outputs: {
						out: ringTensor(
							"tmem acc",
							sim.acc1[R],
							2 * I,
							"(ring_rows, 2 · intermediate)",
							["Token copies", "gate ‖ up"],
							"FP32 accumulators in tensor memory. Each block uses its own expert's W1.",
						),
					},
					equation: {
						title: "First linear layer",
						latex: [
							"z_i = W_1^{(e_i)}\\, \\text{L1}[i] = [\\,g_i \\;\\|\\; u_i\\,]",
						],
					},
				},
			},
			outputs: { acc: "umma.out" },
		},
		epilogue1: {
			type: "op",
			op: "tcgen05.ld, SwiGLU, TMA store",
			label: "SwiGLU × gate weight",
			inputs: { acc: "linear1.acc", weights: "dispatch.weights" },
			outputs: {
				out: ringTensor(
					"l2_token_buffer",
					sim.act[R],
					I,
					"(ring_rows, intermediate)",
					["Token copies", "intermediate"],
					"The L2 ring (BF16 in the kernel). Each finished N block sets a bit in l2_full_mask, so GEMM 2 can start on it.",
				),
			},
			equation: {
				title: "SwiGLU and the gate",
				latex: ["a_i = w_i \\cdot \\operatorname{silu}(g_i) \\odot u_i"],
				note: "The weight is applied here, so the combine only has to add.",
			},
		},
		linear2: {
			type: "function",
			function: "GEMM 2 (down)",
			inputs: { a: "epilogue1.out", b: "l2" },
			operations: {
				tma_a: {
					type: "op",
					op: "TMA load",
					label: "load activation tile",
					inputs: { src: "a" },
					outputs: {
						out: ringTensor(
							"smem A",
							sim.act[R],
							I,
							"(ring_rows, intermediate)",
							["Token copies", "intermediate"],
						),
					},
				},
				tma_b: {
					type: "op",
					op: "TMA load",
					label: "load weight tile",
					inputs: { src: "b" },
					outputs: { out: { ...w2, name: "smem B" } },
				},
				umma: {
					type: "op",
					op: "tcgen05 2-SM UMMA",
					label: "MMA into tensor memory",
					inputs: { a: "tma_a.out", b: "tma_b.out" },
					outputs: {
						out: ringTensor("tmem acc", sim.acc2[R], H, "(ring_rows, hidden)", [
							"Token copies",
							"hidden",
						]),
					},
					equation: {
						title: "Second linear layer",
						latex: ["o_i = W_2^{(e_i)}\\, a_i"],
					},
				},
			},
			outputs: { acc: "umma.out" },
		},
		epilogue2: {
			type: "op",
			op: "stores into peers over NVLink",
			kind: "collective",
			label: "write rows home",
			inputs: { acc: "linear2.acc" },
			outputs: {
				out: {
					name: "combine_token_buffer",
					symbolic_shape: "(top_k, tokens, hidden)",
					shape: [TOPK, TOKENS, H],
					axes: ["(k, token)", "hidden"],
					row_labels: range(TOPK).flatMap((k) =>
						myTokens.map(
							(t) => `k${k} ${t}→E${TOPK_IDX[R][myTokens.indexOf(t)][k]}`,
						),
					),
					col_labels: cols(H),
					values: sim.combine[R].flatMap((slot) => slot.map(row)),
					row_tones: range(TOPK).flatMap((k) =>
						range(TOKENS).map((t) => Math.floor(TOPK_IDX[R][t][k] / LOCAL)),
					),
					note: "Rank 0's combine buffer once every rank has signalled combine_ready_grid_idx. Rows for E0, E1 come from rank 0's own epilogue, rows for E2, E3 from rank 1's.",
				},
			},
			equation: {
				title: "Write each row to its source rank",
				latex: ["\\text{combine}^{(s_i)}[k_i, t_i] = o_i"],
				note: "Uses the metadata the pull stored: source rank, token and top-k slot. Padding rows are skipped. This replaces the combine all-to-all.",
			},
		},
		reduce: {
			type: "op",
			op: "TMA loads, FP32 sum, TMA store",
			label: "sum top-k",
			inputs: { combine: "epilogue2.out" },
			outputs: {
				out: {
					name: "y",
					symbolic_shape: "(tokens, hidden)",
					shape: [TOKENS, H],
					axes: ["Tokens", "hidden"],
					row_labels: myTokens,
					col_labels: cols(H),
					values: sim.y[R].map(row),
					note: "Matches a plain PyTorch MoE on the same tokens and weights.",
				},
			},
			equation: {
				title: "Combine",
				latex: ["y_t = \\sum_{k} \\text{combine}[k, t]"],
				note: "Each epilogue warp waits until every rank holding one of the token's experts has signalled, then adds the top-k rows from local HBM, two loads in flight.",
			},
		},
	},
	outputs: { y: "reduce.out" },
};
