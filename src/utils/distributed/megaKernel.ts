// A tiny worked example of DeepGEMM's Mega MoE kernel (sm100_bf16_mega_moe.cuh), for
// MegaKernelSteps: 2 ranks, 4 experts (2 per rank), top-2, 2 tokens per rank, hidden size 4,
// expert intermediate size 2, token blocks of 4 rows. Every step is one line of the kernel,
// with the tensors it reads and the tensors it writes, computed here in FP32 (the kernel rounds
// to BF16 between stages; that is left out so the numbers stay readable). The final output is
// checked against a plain PyTorch MoE.

export const RANKS = 2;
export const EXPERTS = 4;
export const LOCAL = EXPERTS / RANKS;
export const TOPK = 2;
export const TOKENS = 2; // per rank
export const H = 4;
export const I = 2;
export const BLOCK_M = 4;

/** Each rank's tokens, by local token index. Global token t lives on rank t / 2. */
export const X: number[][][] = [
	[
		[1, 0, 2, -1],
		[0, 1, -1, 2],
	],
	[
		[2, 1, 0, 1],
		[-1, 2, 1, 0],
	],
];
/** Router output per rank: [local token][k] expert ids and gate weights. */
export const TOPK_IDX: number[][][] = [
	[
		[0, 3],
		[1, 0],
	],
	[
		[0, 2],
		[3, 1],
	],
];
export const TOPK_W: number[][][] = [
	[
		[0.6, 0.4],
		[0.7, 0.3],
	],
	[
		[0.5, 0.5],
		[0.8, 0.2],
	],
];

const range = (n: number) => [...Array(n).keys()];
/** W1 of expert e: [2I, H], rows 0..I-1 are the gate projection, rows I..2I-1 the up projection. */
export const W1: number[][][] = range(EXPERTS).map((e) =>
	range(2 * I).map((r) => range(H).map((c) => (((2 * e + 3 * r + 3 * c + 1) % 5) - 2) * 0.5)),
);
/** W2 of expert e: [H, I]. */
export const W2: number[][][] = range(EXPERTS).map((e) =>
	range(H).map((r) => range(I).map((c) => (((e + 3 * r + 2 * c + 2) % 5) - 2) * 0.5)),
);

const silu = (v: number) => v / (1 + Math.exp(-v));
const matvec = (m: number[][], v: number[]) => m.map((row) => row.reduce((a, w, j) => a + w * v[j], 0));

/** One token copy as the kernel sees it. */
export interface CopyRef {
	rank: number; // source rank
	token: number; // local token index on the source rank
	k: number;
	expert: number;
	/** token_topk_idx = token * TOPK + k, the value written into the destination's buffer. */
	tokenTopk: number;
}

const copiesOf = (rank: number): CopyRef[] =>
	range(TOKENS).flatMap((token) =>
		range(TOPK).map((k) => ({ rank, token, k, expert: TOPK_IDX[rank][token][k], tokenTopk: token * TOPK + k })),
	);

/** Reference MoE output: y[rank][token] = sum_k w * W2 (silu(gate) * up). */
export function reference(): number[][][] {
	return range(RANKS).map((r) =>
		range(TOKENS).map((t) => {
			const y = new Array<number>(H).fill(0);
			for (let k = 0; k < TOPK; k++) {
				const e = TOPK_IDX[r][t][k];
				const z = matvec(W1[e], X[r][t]);
				const act = range(I).map((i) => silu(z[i]) * z[I + i] * TOPK_W[r][t][k]);
				const out = matvec(W2[e], act);
				for (let j = 0; j < H; j++) y[j] += out[j];
			}
			return y;
		}),
	);
}

export interface Simulation {
	/** Per source rank: tokens sent to each of the 4 experts. */
	counts: number[][];
	/** Per destination rank: [local expert][source rank] -> list of token_topk indices, in slot order. */
	srcIdx: number[][][][];
	/** Per destination rank: [source rank][local expert] counts. */
	recvCount: number[][][];
	/** Per destination rank: L1 ring rows (null = padding), grouped by local expert in BLOCK_M blocks. */
	l1Rows: (CopyRef | null)[][];
	/** Per destination rank, per row: GEMM 1 accumulators [2I], SwiGLU × weight [I], GEMM 2 output [H]. */
	acc1: (number[] | null)[][];
	act: (number[] | null)[][];
	acc2: (number[] | null)[][];
	/** Per source rank: combine buffer [k][token][H]. */
	combine: number[][][][];
	/** Per source rank: output [token][H]. */
	y: number[][][];
}

export function simulate(): Simulation {
	const counts = range(RANKS).map((r) => range(EXPERTS).map((e) => copiesOf(r).filter((c) => c.expert === e).length));
	// Slots are claimed in token-topk order (the second read_topk_idx pass).
	const srcIdx = range(RANKS).map((d) =>
		range(LOCAL).map((le) =>
			range(RANKS).map((s) =>
				copiesOf(s)
					.filter((c) => c.expert === d * LOCAL + le)
					.map((c) => c.tokenTopk),
			),
		),
	);
	const recvCount = range(RANKS).map((d) => range(RANKS).map((s) => range(LOCAL).map((le) => srcIdx[d][le][s].length)));
	// Pull order within an expert: round-robin over source ranks ("min-peeling").
	const l1Rows = range(RANKS).map((d) =>
		range(LOCAL).flatMap((le) => {
			const queues = range(RANKS).map((s) => [...srcIdx[d][le][s]]);
			const rows: (CopyRef | null)[] = [];
			while (queues.some((q) => q.length > 0))
				for (let s = 0; s < RANKS; s++) {
					const tt = queues[s].shift();
					if (tt === undefined) continue;
					const token = Math.floor(tt / TOPK);
					const k = tt % TOPK;
					rows.push({ rank: s, token, k, expert: d * LOCAL + le, tokenTopk: tt });
				}
			while (rows.length % BLOCK_M !== 0) rows.push(null);
			return rows;
		}),
	);
	const acc1 = l1Rows.map((rows) => rows.map((c) => (c ? matvec(W1[c.expert], X[c.rank][c.token]) : null)));
	const act = l1Rows.map((rows, d) =>
		rows.map((c, i) => {
			const z = acc1[d][i];
			if (!c || !z) return null;
			return range(I).map((j) => silu(z[j]) * z[I + j] * TOPK_W[c.rank][c.token][c.k]);
		}),
	);
	const acc2 = l1Rows.map((rows, d) => rows.map((c, i) => (c && act[d][i] ? matvec(W2[c.expert], act[d][i] as number[]) : null)));
	const combine = range(RANKS).map(() => range(TOPK).map(() => range(TOKENS).map(() => new Array<number>(H).fill(0))));
	l1Rows.forEach((rows, d) => {
		rows.forEach((c, i) => {
			if (c) combine[c.rank][c.k][c.token] = acc2[d][i] as number[];
		});
	});
	const y = combine.map((cb) => range(TOKENS).map((t) => range(H).map((j) => cb.reduce((a, slot) => a + slot[t][j], 0))));
	return { counts, srcIdx, recvCount, l1Rows, acc1, act, acc2, combine, y };
}

// ---------------------------------------------------------------- figure steps

/** A tensor drawn in the figure: a small table of values. */
export interface TensorCard {
	name: string;
	shape: string;
	/** Where it lives, e.g. "rank 0 · symmetric buffer". */
	where: string;
	cols?: string[];
	rows: { label: string; cells: string[]; tone?: number | null; dim?: boolean }[];
}

export interface MkStep {
	line: number;
	head: string;
	body: string;
	op: string;
	inputs: TensorCard[];
	outputs: TensorCard[];
}

const f2 = (v: number) => (Math.abs(v) < 0.005 ? "0.00" : v.toFixed(2));
const vec = (v: number[]) => v.map(f2);
const f3 = (v: number) => (v < 0 ? `(${v.toFixed(3)})` : v.toFixed(3));
/** A number for inline arithmetic: negatives in parentheses. */
const sgn = (v: number) => (v < 0 ? `(${Number.isInteger(v) ? v : f2(v)})` : Number.isInteger(v) ? String(v) : f2(v));
const tokName = (rank: number, token: number) => `t${rank * TOKENS + token}`;
const copyName = (c: CopyRef) => `${tokName(c.rank, c.token)}·k${c.k}`;
/** Tone = the rank that holds the row's expert, as in the other MoE figures. */
const toneOf = (e: number) => Math.floor(e / LOCAL);

export const CODE: string[] = [
	"// dispatch warps",
	"read_topk_idx(...) { atomicAdd_block(&smem.expert_token_count[expert_idx], 1); }",
	"slot = atomic_add(&expert_send_count[e], (1ull << 32) | smem.expert_token_count[e]);",
	"*sym_buffer.map(src_token_topk_idx_ptr(e % kNumExpertsPerRank, my_rank, slot), e / kNumExpertsPerRank) = token_topk_idx;",
	"*sym_buffer.map(expert_recv_count_ptr(my_rank, e % kNumExpertsPerRank), e / kNumExpertsPerRank) = count;",
	"tma_load_1d(pull_buf, sym_buffer.map(x[src_token], src_rank)); tma_store_1d(l1_ring[pool_token_idx], pull_buf);",
	"// GEMM warps",
	"SM100_MMA_F16BF16_2x1SM_SS::fma(b_desc /* W1 tile */, a_desc /* token tile */, tmem_acc, ...);",
	"l2_ring[row] = silu(acc[gate]) * acc[up] * l1_topk_weights[row];   // epilogue 1",
	"SM100_MMA_F16BF16_2x1SM_SS::fma(b_desc /* W2 tile */, a_desc /* L2 tile */, tmem_acc, ...);",
	"*sym_buffer.map(&combine[topk_idx][token_idx], src_rank) = tmem_acc[row];   // epilogue 2",
	"// combine (epilogue warps, after every peer signals combine_ready)",
	"y[token] = sum over k of combine[k][token];   // FP32 accumulate, BF16 store",
];

export function buildSteps(): MkStep[] {
	const S = simulate();
	const r0 = 0;
	const e0 = 0; // the expert block we follow through the GEMMs
	const blockRows = S.l1Rows[r0].slice(0, BLOCK_M);
	const xCard = (r: number): TensorCard => ({
		name: "x",
		shape: `[${TOKENS}, ${H}]`,
		where: `rank ${r} · symmetric buffer`,
		cols: range(H).map((j) => `h${j}`),
		rows: range(TOKENS).map((t) => ({ label: tokName(r, t), cells: vec(X[r][t]) })),
	});
	const topkCard = (r: number): TensorCard => ({
		name: "topk_idx, topk_weights",
		shape: `[${TOKENS}, ${TOPK}]`,
		where: `rank ${r} · symmetric buffer`,
		cols: range(TOPK).map((k) => `k${k}`),
		rows: range(TOKENS).map((t) => ({
			label: tokName(r, t),
			cells: range(TOPK).map((k) => `E${TOPK_IDX[r][t][k]} · ${TOPK_W[r][t][k]}`),
		})),
	});
	const countCard = (r: number, name: string, where: string): TensorCard => ({
		name,
		shape: `[${EXPERTS}]`,
		where,
		cols: range(EXPERTS).map((e) => `E${e}`),
		rows: [{ label: `rank ${r}`, cells: S.counts[r].map(String) }],
	});
	const srcIdxCard = (d: number): TensorCard => ({
		name: "src_token_topk_idx",
		shape: `[${LOCAL} experts, ${RANKS} sources, slots]`,
		where: `rank ${d} · symmetric buffer (written by peers)`,
		cols: ["slot 0", "slot 1"],
		rows: range(LOCAL).flatMap((le) =>
			range(RANKS).map((s) => ({
				label: `E${d * LOCAL + le} ← rank ${s}`,
				tone: s === 0 ? null : 1,
				cells: [0, 1].map((slot) => {
					const v = S.srcIdx[d][le][s][slot];
					return v === undefined ? "·" : `${v} (${tokName(s, Math.floor(v / TOPK))}·k${v % TOPK})`;
				}),
			})),
		),
	});
	const recvCard = (d: number): TensorCard => ({
		name: "expert_recv_count",
		shape: `[${RANKS} sources, ${LOCAL} experts]`,
		where: `rank ${d} · symmetric buffer (written by peers)`,
		cols: range(LOCAL).map((le) => `E${d * LOCAL + le}`),
		rows: range(RANKS).map((s) => ({ label: `from rank ${s}`, cells: S.recvCount[d][s].map(String) })),
	});
	const l1Card = (d: number, rows = S.l1Rows[d], name = "l1_token_buffer (ring)"): TensorCard => ({
		name,
		shape: `[${rows.length}, ${H}]`,
		where: `rank ${d} · symmetric buffer`,
		cols: range(H).map((j) => `h${j}`),
		rows: rows.map((c, i) => ({
			label: c ? `E${c.expert} ← ${copyName(c)}` : `E${S.l1Rows[d][Math.floor(i / BLOCK_M) * BLOCK_M]?.expert ?? ""} pad`,
			tone: c ? c.rank : null,
			dim: !c,
			cells: c ? vec(X[c.rank][c.token]) : range(H).map(() => "0"),
		})),
	});
	const blockLabel = (c: CopyRef | null) => (c ? copyName(c) : "pad");
	const matCard = (name: string, m: number[][], rowNames: string[], colNames: string[], where: string): TensorCard => ({
		name,
		shape: `[${m.length}, ${m[0].length}]`,
		where,
		cols: colNames,
		rows: m.map((row, i) => ({ label: rowNames[i], cells: vec(row) })),
	});
	const rowsCard = (name: string, vals: (number[] | null)[], cols: string[], where: string, rows = blockRows): TensorCard => ({
		name,
		shape: `[${BLOCK_M}, ${cols.length}]`,
		where,
		cols,
		rows: rows.map((c, i) => ({
			label: blockLabel(c),
			tone: c ? c.rank : null,
			dim: !c,
			cells: vals[i] ? vec(vals[i] as number[]) : cols.map(() => "—"),
		})),
	});
	const gateUp = [...range(I).map((i) => `gate${i}`), ...range(I).map((i) => `up${i}`)];
	const weightsCol = (rows: (CopyRef | null)[]) => rows.map((c) => (c ? [TOPK_W[c.rank][c.token][c.k]] : null));
	const combineCard = (r: number): TensorCard => ({
		name: "combine_token_buffer",
		shape: `[${TOPK}, ${TOKENS}, ${H}]`,
		where: `rank ${r} · symmetric buffer (written by expert ranks)`,
		cols: range(H).map((j) => `h${j}`),
		rows: range(TOPK).flatMap((k) =>
			range(TOKENS).map((t) => {
				const e = TOPK_IDX[r][t][k];
				return { label: `k${k} · ${tokName(r, t)} ← E${e}`, tone: toneOf(e), cells: vec(S.combine[r][k][t]) };
			}),
		),
	});
	const ref = reference();
	const maxErr = Math.max(...S.y.flatMap((ry, r) => ry.flatMap((row, t) => row.map((v, j) => Math.abs(v - ref[r][t][j])))));

	const e0Rows = blockRows.filter((c): c is CopyRef => c !== null);
	const l1Block = blockRows.map((c) => (c ? X[c.rank][c.token] : null));
	const acc1Block = S.acc1[r0].slice(0, BLOCK_M);
	const actBlock = S.act[r0].slice(0, BLOCK_M);
	const acc2Block = S.acc2[r0].slice(0, BLOCK_M);
	const rank0Copies = copiesOf(0);

	return [
		{
			line: 1,
			head: "Count tokens per expert",
			body: `Each dispatch warp reads the router's choices, one lane per (token, k), and counts them per expert in shared memory with block-wide atomics. Rank 0's two tokens choose E0 twice and E1 and E3 once: counts ${S.counts[0].join(", ")}.`,
			op: "atomicAdd_block per (token, k)",
			inputs: [topkCard(0)],
			outputs: [countCard(0, "expert_token_count", "rank 0 · shared memory")],
		},
		{
			line: 2,
			head: "Reserve slots with one global atomic per expert",
			body: "Each SM adds (1 << 32) | count to the global counter of every expert and keeps the old value: its low 32 bits are the number of tokens other SMs of this rank already claimed, so they are this SM's first slot. Here one SM does all the work, so every first slot is 0; the counters end at 1·2³² + count.",
			op: "atomic_add, keep the old value",
			inputs: [countCard(0, "expert_token_count", "rank 0 · shared memory")],
			outputs: [
				{
					name: "expert_send_count / first slot",
					shape: `[${EXPERTS}]`,
					where: "rank 0 · symmetric buffer / shared memory",
					cols: range(EXPERTS).map((e) => `E${e}`),
					rows: [
						{ label: "counter after", cells: S.counts[0].map((n) => (n ? `2³²+${n}` : "0")) },
						{ label: "first slot", cells: S.counts[0].map(() => "0") },
					],
				},
			],
		},
		{
			line: 3,
			head: "Tell each expert's rank which copies to fetch",
			body: `Each copy's index, token × ${TOPK} + k, is written straight into the memory of the rank that holds its expert, at the reserved slot. Rank 0 writes ${rank0Copies.map((c) => `${c.tokenTopk} → E${c.expert}`).join(", ")}; rank 1 does the same. The table is what rank 0 holds afterwards: rows written by itself and by rank 1 (tinted).`,
			op: "remote stores through sym_buffer.map",
			inputs: [topkCard(0), topkCard(1)],
			outputs: [srcIdxCard(0)],
		},
		{
			line: 4,
			head: "Publish the per-expert counts",
			body: `After a grid-wide barrier, SM 0 of every rank writes its count for each expert into the expert's rank, and an NVLink barrier waits until all ranks have. Rank 0 now knows it will receive ${S.recvCount[0].map((row, s) => `${row.join(" and ")} copies for E0 and E1 from rank ${s}`).join(", ")}.`,
			op: "remote stores, then NVLink barrier",
			inputs: [countCard(0, "expert_send_count (low 32 bits)", "rank 0"), countCard(1, "expert_send_count (low 32 bits)", "rank 1")],
			outputs: [recvCard(0)],
		},
		{
			line: 5,
			head: "Pull the tokens over NVLink",
			body: `Each dispatch warp takes one slot, reads which (rank, token) it holds, and copies that row from the source rank's x into rank 0's L1 ring, by TMA load (peer HBM → shared memory) and TMA store (shared memory → ring). Within an expert the sources alternate (E0: ${S.l1Rows[0]
				.slice(0, BLOCK_M)
				.filter(Boolean)
				.map((c) => copyName(c as CopyRef))
				.join(", ")}), and each expert's rows fill a block of ${BLOCK_M}, padded. When a block is full, l1_full_count tells the GEMM warps.`,
			op: "TMA pull, round-robin over source ranks",
			inputs: [xCard(0), xCard(1)],
			outputs: [l1Card(0)],
		},
		{
			line: 7,
			head: "GEMM 1 on E0's block",
			body: `The MMA warp multiplies E0's block of tokens by E0's W1, the gate and up projections stacked, accumulating in tensor memory. Check one entry, ${copyName(e0Rows[0])}'s gate0: ${X[e0Rows[0].rank][e0Rows[0].token].map((v, j) => `${sgn(v)} × ${sgn(W1[e0][0][j])}`).join(" + ")} = ${f2(acc1Block[0]?.[0] ?? 0)}. The padding row is never stored.`,
			op: "acc = tokens × W1ᵀ (tensor cores)",
			inputs: [
				rowsCard("token tile (from the L1 ring)", l1Block, range(H).map((j) => `h${j}`), "rank 0 · shared memory"),
				matCard("W1 of E0", W1[e0], gateUp, range(H).map((j) => `h${j}`), "rank 0 · HBM → shared memory"),
			],
			outputs: [rowsCard("accumulator", acc1Block, gateUp, "rank 0 · tensor memory")],
		},
		{
			line: 8,
			head: "Epilogue 1: SwiGLU and the gate weight",
			body: `The epilogue warps read the accumulator, compute silu(gate) × up, multiply by the copy's top-k weight (loaded during the pull) and store the result into the L2 ring. For ${copyName(e0Rows[0])}: silu(${sgn(acc1Block[0]?.[0] ?? 0)}) × ${sgn(acc1Block[0]?.[I] ?? 0)} × ${TOPK_W[e0Rows[0].rank][e0Rows[0].token][e0Rows[0].k]} = ${f2(actBlock[0]?.[0] ?? 0)}.`,
			op: "silu(gate) · up · weight",
			inputs: [
				rowsCard("accumulator", acc1Block, gateUp, "rank 0 · tensor memory"),
				rowsCard("l1_topk_weights", weightsCol(blockRows), ["w"], "rank 0 · symmetric buffer"),
			],
			outputs: [rowsCard("l2_token_buffer (ring)", actBlock, range(I).map((i) => `i${i}`), "rank 0 · symmetric buffer")],
		},
		{
			line: 9,
			head: "GEMM 2 on the same block",
			body: "As soon as the bits for the K blocks it needs are set in l2_full_mask, a second-GEMM task multiplies the block by E0's W2, again into tensor memory.",
			op: "acc = act × W2ᵀ (tensor cores)",
			inputs: [
				rowsCard("L2 tile", actBlock, range(I).map((i) => `i${i}`), "rank 0 · shared memory"),
				matCard("W2 of E0", W2[e0], range(H).map((j) => `h${j}`), range(I).map((i) => `i${i}`), "rank 0 · HBM → shared memory"),
			],
			outputs: [rowsCard("accumulator", acc2Block, range(H).map((j) => `h${j}`), "rank 0 · tensor memory")],
		},
		{
			line: 10,
			head: "Epilogue 2: write each row home",
			body: `Each output row goes straight to the rank its token came from, into that rank's combine buffer at [k][token]. ${e0Rows.map((c) => `${copyName(c)} → rank ${c.rank}, slot k${c.k}`).join("; ")}. Rank 0's E1 block and rank 1's E2 and E3 blocks do the same, so rank 0's buffer fills with rows computed on both ranks.`,
			op: "16-byte remote stores via sym_buffer.map",
			inputs: [rowsCard("accumulator", acc2Block, range(H).map((j) => `h${j}`), "rank 0 · tensor memory")],
			outputs: [combineCard(0)],
		},
		{
			line: 12,
			head: "Combine: add each token's top-k rows",
			body: `When every rank holding one of a token's experts has signalled combine_ready, the epilogue warps load the token's ${TOPK} rows from local memory and add them in FP32. Check one entry: y[t0][h0] = ${f3(S.combine[0][0][0][0])} + ${f3(S.combine[0][1][0][0])} = ${S.y[0][0][0].toFixed(3)}. Against a plain PyTorch-style MoE on the same inputs, the largest difference is ${maxErr.toExponential(1)}.`,
			op: "sum over k, FP32 → BF16",
			inputs: [combineCard(0)],
			outputs: [
				{
					name: "y",
					shape: `[${TOKENS}, ${H}]`,
					where: "rank 0 · output",
					cols: range(H).map((j) => `h${j}`),
					rows: range(TOKENS).map((t) => ({ label: tokName(0, t), cells: vec(S.y[0][t]) })),
				},
			],
		},
	];
}
