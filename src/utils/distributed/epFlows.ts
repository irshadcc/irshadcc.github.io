// The forward pass of one expert-parallel MoE layer, as three frameworks run it on the same toy
// setup (epDispatch.ts: EP = 4, 8 experts, top-2, 8 tokens). Each flow describes what
// MoeEpSteps draws: the graph (one column per rank, plus shared collective nodes), the chips that
// move through it, one step per code line, the code itself and a hover card per node.
//   megatron – Megatron-LM's MoEAlltoAllTokenDispatcher: count, permute, all-to-all, sort, experts,
//              and the same in reverse.
//   vllm     – vLLM's default EP backend (allgather_reducescatter): every rank routes its own
//              tokens, all-gathers everyone's, runs its local experts on the copies it owns and
//              reduce-scatters the partial outputs.
//   sglang   – SGLang's default (--moe-a2a-backend none) with tensor-parallel attention: every rank
//              already holds every token, runs its local experts, and an all-reduce adds the
//              partial outputs.
//   deepspeed – DeepSpeed's MOELayer (GShard style): each rank fills a fixed [experts, capacity, h]
//              buffer, dropping copies past the capacity and padding the rest, so both all-to-alls
//              exchange equal chunks.
//   deepseek – DeepEP's hybrid dispatch as in DeepSeek-V3, on 2 nodes of 2 GPUs: a token crosses
//              InfiniBand once per target node, to the GPU with its own local index, and NVLink
//              forwards it from there; the combine sums partial results on the way back.
import type { NodeKind } from "../neuralnetwork/NNGraph";
import {
	COPIES,
	type Copy,
	EP,
	EXPERTS,
	GATES,
	LOCAL,
	CODE as MEGATRON_CODE,
	ROUTING,
	TOKENS,
	TOKENS_PER_RANK,
	TOP_K,
	TRAY_COLS,
	type TensorRow,
	type TensorView,
	buildSteps,
	node,
	tensorViews,
} from "./epDispatch";

export type FlowName =
	| "megatron"
	| "vllm"
	| "sglang"
	| "deepspeed"
	| "deepseek";

export interface FlowNode {
	id: string;
	kind: NodeKind;
	label: string;
	detail?: string;
	/** Chip slots: columns × rows. */
	tray?: [number, number];
	from?: string[];
}

export interface FlowChip {
	/** Token number shown on the chip. */
	t: number;
	/** Colour: the rank that holds the copy's expert. */
	dest: number;
	/** An empty capacity slot (padding) rather than a token copy. */
	pad?: boolean;
}

/** Where a chip is in one step. */
export interface ChipPlace {
	node: string;
	col: number;
	row: number;
	/** on: a live copy; dim: a copy this rank skips; plain: a whole token (grey); hidden. */
	state: "on" | "dim" | "plain" | "hidden";
	/** Small offset so stacked chips (a token's copies being summed) stay visible. */
	nudge?: number;
}

export interface FlowStep {
	line: number;
	head: string;
	body: string;
	active: string[];
	places: ChipPlace[];
}

export interface Flow {
	nodes: FlowNode[];
	chips: FlowChip[];
	steps: FlowStep[];
	code: string[];
	views: Record<string, TensorView>;
	/** What the code was condensed from. */
	source: string;
	/** Rows of h numbers each rank sends or receives per layer, forward pass only. */
	traffic: number[];
}

const range = (n: number) => [...Array(n).keys()];
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const ranks = range(EP);
const g2 = (v: number) => v.toFixed(2);
const rowsOf = (n: number) => `${n} row${n === 1 ? "" : "s"}`;
const fmt = (xs: number[]) => `[${xs.join(", ")}]`;
const tokSlot = (t: number, cols: number) => ({
	col: t % cols,
	row: Math.floor(t / cols),
});
const localCopies = (r: number) => COPIES.filter((c) => c.dest === r);
const homeTokens = (r: number) =>
	range(TOKENS_PER_RANK).map((i) => r * TOKENS_PER_RANK + i);
/** A rank's copies grouped by local expert, token order within each. */
const byLocalExpert = (r: number) =>
	range(LOCAL).map((le) =>
		localCopies(r)
			.filter((c) => c.expert === r * LOCAL + le)
			.sort((a, b) => a.t - b.t),
	);
/** Copies sent to each rank's experts; the busiest and quietest ranks. */
const load = ranks.map((r) => localCopies(r).length);
const busy = load.indexOf(Math.max(...load));
const quiet = load.indexOf(Math.min(...load));

// ---------------------------------------------------------------- Megatron-LM

function megatronFlow(): Flow {
	const nodes: FlowNode[] = [
		...ranks.map((r) => ({
			id: node.tok(r),
			kind: "input" as const,
			label: `Rank ${r}`,
			detail: "2 tokens",
			tray: [2, 1] as [number, number],
		})),
		...ranks.map((r) => ({
			id: node.router(r),
			kind: "linear" as const,
			label: "Router",
			tray: [TRAY_COLS, 1] as [number, number],
			from: [node.tok(r)],
		})),
		{
			id: node.gather,
			kind: "other",
			label: "Count copies, all-gather counts",
			from: ranks.map(node.router),
		},
		...ranks.map((r) => ({
			id: node.perm(r),
			kind: "op" as const,
			label: "Permute",
			tray: [TRAY_COLS, 1] as [number, number],
			from: [node.gather],
		})),
		{
			id: node.dispatch,
			kind: "other",
			label: "All-to-all: dispatch",
			from: ranks.map(node.perm),
		},
		...ranks.map((r) => ({
			id: node.recv(r),
			kind: "op" as const,
			label: "Received",
			tray: [TRAY_COLS, 2] as [number, number],
			from: [node.dispatch],
		})),
		...ranks.map((r) => ({
			id: node.experts(r),
			kind: "linear" as const,
			label: "Experts",
			detail: `E${r * LOCAL} / E${r * LOCAL + 1}`,
			tray: [TRAY_COLS, LOCAL] as [number, number],
			from: [node.recv(r)],
		})),
		...ranks.map((r) => ({
			id: node.unsort(r),
			kind: "op" as const,
			label: "Unsort",
			tray: [TRAY_COLS, 2] as [number, number],
			from: [node.experts(r)],
		})),
		{
			id: node.combine,
			kind: "other",
			label: "All-to-all: combine",
			from: ranks.map(node.unsort),
		},
		...ranks.map((r) => ({
			id: node.out(r),
			kind: "output" as const,
			label: "Unpermute",
			tray: [TRAY_COLS, 1] as [number, number],
			from: [node.combine],
		})),
	];
	const steps = buildSteps().map((s) => ({
		line: s.line,
		head: s.head,
		body: s.body,
		active: s.active,
		places: s.places.map((p, id) => ({
			...p,
			state: s.phase === "copies" ? ("on" as const) : ("plain" as const),
			nudge: s.phase === "copies" ? 0 : (id % 2) * 1.5,
		})),
	}));
	// Rows each rank sends in the dispatch and in the combine (copies to other ranks only).
	const sent = ranks.map(
		(r) => COPIES.filter((c) => c.home === r && c.dest !== r).length,
	);
	const back = ranks.map(
		(r) => COPIES.filter((c) => c.dest === r && c.home !== r).length,
	);
	return {
		nodes,
		chips: COPIES.map((c) => ({ t: c.t, dest: c.dest })),
		steps,
		code: MEGATRON_CODE,
		views: tensorViews(),
		source:
			"Condensed from Megatron-LM's MoELayer.forward and MoEAlltoAllTokenDispatcher (TP = 1, dropless).",
		traffic: ranks.map((r) => sent[r] + back[r]),
	};
}

// ---------------------------------------------------------------- shared helpers for replicated flows

/** Chip id for the copy `c` as held by rank `r` (these flows keep a replica of copies per rank). */
const rid = (r: number, c: number) => r * COPIES.length + c;
const replicatedChips = (): FlowChip[] =>
	ranks.flatMap(() => COPIES.map((c) => ({ t: c.t, dest: c.dest })));

/** Places for every replica, built from a function of (rank, copy). */
function placeAll(f: (r: number, c: Copy) => ChipPlace): ChipPlace[] {
	const out: ChipPlace[] = [];
	for (const r of ranks) for (const c of COPIES) out[rid(r, c.id)] = f(r, c);
	return out;
}

const gridSlot = (i: number) => ({
	col: i % TRAY_COLS,
	row: Math.floor(i / TRAY_COLS),
});

/** Local copies laid out in the experts node: one row per local expert. */
function expertSlot(r: number, c: Copy) {
	const le = c.expert - r * LOCAL;
	return { col: byLocalExpert(r)[le].indexOf(c), row: le };
}

function expertsView(r: number, code: string, note: string): TensorView {
	const groups = byLocalExpert(r);
	return {
		title: `Rank ${r}'s experts at work`,
		code,
		shape: `[${TOKENS}, ${TOP_K}, h]`,
		shapeWords: `a row per (token, choice); only ${rowsOf(sum(groups.map((g) => g.length)))} are computed here`,
		note,
		groups: groups.map((cs, le) => ({
			label: `expert E${r * LOCAL + le} · ${rowsOf(cs.length)}`,
			rows: cs.map((c) => ({
				t: c.t,
				rank: c.dest,
				note: `${g2(c.gate)} × E${c.expert}(t${c.t})`,
			})),
		})),
	};
}

function partialView(r: number, title: string): TensorView {
	const rows: TensorRow[] = range(TOKENS).map((t) => {
		const mine = localCopies(r).filter((c) => c.t === t);
		return {
			t,
			rank: mine.length ? r : null,
			note: mine.length
				? mine.map((c) => `${g2(c.gate)} × E${c.expert}(t${t})`).join(" + ")
				: "0 (no expert here)",
		};
	});
	return {
		title,
		code: "out",
		shape: `[${TOKENS}, h]`,
		shapeWords: `${TOKENS} rows × h numbers, one per token of the whole group`,
		note: "Each row holds only this rank's experts' share of the token. The other ranks hold the rest; the next collective adds them up.",
		groups: [{ label: "", rows }],
	};
}

function finalRow(t: number): TensorRow {
	return {
		t,
		rank: null,
		note: ROUTING[t]
			.map((e, k) => `${g2(GATES[t][k])} × E${e}(t${t})`)
			.join(" + "),
	};
}

function routerView(
	r: number,
	toks: number[],
	title: string,
	note: string,
): TensorView {
	return {
		title,
		code: "topk_weights, topk_ids",
		shape: `[${toks.length}, ${TOP_K}]`,
		shapeWords: `one row per token, its ${TOP_K} experts and their gate weights`,
		note,
		groups: [
			{
				label: "",
				rows: toks.flatMap((t) =>
					COPIES.filter((c) => c.t === t).map((c) => ({
						t,
						rank: c.dest,
						note: `expert ${c.expert} · gate ${g2(c.gate)}`,
					})),
				),
			},
		],
	};
}

function collectiveView(
	title: string,
	code: string,
	rowLabel: string,
	perRank: number[],
	note: string,
): TensorView {
	return {
		title,
		code,
		shape: `[${EP}]`,
		shapeWords: "one entry per rank",
		note,
		grid: {
			rowAxis: rowLabel,
			colAxis: "rank",
			rowLabels: [rowLabel],
			colLabels: ranks.map((r) => `rank ${r}`),
			colRanks: ranks,
			values: [perRank],
			decimals: 0,
		},
	};
}

// ---------------------------------------------------------------- vLLM

const V = {
	tok: (r: number) => `v-tok${r}`,
	router: (r: number) => `v-router${r}`,
	gather: "v-allgather",
	all: (r: number) => `v-all${r}`,
	experts: (r: number) => `v-exp${r}`,
	partial: (r: number) => `v-partial${r}`,
	rs: "v-reducescatter",
	out: (r: number) => `v-out${r}`,
};

function vllmFlow(): Flow {
	const nodes: FlowNode[] = [
		...ranks.map((r) => ({
			id: V.tok(r),
			kind: "input" as const,
			label: `Rank ${r}`,
			detail: "2 tokens",
			tray: [2, 1] as [number, number],
		})),
		...ranks.map((r) => ({
			id: V.router(r),
			kind: "linear" as const,
			label: "Router",
			tray: [TRAY_COLS, 1] as [number, number],
			from: [V.tok(r)],
		})),
		{
			id: V.gather,
			kind: "other",
			label: "All-gather: tokens + top-k",
			from: ranks.map(V.router),
		},
		...ranks.map((r) => ({
			id: V.all(r),
			kind: "op" as const,
			label: "All 8 tokens",
			detail: "16 copies",
			tray: [TRAY_COLS, 4] as [number, number],
			from: [V.gather],
		})),
		...ranks.map((r) => ({
			id: V.experts(r),
			kind: "linear" as const,
			label: "Fused MoE kernel",
			detail: `E${r * LOCAL} / E${r * LOCAL + 1}`,
			tray: [TRAY_COLS, LOCAL] as [number, number],
			from: [V.all(r)],
		})),
		...ranks.map((r) => ({
			id: V.partial(r),
			kind: "op" as const,
			label: "Partial output",
			detail: "8 tokens",
			tray: [TRAY_COLS, 2] as [number, number],
			from: [V.experts(r)],
		})),
		{
			id: V.rs,
			kind: "other",
			label: "Reduce-scatter",
			from: ranks.map(V.partial),
		},
		...ranks.map((r) => ({
			id: V.out(r),
			kind: "output" as const,
			label: "Output",
			detail: "2 tokens",
			tray: [2, 1] as [number, number],
			from: [V.rs],
		})),
	];

	const hiddenAtHome = (
		c: Copy,
		n: (r: number) => string,
		slot: { col: number; row: number },
	): ChipPlace => ({
		node: n(c.home),
		...slot,
		state: "hidden",
	});
	const atTokens = placeAll((r, c) =>
		r === c.home
			? {
					node: V.tok(r),
					col: c.t % TOKENS_PER_RANK,
					row: 0,
					state: "plain",
					nudge: c.k * 1.5,
				}
			: hiddenAtHome(c, V.tok, { col: c.t % TOKENS_PER_RANK, row: 0 }),
	);
	const routerSlot = (c: Copy) =>
		gridSlot(c.id - c.home * TOKENS_PER_RANK * TOP_K);
	const atRouter = placeAll((r, c) =>
		r === c.home
			? { node: V.router(r), ...routerSlot(c), state: "on" }
			: hiddenAtHome(c, V.router, routerSlot(c)),
	);
	const atAll = placeAll((_r, c) => ({
		node: V.all(_r),
		...gridSlot(c.id),
		state: "on",
	}));
	const atExperts = placeAll((r, c) =>
		c.dest === r
			? { node: V.experts(r), ...expertSlot(r, c), state: "on" }
			: { node: V.all(r), ...gridSlot(c.id), state: "dim" },
	);
	const atPartial = placeAll((r, c) =>
		c.dest === r
			? {
					node: V.partial(r),
					...tokSlot(c.t, TRAY_COLS),
					state: "on",
					nudge: c.k * 1.5,
				}
			: { node: V.all(r), ...gridSlot(c.id), state: "dim" },
	);
	const atOut = placeAll((r, c) =>
		c.dest === r
			? {
					node: V.out(c.home),
					col: c.t % TOKENS_PER_RANK,
					row: 0,
					state: "plain",
					nudge: c.k * 1.5,
				}
			: { node: V.all(r), ...gridSlot(c.id), state: "hidden" },
	);

	const gathered = (EP - 1) * TOKENS_PER_RANK;
	const code = [
		`def forward(x, router_logits):  # x: [${TOKENS_PER_RANK}, h] on each of ${EP} DP ranks; the EP group is all ${EP}`,
		`    topk_weights, topk_ids = select_experts(x, router_logits)  # top-${TOP_K} of ${EXPERTS}, for my tokens only`,
		`    x, topk_weights, topk_ids = all_gatherv([x, topk_weights, topk_ids], group=ep_group)  # [${TOKENS}, h]`,
		`    sorted_ids, expert_ids, _ = moe_align_block_size(topk_ids, BLOCK_M, ${EXPERTS}, expert_map)  # -1: not mine`,
		"    cache1 = fused_moe_kernel(x, w1, sorted_ids, expert_ids)  # reads rows of x by index",
		"    cache2 = silu_and_mul(cache1)",
		"    cache3 = fused_moe_kernel(cache2, w2, sorted_ids, expert_ids, topk_weights)  # gate applied",
		`    out = moe_sum(cache3)  # [${TOKENS}, h]: add each token's top-${TOP_K} rows`,
		`    return reduce_scatterv(out, group=ep_group)  # [${TOKENS_PER_RANK}, h]: sum over ranks, keep my rows`,
	];
	const steps: FlowStep[] = [
		{
			line: 0,
			head: "Each rank serves its own requests",
			body: `vLLM runs data-parallel attention: each of the ${EP} ranks holds different requests, here ${TOKENS_PER_RANK} tokens each. With --enable-expert-parallel, the MoE layers of all ${EP} ranks form one EP group of size TP × DP = ${EP}, each holding ${LOCAL} experts. Hover over a node to see its tensor.`,
			active: ranks.map(V.tok),
			places: atTokens,
		},
		{
			line: 1,
			head: "Route locally",
			body: `Each rank runs the router on its own tokens only. Rank 0's t0 picks experts ${fmt(ROUTING[0])}, t1 picks ${fmt(ROUTING[1])}. Chip colours are the ranks that hold those experts.`,
			active: ranks.map(V.router),
			places: atRouter,
		},
		{
			line: 2,
			head: "All-gather everyone's tokens",
			body: `Instead of sending each copy to its expert, the default backend sends every token to every rank: an all-gather of the hidden states with their top-k ids and weights. Each rank receives ${gathered} rows from the others and now holds all ${TOKENS} tokens, ${COPIES.length} copies.`,
			active: [V.gather, ...ranks.map(V.all)],
			places: atAll,
		},
		{
			line: 3,
			head: "Keep only my experts' copies",
			body: `moe_align_block_size sorts the copies by expert and pads each expert's group to the kernel's block size. expert_map turns the other ranks' experts into -1, so their copies (dimmed) are skipped. Rank ${busy} keeps ${load[busy]} copies, rank ${quiet} only ${load[quiet]}.`,
			active: ranks.map(V.experts),
			places: atExperts,
		},
		{
			line: 4,
			head: "First GEMM",
			body: "The Triton kernel launches one program per block of sorted copies. Each block belongs to one expert and reads its token rows straight from x by index, so the tokens are never physically permuted. Blocks of -1 experts write nothing.",
			active: ranks.map(V.experts),
			places: atExperts,
		},
		{
			line: 5,
			head: "Activation",
			body: "SiLU of the gate half times the up half, the usual gated MLP, on every computed row.",
			active: ranks.map(V.experts),
			places: atExperts,
		},
		{
			line: 6,
			head: "Second GEMM, scaled by the gate",
			body: "The down projection runs on the same blocks, and multiplies each row by its gate weight as it writes it into a [tokens, top-k, h] buffer, at the copy's own position.",
			active: ranks.map(V.experts),
			places: atExperts,
		},
		{
			line: 7,
			head: "Sum each token's rows",
			body: `moe_sum adds each token's top-${TOP_K} rows. Every rank now has an output row for all ${TOKENS} tokens, but each row only holds this rank's experts' share: a partial sum, zero for tokens none of its experts saw.`,
			active: ranks.map(V.partial),
			places: atPartial,
		},
		{
			line: 8,
			head: "Reduce-scatter",
			body: `One collective does the combine: it adds the ${EP} partial outputs and gives each rank back only its own ${TOKENS_PER_RANK} rows. Each rank sends ${gathered} rows. Every token's expert outputs have been summed, wherever they were computed.`,
			active: [V.rs, ...ranks.map(V.out)],
			places: atOut,
		},
	];

	const views: Record<string, TensorView> = {};
	for (const r of ranks) {
		const toks = homeTokens(r);
		views[V.tok(r)] = {
			title: `Rank ${r}'s tokens`,
			code: "x",
			shape: `[${TOKENS_PER_RANK}, h]`,
			shapeWords: `${TOKENS_PER_RANK} rows × h numbers`,
			note: "This rank's own requests, after its own attention layer (data-parallel attention).",
			groups: [
				{
					label: "",
					rows: toks.map((t) => ({ t, rank: null, note: `token ${t}` })),
				},
			],
		};
		views[V.router(r)] = routerView(
			r,
			toks,
			`Rank ${r}'s routing decision`,
			"Computed only for this rank's tokens, before any communication.",
		);
		views[V.all(r)] = {
			title: `Rank ${r} after the all-gather`,
			code: "x, topk_ids",
			shape: `[${TOKENS}, h]`,
			shapeWords: `${TOKENS} rows × h numbers, every token of the group`,
			note: `Every rank holds the same ${TOKENS} tokens. Only the copies for E${r * LOCAL} and E${r * LOCAL + 1} will be computed here.`,
			groups: ranks.map((src) => ({
				label: `from rank ${src}`,
				rows: COPIES.filter((c) => c.home === src).map((c) => ({
					t: c.t,
					rank: c.dest,
					note:
						c.dest === r
							? `expert ${c.expert} · computed here`
							: `expert ${c.expert} · skipped (-1)`,
				})),
			})),
		};
		views[V.experts(r)] = expertsView(
			r,
			"intermediate_cache3",
			"Rows are grouped by expert in blocks of BLOCK_M (padded); the copies of other ranks' experts are never computed.",
		);
		views[V.partial(r)] = partialView(r, `Rank ${r}'s partial output`);
		views[V.out(r)] = {
			title: `Rank ${r}'s output`,
			code: "output",
			shape: `[${TOKENS_PER_RANK}, h]`,
			shapeWords: `${TOKENS_PER_RANK} rows × h numbers`,
			note: "The reduce-scatter added the 4 partial rows of each token, so each holds the gate-weighted sum of both its experts.",
			groups: [{ label: "", rows: toks.map(finalRow) }],
		};
	}
	views[V.gather] = collectiveView(
		"Rows received by the all-gather",
		"all_gatherv",
		"rows in",
		ranks.map(() => gathered),
		`Every rank receives every other rank's ${TOKENS_PER_RANK} tokens, whatever experts they chose: the volume doesn't depend on the routing.`,
	);
	views[V.rs] = collectiveView(
		"Rows sent by the reduce-scatter",
		"reduce_scatterv",
		"rows out",
		ranks.map(() => gathered),
		"Each rank sends its partial rows for every other rank's tokens, and receives the sum for its own.",
	);
	return {
		nodes,
		chips: replicatedChips(),
		steps,
		code,
		views,
		source:
			"Condensed from vLLM's MoERunner, MoEPrepareAndFinalizeNaiveDPEPModular, AgRsAll2AllManager and fused_experts_impl (default --all2all-backend allgather_reducescatter).",
		traffic: ranks.map(() => 2 * gathered),
	};
}

// ---------------------------------------------------------------- SGLang

const G = {
	tok: (r: number) => `g-tok${r}`,
	router: (r: number) => `g-router${r}`,
	experts: (r: number) => `g-exp${r}`,
	partial: (r: number) => `g-partial${r}`,
	ar: "g-allreduce",
	out: (r: number) => `g-out${r}`,
};

function sglangFlow(): Flow {
	const nodes: FlowNode[] = [
		...ranks.map((r) => ({
			id: G.tok(r),
			kind: "input" as const,
			label: `Rank ${r}`,
			detail: "all 8 tokens",
			tray: [TRAY_COLS, 2] as [number, number],
		})),
		...ranks.map((r) => ({
			id: G.router(r),
			kind: "linear" as const,
			label: "Router",
			detail: "16 copies",
			tray: [TRAY_COLS, 4] as [number, number],
			from: [G.tok(r)],
		})),
		...ranks.map((r) => ({
			id: G.experts(r),
			kind: "linear" as const,
			label: "Fused MoE kernel",
			detail: `E${r * LOCAL} / E${r * LOCAL + 1}`,
			tray: [TRAY_COLS, LOCAL] as [number, number],
			from: [G.router(r)],
		})),
		...ranks.map((r) => ({
			id: G.partial(r),
			kind: "op" as const,
			label: "Partial output",
			detail: "8 tokens",
			tray: [TRAY_COLS, 2] as [number, number],
			from: [G.experts(r)],
		})),
		{
			id: G.ar,
			kind: "other",
			label: "All-reduce",
			from: ranks.map(G.partial),
		},
		...ranks.map((r) => ({
			id: G.out(r),
			kind: "output" as const,
			label: "Output",
			detail: "all 8 tokens",
			tray: [TRAY_COLS, 2] as [number, number],
			from: [G.ar],
		})),
	];
	const atTokens = placeAll((r, c) => ({
		node: G.tok(r),
		...tokSlot(c.t, TRAY_COLS),
		state: "plain",
		nudge: c.k * 1.5,
	}));
	const atRouter = placeAll((r, c) => ({
		node: G.router(r),
		...gridSlot(c.id),
		state: "on",
	}));
	const atExperts = placeAll((r, c) =>
		c.dest === r
			? { node: G.experts(r), ...expertSlot(r, c), state: "on" }
			: { node: G.router(r), ...gridSlot(c.id), state: "dim" },
	);
	const atPartial = placeAll((r, c) =>
		c.dest === r
			? {
					node: G.partial(r),
					...tokSlot(c.t, TRAY_COLS),
					state: "on",
					nudge: c.k * 1.5,
				}
			: { node: G.router(r), ...gridSlot(c.id), state: "dim" },
	);
	const atOut = placeAll((r, c) => ({
		node: G.out(r),
		...tokSlot(c.t, TRAY_COLS),
		state: "plain",
		nudge: c.k * 1.5,
	}));
	// A ring all-reduce of an [8, h] buffer over 4 ranks: each sends 2 (N - 1) / N of it.
	const arRows = (2 * (EP - 1) * TOKENS) / EP;
	const code = [
		`def forward_normal(self, hidden_states):  # [${TOKENS}, h], the same on all ${EP} ranks (TP attention)`,
		`    router_logits = self.gate(hidden_states)  # [${TOKENS}, ${EXPERTS}]: every rank routes every token`,
		`    topk_output = self.topk(hidden_states, router_logits)  # top-${TOP_K} of ${EXPERTS} experts`,
		"    topk_ids = self.local_expert_mapping[topk_output.topk_ids]  # StandardDispatcher: no communication, -1 = not mine",
		`    out = self.experts(hidden_states, topk_ids, topk_weights)  # Triton fused MoE, sums the top-${TOP_K}`,
		"    return tensor_model_parallel_all_reduce(out)  # add the partial outputs of all ranks",
	];
	const steps: FlowStep[] = [
		{
			line: 0,
			head: "Every rank already has every token",
			body: `With --tp ${EP} --ep ${EP}, attention is tensor-parallel: it ends with an all-reduce, so all ${EP} ranks hold the same ${TOKENS} tokens when the MoE layer starts. There is nothing to dispatch. Hover over a node to see its tensor.`,
			active: ranks.map(G.tok),
			places: atTokens,
		},
		{
			line: 1,
			head: "Router logits",
			body: `The router is replicated, so every rank scores all ${TOKENS} tokens against all ${EXPERTS} experts, and they all get the same answer.`,
			active: ranks.map(G.router),
			places: atTokens,
		},
		{
			line: 2,
			head: "Top-k on every rank",
			body: `Each token becomes ${TOP_K} copies, ${COPIES.length} in all, on every rank. Chip colours are the ranks that hold the experts.`,
			active: ranks.map(G.router),
			places: atRouter,
		},
		{
			line: 3,
			head: "Map to local experts",
			body: `The default dispatcher only renames experts: local_expert_mapping turns E${busy * LOCAL} and E${busy * LOCAL + 1} into 0 and 1 on rank ${busy}, and every other expert into -1. The dimmed copies are skipped. Rank ${busy} keeps ${load[busy]} copies, rank ${quiet} ${load[quiet]}.`,
			active: ranks.map(G.experts),
			places: atExperts,
		},
		{
			line: 4,
			head: "Run the local experts",
			body: `The Triton fused MoE kernel (the same algorithm as vLLM's) computes the local copies, scales them by their gates and adds each token's rows: a partial output for all ${TOKENS} tokens.`,
			active: ranks.map(G.partial),
			places: atPartial,
		},
		{
			line: 5,
			head: "All-reduce",
			body: `An all-reduce adds the ${EP} partial outputs, exactly as after a tensor-parallel MLP. Every rank ends with the full output of all ${TOKENS} tokens, ready for the next layer's tensor-parallel attention.`,
			active: [G.ar, ...ranks.map(G.out)],
			places: atOut,
		},
	];
	const views: Record<string, TensorView> = {};
	for (const r of ranks) {
		views[G.tok(r)] = {
			title: `Rank ${r}'s input`,
			code: "hidden_states",
			shape: `[${TOKENS}, h]`,
			shapeWords: `${TOKENS} rows × h numbers, identical on every rank`,
			note: "Tensor-parallel attention leaves the whole batch on every rank of the TP group.",
			groups: [
				{
					label: "",
					rows: range(TOKENS).map((t) => ({
						t,
						rank: null,
						note: `token ${t}`,
					})),
				},
			],
		};
		views[G.router(r)] = routerView(
			r,
			range(TOKENS),
			`Rank ${r}'s routing (same on every rank)`,
			"The router runs on all tokens on every rank: redundant, but cheap next to the experts.",
		);
		views[G.experts(r)] = expertsView(
			r,
			"fused_experts output",
			"Copies whose expert maps to -1 are never computed; the rest are grouped by expert for the Triton kernel.",
		);
		views[G.partial(r)] = partialView(r, `Rank ${r}'s partial output`);
		views[G.out(r)] = {
			title: `Rank ${r}'s output`,
			code: "final_hidden_states",
			shape: `[${TOKENS}, h]`,
			shapeWords: `${TOKENS} rows × h numbers, identical on every rank`,
			note: "After the all-reduce every token holds the gate-weighted sum of both its experts, on every rank.",
			groups: [{ label: "", rows: range(TOKENS).map(finalRow) }],
		};
	}
	views[G.ar] = collectiveView(
		"Rows sent by the all-reduce",
		"tensor_model_parallel_all_reduce",
		"rows out",
		ranks.map(() => arRows),
		`A ring all-reduce of the [${TOKENS}, h] output: each rank sends 2 × (N − 1) / N of it, ${arRows} rows here, however the tokens were routed.`,
	);
	return {
		nodes,
		chips: replicatedChips(),
		steps,
		code,
		views,
		source:
			"Condensed from SGLang's DeepseekV2MoE.forward_normal, StandardDispatcher and the Triton MoE runner (default --moe-a2a-backend none).",
		traffic: ranks.map(() => arRows),
	};
}

// ---------------------------------------------------------------- DeepSpeed

/** DeepSpeed example settings: capacity factor 1 and min_capacity 0 (the default is 4). */
export const DS_CAPACITY = Math.ceil((TOKENS_PER_RANK / EXPERTS) * 2 * 1.0);

const D = {
	tok: (r: number) => `ds-tok${r}`,
	gate: (r: number) => `ds-gate${r}`,
	buf: (r: number) => `ds-buf${r}`,
	a2a1: "ds-a2a1",
	recv: (r: number) => `ds-recv${r}`,
	exp: (r: number) => `ds-exp${r}`,
	a2a2: "ds-a2a2",
	back: (r: number) => `ds-back${r}`,
	out: (r: number) => `ds-out${r}`,
};

/**
 * GShard / DeepSpeed slot assignment on each rank: first choices take slots before second
 * choices, each in token order; copies past the capacity are dropped.
 * Returns, per rank, the copy (or -1) in every [expert, slot] cell, and the dropped copies.
 */
export function deepspeedSlots(capacity = DS_CAPACITY) {
	const cells = ranks.map(() =>
		range(EXPERTS).map(() => new Array<number>(capacity).fill(-1)),
	);
	const dropped: number[] = [];
	for (const r of ranks)
		for (let k = 0; k < TOP_K; k++)
			for (const c of COPIES.filter((c) => c.home === r && c.k === k)) {
				const slot = cells[r][c.expert].indexOf(-1);
				if (slot < 0) dropped.push(c.id);
				else cells[r][c.expert][slot] = c.id;
			}
	return { cells, dropped };
}

function deepspeedFlow(): Flow {
	const C = DS_CAPACITY;
	const { cells, dropped } = deepspeedSlots();
	// Chips: the 16 copies, then one padding chip for every empty (rank, expert, slot) cell.
	const chips: FlowChip[] = COPIES.map((c) => ({ t: c.t, dest: c.dest }));
	const padId = new Map<string, number>();
	for (const r of ranks)
		for (let e = 0; e < EXPERTS; e++)
			for (let j = 0; j < C; j++)
				if (cells[r][e][j] < 0) {
					padId.set(`${r}:${e}:${j}`, chips.length);
					chips.push({ t: -1, dest: Math.floor(e / LOCAL), pad: true });
				}
	/** Each (rank, expert, slot) cell's chip id: the copy in it, or its padding chip. */
	const cellChip = (r: number, e: number, j: number) =>
		cells[r][e][j] >= 0
			? cells[r][e][j]
			: (padId.get(`${r}:${e}:${j}`) as number);
	const bufSlot = (e: number, j: number) => gridSlot(e * C + j);
	/** After the first all-to-all, rank d holds, for each local expert, C rows from each source. */
	const recvSlot = (src: number, e: number, j: number) => ({
		col: src * C + j,
		row: e % LOCAL,
	});

	const nodes: FlowNode[] = [
		...ranks.map((r) => ({
			id: D.tok(r),
			kind: "input" as const,
			label: `Rank ${r}`,
			detail: "2 tokens",
			tray: [2, 1] as [number, number],
		})),
		...ranks.map((r) => ({
			id: D.gate(r),
			kind: "linear" as const,
			label: "Gate",
			detail: `top-2, C = ${C}`,
			tray: [TRAY_COLS, 1] as [number, number],
			from: [D.tok(r)],
		})),
		...ranks.map((r) => ({
			id: D.buf(r),
			kind: "op" as const,
			label: "Capacity buffer",
			detail: `[${EXPERTS} experts, ${C}, h]`,
			tray: [TRAY_COLS, (EXPERTS * C) / TRAY_COLS] as [number, number],
			from: [D.gate(r)],
		})),
		{
			id: D.a2a1,
			kind: "other",
			label: "All-to-all: equal chunks",
			from: ranks.map(D.buf),
		},
		...ranks.map((r) => ({
			id: D.recv(r),
			kind: "op" as const,
			label: "Received",
			detail: `[${EP} ranks, ${LOCAL}, ${C}, h]`,
			tray: [EP * C, LOCAL] as [number, number],
			from: [D.a2a1],
		})),
		...ranks.map((r) => ({
			id: D.exp(r),
			kind: "linear" as const,
			label: "Experts",
			detail: `E${r * LOCAL} / E${r * LOCAL + 1}`,
			tray: [EP * C, LOCAL] as [number, number],
			from: [D.recv(r)],
		})),
		{
			id: D.a2a2,
			kind: "other",
			label: "All-to-all: back",
			from: ranks.map(D.exp),
		},
		...ranks.map((r) => ({
			id: D.back(r),
			kind: "op" as const,
			label: "Returned",
			detail: `[${EXPERTS} experts, ${C}, h]`,
			tray: [TRAY_COLS, (EXPERTS * C) / TRAY_COLS] as [number, number],
			from: [D.a2a2],
		})),
		...ranks.map((r) => ({
			id: D.out(r),
			kind: "output" as const,
			label: "Combine",
			detail: "2 tokens",
			tray: [2, 1] as [number, number],
			from: [D.back(r)],
		})),
	];

	/** Places for every chip, from a function of the cell (rank, expert, slot) it occupies. */
	const byCell = (
		f: (r: number, e: number, j: number, isPad: boolean) => ChipPlace,
		other: (c: Copy) => ChipPlace,
	): ChipPlace[] => {
		const out: ChipPlace[] = [];
		for (const c of COPIES) out[c.id] = other(c);
		for (const r of ranks)
			for (let e = 0; e < EXPERTS; e++)
				for (let j = 0; j < C; j++)
					out[cellChip(r, e, j)] = f(r, e, j, cells[r][e][j] < 0);
		return out;
	};
	const gateSlot = (c: Copy) =>
		gridSlot(c.id - c.home * TOKENS_PER_RANK * TOP_K);
	const atGate = (c: Copy): ChipPlace => ({
		node: D.gate(c.home),
		...gateSlot(c),
		state: "on",
	});
	const hiddenAt = (
		n: string,
		slot: { col: number; row: number },
	): ChipPlace => ({ node: n, ...slot, state: "hidden" });
	const isDropped = (c: Copy) => dropped.includes(c.id);

	const atTokens = byCell(
		(r, e, j) => hiddenAt(D.buf(r), bufSlot(e, j)),
		(c) => ({
			node: D.tok(c.home),
			col: c.t % TOKENS_PER_RANK,
			row: 0,
			state: "plain",
			nudge: c.k * 1.5,
		}),
	);
	const atGateStep = byCell(
		(r, e, j, isPad) =>
			isPad
				? hiddenAt(D.buf(r), bufSlot(e, j))
				: atGate(COPIES[cells[r][e][j]]),
		atGate,
	);
	const inBuf = (r: number, e: number, j: number): ChipPlace => ({
		node: D.buf(r),
		...bufSlot(e, j),
		state: "on",
	});
	const droppedAtGate = (c: Copy): ChipPlace => ({
		...atGate(c),
		state: isDropped(c) ? "dim" : "on",
	});
	const atBuf = byCell(inBuf, droppedAtGate);
	const atRecv = byCell(
		(r, e, j) => ({
			node: D.recv(Math.floor(e / LOCAL)),
			...recvSlot(r, e, j),
			state: "on",
		}),
		droppedAtGate,
	);
	const atExp = byCell(
		(r, e, j) => ({
			node: D.exp(Math.floor(e / LOCAL)),
			...recvSlot(r, e, j),
			state: "on",
		}),
		droppedAtGate,
	);
	const atBack = byCell(
		(r, e, j) => ({ node: D.back(r), ...bufSlot(e, j), state: "on" }),
		droppedAtGate,
	);
	const atOut = byCell(
		(r, e, j, isPad) => {
			if (isPad) return hiddenAt(D.back(r), bufSlot(e, j));
			const c = COPIES[cells[r][e][j]];
			return {
				node: D.out(r),
				col: c.t % TOKENS_PER_RANK,
				row: 0,
				state: "plain",
				nudge: c.k * 1.5,
			};
		},
		(c) => ({ ...atGate(c), state: "hidden" }),
	);

	const pads = ranks.map((r) => cells[r].flat().filter((x) => x < 0).length);
	const droppedTxt = dropped
		.map((id) => `t${COPIES[id].t}'s choice of E${COPIES[id].expert}`)
		.join(", ");
	const code = [
		`def forward(self, x):  # x: [${TOKENS_PER_RANK}, h] tokens on each of the ${EP} EP ranks`,
		`    l_aux, C, E, indices, locations, gates, _ = self.gate(x)  # top-2; C = ceil(2 S / E × cf) = ${C}`,
		"    dispatched = _sparse_encode(x, _route_slots(indices, locations, E, C), E, C)  # [E, C, h]",
		"    dispatched = _AllToAll.apply(self.ep_group, dispatched)  # equal chunks, no counts needed",
		"    expert_output = self.experts(dispatched.reshape(ep_size, num_local_experts, -1, h))",
		"    expert_output = _AllToAll.apply(self.ep_group, expert_output)",
		"    return _sparse_decode(expert_output.view(E * C, h), slots, gates, S)  # gather, weight, sum",
	];
	const steps: FlowStep[] = [
		{
			line: 0,
			head: "Tokens on each rank",
			body: `The same example as before: ${EP} ranks, ${TOKENS_PER_RANK} tokens each, ${EXPERTS} experts, top-2. DeepSpeed's MoE layer follows GShard: every expert gets a fixed number of slots per rank. Hover over a node to see its tensor.`,
			active: ranks.map(D.tok),
			places: atTokens,
		},
		{
			line: 1,
			head: "Gate: top-2 and a capacity",
			body: `The gate picks each token's 2 experts and sizes every expert's buffer from the local token count S = ${TOKENS_PER_RANK}: C = ceil(2 × ${TOKENS_PER_RANK} / ${EXPERTS} × 1.0) = ${C} slot (min_capacity is set to 0 here; DeepSpeed's default of 4 would leave even more padding). Slots are numbered first choices first, then second choices.`,
			active: ranks.map(D.gate),
			places: atGateStep,
		},
		{
			line: 2,
			head: "Fill the capacity buffers",
			body: `Each copy goes to its expert's slot in an [${EXPERTS}, ${C}, h] buffer; empty slots are zero padding (dashed). On rank 0, t1 takes expert 2's only slot with its first choice, so ${droppedTxt} is dropped (dimmed): that token keeps only its other expert. Ranks carry ${pads.join(", ")} padding rows.`,
			active: ranks.map(D.buf),
			places: atBuf,
		},
		{
			line: 3,
			head: "All-to-all with equal chunks",
			body: `Every rank sends rank j the slots of experts ${LOCAL}j and ${LOCAL}j + 1: always ${LOCAL * C} rows, padding included. Because every chunk has the same size, no counts need to be exchanged first, and the shapes are static.`,
			active: [D.a2a1, ...ranks.map(D.recv)],
			places: atRecv,
		},
		{
			line: 4,
			head: "Run the experts on full buffers",
			body: `Each rank runs its ${LOCAL} experts on ${EP * C} rows each, one from every rank, padding included. Rank 1, which holds the popular experts, gets no more rows than any other rank: the capacity has already capped its load by dropping.`,
			active: ranks.map(D.exp),
			places: atExp,
		},
		{
			line: 5,
			head: "All-to-all back",
			body: "The same equal exchange in reverse: every slot returns to the rank its token came from, in the same [experts, C, h] layout.",
			active: [D.a2a2, ...ranks.map(D.back)],
			places: atBack,
		},
		{
			line: 6,
			head: "Combine: gather, weight, sum",
			body: "_sparse_decode gathers each token's slots, multiplies them by the gate weights and sums them. Padding is ignored; a dropped copy contributes nothing, and the residual connection carries the token through.",
			active: ranks.map(D.out),
			places: atOut,
		},
	];

	const views: Record<string, TensorView> = {};
	const slotRows = (r: number, f: (c: Copy) => string) =>
		range(EXPERTS).map((e) => ({
			label: `expert E${e}`,
			rows: range(C).map((j) => {
				const id = cells[r][e][j];
				return id < 0
					? { t: -1, rank: null, note: "padding (zeros)" }
					: { t: COPIES[id].t, rank: COPIES[id].dest, note: f(COPIES[id]) };
			}),
		}));
	for (const r of ranks) {
		const toks = homeTokens(r);
		views[D.tok(r)] = {
			title: `Rank ${r}'s tokens`,
			code: "x",
			shape: `[${TOKENS_PER_RANK}, h]`,
			shapeWords: `${TOKENS_PER_RANK} rows × h numbers`,
			note: "This rank's own tokens.",
			groups: [
				{
					label: "",
					rows: toks.map((t) => ({ t, rank: null, note: `token ${t}` })),
				},
			],
		};
		views[D.gate(r)] = {
			...routerView(
				r,
				toks,
				`Rank ${r}'s gate`,
				`Top-2 gating with capacity C = ${C} slot per expert on this rank.`,
			),
			groups: [
				{
					label: "",
					rows: toks.flatMap((t) =>
						COPIES.filter((c) => c.t === t).map((c) => ({
							t,
							rank: c.dest,
							note: `${c.k === 0 ? "1st" : "2nd"} choice E${c.expert} · gate ${g2(c.gate)}${isDropped(c) ? " · dropped" : ""}`,
						})),
					),
				},
			],
		};
		views[D.buf(r)] = {
			title: `Rank ${r}'s capacity buffer`,
			code: "dispatched_input",
			shape: `[${EXPERTS}, ${C}, h]`,
			shapeWords: `${EXPERTS * C} rows × h numbers, ${pads[r]} of them padding`,
			note: "One block of C rows per expert, whether or not anyone chose it.",
			groups: slotRows(r, (c) => `slot for E${c.expert}`),
		};
		views[D.recv(r)] = {
			title: `Rank ${r} after the first all-to-all`,
			code: "dispatched_input",
			shape: `[${EP}, ${LOCAL}, ${C}, h]`,
			shapeWords: `${EP * LOCAL * C} rows: ${C} per (source rank, local expert)`,
			note: "Every rank sent exactly the same number of rows; padding travels too.",
			groups: range(LOCAL).map((le) => ({
				label: `expert E${r * LOCAL + le}`,
				rows: ranks.flatMap((src) =>
					range(C).map((j) => {
						const id = cells[src][r * LOCAL + le][j];
						return id < 0
							? { t: -1, rank: null, note: `from rank ${src} · padding` }
							: { t: COPIES[id].t, rank: r, note: `from rank ${src}` };
					}),
				),
			})),
		};
		views[D.exp(r)] = {
			...views[D.recv(r)],
			title: `Rank ${r}'s experts`,
			code: "expert_output",
			note: "Each local expert multiplies all of its rows, padding included: a batched matrix multiply over equal-sized blocks.",
		};
		views[D.back(r)] = {
			title: `Rank ${r} after the second all-to-all`,
			code: "expert_output",
			shape: `[${EXPERTS}, ${C}, h]`,
			shapeWords: `${EXPERTS * C} rows, back in this rank's slot layout`,
			note: "The expert outputs for this rank's own copies, in the slots they left from.",
			groups: slotRows(r, (c) => `E${c.expert}(t${c.t})`),
		};
		views[D.out(r)] = {
			title: `Rank ${r}'s output`,
			code: "combined_output",
			shape: `[${TOKENS_PER_RANK}, h]`,
			shapeWords: `${TOKENS_PER_RANK} rows × h numbers`,
			note: "Gate-weighted sum of each token's surviving copies. The gates are not renormalized after a drop.",
			groups: [
				{
					label: "",
					rows: toks.map((t) => ({
						t,
						rank: null,
						note: COPIES.filter((c) => c.t === t && !isDropped(c))
							.map((c) => `${g2(c.gate)} × E${c.expert}(t${t})`)
							.join(" + "),
					})),
				},
			],
		};
	}
	const sent = ranks.map(() => (EP - 1) * LOCAL * C);
	views[D.a2a1] = collectiveView(
		"Rows sent by the first all-to-all",
		"_AllToAll",
		"rows out",
		sent,
		`Each rank sends ${LOCAL * C} rows to each other rank, whatever the routing: the price of static shapes is padding and dropping.`,
	);
	views[D.a2a2] = collectiveView(
		"Rows sent by the second all-to-all",
		"_AllToAll",
		"rows out",
		sent,
		"The same equal exchange, in reverse.",
	);
	return {
		nodes,
		chips,
		steps,
		code,
		views,
		source:
			"Condensed from DeepSpeed's MOELayer.forward and top2gating (deepspeed/moe/sharded_moe.py), with capacity factor 1 and min_capacity 0.",
		traffic: sent.map((x) => 2 * x),
	};
}

// ---------------------------------------------------------------- DeepSeek-V3 / DeepEP hybrid

/** Ranks are numbered node × 2 + local index: 2 nodes of 2 GPUs. */
export const GPUS_PER_NODE = 2;
const nodeOf = (r: number) => Math.floor(r / GPUS_PER_NODE);
const idxOf = (r: number) => r % GPUS_PER_NODE;
/** The GPU a cross-node copy first lands on: same local index as its home, on the target node. */
const viaOf = (c: Copy) => nodeOf(c.dest) * GPUS_PER_NODE + idxOf(c.home);
const crossNode = (c: Copy) => nodeOf(c.dest) !== nodeOf(c.home);

const K = {
	tok: (r: number) => `dk-tok${r}`,
	router: (r: number) => `dk-router${r}`,
	ib: "dk-ib",
	rdma: (r: number) => `dk-rdma${r}`,
	nvl: "dk-nvl",
	exp: (r: number) => `dk-exp${r}`,
	nvl2: "dk-nvl2",
	fwd: (r: number) => `dk-fwd${r}`,
	ib2: "dk-ib2",
	out: (r: number) => `dk-out${r}`,
};

/** Tokens that cross InfiniBand: one entry per (token, target node). */
export function deepseekIbSends() {
	const sends: { t: number; from: number; to: number; copies: number[] }[] = [];
	for (const c of COPIES.filter(crossNode)) {
		const to = viaOf(c);
		const s = sends.find((x) => x.t === c.t && x.to === to);
		if (s) s.copies.push(c.id);
		else sends.push({ t: c.t, from: c.home, to, copies: [c.id] });
	}
	return sends;
}

function deepseekFlow(): Flow {
	const sends = deepseekIbSends();
	const gpuName = (r: number) => `Node ${nodeOf(r)} · GPU ${idxOf(r)}`;
	const nodes: FlowNode[] = [
		...ranks.map((r) => ({
			id: K.tok(r),
			kind: "input" as const,
			label: gpuName(r),
			detail: "2 tokens",
			tray: [2, 1] as [number, number],
		})),
		...ranks.map((r) => ({
			id: K.router(r),
			kind: "linear" as const,
			label: "Router",
			tray: [TRAY_COLS, 1] as [number, number],
			from: [K.tok(r)],
		})),
		{
			id: K.ib,
			kind: "other",
			label: "RDMA: once per token and node",
			from: ranks.map(K.router),
		},
		...ranks.map((r) => ({
			id: K.rdma(r),
			kind: "op" as const,
			label: "Arrived over RDMA",
			tray: [TRAY_COLS, 1] as [number, number],
			from: [K.ib],
		})),
		{
			id: K.nvl,
			kind: "other",
			label: "NVLink: forward to the experts",
			from: ranks.map(K.rdma),
		},
		...ranks.map((r) => ({
			id: K.exp(r),
			kind: "linear" as const,
			label: "Experts",
			detail: `E${r * LOCAL} / E${r * LOCAL + 1}`,
			tray: [TRAY_COLS, LOCAL] as [number, number],
			from: [K.nvl],
		})),
		{
			id: K.nvl2,
			kind: "other",
			label: "NVLink: back, summed per token",
			from: ranks.map(K.exp),
		},
		...ranks.map((r) => ({
			id: K.fwd(r),
			kind: "op" as const,
			label: "Partial sums",
			detail: "for other nodes",
			tray: [TRAY_COLS, 1] as [number, number],
			from: [K.nvl2],
		})),
		{
			id: K.ib2,
			kind: "other",
			label: "RDMA: back home",
			from: ranks.map(K.fwd),
		},
		...ranks.map((r) => ({
			id: K.out(r),
			kind: "output" as const,
			label: "Output",
			detail: "2 tokens",
			tray: [2, 1] as [number, number],
			from: [K.ib2],
		})),
	];
	/** Slot of a token arriving over RDMA at its forwarding GPU (copies of one token share it). */
	const rdmaSlot = (c: Copy) => {
		const at = sends.filter((s) => s.to === viaOf(c));
		const i = at.findIndex((s) => s.copies.includes(c.id));
		return gridSlot(i);
	};
	const routerSlot = (c: Copy) =>
		gridSlot(c.id - c.home * TOKENS_PER_RANK * TOP_K);
	const outSlot = (c: Copy) => ({ col: c.t % TOKENS_PER_RANK, row: 0 });
	const stack = (c: Copy) =>
		(sends.find((s) => s.copies.includes(c.id))?.copies.indexOf(c.id) ?? 0) *
		1.5;
	const place = (f: (c: Copy) => ChipPlace) => COPIES.map(f);

	const atTokens = place((c) => ({
		node: K.tok(c.home),
		...outSlot(c),
		state: "plain",
		nudge: c.k * 1.5,
	}));
	const atRouter = place((c) => ({
		node: K.router(c.home),
		...routerSlot(c),
		state: "on",
	}));
	const atRdma = place((c) =>
		crossNode(c)
			? { node: K.rdma(viaOf(c)), ...rdmaSlot(c), state: "on", nudge: stack(c) }
			: { node: K.router(c.home), ...routerSlot(c), state: "on" },
	);
	const atExp = place((c) => ({
		node: K.exp(c.dest),
		...expertSlot(c.dest, c),
		state: "on",
	}));
	const atFwd = place((c) =>
		crossNode(c)
			? {
					node: K.fwd(viaOf(c)),
					...rdmaSlot(c),
					state: "plain",
					nudge: stack(c),
				}
			: {
					node: K.out(c.home),
					...outSlot(c),
					state: "plain",
					nudge: c.k * 1.5,
				},
	);
	const atOut = place((c) => ({
		node: K.out(c.home),
		...outSlot(c),
		state: "plain",
		nudge: c.k * 1.5,
	}));

	const cross = COPIES.filter(crossNode);
	const dedup = sends.filter((s) => s.copies.length > 1);
	const relay = cross.filter((c) => viaOf(c) !== c.dest);
	const code = [
		`x = attention(x)  # [${TOKENS_PER_RANK}, h] tokens per GPU; 2 nodes × 2 GPUs, EP = ${EP}`,
		"topk_idx, topk_weights = gate(x)  # top-2; node-limited: experts on at most M nodes",
		"event = buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights, num_experts=E, ...)",
		"recv_x, _, recv_topk_weights, handle = event.current_stream_wait()  # expanded layout",
		"expert_output = grouped_gemm(recv_x) * recv_topk_weights  # the experts apply the gates",
		"event = buffer.combine(expert_output, handle)",
		"combined_x, _ = event.current_stream_wait()",
	];
	const steps: FlowStep[] = [
		{
			line: 0,
			head: "Two nodes of two GPUs",
			body: `The same ${EP}-rank example, now on 2 nodes: GPUs in a node talk over NVLink (160 GB/s on DeepSeek's H800s), nodes over InfiniBand RDMA (50 GB/s per GPU). Experts E0–E3 live on node 0, E4–E7 on node 1. Hover over a node to see its tensor.`,
			active: ranks.map(K.tok),
			places: atTokens,
		},
		{
			line: 1,
			head: "Route, at most M nodes per token",
			body: "DeepSeek-V3's router first keeps the best M nodes for each token (M = 4 of its 8), then picks the top 8 experts among them, so a token crosses InfiniBand at most M times. With 2 nodes here, every route is allowed.",
			active: ranks.map(K.router),
			places: atRouter,
		},
		{
			line: 2,
			head: "Dispatch, part 1: RDMA",
			body: `Each token crosses InfiniBand once per target node, to the GPU with its own local index there. ${cross.length} copies need another node, but only ${sends.length} RDMA transfers happen: ${dedup.map((s) => `t${s.t}'s ${s.copies.length} copies for node ${nodeOf(s.to)} travel together`).join("; ")}. Copies for the home node stay put.`,
			active: [K.ib, ...ranks.map(K.rdma)],
			places: atRdma,
		},
		{
			line: 3,
			head: "Dispatch, part 2: NVLink",
			body: `Inside each node, NVLink forwards every copy to the GPU that holds its expert: ${relay.length} of the RDMA arrivals are relayed one more hop, and the home node's copies go directly. In DeepEP both parts run in one kernel, with separate warps sending over RDMA, forwarding and receiving.`,
			active: [K.nvl, ...ranks.map(K.exp)],
			places: atExp,
		},
		{
			line: 4,
			head: "Run the experts",
			body: "Each GPU runs a grouped GEMM over its experts' rows, already laid out per expert and aligned for the GEMM (the expanded layout), and multiplies the rows by their gate weights.",
			active: ranks.map(K.exp),
			places: atExp,
		},
		{
			line: 5,
			head: "Combine, part 1: NVLink",
			body: "The results retrace the path. Within a node they go back over NVLink: straight home if the token lives on this node, otherwise to the GPU that received it over RDMA, which adds up that token's results from this node into one row.",
			active: [K.nvl2, ...ranks.map(K.fwd)],
			places: atFwd,
		},
		{
			line: 6,
			head: "Combine, part 2: RDMA",
			body: `One partial sum per token and node crosses InfiniBand back home, ${sends.length} transfers again, and is added to the token's other results. Each GPU ends with its ${TOKENS_PER_RANK} tokens.`,
			active: [K.ib2, ...ranks.map(K.out)],
			places: atOut,
		},
	];

	const views: Record<string, TensorView> = {};
	for (const r of ranks) {
		const toks = homeTokens(r);
		views[K.tok(r)] = {
			title: `${gpuName(r)}: tokens`,
			code: "x",
			shape: `[${TOKENS_PER_RANK}, h]`,
			shapeWords: `${TOKENS_PER_RANK} rows × h numbers`,
			note: "This GPU's own tokens after attention.",
			groups: [
				{
					label: "",
					rows: toks.map((t) => ({ t, rank: null, note: `token ${t}` })),
				},
			],
		};
		views[K.router(r)] = {
			...routerView(
				r,
				toks,
				`${gpuName(r)}: routing`,
				"Where each copy has to go, and how.",
			),
			groups: [
				{
					label: "",
					rows: toks.flatMap((t) =>
						COPIES.filter((c) => c.t === t).map((c) => ({
							t,
							rank: c.dest,
							note: `E${c.expert} on GPU ${c.dest}: ${c.dest === r ? "local" : !crossNode(c) ? "NVLink" : viaOf(c) === c.dest ? "RDMA" : `RDMA to GPU ${viaOf(c)}, then NVLink`}`,
						})),
					),
				},
			],
		};
		const arrivals = sends.filter((s) => s.to === r);
		views[K.rdma(r)] = {
			title: `${gpuName(r)}: RDMA receive buffer`,
			code: "(inside the dispatch kernel)",
			shape: `[${arrivals.length}, h]`,
			shapeWords: `${rowsOf(arrivals.length)}: one per (token, this node)`,
			note: "Tokens from the other node that arrived at this GPU because it has their home GPU's local index.",
			groups: [
				{
					label: "",
					rows: arrivals.map((s) => ({
						t: s.t,
						rank: null,
						note: `from GPU ${s.from}, for ${s.copies.map((id) => `E${COPIES[id].expert} (GPU ${COPIES[id].dest})`).join(" and ")}`,
					})),
				},
			],
		};
		views[K.exp(r)] = expertsView(
			r,
			"recv_x → expert_output",
			"Rows grouped by local expert (the expanded layout), each multiplied by its gate weight.",
		);
		const fwdRows = sends.filter((s) => s.to === r);
		views[K.fwd(r)] = {
			title: `${gpuName(r)}: partial sums for the other node`,
			code: "(inside the combine kernel)",
			shape: `[${fwdRows.length}, h]`,
			shapeWords: `${rowsOf(fwdRows.length)}: one per token received over RDMA`,
			note: "Results from this node for tokens that came from the other node, summed per token before crossing InfiniBand.",
			groups: [
				{
					label: "",
					rows: fwdRows.map((s) => ({
						t: s.t,
						rank: null,
						note: s.copies
							.map(
								(id) =>
									`${g2(COPIES[id].gate)} × E${COPIES[id].expert}(t${s.t})`,
							)
							.join(" + "),
					})),
				},
			],
		};
		views[K.out(r)] = {
			title: `${gpuName(r)}: output`,
			code: "combined_x",
			shape: `[${TOKENS_PER_RANK}, h]`,
			shapeWords: `${TOKENS_PER_RANK} rows × h numbers`,
			note: "Each token's expert outputs, gate-weighted and summed, from both nodes.",
			groups: [{ label: "", rows: toks.map(finalRow) }],
		};
	}
	const ibOut = ranks.map((r) => sends.filter((s) => s.from === r).length);
	views[K.ib] = collectiveView(
		"Rows each GPU sends over RDMA",
		"dispatch (RDMA part)",
		"rows out",
		ibOut,
		`${sends.length} transfers for ${cross.length} cross-node copies: a token going to two experts on the same node crosses InfiniBand once.`,
	);
	views[K.nvl] = collectiveView(
		"Copies each GPU receives over NVLink",
		"dispatch (NVLink part)",
		"rows in",
		ranks.map(
			(r) =>
				COPIES.filter(
					(c) =>
						c.dest === r && c.home !== r && !(crossNode(c) && viaOf(c) === r),
				).length,
		),
		"Forwarded RDMA arrivals plus copies from the same node.",
	);
	views[K.nvl2] = collectiveView(
		"Rows each GPU sends back over NVLink",
		"combine (NVLink part)",
		"rows out",
		ranks.map(
			(r) =>
				COPIES.filter(
					(c) =>
						c.dest === r && c.home !== r && !(crossNode(c) && viaOf(c) === r),
				).length,
		),
		"The mirror image of the NVLink dispatch.",
	);
	views[K.ib2] = collectiveView(
		"Rows each GPU sends back over RDMA",
		"combine (RDMA part)",
		"rows out",
		ranks.map((r) => sends.filter((s) => s.to === r).length),
		"One partial sum per token and node.",
	);
	return {
		nodes,
		chips: COPIES.map((c) => ({ t: c.t, dest: c.dest })),
		steps,
		code,
		views,
		source:
			"Condensed from DeepEP's README usage of EPBuffer.dispatch / combine (hybrid mode); the two-hop path follows the DeepSeek-V3 report, §3.2.2. In DeepEP the RDMA and NVLink parts run in one kernel.",
		traffic: ranks.map(
			(r) => ibOut[r] + sends.filter((x) => x.to === r).length,
		),
	};
}

export function buildFlow(name: FlowName): Flow {
	if (name === "vllm") return vllmFlow();
	if (name === "sglang") return sglangFlow();
	if (name === "deepspeed") return deepspeedFlow();
	if (name === "deepseek") return deepseekFlow();
	return megatronFlow();
}
