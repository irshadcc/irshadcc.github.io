// Megatron-LM's MoEAlltoAllTokenDispatcher (TP = 1, dropless, unfused path) on a toy setup:
// EP = 4 ranks, 8 experts (2 per rank), top-2, 8 tokens (2 per rank). Follows each token copy
// through every line of the forward pass, and describes the tensor at every graph node, for
// MoeEpSteps. The routing comes from a PyTorch run (seed 4, experts 2 and 3 made popular); the
// data path below was checked against the same algorithm running on 4 gloo processes.

export const EP = 4;
export const EXPERTS = 8;
export const LOCAL = EXPERTS / EP;
export const TOP_K = 2;
export const TOKENS_PER_RANK = 2;
export const TOKENS = EP * TOKENS_PER_RANK;

/** Expert ids chosen for each token (token t lives on rank floor(t / 2)), best first. */
export const ROUTING: number[][] = [
	[1, 2],
	[2, 3],
	[0, 6],
	[5, 3],
	[2, 5],
	[7, 3],
	[1, 3],
	[4, 2],
];

/** The matching gate weights (softmax over all experts, top-2 renormalized), rounded. */
export const GATES: number[][] = [
	[0.61, 0.39],
	[0.53, 0.47],
	[0.58, 0.42],
	[0.62, 0.38],
	[0.84, 0.16],
	[0.54, 0.46],
	[0.76, 0.24],
	[0.65, 0.35],
];

/** One token copy: token t's k-th choice, bound for `expert` on rank `dest`. */
export interface Copy {
	id: number;
	t: number;
	k: number;
	expert: number;
	home: number;
	dest: number;
	gate: number;
}

export const COPIES: Copy[] = ROUTING.flatMap((es, t) =>
	es.map((expert, k) => ({
		id: t * TOP_K + k,
		t,
		k,
		expert,
		home: Math.floor(t / TOKENS_PER_RANK),
		dest: Math.floor(expert / LOCAL),
		gate: GATES[t][k],
	})),
);

const range = (n: number) => [...Array(n).keys()];
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

export interface RankState {
	/** num_local_tokens_per_expert: copies this rank sends to each expert. */
	perExpert: number[];
	/** input_splits: copies this rank sends to each rank. */
	inputSplits: number[];
	/** output_splits: copies this rank receives from each rank. */
	outputSplits: number[];
	/** num_global_tokens_per_local_expert: [source rank][local expert]. */
	globalPerLocal: number[][];
	/** tokens_per_expert for the grouped GEMM. */
	tokensPerExpert: number[];
	/** Copy ids in each buffer, in row order. */
	permuted: number[];
	received: number[];
	sorted: number[];
	unsorted: number[];
	returned: number[];
}

/** Every rank's buffers after each step of the dispatcher. */
export function simulate(): RankState[] {
	const states = range(EP).map((r) => {
		const mine = COPIES.filter((c) => c.home === r);
		const perExpert = range(EXPERTS).map(
			(e) => mine.filter((c) => c.expert === e).length,
		);
		// permute: argsort of routing_map.T, i.e. expert-major, then token order (stable).
		const permuted = [...mine]
			.sort((a, b) => a.expert - b.expert || a.t - b.t)
			.map((c) => c.id);
		const inputSplits = range(EP).map((j) =>
			sum(perExpert.slice(j * LOCAL, (j + 1) * LOCAL)),
		);
		return { perExpert, inputSplits, permuted };
	});
	return states
		.map((s, r) => {
			// all_gather of the counts: every rank sees every rank's perExpert.
			const globalPerLocal = states.map((o) =>
				o.perExpert.slice(r * LOCAL, (r + 1) * LOCAL),
			);
			const outputSplits = globalPerLocal.map(sum);
			// all_to_all: chunk r of every source rank, in source-rank order.
			const received = states.flatMap((o) => {
				const start = sum(o.inputSplits.slice(0, r));
				return o.permuted.slice(start, start + o.inputSplits[r]);
			});
			// sort_chunks_by_idxs: chunks are (source rank, local expert); regroup local-expert-major.
			const chunks: number[][] = [];
			let i = 0;
			for (const row of globalPerLocal)
				for (const n of row) {
					chunks.push(received.slice(i, i + n));
					i += n;
				}
			const sortIdx = range(LOCAL).flatMap((le) =>
				range(EP).map((src) => src * LOCAL + le),
			);
			const sorted = sortIdx.flatMap((c) => chunks[c]);
			const tokensPerExpert = range(LOCAL).map((le) =>
				sum(globalPerLocal.map((row) => row[le])),
			);
			// Restore: back to (source rank, local expert) order, which is the received order.
			const unsorted = received;
			return {
				...s,
				globalPerLocal,
				outputSplits,
				received,
				sorted,
				tokensPerExpert,
				unsorted,
				returned: [],
			};
		})
		.map((s, r, all) => {
			// Reverse all_to_all: from each rank j, the chunk that came from r.
			const returned = all.flatMap((o) => {
				const start = sum(o.outputSplits.slice(0, r));
				return o.unsorted.slice(start, start + o.outputSplits[r]);
			});
			return { ...s, returned };
		});
}

/** Where a copy is drawn: a graph node, and a cell of that node's tray. */
export interface Place {
	node: string;
	col: number;
	row: number;
}

export interface Step {
	/** Index of the highlighted code line. */
	line: number;
	head: string;
	body: string;
	/** Graph nodes to highlight. */
	active: string[];
	/** Position of every copy, by copy id. */
	places: Place[];
	/** Copies not yet routed (drawn grey) or already summed back into their token. */
	phase: "tokens" | "copies" | "summed";
}

export const node = {
	tok: (r: number) => `tok${r}`,
	router: (r: number) => `router${r}`,
	perm: (r: number) => `perm${r}`,
	recv: (r: number) => `recv${r}`,
	experts: (r: number) => `exp${r}`,
	unsort: (r: number) => `unsort${r}`,
	out: (r: number) => `out${r}`,
	gather: "gather",
	dispatch: "a2a-dispatch",
	combine: "a2a-combine",
};

/** Chips per tray row. */
export const TRAY_COLS = 4;

/** The code shown under the graph, condensed from Megatron-LM's moe_layer.py and token_dispatcher.py. */
export const CODE: string[] = [
	`def forward(self, hidden_states):  # [${TOKENS_PER_RANK} tokens, hidden] on each of the ${EP} EP ranks`,
	`    probs, routing_map = self.router(hidden_states)  # top-${TOP_K} of ${EXPERTS} experts`,
	`    num_local_tokens_per_expert = routing_map.sum(dim=0)  # [${EXPERTS}]`,
	"    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(axis=1)",
	"    num_global_tokens_per_expert = all_gather(num_local_tokens_per_expert, group=ep_group)",
	"    output_splits = num_global_tokens_per_local_expert.sum(axis=-1)",
	"    tokens, probs, mapping = permute(hidden_states, routing_map, probs)",
	"    tokens = all_to_all(ep_group, tokens, output_splits, input_splits)",
	"    probs = all_to_all(ep_group, probs, output_splits, input_splits)",
	"    tokens = sort_chunks_by_idxs(tokens, num_global_tokens_per_local_expert, sort_input_by_local_experts)",
	"    out = self.experts(tokens, tokens_per_expert, probs)  # grouped GEMM, probs scale the activation",
	"    out = sort_chunks_by_idxs(out, num_global_tokens_per_local_expert.T, restore_output_by_local_experts)",
	"    out = all_to_all(ep_group, out, input_splits, output_splits)",
	`    return unpermute(out, mapping, restore_shape)  # scatter-add each token's ${TOP_K} copies`,
];

const fmt = (xs: number[]) => `[${xs.join(", ")}]`;
const tokName = (c: Copy) => `t${c.t}`;

export function buildSteps(): Step[] {
	const S = simulate();
	const all = (f: (r: number) => string) => range(EP).map(f);
	const grid = (n: string, ids: number[], places: Place[]) =>
		ids.forEach((id, i) => {
			places[id] = {
				node: n,
				col: i % TRAY_COLS,
				row: Math.floor(i / TRAY_COLS),
			};
		});
	const placesFor = (f: (r: number, places: Place[]) => void): Place[] => {
		const places: Place[] = new Array(COPIES.length);
		for (let r = 0; r < EP; r++) f(r, places);
		return places;
	};
	const atTokens = placesFor((r, p) => {
		for (const c of COPIES.filter((c) => c.home === r))
			p[c.id] = { node: node.tok(r), col: c.t % TOKENS_PER_RANK, row: 0 };
	});
	const atRouter = placesFor((r, p) =>
		grid(
			node.router(r),
			COPIES.filter((c) => c.home === r).map((c) => c.id),
			p,
		),
	);
	const atPerm = placesFor((r, p) => grid(node.perm(r), S[r].permuted, p));
	const atRecv = placesFor((r, p) => grid(node.recv(r), S[r].received, p));
	const atSorted = placesFor((r, p) => grid(node.recv(r), S[r].sorted, p));
	const atExperts = placesFor((r, p) => {
		let i = 0;
		S[r].tokensPerExpert.forEach((n, le) => {
			S[r].sorted.slice(i, i + n).forEach((id, j) => {
				p[id] = { node: node.experts(r), col: j, row: le };
			});
			i += n;
		});
	});
	const atUnsort = placesFor((r, p) => grid(node.unsort(r), S[r].unsorted, p));
	const atReturned = placesFor((r, p) => grid(node.out(r), S[r].returned, p));
	const atSummed = placesFor((r, p) => {
		for (const c of COPIES.filter((c) => c.home === r))
			p[c.id] = { node: node.out(r), col: c.t % TOKENS_PER_RANK, row: 0 };
	});

	const r0 = S[0];
	const c = (id: number) => COPIES[id];
	const busiest = S.map((s) => sum(s.outputSplits));
	const maxRank = busiest.indexOf(Math.max(...busiest));
	const minRecv = Math.min(...busiest);
	const rb = S[maxRank];
	const last = EP - 1;

	return [
		{
			line: 0,
			head: `${TOKENS} tokens on ${EP} ranks`,
			body: `Each rank of the EP group holds its own ${TOKENS_PER_RANK} tokens, as in data parallelism: rank 0 has t0 and t1, rank ${last} has t${TOKENS - 2} and t${TOKENS - 1}. Every rank also holds the full router and ${LOCAL} of the ${EXPERTS} experts. Hover over a node for its equation, or an edge for the tensor on it.`,
			active: all(node.tok),
			places: atTokens,
			phase: "tokens",
		},
		{
			line: 1,
			head: "Route",
			body: `Each rank runs the router on its own tokens. Every token picks 2 experts, so it becomes 2 copies. On rank 0, t0 goes to experts ${fmt(ROUTING[0])} and t1 to ${fmt(ROUTING[1])}.`,
			active: all(node.router),
			places: atRouter,
			phase: "copies",
		},
		{
			line: 2,
			head: "Count copies per expert",
			body: `routing_map is a [${TOKENS_PER_RANK}, ${EXPERTS}] boolean mask; summing over its 2 rows counts the copies for each expert. Rank 0 sends ${r0.perExpert
				.map((n, e) => [n, e])
				.filter(([n]) => n > 0)
				.map(([n, e]) => `${n} to expert ${e}`)
				.join(", ")}.`,
			active: all(node.router),
			places: atRouter,
			phase: "copies",
		},
		{
			line: 3,
			head: "input_splits: copies per destination rank",
			body: `Experts 2r and 2r + 1 live on rank r, so adding the counts in pairs gives how many copies go to each rank. Rank 0's input_splits = ${fmt(r0.inputSplits)}: ${r0.inputSplits
				.map((n, j) => [n, j])
				.filter(([n]) => n > 0)
				.map(([n, j]) => `${n} to rank ${j}`)
				.join(", ")}.`,
			active: [node.router(0)],
			places: atRouter,
			phase: "copies",
		},
		{
			line: 4,
			head: "All-gather the counts",
			body: `Before any token moves, each rank must know how much it will receive. An all-gather over the EP group gives every rank the full ${EP} × ${EXPERTS} matrix of counts. It's tiny, but its result must reach the CPU to size the receive buffer: this is the synchronization point.`,
			active: [node.gather],
			places: atRouter,
			phase: "copies",
		},
		{
			line: 5,
			head: "output_splits: copies per source rank",
			body: `Each rank reads the columns of its own ${LOCAL} experts. Rank ${maxRank} holds experts ${maxRank * LOCAL} and ${maxRank * LOCAL + 1}, and its output_splits = ${fmt(rb.outputSplits)}: it will receive ${sum(rb.outputSplits)} copies. Rank 0 will receive ${sum(r0.outputSplits)}.`,
			active: [node.gather],
			places: atRouter,
			phase: "copies",
		},
		{
			line: 6,
			head: "Permute: sort the copies by expert",
			body: `permute gathers the rows in expert order, so the copies for each destination rank are contiguous. Rank 0's buffer is now ${r0.permuted.map((id) => `${tokName(c(id))}→E${c(id).expert}`).join(", ")}. mapping remembers where each row came from.`,
			active: all(node.perm),
			places: atPerm,
			phase: "copies",
		},
		{
			line: 7,
			head: "Dispatch: all-to-all",
			body: `Chunk j of every rank's buffer goes to rank j, so after the exchange each rank holds only copies for its own experts, grouped by sender. The chunks are uneven: rank ${maxRank} receives ${Math.max(...busiest)} copies, and rank ${busiest.indexOf(minRecv)} only ${minRecv}.`,
			active: [node.dispatch],
			places: atRecv,
			phase: "copies",
		},
		{
			line: 8,
			head: "Dispatch the gate weights too",
			body: "A second all-to-all with the same splits sends each copy's gate weight along with it. Megatron multiplies them in inside the experts, not after the combine.",
			active: [node.dispatch],
			places: atRecv,
			phase: "copies",
		},
		{
			line: 9,
			head: "Regroup by local expert",
			body: `The received buffer is ordered by sender, then by expert. sort_chunks_by_idxs reorders the (sender, expert) chunks so that all of expert 2r's rows come first, then expert 2r + 1's. On rank ${maxRank}: ${rb.received.map((id) => `t${c(id).t}`).join(" ")} becomes ${rb.sorted.map((id) => `t${c(id).t}`).join(" ")}.`,
			active: all(node.recv),
			places: atSorted,
			phase: "copies",
		},
		{
			line: 10,
			head: "Run the local experts",
			body: `A grouped GEMM runs each local expert on its own rows: tokens_per_expert is ${fmt(rb.tokensPerExpert)} on rank ${maxRank} and ${fmt(S[0].tokensPerExpert)} on rank 0. The activation is scaled by each row's gate weight. Rank ${maxRank} has the most work, and the layer waits for it.`,
			active: all(node.experts),
			places: atExperts,
			phase: "copies",
		},
		{
			line: 11,
			head: "Undo the regrouping",
			body: "The inverse sort puts the output rows back in (sender, expert) order, so that each sender's chunk is contiguous again.",
			active: all(node.unsort),
			places: atUnsort,
			phase: "copies",
		},
		{
			line: 12,
			head: "Combine: the reverse all-to-all",
			body: "The same exchange with the splits swapped: every output row returns to the rank its token came from, in the order that rank sent it. In the backward pass, this all-to-all and the dispatch swap roles.",
			active: [node.combine],
			places: atReturned,
			phase: "copies",
		},
		{
			line: 13,
			head: "Unpermute: add up each token's copies",
			body: `unpermute scatter-adds the rows into their tokens' positions using mapping, so each token's ${TOP_K} expert outputs, already scaled by their gates, are summed. Each rank ends with ${TOKENS_PER_RANK} tokens, as it started.`,
			active: all(node.out),
			places: atSummed,
			phase: "summed",
		},
	];
}

/** One row of a row-per-token tensor, as a hover card draws it. */
export interface TensorRow {
	/** Token whose (copy of the) hidden vector this row is. */
	t: number;
	/** Colour: the rank that holds the row's expert, or null for a plain token. */
	rank: number | null;
	/** Plain-language note beside the row, e.g. "for expert 2 · gate 0.84". */
	note: string;
}

/** What a node's hover card shows: the tensor at that point of the forward pass. */
export interface TensorView {
	/** Plain-language title, e.g. "Rank 1, after the dispatch". */
	title: string;
	/** The variable's name in the code. */
	code: string;
	shape: string;
	/** The shape in words, e.g. "8 rows × h numbers". */
	shapeWords: string;
	/** What to take away from it. */
	note: string;
	/** A tensor with one row per token (copy), in labelled groups. */
	groups?: { label: string; rows: TensorRow[] }[];
	/** A small matrix drawn as a grid of numbers, shaded by value. */
	grid?: {
		rowAxis: string;
		colAxis: string;
		rowLabels: string[];
		colLabels: string[];
		/** Rank colour of each column (experts are coloured by the rank that holds them). */
		colRanks?: number[];
		values: number[][];
		/** Show 2 decimals (gates) or integers (counts). */
		decimals: number;
		/** Add a row and a column of sums. */
		totals?: boolean;
	};
}

const g2 = (v: number) => v.toFixed(2);
const rowsOf = (n: number) => `${n} row${n === 1 ? "" : "s"}`;

/** The tensor at every node of the graph, by node id. */
export function tensorViews(): Record<string, TensorView> {
	const S = simulate();
	const views: Record<string, TensorView> = {};
	const cp = (id: number) => COPIES[id];
	const expertLabels = range(EXPERTS).map((e) => `E${e}`);
	const expertRanks = range(EXPERTS).map((e) => Math.floor(e / LOCAL));
	const sendMatrix = S.map((s) => s.inputSplits);
	const copyRow = (id: number, note: (c: Copy) => string): TensorRow => ({
		t: cp(id).t,
		rank: cp(id).dest,
		note: note(cp(id)),
	});
	/** Split a buffer into runs of rows that share a key, keeping order. */
	const runs = (ids: number[], key: (c: Copy) => number) => {
		const out: { key: number; ids: number[] }[] = [];
		for (const id of ids) {
			const k = key(cp(id));
			if (out.at(-1)?.key === k) out[out.length - 1].ids.push(id);
			else out.push({ key: k, ids: [id] });
		}
		return out;
	};
	const gateNote = (c: Copy) => `for expert ${c.expert} · gate ${g2(c.gate)}`;
	const outNote = (c: Copy) => `${g2(c.gate)} × E${c.expert}(t${c.t})`;

	for (let r = 0; r < EP; r++) {
		const toks = range(TOKENS_PER_RANK).map((i) => r * TOKENS_PER_RANK + i);
		const s = S[r];
		views[node.tok(r)] = {
			title: `Rank ${r}'s tokens`,
			code: "hidden_states",
			shape: `[${TOKENS_PER_RANK}, h]`,
			shapeWords: `${TOKENS_PER_RANK} rows × h numbers`,
			note: "One row per token: its hidden vector, h numbers long. Every rank starts with its own tokens, as in data parallelism.",
			groups: [
				{
					label: "",
					rows: toks.map((t) => ({ t, rank: null, note: `token ${t}` })),
				},
			],
		};
		views[node.router(r)] = {
			title: `Rank ${r}'s routing decision`,
			code: "probs",
			shape: `[${TOKENS_PER_RANK}, ${EXPERTS}]`,
			shapeWords: "one row per token, one column per expert",
			note: `Each token keeps its top ${TOP_K} experts; their gate weights add up to 1 and every other entry is 0. routing_map marks the same cells as true/false.`,
			grid: {
				rowAxis: "token",
				colAxis: "expert",
				rowLabels: toks.map((t) => `t${t}`),
				colLabels: expertLabels,
				colRanks: expertRanks,
				values: toks.map((t) =>
					range(EXPERTS).map((e) => {
						const k = ROUTING[t].indexOf(e);
						return k < 0 ? 0 : GATES[t][k];
					}),
				),
				decimals: 2,
			},
		};
		views[node.perm(r)] = {
			title: `Rank ${r}'s send buffer`,
			code: "permutated_local_input_tokens",
			shape: `[${s.permuted.length}, h]`,
			shapeWords: `${rowsOf(s.permuted.length)} × h numbers, one per token copy`,
			note: "Each token appears once per chosen expert. The rows are sorted by expert, so all the rows for one rank sit together and can be sent as one chunk.",
			groups: runs(s.permuted, (c) => c.dest).map(({ key, ids }) => ({
				label: `chunk for rank ${key} · ${rowsOf(ids.length)}`,
				rows: ids.map((id) => copyRow(id, gateNote)),
			})),
		};
		views[node.recv(r)] = {
			title: `Rank ${r} after the dispatch`,
			code: "global_input_tokens",
			shape: `[${s.received.length}, h]`,
			shapeWords: `${rowsOf(s.received.length)} × h numbers`,
			note: `Every row is for one of this rank's experts (E${r * LOCAL}, E${r * LOCAL + 1}). They arrive grouped by the rank that sent them.`,
			groups: runs(s.received, (c) => c.home).map(({ key, ids }) => ({
				label: `from rank ${key} · ${rowsOf(ids.length)}`,
				rows: ids.map((id) => copyRow(id, gateNote)),
			})),
		};
		views[node.experts(r)] = {
			title: `Rank ${r}'s experts at work`,
			code: "dispatched_input → expert_output",
			shape: `[${s.sorted.length}, h]`,
			shapeWords: `${rowsOf(s.sorted.length)} in, ${rowsOf(s.sorted.length)} out`,
			note: "Rows are regrouped so each expert gets one contiguous block; one grouped GEMM runs both. Each output row is the expert's output scaled by the gate.",
			groups: runs(s.sorted, (c) => c.expert).map(({ key, ids }) => ({
				label: `expert E${key} · ${rowsOf(ids.length)}`,
				rows: ids.map((id) => copyRow(id, outNote)),
			})),
		};
		views[node.unsort(r)] = {
			title: `Rank ${r}'s results, ready to send back`,
			code: "hidden_states",
			shape: `[${s.unsorted.length}, h]`,
			shapeWords: `${rowsOf(s.unsorted.length)} × h numbers`,
			note: "The expert outputs, put back in the order they arrived, so each chunk can return to the rank it came from.",
			groups: runs(s.unsorted, (c) => c.home).map(({ key, ids }) => ({
				label: `back to rank ${key} · ${rowsOf(ids.length)}`,
				rows: ids.map((id) => copyRow(id, outNote)),
			})),
		};
		views[node.out(r)] = {
			title: `Rank ${r}'s output`,
			code: "output",
			shape: `[${TOKENS_PER_RANK}, h]`,
			shapeWords: `${TOKENS_PER_RANK} rows × h numbers, the same shape as the input`,
			note: `The ${s.returned.length} returned rows are added into their tokens: each token is the gate-weighted sum of its ${TOP_K} experts' outputs.`,
			groups: [
				{
					label: "",
					rows: toks.map((t) => ({
						t,
						rank: null,
						note: ROUTING[t]
							.map((e, k) => `${g2(GATES[t][k])} × E${e}(t${t})`)
							.join(" + "),
					})),
				},
			],
		};
	}
	views[node.gather] = {
		title: "Who sends how many copies to which expert",
		code: "num_global_tokens_per_expert",
		shape: `[${EP}, ${EXPERTS}]`,
		shapeWords: "one row per sending rank, one column per expert",
		note: "Each rank counts its own copies per expert, then an all-gather gives every rank this whole table. A rank's own two columns tell it how many rows it will receive.",
		grid: {
			rowAxis: "sender",
			colAxis: "expert",
			rowLabels: range(EP).map((i) => `rank ${i}`),
			colLabels: expertLabels,
			colRanks: expertRanks,
			values: S.map((s) => s.perExpert),
			decimals: 0,
			totals: true,
		},
	};
	views[node.dispatch] = {
		title: "Copies moved by the dispatch",
		code: "input_splits / output_splits",
		shape: `[${EP}, ${EP}]`,
		shapeWords: "one row per sender, one column per receiver",
		note: "A row is what one rank sends (its input_splits); a column is what one rank receives (its output_splits). The column totals are uneven: that's the load imbalance.",
		grid: {
			rowAxis: "sender",
			colAxis: "receiver",
			rowLabels: range(EP).map((i) => `rank ${i}`),
			colLabels: range(EP).map((j) => `rank ${j}`),
			colRanks: range(EP),
			values: sendMatrix,
			decimals: 0,
			totals: true,
		},
	};
	views[node.combine] = {
		title: "Results moved by the combine",
		code: "output_splits / input_splits",
		shape: `[${EP}, ${EP}]`,
		shapeWords: "one row per sender, one column per receiver",
		note: "The dispatch table flipped: every result row goes back to the rank its token came from.",
		grid: {
			rowAxis: "sender",
			colAxis: "receiver",
			rowLabels: range(EP).map((i) => `rank ${i}`),
			colLabels: range(EP).map((j) => `rank ${j}`),
			colRanks: range(EP),
			values: range(EP).map((i) => range(EP).map((j) => sendMatrix[j][i])),
			decimals: 0,
			totals: true,
		},
	};
	return views;
}
