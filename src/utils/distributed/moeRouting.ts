// Routing, dispatch and combine for a toy MoE layer: 8 tokens, 4 experts. Used by MoeRouters
// (which experts each router picks) and MoeDispatchSteps (how the token copies are laid out
// for the experts and summed back). Everything is computed here, so it can be checked with
// `node --experimental-strip-types` against a PyTorch reference.

/** The example batch: a sentence's tokens, their GPT-2 ids and the router's logits (one row per token). */
export const EXAMPLE = {
	words: ["The", "cat", "sat", "on", "the", "mat", ".", "It"],
	ids: [464, 3797, 3332, 319, 262, 2603, 13, 632],
	logits: [
		[2.1, 0.3, 1.6, -0.4],
		[0.2, 1.1, 1.9, 0.0],
		[1.4, -0.2, 1.7, 0.5],
		[-0.3, 0.6, 1.2, 1.4],
		[0.9, 1.8, 0.1, -0.6],
		[0.4, 0.1, 2.2, 0.8],
		[1.3, 0.2, 0.9, -0.1],
		[-0.5, 0.7, 1.5, 1.0],
	],
};

export type Matrix = number[][];

export const softmax = (row: number[]): number[] => {
	const m = Math.max(...row);
	const e = row.map((v) => Math.exp(v - m));
	const s = e.reduce((a, b) => a + b, 0);
	return e.map((v) => v / s);
};

export const sigmoid = (v: number): number => 1 / (1 + Math.exp(-v));

/** Indices of the k largest values, largest first (ties go to the lower index, like torch.topk on CPU). */
export function topk(row: number[], k: number): number[] {
	return row
		.map((v, i) => [v, i] as const)
		.sort((a, b) => b[0] - a[0] || a[1] - b[1])
		.slice(0, k)
		.map(([, i]) => i);
}

/** One router's decision: the scores it shows, and gates[t][e] (0 where token t doesn't use expert e). */
export interface Routing {
	scores: Matrix;
	gates: Matrix;
	/** Expert ids per token, in order of preference. */
	choice: number[][];
	load: number[];
	bias?: number[];
}

function finish(
	scores: Matrix,
	choice: number[][],
	gateOf: (t: number, e: number, rank: number) => number,
	bias?: number[],
): Routing {
	const E = scores[0].length;
	const gates = scores.map(() => new Array<number>(E).fill(0));
	const load = new Array<number>(E).fill(0);
	choice.forEach((es, t) =>
		es.forEach((e, r) => {
			gates[t][e] = gateOf(t, e, r);
			load[e] += 1;
		}),
	);
	return { scores, gates, choice, load, bias };
}

/** Token choice with a softmax: keep the top k probabilities, renormalized to sum to 1 (Mixtral). */
export function softmaxTopK(
	logits: Matrix,
	k: number,
	renormalize = true,
): Routing {
	const p = logits.map(softmax);
	const choice = p.map((row) => topk(row, k));
	return finish(p, choice, (t, e) => {
		const sum = renormalize ? choice[t].reduce((a, j) => a + p[t][j], 0) : 1;
		return p[t][e] / sum;
	});
}

/** DeepSeek-V3: sigmoid affinities; pick the top k of s + b, gate with s normalized over the chosen. */
export function sigmoidBiasTopK(
	logits: Matrix,
	k: number,
	bias: number[],
): Routing {
	const s = logits.map((row) => row.map(sigmoid));
	const choice = s.map((row) =>
		topk(
			row.map((v, e) => v + bias[e]),
			k,
		),
	);
	return finish(
		s,
		choice,
		(t, e) => s[t][e] / choice[t].reduce((a, j) => a + s[t][j], 0),
		bias,
	);
}

/**
 * The auxiliary-loss-free bias update (Wang et al., 2024, Algorithm 1), run on one fixed batch:
 * after each step, b_e += gamma * sign(mean load - load_e). Returns the routing at every step.
 */
export function biasSteps(
	logits: Matrix,
	k: number,
	gamma: number,
	steps: number,
): Routing[] {
	const E = logits[0].length;
	let bias = new Array<number>(E).fill(0);
	const out: Routing[] = [];
	for (let i = 0; i <= steps; i++) {
		const r = sigmoidBiasTopK(logits, k, bias);
		out.push(r);
		const mean = r.load.reduce((a, b) => a + b, 0) / E;
		// Round to keep the printed biases exact multiples of gamma.
		bias = bias.map(
			(b, e) =>
				Math.round((b + gamma * Math.sign(mean - r.load[e])) * 1e9) / 1e9,
		);
	}
	return out;
}

/** Expert choice (Zhou et al., 2022): each expert takes the `capacity` tokens with the highest softmax score. */
export function expertChoice(logits: Matrix, capacity: number): Routing {
	const p = logits.map(softmax);
	const E = p[0].length;
	const choice: number[][] = p.map(() => []);
	for (let e = 0; e < E; e++)
		for (const t of topk(
			p.map((row) => row[e]),
			capacity,
		))
			choice[t].push(e);
	return finish(p, choice, (t, e) => p[t][e]);
}

/** Sinkhorn normalization as in Megatron-LM's moe_utils.sinkhorn: rows and columns of exp(cost) rescaled to equal sums. */
export function sinkhorn(cost: Matrix, tol = 1e-4): Matrix {
	const c = cost.map((row) => row.map(Math.exp));
	const T = c.length;
	const E = c[0].length;
	let d0 = new Array<number>(T).fill(1);
	let d1 = new Array<number>(E).fill(1);
	let d1Old = d1;
	let error = 1e9;
	const eps = 1e-8;
	while (error > tol) {
		d0 = c.map(
			(row) => 1 / T / (row.reduce((a, v, e) => a + d1[e] * v, 0) + eps),
		);
		d1 = d1.map(
			(_, e) => 1 / E / (c.reduce((a, row, t) => a + d0[t] * row[e], 0) + eps),
		);
		error = d1.reduce((a, v, e) => a + Math.abs(d1Old[e] - v), 0) / E;
		d1Old = d1;
	}
	return c.map((row, t) => row.map((v, e) => d1[e] * v * d0[t]));
}

/** Sinkhorn routing (Megatron-LM, k > 1): choose by the balanced matrix, gate with the softmax. */
export function sinkhornTopK(logits: Matrix, k: number): Routing {
	const balanced = sinkhorn(logits).map((row) =>
		row.map((v) => v * logits.length),
	);
	const p = logits.map(softmax);
	const choice = balanced.map((row) => topk(row, k));
	return { ...finish(balanced, choice, (t, e) => p[t][e]), scores: balanced };
}

/** Hash routing (Roller et al., 2021): a fixed function of the token id, here id mod E, with gate 1. */
export function hashRoute(ids: number[], E: number): Routing {
	const scores = ids.map((id) =>
		Array.from({ length: E }, (_, e) => (id % E === e ? 1 : 0)),
	);
	return finish(
		scores,
		ids.map((id) => [id % E]),
		() => 1,
	);
}

/** One token copy: token t sent to expert e as its rank-th choice, with gate g. */
export interface Copy {
	t: number;
	e: number;
	rank: number;
	g: number;
	/** Slot in the expert's buffer, or -1 if dropped. */
	slot: number;
}

export interface DispatchPlan {
	copies: Copy[];
	/** Rows of each expert's buffer: a copy index, or -1 for padding. */
	buffers: number[][];
	/** Copy indices in the order they're laid out for the experts (sorted by expert, dropless). */
	order: number[];
	capacity: number | null;
	dropped: number[];
}

/**
 * Lay the copies out for the experts.
 * With a capacity, GShard-style: first choices take slots before second choices, each in token
 * order (a cumulative sum over the dispatch mask), and copies past the capacity are dropped.
 * Without one (dropless), a stable sort by expert id, as in Megatron-LM's permute.
 */
export function dispatch(r: Routing, capacity: number | null): DispatchPlan {
	const E = r.load.length;
	const copies: Copy[] = [];
	r.choice.forEach((es, t) =>
		es.forEach((e, rank) =>
			copies.push({ t, e, rank, g: r.gates[t][e], slot: -1 }),
		),
	);
	const order = copies
		.map((_, i) => i)
		.sort((a, b) => copies[a].e - copies[b].e || a - b);
	const buffers: number[][] = Array.from({ length: E }, () => []);
	const ranks = Math.max(...r.choice.map((c) => c.length));
	const byPriority =
		capacity === null
			? order
			: [...Array(ranks).keys()].flatMap((k) =>
					copies
						.map((c, i) => [c, i] as const)
						.filter(([c]) => c.rank === k)
						.map(([, i]) => i),
				);
	for (const i of byPriority) {
		const c = copies[i];
		if (capacity === null || buffers[c.e].length < capacity) {
			c.slot = buffers[c.e].length;
			buffers[c.e].push(i);
		}
	}
	if (capacity !== null)
		for (const b of buffers) while (b.length < capacity) b.push(-1);
	const dropped = copies
		.map((c, i) => (c.slot < 0 ? i : -1))
		.filter((i) => i >= 0);
	return { copies, buffers, order, capacity, dropped };
}

/** Expert capacity as in Switch Transformer and Megatron-LM's get_capacity: ceil(T k / E * factor). */
export const capacityOf = (
	tokens: number,
	k: number,
	experts: number,
	factor: number,
): number => Math.ceil(((tokens * k) / experts) * factor);

/** Switch Transformer's load-balancing loss, generalized to top-k as in Megatron-LM: alpha E sum_i f_i P_i. */
export function switchAuxLoss(
	logits: Matrix,
	r: Routing,
	alpha = 1,
): { f: number[]; P: number[]; loss: number } {
	const T = logits.length;
	const p = logits.map(softmax);
	const E = p[0].length;
	const k = r.choice[0].length;
	const f = r.load.map((c) => c / (T * k));
	const P = Array.from(
		{ length: E },
		(_, e) => p.reduce((a, row) => a + row[e], 0) / T,
	);
	return { f, P, loss: alpha * E * f.reduce((a, fi, i) => a + fi * P[i], 0) };
}

/** Load imbalance as in Wang et al. (2024): MaxVio = (max load - mean load) / mean load. */
export const maxVio = (load: number[]): number => {
	const mean = load.reduce((a, b) => a + b, 0) / load.length;
	return (Math.max(...load) - mean) / mean;
};
