// Step-by-step schedules for the classic collectives, used by CollectiveSteps.astro.
//
// Each of the n ranks holds a buffer of M bytes split into n chunks. A schedule is a list of
// synchronous steps; in a step every rank sends at most one message and all messages read the
// state from before the step. The state records, for every chunk slot, which ranks' data it
// holds (for reductions, the set of ranks summed into it).
//
//   broadcast       binomial tree from rank 0, whole buffer per message, ceil(log2 n) steps
//   reduce          binomial tree into rank 0, whole buffer per message, ceil(log2 n) steps
//   reduce-scatter  ring, one chunk per message, n - 1 steps; rank i ends with chunk i summed
//   all-gather      ring, one chunk per message, n - 1 steps; rank i starts with chunk i
//   all-reduce      ring reduce-scatter then ring all-gather, 2(n - 1) steps
//   all-to-all      pairwise exchange, n - 1 steps; chunk j of rank i's send buffer goes to
//                   slot i of rank j's receive buffer
//
// Costs follow the alpha-beta model: a step costs alpha + (largest message) * beta, so the
// schedule's time is (steps) alpha + (sum of per-step largest messages) beta.

export type CollectiveOp =
	| "broadcast"
	| "reduce"
	| "reduce-scatter"
	| "all-gather"
	| "all-reduce"
	| "all-to-all";

export const COLLECTIVE_OPS: readonly CollectiveOp[] = [
	"broadcast",
	"reduce",
	"reduce-scatter",
	"all-gather",
	"all-reduce",
	"all-to-all",
];

/** What a chunk slot holds: the ranks whose data it contains, and for all-to-all its destination. */
export interface Piece {
	src: number[];
	dst?: number;
}

/** state[rank][buffer][chunk] */
export type State = (Piece | null)[][][];

export interface Move {
	fromBuf: number;
	fromChunk: number;
	toBuf: number;
	toChunk: number;
}

export interface Transfer {
	from: number;
	to: number;
	moves: Move[];
	/** The receiver adds the data into its slot instead of overwriting it. */
	reduce: boolean;
}

export interface Step {
	phase: string;
	text: string;
	transfers: Transfer[];
	/** State after the step's transfers. */
	state: State;
	/** Largest message any rank sends this step, in chunks (M / n each). */
	chunks: number;
}

export interface Schedule {
	op: CollectiveOp;
	n: number;
	buffers: string[];
	/** steps[0] is the initial state, with no transfers. */
	steps: Step[];
	/** Cost of the whole schedule as LaTeX, in alpha, beta (and gamma for reductions). */
	cost: string;
}

const clone = (s: State): State =>
	s.map((bufs) =>
		bufs.map((b) => b.map((p) => (p ? { ...p, src: [...p.src] } : null))),
	);

function apply(prev: State, transfers: Transfer[]): State {
	const next = clone(prev);
	for (const t of transfers) {
		for (const m of t.moves) {
			const p = prev[t.from][m.fromBuf][m.fromChunk];
			if (!p)
				throw new Error(
					`collectives: rank ${t.from} sends empty chunk ${m.fromChunk}`,
				);
			const cur = next[t.to][m.toBuf][m.toChunk];
			const src =
				t.reduce && cur
					? [...new Set([...cur.src, ...p.src])].sort((a, b) => a - b)
					: [...p.src];
			next[t.to][m.toBuf][m.toChunk] = { ...p, src };
		}
	}
	return next;
}

const whole = (n: number): Move[] =>
	Array.from({ length: n }, (_, c) => ({
		fromBuf: 0,
		fromChunk: c,
		toBuf: 0,
		toChunk: c,
	}));
const one = (c: number): Move[] => [
	{ fromBuf: 0, fromChunk: c, toBuf: 0, toChunk: c },
];
const mod = (a: number, n: number) => ((a % n) + n) % n;
const log2ceil = (n: number) => Math.ceil(Math.log2(n));

export function buildSchedule(op: CollectiveOp, n: number): Schedule {
	if (!Number.isInteger(n) || n < 2 || n > 8)
		throw new Error(`collectives: n must be 2..8, got ${n}`);
	const ranks = Array.from({ length: n }, (_, r) => r);
	const steps: Step[] = [];
	let state: State;
	const push = (phase: string, text: string, transfers: Transfer[]) => {
		state = apply(state, transfers);
		const chunks = Math.max(0, ...transfers.map((t) => t.moves.length));
		steps.push({ phase, text, transfers, state, chunks });
	};
	const full = (r: number) => Array.from({ length: n }, () => ({ src: [r] }));

	if (op === "broadcast") {
		state = ranks.map((r) => [r === 0 ? full(0) : Array(n).fill(null)]);
		steps.push({
			phase: "start",
			text: "Rank a holds the whole buffer; everyone else is empty.",
			transfers: [],
			state,
			chunks: 0,
		});
		for (let k = 0; k < log2ceil(n); k++) {
			const span = 2 ** k;
			const ts = ranks
				.filter((i) => i < span && i + span < n)
				.map((i) => ({
					from: i,
					to: i + span,
					moves: whole(n),
					reduce: false,
				}));
			push(
				"tree",
				`Every rank that has the data sends the whole buffer ${span} rank${span > 1 ? "s" : ""} to the right. The number of ranks holding it doubles.`,
				ts,
			);
		}
		return {
			op,
			n,
			buffers: [""],
			steps,
			cost: String.raw`\lceil \log_2 N \rceil \,(\alpha + M\beta)`,
		};
	}

	if (op === "reduce") {
		state = ranks.map((r) => [full(r)]);
		steps.push({
			phase: "start",
			text: "Every rank holds its own full buffer.",
			transfers: [],
			state,
			chunks: 0,
		});
		for (let k = 0; k < log2ceil(n); k++) {
			const span = 2 ** k;
			const ts = ranks
				.filter((i) => i % (2 * span) === span)
				.map((i) => ({ from: i, to: i - span, moves: whole(n), reduce: true }));
			push(
				"tree",
				`Ranks pair up ${span} apart; the right one sends its whole partial sum to the left one, which adds it in. Half the ranks drop out.`,
				ts,
			);
		}
		return {
			op,
			n,
			buffers: [""],
			steps,
			cost: String.raw`\lceil \log_2 N \rceil \,(\alpha + M\beta + M\gamma)`,
		};
	}

	const reduceScatter = () => {
		for (let s = 0; s < n - 1; s++) {
			const ts = ranks.map((i) => ({
				from: i,
				to: mod(i + 1, n),
				moves: one(mod(i - s - 1, n)),
				reduce: true,
			}));
			const last = s === n - 2;
			push(
				"reduce-scatter",
				`Every rank sends one chunk to its right neighbour, which adds it to its own copy of that chunk.${last ? " Each chunk now holds the full sum at exactly one rank." : ""}`,
				ts,
			);
		}
	};
	const allGather = () => {
		for (let s = 0; s < n - 1; s++) {
			const ts = ranks.map((i) => ({
				from: i,
				to: mod(i + 1, n),
				moves: one(mod(i - s, n)),
				reduce: false,
			}));
			const last = s === n - 2;
			push(
				"all-gather",
				`Every rank forwards the chunk it received last (or its own, at first) to its right neighbour, which stores it.${last ? " Every rank now has every chunk." : ""}`,
				ts,
			);
		}
	};

	if (op === "reduce-scatter") {
		state = ranks.map((r) => [full(r)]);
		steps.push({
			phase: "start",
			text: "Every rank holds its own full buffer, split into N chunks.",
			transfers: [],
			state,
			chunks: 0,
		});
		reduceScatter();
		return {
			op,
			n,
			buffers: [""],
			steps,
			cost: String.raw`(N-1)\,\alpha + \tfrac{N-1}{N} M\beta + \tfrac{N-1}{N} M\gamma`,
		};
	}

	if (op === "all-gather") {
		state = ranks.map((r) => [
			Array.from({ length: n }, (_, c) => (c === r ? { src: [r] } : null)),
		]);
		steps.push({
			phase: "start",
			text: "Rank i holds only chunk i.",
			transfers: [],
			state,
			chunks: 0,
		});
		allGather();
		return {
			op,
			n,
			buffers: [""],
			steps,
			cost: String.raw`(N-1)\,\alpha + \tfrac{N-1}{N} M\beta`,
		};
	}

	if (op === "all-reduce") {
		state = ranks.map((r) => [full(r)]);
		steps.push({
			phase: "start",
			text: "Every rank holds its own full buffer, split into N chunks.",
			transfers: [],
			state,
			chunks: 0,
		});
		reduceScatter();
		allGather();
		return {
			op,
			n,
			buffers: [""],
			steps,
			cost: String.raw`2(N-1)\,\alpha + 2\tfrac{N-1}{N} M\beta + \tfrac{N-1}{N} M\gamma`,
		};
	}

	// all-to-all: buffer 0 = send, buffer 1 = receive. A rank's own chunk is copied locally.
	state = ranks.map((r) => [
		Array.from({ length: n }, (_, j) => ({ src: [r], dst: j })),
		Array.from({ length: n }, (_, j) =>
			j === r ? { src: [r], dst: r } : null,
		),
	]);
	steps.push({
		phase: "start",
		text: "Chunk j of each rank's send buffer is meant for rank j. Each rank's chunk for itself is copied locally.",
		transfers: [],
		state,
		chunks: 0,
	});
	for (let s = 1; s < n; s++) {
		const ts = ranks.map((i) => {
			const j = mod(i + s, n);
			return {
				from: i,
				to: j,
				moves: [{ fromBuf: 0, fromChunk: j, toBuf: 1, toChunk: i }],
				reduce: false,
			};
		});
		const who =
			s === 1
				? "its right neighbour"
				: `the rank ${s} places to its right (wrapping around)`;
		push(
			"pairwise",
			`Every rank sends the chunk meant for ${who} straight to it.`,
			ts,
		);
	}
	return {
		op,
		n,
		buffers: ["send", "recv"],
		steps,
		cost: String.raw`(N-1)\,\alpha + \tfrac{N-1}{N} M\beta`,
	};
}
