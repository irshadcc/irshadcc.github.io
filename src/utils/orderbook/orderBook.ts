// Logic behind the order book figures: a price-time matching engine (a port of the C++ Engine in
// the order book post), and the step lists that the structure figures walk through (sorted
// vector search, flat-array walk, two-level bitmap search). Drawing lives in orderBookDraw.ts.

export type Side = "buy" | "sell";

export interface LimitMsg {
	kind: "limit";
	id: number;
	side: Side;
	price: number;
	qty: number;
}
export interface CancelMsg {
	kind: "cancel";
	id: number;
}
export type Msg = LimitMsg | CancelMsg;

export interface RestingOrder {
	id: number;
	qty: number;
	side: Side;
}
export interface Trade {
	taker: number;
	maker: number;
	price: number;
	qty: number;
}
export interface BookFrame {
	msg: Msg;
	trades: Trade[];
	/** price -> FIFO queue, oldest first */
	levels: Map<number, RestingOrder[]>;
	bestBid: number | null;
	bestAsk: number | null;
	/** orders touched by this message: filled (fully or partly), added, or cancelled */
	filled: Map<number, number>;
	added: number | null;
	cancelled: (RestingOrder & { price: number }) | null;
	/** orders consumed completely by this message, drawn as ghosts at their old place */
	ghosts: { price: number; order: RestingOrder }[];
	label: string;
	note: string;
}

const cloneLevels = (l: Map<number, RestingOrder[]>) =>
	new Map([...l].map(([p, q]) => [p, q.map((o) => ({ ...o }))]));

function best(levels: Map<number, RestingOrder[]>, side: Side): number | null {
	const ps = [...levels]
		.filter(([, q]) => q.length > 0 && q[0].side === side)
		.map(([p]) => p);
	if (ps.length === 0) return null;
	return side === "buy" ? Math.max(...ps) : Math.min(...ps);
}

/** Price-time priority matching, trades at the resting order's price (as the C++ engine). */
export function runEngine(msgs: Msg[]): BookFrame[] {
	let levels = new Map<number, RestingOrder[]>();
	const where = new Map<number, number>(); // order id -> price
	const frames: BookFrame[] = [];
	for (const msg of msgs) {
		const before = cloneLevels(levels);
		levels = cloneLevels(levels);
		const trades: Trade[] = [];
		const filled = new Map<number, number>();
		const ghosts: BookFrame["ghosts"] = [];
		let added: number | null = null;
		let cancelled: BookFrame["cancelled"] = null;
		if (msg.kind === "limit") {
			let qty = msg.qty;
			const other: Side = msg.side === "buy" ? "sell" : "buy";
			for (;;) {
				const p = best(levels, other);
				if (qty === 0 || p === null) break;
				if (msg.side === "buy" ? p > msg.price : p < msg.price) break;
				const q = levels.get(p) as RestingOrder[];
				while (qty > 0 && q.length > 0) {
					const m = q[0];
					const fill = Math.min(m.qty, qty);
					trades.push({ taker: msg.id, maker: m.id, price: p, qty: fill });
					filled.set(m.id, (filled.get(m.id) ?? 0) + fill);
					m.qty -= fill;
					qty -= fill;
					if (m.qty === 0) {
						const orig = before.get(p)?.find((o) => o.id === m.id);
						ghosts.push({ price: p, order: { ...(orig ?? m) } });
						q.shift();
						where.delete(m.id);
					}
				}
				if (q.length === 0) levels.delete(p);
			}
			if (qty > 0) {
				const q = levels.get(msg.price) ?? [];
				q.push({ id: msg.id, qty, side: msg.side });
				levels.set(msg.price, q);
				where.set(msg.id, msg.price);
				added = msg.id;
			}
		} else {
			const p = where.get(msg.id);
			if (p !== undefined) {
				const q = levels.get(p) as RestingOrder[];
				const i = q.findIndex((o) => o.id === msg.id);
				cancelled = { ...q[i], price: p };
				q.splice(i, 1);
				if (q.length === 0) levels.delete(p);
				where.delete(msg.id);
			}
		}
		frames.push({
			msg,
			trades,
			levels,
			bestBid: best(levels, "buy"),
			bestAsk: best(levels, "sell"),
			filled,
			added,
			cancelled,
			ghosts,
			label: msg.kind === "limit" ? `#${msg.id}` : `cancel #${msg.id}`,
			note: "",
		});
	}
	return frames;
}

/** The worked example of the post (same messages as engine.cpp). */
export const EXAMPLE_MSGS: Msg[] = [
	{ kind: "limit", id: 1, side: "buy", price: 99, qty: 5 },
	{ kind: "limit", id: 2, side: "buy", price: 98, qty: 3 },
	{ kind: "limit", id: 3, side: "sell", price: 101, qty: 4 },
	{ kind: "limit", id: 4, side: "sell", price: 101, qty: 2 },
	{ kind: "limit", id: 5, side: "sell", price: 102, qty: 6 },
	{ kind: "limit", id: 6, side: "buy", price: 99, qty: 2 },
	{ kind: "limit", id: 7, side: "sell", price: 104, qty: 3 },
	{ kind: "cancel", id: 2 },
	{ kind: "limit", id: 8, side: "buy", price: 102, qty: 7 },
	{ kind: "limit", id: 9, side: "sell", price: 98, qty: 8 },
	{ kind: "limit", id: 10, side: "buy", price: 97, qty: 4 },
];

export const EXAMPLE_NOTES: string[] = [
	"Buy 5 at 99 into an empty book: nothing to match, so it rests and 99 becomes the best bid.",
	"Buy 3 at 98 is below the best bid; it opens a new level one tick lower.",
	"Sell 4 at 101. There are no bids at 101 or above, so it rests and opens the ask side. Spread: 101 − 99 = 2 ticks.",
	"Sell 2 at 101 joins the same level behind #3: same price, later time, lower priority.",
	"Sell 6 at 102 opens a second ask level.",
	"Buy 2 at 99 queues behind #1. The level total is now 5 + 2 = 7.",
	"Sell 3 at 104 opens a level two ticks behind 102; 103 stays empty.",
	"Cancel #2. The 98 level had only #2, so the level disappears. It was not the best bid, so the best prices do not move.",
	"Buy 7 at 102 crosses the spread. It takes #3 (4 at 101), then #4 (2 at 101), then 1 of #5's 6 at 102. Three trades, each at the resting order's price; #5 keeps 5.",
	"Sell 8 at 98 takes #1 (5 at 99) and #6 (2 at 99). Nothing bids at 98 any more, so the last 1 rests as an ask at 98, and the best ask drops from 102 to 98.",
	"Buy 4 at 97 is below the best ask of 98, so it rests. Spread: 98 − 97 = 1 tick.",
];

export function exampleFrames(): BookFrame[] {
	const frames = runEngine(EXAMPLE_MSGS);
	frames.forEach((f, i) => {
		f.note = EXAMPLE_NOTES[i] ?? "";
	});
	return frames;
}

// ------------------------------------------------------------------ sorted vector search

export type SearchKind = "linear" | "binary" | "branchless";

export interface SearchStep {
	/** index compared at this step, or null for the final answer */
	probe: number | null;
	/** [first, first + length) still possible, for binary searches */
	range: [number, number];
	result: number | null;
	note: string;
}

/**
 * Steps of finding the first key >= target in an ascending vector, the way each variant in the
 * post does it. Linear scans from the back (where the best price is), binary is std::lower_bound,
 * branchless is Gross's loop with a fixed number of halvings.
 */
export function searchSteps(
	keys: number[],
	target: number,
	kind: SearchKind,
): SearchStep[] {
	const steps: SearchStep[] = [];
	const n = keys.length;
	if (kind === "linear") {
		let it = n;
		while (it > 0) {
			const k = keys[it - 1];
			const go = k >= target;
			steps.push({
				probe: it - 1,
				range: [0, n],
				result: null,
				note: `key[${it - 1}] = ${k} ${go ? "≥" : "<"} ${target}: ${go ? "keep walking towards the front" : "stop"}.`,
			});
			if (!go) break;
			it -= 1;
		}
		steps.push({ probe: null, range: [0, n], result: it, note: "" });
	} else if (kind === "binary") {
		let first = 0;
		let count = n;
		while (count > 0) {
			const step = Math.floor(count / 2);
			const it = first + step;
			const less = keys[it] < target;
			steps.push({
				probe: it,
				range: [first, first + count],
				result: null,
				note: `key[${it}] = ${keys[it]} ${less ? "<" : "≥"} ${target}: ${less ? `answer is after ${it}` : `answer is at or before ${it}`}.`,
			});
			if (less) {
				first = it + 1;
				count -= step + 1;
			} else count = step;
		}
		steps.push({ probe: null, range: [first, first], result: first, note: "" });
	} else {
		let first = 0;
		let length = n;
		while (length > 0) {
			const half = Math.floor(length / 2);
			const it = first + half;
			const less = keys[it] < target;
			steps.push({
				probe: it,
				range: [first, first + length],
				result: null,
				note: `key[${it}] = ${keys[it]} ${less ? "<" : "≥"} ${target}: first += ${less ? length - half : 0} (no branch, just a multiply), length = ${half}.`,
			});
			if (less) first += length - half;
			length = half;
		}
		steps.push({ probe: null, range: [first, first], result: first, note: "" });
	}
	const last = steps[steps.length - 1];
	const r = last.result as number;
	const compares = steps.length - 1;
	const where =
		r < n && keys[r] === target
			? `key[${r}] = ${target}: the level exists, add the order to it`
			: `no level at ${target}: insert it at index ${r}, shifting ${n - r} element${n - r === 1 ? "" : "s"} right`;
	last.note = `Done after ${compares} comparison${compares === 1 ? "" : "s"}: ${where}.`;
	return steps;
}

// ------------------------------------------------------------------ next level in a flat array

export interface WalkStep {
	/** tick indices examined at this step */
	cells: number[];
	/** bitmap words examined (level 0 word index) */
	words: number[];
	/** summary bits examined */
	summary: boolean;
	found: number | null;
	note: string;
}

/** Walk down from `from - 1` to the next occupied tick, one cell per step (the QuantCup walk). */
export function walkSteps(
	occupied: boolean[],
	from: number,
	lo = 0,
): WalkStep[] {
	const steps: WalkStep[] = [];
	let i = from - 1;
	while (i >= 0 && !occupied[i]) {
		steps.push({
			cells: [i],
			words: [],
			summary: false,
			found: null,
			note: `tick ${lo + i} is empty, keep walking.`,
		});
		i -= 1;
	}
	steps.push({
		cells: i >= 0 ? [i] : [],
		words: [],
		summary: false,
		found: i >= 0 ? i : null,
		note:
			i >= 0
				? `tick ${lo + i} has orders: new best after looking at ${from - i} cells.`
				: "no occupied tick: the side is empty.",
	});
	return steps;
}

/**
 * The same search with a two-level bitmap of `wordBits`-bit words (8 in the figure, 64 in the
 * code): mask the current word, else find the highest set summary bit below it, then the highest
 * set bit of that word. At most three word reads whatever the gap.
 */
export function bitmapSteps(
	occupied: boolean[],
	from: number,
	wordBits = 8,
	lo = 0,
): WalkStep[] {
	const steps: WalkStep[] = [];
	const i = from - 1;
	const w = Math.floor(i / wordBits);
	const inWord: number[] = [];
	for (let b = w * wordBits; b <= i; b += 1) inWord.push(b);
	const hitInWord = [...inWord].reverse().find((b) => occupied[b]);
	if (hitInWord !== undefined) {
		steps.push({
			cells: inWord,
			words: [w],
			summary: false,
			found: hitInWord,
			note: `word ${w} masked to ticks ≤ ${lo + i} is non-zero: the highest set bit, tick ${lo + hitInWord}, is the new best.`,
		});
		return steps;
	}
	steps.push({
		cells: inWord,
		words: [w],
		summary: false,
		found: null,
		note: `word ${w} masked to ticks ≤ ${lo + i} is zero, so nothing nearby: go up one level.`,
	});
	const words = Math.ceil(occupied.length / wordBits);
	const nonEmpty = (k: number) =>
		occupied.slice(k * wordBits, (k + 1) * wordBits).some(Boolean);
	let ww = -1;
	for (let k = w - 1; k >= 0; k -= 1)
		if (nonEmpty(k)) {
			ww = k;
			break;
		}
	if (ww < 0) {
		steps.push({
			cells: [],
			words: [],
			summary: true,
			found: null,
			note: "summary has no bit below: the side is empty.",
		});
		return steps;
	}
	steps.push({
		cells: [],
		words: [],
		summary: true,
		found: null,
		note: `summary word masked to words < ${w}: the highest set bit says word ${ww} is the nearest non-empty one (count-leading-zeros, one instruction).`,
	});
	let hit = ww * wordBits;
	for (let b = (ww + 1) * wordBits - 1; b >= ww * wordBits; b -= 1)
		if (occupied[b]) {
			hit = b;
			break;
		}
	const cells: number[] = [];
	for (
		let b = ww * wordBits;
		b < Math.min((ww + 1) * wordBits, words * wordBits);
		b += 1
	)
		cells.push(b);
	steps.push({
		cells,
		words: [ww],
		summary: false,
		found: hit,
		note: `highest set bit of word ${ww} is tick ${lo + hit}: the new best, after 3 word reads instead of ${from - hit} cells.`,
	});
	return steps;
}
