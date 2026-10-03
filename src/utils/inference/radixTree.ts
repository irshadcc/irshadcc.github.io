// A small model of SGLang's RadixCache with page_size = 1 (the CUDA default), for
// RadixTreeSteps.astro. It follows sglang/srt/mem_cache/radix_cache.py, common.py and
// allocator/token.py:
// - the KV pool is a flat array of token slots; slot 0 is reserved for padded tokens, and the
//   allocator hands out slots from the front of its free list and appends freed ones;
// - each request owns a row of req_to_token: the slot of every one of its tokens;
// - a new request matches its prompt against the tree (at most prompt length - 1 tokens),
//   splitting a node when the match ends inside it, and locks the matched path (lock_ref + 1
//   on every node up to the root) so it cannot be evicted while the request runs;
// - a finished request inserts the tokens whose KV it computed (prompt + all outputs but the
//   last), split at the prompt boundary; the tree keeps one copy of each slot and frees the
//   request's duplicates; then the request unlocks its path;
// - when the pool is short, unlocked leaves are evicted least recently used first, and a parent
//   left childless and unlocked becomes a candidate too.

export interface RNode {
	id: number;
	key: string[];
	value: number[];
	children: RNode[];
	parent?: RNode;
	lock: number;
	access: number;
	owner: string;
}

export interface RSnapNode {
	id: number;
	key: string[];
	value: number[];
	lock: number;
	owner: string;
	depth: number;
	parent: number;
	access: number;
}

export interface RFrame {
	label: string;
	note: string;
	nodes: RSnapNode[];
	/** Owner of every slot: a request id, "tree", "free" or "pad". */
	slots: string[];
	rows: { id: string; slots: number[]; prefix: number }[];
	focus: number[];
	focusSlots: number[];
}

export type REvent =
	| {
			kind: "admit";
			req: string;
			tokens: string[];
			label: string;
			note: string;
	  }
	| {
			kind: "decode";
			req: string;
			tokens: string[];
			label: string;
			note: string;
	  }
	| { kind: "finish"; req: string; label: string; note: string };

interface Live {
	id: string;
	tokens: string[];
	prompt: number;
	row: number[];
	prefix: number;
	last: RNode;
	/** Tokens whose KV has been written (the last sampled token has none yet). */
	computed: number;
}

export class RadixModel {
	root: RNode;
	free: number[];
	live = new Map<string, Live>();
	clock = 0;
	nextId = 1;
	evicted: number[] = [];

	readonly size: number;

	constructor(size: number) {
		this.size = size;
		this.root = {
			id: 0,
			key: [],
			value: [],
			children: [],
			lock: 1,
			access: 0,
			owner: "",
		};
		this.free = Array.from({ length: size - 1 }, (_, i) => i + 1);
	}

	private tick(): number {
		this.clock += 1;
		return this.clock;
	}

	private split(child: RNode, at: number): RNode {
		const parent = child.parent as RNode;
		const top: RNode = {
			id: this.nextId++,
			key: child.key.slice(0, at),
			value: child.value.slice(0, at),
			children: [child],
			parent,
			lock: child.lock,
			access: child.access,
			owner: child.owner,
		};
		parent.children[parent.children.indexOf(child)] = top;
		child.key = child.key.slice(at);
		child.value = child.value.slice(at);
		child.parent = top;
		return top;
	}

	match(key: string[]): { value: number[]; node: RNode } {
		const now = this.tick();
		let node = this.root;
		const value: number[] = [];
		let rest = key;
		while (rest.length) {
			const child = node.children.find((c) => c.key[0] === rest[0]);
			if (!child) break;
			child.access = now;
			let n = 0;
			while (
				n < child.key.length &&
				n < rest.length &&
				child.key[n] === rest[n]
			)
				n++;
			if (n < child.key.length) {
				const top = this.split(child, n);
				value.push(...top.value);
				node = top;
				break;
			}
			value.push(...child.value);
			node = child;
			rest = rest.slice(n);
		}
		return { value, node };
	}

	/** Insert key -> value; returns how many leading tokens were already in the tree. */
	insert(key: string[], value: number[], owner: string): number {
		const now = this.tick();
		let node = this.root;
		let rest = key;
		let vals = value;
		let had = 0;
		while (rest.length) {
			const child = node.children.find((c) => c.key[0] === rest[0]);
			if (!child) {
				const leaf: RNode = {
					id: this.nextId++,
					key: rest,
					value: vals,
					children: [],
					parent: node,
					lock: 0,
					access: now,
					owner,
				};
				node.children.push(leaf);
				return had;
			}
			child.access = now;
			let n = 0;
			while (
				n < child.key.length &&
				n < rest.length &&
				child.key[n] === rest[n]
			)
				n++;
			const target = n < child.key.length ? this.split(child, n) : child;
			had += n;
			node = target;
			rest = rest.slice(n);
			vals = vals.slice(n);
		}
		return had;
	}

	private lockPath(node: RNode, d: number): void {
		for (let x: RNode | undefined = node; x && x !== this.root; x = x.parent)
			x.lock += d;
	}

	private evict(need: number): number[] {
		const out: number[] = [];
		const leaves = (n: RNode): RNode[] =>
			n.children.length
				? n.children.flatMap(leaves)
				: n === this.root
					? []
					: [n];
		while (out.length < need) {
			const cand = leaves(this.root).filter((l) => l.lock === 0);
			if (!cand.length) break;
			cand.sort((a, b) => a.access - b.access);
			const x = cand[0];
			out.push(...x.value);
			this.evicted.push(x.id);
			const p = x.parent as RNode;
			p.children.splice(p.children.indexOf(x), 1);
		}
		this.free.push(...out);
		return out;
	}

	private alloc(n: number): number[] {
		if (n > this.free.length) this.evict(n - this.free.length);
		if (n > this.free.length) throw new Error("out of KV slots");
		return this.free.splice(0, n);
	}

	admit(id: string, tokens: string[]): Live {
		const { value, node } = this.match(tokens.slice(0, tokens.length - 1));
		this.lockPath(node, 1);
		const fresh = this.alloc(tokens.length - value.length);
		const l: Live = {
			id,
			tokens: [...tokens],
			prompt: tokens.length,
			row: [...value, ...fresh],
			prefix: value.length,
			last: node,
			computed: tokens.length,
		};
		this.live.set(id, l);
		return l;
	}

	/** Sample `tokens`: each one but the first gets a slot when it is fed back in. */
	decode(id: string, tokens: string[]): number[] {
		const l = this.live.get(id);
		if (!l) throw new Error(`unknown request ${id}`);
		const fresh: number[] = [];
		for (const t of tokens) {
			if (l.tokens.length > l.computed) {
				const [s] = this.alloc(1);
				l.row.push(s);
				fresh.push(s);
				l.computed++;
			}
			l.tokens.push(t);
		}
		return fresh;
	}

	finish(id: string): { freed: number[] } {
		const l = this.live.get(id);
		if (!l) throw new Error(`unknown request ${id}`);
		const key = l.tokens.slice(0, l.computed);
		const had = this.insert(key, l.row.slice(0, l.computed), id);
		if (l.prompt < key.length)
			this.insert(key.slice(0, l.prompt), l.row.slice(0, l.prompt), id);
		const freed = l.row.slice(l.prefix, had);
		this.free.push(...freed);
		this.lockPath(l.last, -1);
		this.live.delete(id);
		return { freed };
	}

	snapshot(
		label: string,
		note: string,
		focus: number[],
		focusSlots: number[],
	): RFrame {
		const nodes: RSnapNode[] = [];
		const walk = (n: RNode, depth: number) => {
			for (const c of n.children) {
				nodes.push({
					id: c.id,
					key: [...c.key],
					value: [...c.value],
					lock: c.lock,
					owner: c.owner,
					depth,
					parent: n.id,
					access: c.access,
				});
				walk(c, depth + 1);
			}
		};
		walk(this.root, 1);
		const slots = Array.from({ length: this.size }, () => "free");
		slots[0] = "pad";
		for (const n of nodes) for (const s of n.value) slots[s] = "tree";
		for (const l of this.live.values())
			l.row.forEach((s, i) => {
				if (i >= l.prefix) slots[s] = l.id;
			});
		return {
			label,
			note,
			nodes,
			slots,
			rows: [...this.live.values()].map((l) => ({
				id: l.id,
				slots: [...l.row],
				prefix: l.prefix,
			})),
			focus,
			focusSlots,
		};
	}
}

export function runRadix(size: number, events: readonly REvent[]): RFrame[] {
	const m = new RadixModel(size);
	const frames = [
		m.snapshot(
			"start",
			`A pool of ${size} token slots. Slot 0 is reserved for padded tokens in CUDA-graph batches; the tree holds only its root.`,
			[],
			[],
		),
	];
	for (const e of events) {
		const before = new Set<number>();
		const walk = (n: RNode) => {
			before.add(n.id);
			n.children.forEach(walk);
		};
		walk(m.root);
		let slots: number[] = [];
		if (e.kind === "admit") {
			const l = m.admit(e.req, e.tokens);
			slots = l.row.slice(l.prefix);
			frames.push(m.snapshot(e.label, e.note, pathIds(l.last), slots));
			continue;
		}
		if (e.kind === "decode") slots = m.decode(e.req, e.tokens);
		else slots = m.finish(e.req).freed;
		const now: number[] = [];
		const walk2 = (n: RNode) => {
			if (!before.has(n.id)) now.push(n.id);
			n.children.forEach(walk2);
		};
		walk2(m.root);
		frames.push(m.snapshot(e.label, e.note, now, slots));
	}
	return frames;
}

function pathIds(node: RNode): number[] {
	const out: number[] = [];
	for (let x: RNode | undefined = node; x && x.id !== 0; x = x.parent)
		out.push(x.id);
	return out;
}

const SYS = ["<s>", "You", "are", "terse", "."];

/** The scenario used in the SGLang post. */
export const RADIX_EVENTS: REvent[] = [
	{
		kind: "admit",
		req: "A",
		tokens: [...SYS, "Hi", "?"],
		label: "A arrives",
		note: "The tree is empty, so A matches nothing. The allocator gives A's 7 prompt tokens slots 1 to 7, and its req_to_token row records them in order.",
	},
	{
		kind: "decode",
		req: "A",
		tokens: ["Hello", "!"],
		label: "A decodes",
		note: 'Prefill sampled "Hello". The next step feeds "Hello" back in, writes its KV to slot 8 and samples "!", which ends the reply. "!" is never fed back, so it never gets a slot.',
	},
	{
		kind: "finish",
		req: "A",
		label: "A finishes",
		note: "A inserts the 8 tokens that have KV, split at the prompt boundary into a prompt node and an output leaf. The slots now belong to the tree: unlocked, so evictable, but kept for later requests.",
	},
	{
		kind: "admit",
		req: "B",
		tokens: [...SYS, "Bye", "?"],
		label: "B matches",
		note: 'B shares A\'s first 5 tokens. The match ends inside the prompt node, so the node is split into "<s> You are terse ." and "Hi ?". B reuses slots 1 to 5, locks the shared node, and gets only 2 new slots for "Bye ?".',
	},
	{
		kind: "decode",
		req: "B",
		tokens: ["See", "ya"],
		label: "B decodes",
		note: 'B feeds "See" back in (slot 11) and samples "ya".',
	},
	{
		kind: "finish",
		req: "B",
		label: "B finishes",
		note: 'B inserts its 8 tokens. The first 5 are already in the tree, so they cost nothing; "Bye ?" and "See" become new nodes under the shared prefix. Two conversations now share one copy of the system prompt.',
	},
	{
		kind: "admit",
		req: "C",
		tokens: [
			"<s>",
			"Sum",
			"1",
			"+",
			"2",
			"+",
			"3",
			"+",
			"4",
			"+",
			"5",
			"+",
			"6",
			"=",
		],
		label: "C evicts",
		note: 'C shares only "<s>", which is not a whole node, so the root\'s child is split again and C reuses slot 1. It needs 13 new slots but only 12 are free. Eviction removes the least recently used unlocked leaf, A\'s "Hello" (slot 8), which is enough. Slot 8 went to the end of the free list, so it is C\'s last slot.',
	},
];
export const RADIX_SLOTS = 24;

const esc = (s: string) =>
	s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const ranges = (v: number[]) => {
	const out: string[] = [];
	let i = 0;
	while (i < v.length) {
		let j = i;
		while (j + 1 < v.length && v[j + 1] === v[j] + 1) j++;
		out.push(j > i ? `${v[i]}–${v[j]}` : `${v[i]}`);
		i = j + 1;
	}
	return out.join(",");
};

/** One frame: the tree (left to right by depth), the slot pool, then req_to_token rows. */
export function drawRadix(f: RFrame, colors: Record<string, string>): string {
	const out: string[] = [];
	const nodeH = 34;
	const vGap = 10;
	const valueLine = (n: RSnapNode) =>
		`slots ${ranges(n.value)}${n.lock ? ` · lock ${n.lock}` : ""}`;
	const width = (n: RSnapNode) =>
		Math.max(
			64,
			n.key.join(" ").length * 6.9 + 18,
			valueLine(n).length * 5.8 + 18,
		);
	const byId = new Map(f.nodes.map((n) => [n.id, n]));
	const kids = (id: number) => f.nodes.filter((n) => n.parent === id);
	const maxDepth = Math.max(0, ...f.nodes.map((n) => n.depth));
	const colW: number[] = [36];
	for (let d = 1; d <= maxDepth; d++)
		colW[d] = Math.max(0, ...f.nodes.filter((n) => n.depth === d).map(width));
	const colX: number[] = [0];
	for (let d = 1; d <= maxDepth; d++) colX[d] = colX[d - 1] + colW[d - 1] + 22;
	// y: leaves get consecutive rows, a parent sits at the middle of its children.
	const y = new Map<number, number>();
	let row = 0;
	const place = (id: number): number => {
		const ks = kids(id);
		if (!ks.length) {
			const v = row * (nodeH + vGap);
			row++;
			y.set(id, v);
			return v;
		}
		const ys = ks.map((k) => place(k.id));
		const v = (ys[0] + ys[ys.length - 1]) / 2;
		y.set(id, v);
		return v;
	};
	place(0);
	const treeH = Math.max(1, row) * (nodeH + vGap);
	const top = 6;
	const rootY = (y.get(0) ?? 0) + top;
	out.push(
		`<g class="rn root"><rect x="0" y="${rootY}" width="36" height="${nodeH}" rx="6"/><text class="rn-key" x="18" y="${rootY + nodeH / 2 + 4}" text-anchor="middle">root</text></g>`,
	);
	for (const n of f.nodes) {
		const px =
			n.parent === 0
				? 36
				: colX[(byId.get(n.parent) as RSnapNode).depth] +
					width(byId.get(n.parent) as RSnapNode);
		const py = (y.get(n.parent) ?? 0) + top + nodeH / 2;
		const x = colX[n.depth];
		const yy = (y.get(n.id) ?? 0) + top;
		out.push(
			`<path class="re" d="M${px} ${py} C ${px + 11} ${py}, ${x - 11} ${yy + nodeH / 2}, ${x} ${yy + nodeH / 2}"/>`,
		);
	}
	for (const n of f.nodes) {
		const x = colX[n.depth];
		const yy = (y.get(n.id) ?? 0) + top;
		const w = width(n);
		const cls = [
			"rn",
			n.lock > 0 ? "locked" : "",
			f.focus.includes(n.id) ? "focus" : "",
		].join(" ");
		const tip = `Node ${n.id}: tokens "${n.key.join(" ")}", slots ${ranges(n.value)}, lock_ref ${n.lock}${n.lock ? " (in use, cannot be evicted)" : ""}`;
		out.push(
			`<g class="${cls}"><title>${esc(tip)}</title><rect x="${x}" y="${yy}" width="${w}" height="${nodeH}" rx="6" style="fill:${colors[n.owner] ?? "#ccc"}"/>`,
			`<text class="rn-key" x="${x + 8}" y="${yy + 14}">${esc(n.key.join(" "))}</text>`,
			`<text class="rn-val" x="${x + 8}" y="${yy + 27}">${valueLine(n)}</text></g>`,
		);
	}
	const treeW = colX[maxDepth] + (colW[maxDepth] ?? 36);
	// Slot pool.
	const sw = 22;
	let yy = top + treeH + 22;
	out.push(`<text class="rl" x="0" y="${yy}">KV pool slots</text>`);
	yy += 8;
	f.slots.forEach((o, i) => {
		const x = i * (sw + 2);
		const fill = colors[o] ? ` style="fill:${colors[o]}"` : "";
		const cls = [
			"rs",
			o === "tree" || o === "free" || o === "pad" ? o : "req",
			f.focusSlots.includes(i) ? "focus" : "",
		].join(" ");
		const tip =
			o === "pad"
				? "Slot 0: reserved"
				: o === "free"
					? `Slot ${i}: free`
					: o === "tree"
						? `Slot ${i}: owned by the radix tree`
						: `Slot ${i}: written by request ${o}`;
		out.push(
			`<g class="${cls}"><title>${tip}</title><rect x="${x}" y="${yy}" width="${sw}" height="${sw}" rx="3"${fill}/><text x="${x + sw / 2}" y="${yy + 15}">${i}</text></g>`,
		);
	});
	const poolW = f.slots.length * (sw + 2);
	yy += sw + 24;
	out.push(`<text class="rl" x="0" y="${yy}">req_to_token rows</text>`);
	yy += 6;
	f.rows.forEach((r, k) => {
		const ry = yy + k * 26;
		out.push(
			`<rect x="0" y="${ry}" width="22" height="20" rx="4" style="fill:${colors[r.id] ?? "#ccc"}"/><text class="rr-name" x="11" y="${ry + 14}">${esc(r.id)}</text>`,
		);
		r.slots.forEach((s, i) => {
			const x = 30 + i * 24;
			out.push(
				`<g class="rq ${i < r.prefix ? "hit" : ""}"><rect x="${x}" y="${ry}" width="22" height="20" rx="3"/><text x="${x + 11}" y="${ry + 14}">${s}</text></g>`,
			);
		});
	});
	if (!f.rows.length)
		out.push(
			`<text class="rl" x="0" y="${yy + 14}">(no running requests)</text>`,
		);
	const H = yy + Math.max(1, f.rows.length) * 26 + 4;
	const W = Math.max(treeW, poolW, 360);
	return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Radix tree and KV pool">${out.join("")}</svg>`;
}
