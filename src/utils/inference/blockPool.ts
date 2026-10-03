// A small model of vLLM's KV block pool with automatic prefix caching, for BlockPoolSteps.astro.
// It follows vllm/v1/core/block_pool.py and kv_cache_utils.py:
// - block 0 is the null block and is never handed out;
// - a full block gets a hash of (parent block's hash, its tokens) and goes into a
//   hash -> block map, so a later request with the same prefix finds it;
// - a hit at most covers num_tokens - 1 tokens, so the last prompt token is always recomputed;
// - a hit block is "touched": ref_cnt + 1, and out of the free queue if it was there;
// - new blocks come from the front of the free queue, and a cached block taken this way is
//   evicted from the map;
// - a finished request frees its blocks tail first; a block whose ref_cnt drops to 0 goes to
//   the front of the free queue if it has no hash (reuse first) or to the back if it has one
//   (kept as long as possible, so least recently used cached blocks are evicted first).
// The real pool stores hashes as bytes (SHA-256 by default); here a 16-bit FNV hash keeps the
// labels short.

export interface PoolBlock {
	id: number;
	ref: number;
	hash?: string;
	tokens: string[];
	/** Request that last wrote the block; used for colour. */
	owner?: string;
}

export interface PoolFrame {
	label: string;
	note: string;
	blocks: PoolBlock[];
	/** Free block ids, the next one to hand out first. */
	free: number[];
	/** Block table of each live request. */
	tables: { id: string; blocks: number[]; hit: number }[];
	/** Blocks to highlight in this frame. */
	focus: number[];
}

export type PoolEvent =
	| {
			kind: "admit";
			req: string;
			tokens: string[];
			label: string;
			note?: string;
	  }
	| {
			kind: "decode";
			req: string;
			tokens: string[];
			label: string;
			note?: string;
	  }
	| { kind: "finish"; req: string; label: string; note?: string };

const NONE_HASH = "root";

/** FNV-1a over the text, folded to 4 hex digits. */
export function shortHash(text: string): string {
	let h = 0x811c9dc5;
	for (let i = 0; i < text.length; i++) {
		h ^= text.charCodeAt(i);
		h = Math.imul(h, 0x01000193) >>> 0;
	}
	return ((h ^ (h >>> 16)) & 0xffff).toString(16).padStart(4, "0");
}

export const blockHash = (parent: string | undefined, tokens: string[]) =>
	shortHash(`${parent ?? NONE_HASH}|${tokens.join(" ")}`);

interface Live {
	id: string;
	tokens: string[];
	blocks: number[];
	hit: number;
}

export class BlockPoolModel {
	blocks: PoolBlock[];
	free: number[];
	cached = new Map<string, number>();
	live = new Map<string, Live>();

	readonly blockSize: number;

	constructor(numBlocks: number, blockSize: number) {
		this.blockSize = blockSize;
		this.blocks = Array.from({ length: numBlocks }, (_, id) => ({
			id,
			ref: 0,
			tokens: [],
		}));
		this.free = this.blocks.map((b) => b.id).slice(1); // block 0 is the null block
	}

	/** Hashes of the full blocks of `tokens`, chained from the first block. */
	hashes(tokens: string[]): string[] {
		const out: string[] = [];
		for (let i = 0; i + this.blockSize <= tokens.length; i += this.blockSize)
			out.push(blockHash(out.at(-1), tokens.slice(i, i + this.blockSize)));
		return out;
	}

	private take(): number {
		const id = this.free.shift();
		if (id === undefined) throw new Error("out of KV blocks");
		const b = this.blocks[id];
		if (b.hash) this.cached.delete(b.hash);
		b.hash = undefined;
		b.tokens = [];
		b.ref = 1;
		return id;
	}

	/** Give every full, unhashed block of the request its hash. */
	private cacheFull(l: Live): void {
		const hs = this.hashes(l.tokens);
		hs.forEach((h, i) => {
			const b = this.blocks[l.blocks[i]];
			if (b.hash) return;
			b.hash = h;
			if (!this.cached.has(h)) this.cached.set(h, b.id);
		});
	}

	private write(l: Live, from: number): void {
		for (let i = from; i < l.tokens.length; i++) {
			const b = this.blocks[l.blocks[Math.floor(i / this.blockSize)]];
			b.tokens[i % this.blockSize] = l.tokens[i];
			b.owner = l.id;
		}
	}

	admit(id: string, tokens: string[]): number[] {
		const maxHit = tokens.length - 1;
		const hit: number[] = [];
		for (const h of this.hashes(tokens)) {
			if ((hit.length + 1) * this.blockSize > maxHit) break;
			const bid = this.cached.get(h);
			if (bid === undefined) break;
			hit.push(bid);
		}
		for (const bid of hit) {
			const b = this.blocks[bid];
			if (b.ref === 0) this.free.splice(this.free.indexOf(bid), 1);
			b.ref++;
		}
		const need = Math.ceil(tokens.length / this.blockSize) - hit.length;
		if (need > this.free.length) throw new Error(`cannot admit ${id}`);
		const fresh = Array.from({ length: need }, () => this.take());
		const l: Live = {
			id,
			tokens: [...tokens],
			blocks: [...hit, ...fresh],
			hit: hit.length,
		};
		this.live.set(id, l);
		this.write(l, hit.length * this.blockSize);
		this.cacheFull(l);
		return fresh;
	}

	decode(id: string, tokens: string[]): number[] {
		const l = this.live.get(id);
		if (!l) throw new Error(`unknown request ${id}`);
		const fresh: number[] = [];
		const from = l.tokens.length;
		for (const t of tokens) {
			if (l.tokens.length === l.blocks.length * this.blockSize) {
				const nb = this.take();
				l.blocks.push(nb);
				fresh.push(nb);
			}
			l.tokens.push(t);
		}
		this.write(l, from);
		this.cacheFull(l);
		return fresh;
	}

	finish(id: string): number[] {
		const l = this.live.get(id);
		if (!l) throw new Error(`unknown request ${id}`);
		const front: number[] = [];
		const back: number[] = [];
		for (const bid of [...l.blocks].reverse()) {
			const b = this.blocks[bid];
			b.ref--;
			if (b.ref === 0) (b.hash ? back : front).push(bid);
		}
		this.free = [...front, ...this.free, ...back];
		this.live.delete(id);
		return l.blocks;
	}

	snapshot(label: string, note: string, focus: number[]): PoolFrame {
		return {
			label,
			note,
			focus,
			blocks: this.blocks.map((b) => ({ ...b, tokens: [...b.tokens] })),
			free: [...this.free],
			tables: [...this.live.values()].map((l) => ({
				id: l.id,
				blocks: [...l.blocks],
				hit: l.hit,
			})),
		};
	}
}

export function runPool(
	numBlocks: number,
	blockSize: number,
	events: readonly PoolEvent[],
): PoolFrame[] {
	const m = new BlockPoolModel(numBlocks, blockSize);
	const frames = [
		m.snapshot(
			"start",
			`${numBlocks} blocks of ${blockSize} token slots. Block 0 is the null block; the other ${numBlocks - 1} start in the free queue.`,
			[],
		),
	];
	for (const e of events) {
		let focus: number[] = [];
		if (e.kind === "admit") focus = m.admit(e.req, e.tokens);
		else if (e.kind === "decode") focus = m.decode(e.req, e.tokens);
		else focus = m.finish(e.req);
		if (e.kind === "admit") {
			const l = m.live.get(e.req) as Live;
			focus = [...l.blocks];
		}
		frames.push(m.snapshot(e.label, e.note ?? "", focus));
	}
	return frames;
}

const SYS = ["<s>", "You", "are", "a", "helpful", "bot", ".", "Q:"];

/** The scenario used in the vLLM post: two requests that share an 8-token system prompt. */
export const POOL_EVENTS: PoolEvent[] = [
	{
		kind: "admit",
		req: "A",
		tokens: [...SYS, "what", "is", "2+2", "?"],
		label: "A arrives",
		note: "A's 12 prompt tokens fill three blocks. Nothing is cached yet, so all three come from the front of the free queue, and each full block gets a hash chained from the block before it.",
	},
	{
		kind: "decode",
		req: "A",
		tokens: ["4", "."],
		label: "A decodes",
		note: "The first generated token has no free slot in A's blocks, so A gets block 4. It holds 2 tokens, is not full and so has no hash.",
	},
	{
		kind: "finish",
		req: "A",
		label: "A finishes",
		note: "A's blocks are freed tail first. Block 4 has no hash and nobody can reuse its contents, so it goes to the front of the queue. Blocks 3, 2, 1 keep their hashes and go to the back, the prefix root last: they stay cached until the pool runs out of other blocks.",
	},
	{
		kind: "admit",
		req: "B",
		tokens: [...SYS, "name", "a", "prime", "?"],
		label: "B hits",
		note: "B shares A's first 8 tokens, so the hashes of its first two blocks match blocks 1 and 2. They are touched (ref_cnt 0 → 1, removed from the free queue) and B prefills only its last 4 tokens, into block 4, the first block in the queue.",
	},
	{
		kind: "admit",
		req: "C",
		tokens: [
			"<s>",
			"Translate",
			"to",
			"French",
			":",
			"the",
			"cat",
			"sat",
			"on",
			"the",
			"mat",
			"and",
			"looked",
			"at",
			"the",
			"dog",
			"who",
			"was",
			"asleep",
			"and",
			"dreamt",
			"of",
			"fish",
			".",
		],
		label: "C evicts",
		note: "C shares only the first token with A and B, so it hits nothing and needs 6 new blocks. The queue holds exactly 5, 6, 7, 8, 9, 3. Blocks 5 to 9 are empty; block 3 (A's \"what is 2+2 ?\") is still cached, so handing it to C evicts its hash. A request repeating A's prompt would now hit only 2 blocks.",
	},
	{
		kind: "finish",
		req: "B",
		label: "B finishes",
		note: "B frees blocks 4, 2, 1, tail first. All three are full and hashed, so all three go to the back of the queue in that order: the next allocation would evict B's question (block 4) before the shared prefix (blocks 2 and 1).",
	},
];
export const POOL_BLOCKS = 10;
export const POOL_BLOCK_SIZE = 4;

const esc = (s: string) =>
	s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** One frame: the blocks in a row, then the free queue, then each block table. */
export function drawPool(
	f: PoolFrame,
	colors: Record<string, string>,
	blockSize: number,
): string {
	const bw = 62;
	const bh = 74;
	const gap = 6;
	const n = f.blocks.length;
	const W = Math.max(n * (bw + gap) - gap, 460);
	const out: string[] = [];
	const y0 = 16;
	f.blocks.forEach((b, i) => {
		const x = i * (bw + gap);
		const focus = f.focus.includes(b.id);
		const used = b.ref > 0;
		const cls = [
			"pb",
			b.id === 0 ? "null" : used ? "used" : b.hash ? "cached" : "empty",
			focus ? "focus" : "",
		].join(" ");
		const fill =
			b.id !== 0 && (used || b.hash) && b.owner
				? ` style="fill:${colors[b.owner] ?? "#ccc"}"`
				: "";
		const tip =
			b.id === 0
				? "Block 0: the null block, never allocated"
				: `Block ${b.id}: ref_cnt ${b.ref}${b.hash ? `, hash ${b.hash}` : ", no hash"}${b.tokens.length ? `, tokens: ${b.tokens.join(" ")}` : ""}`;
		out.push(`<g class="${cls}"><title>${esc(tip)}</title>`);
		out.push(
			`<text class="pb-id" x="${x + bw / 2}" y="${y0 - 4}">${b.id}</text>`,
		);
		out.push(
			`<rect x="${x}" y="${y0}" width="${bw}" height="${bh}" rx="5"${fill}/>`,
		);
		if (b.id === 0) {
			out.push(
				`<text class="pb-null" x="${x + bw / 2}" y="${y0 + bh / 2 + 4}">null</text>`,
			);
		} else {
			for (let s = 0; s < blockSize; s++) {
				const t = b.tokens[s];
				out.push(
					`<text class="pb-tok" x="${x + 6}" y="${y0 + 14 + s * 11}">${t ? esc(t.length > 8 ? `${t.slice(0, 7)}…` : t) : "·"}</text>`,
				);
			}
			out.push(
				`<text class="pb-meta" x="${x + 6}" y="${y0 + bh - 6}">r${b.ref} ${b.hash ? `#${b.hash}` : ""}</text>`,
			);
		}
		out.push("</g>");
	});
	let y = y0 + bh + 26;
	out.push(
		`<text class="pb-label" x="0" y="${y}">free queue (next out first)</text>`,
	);
	y += 8;
	f.free.forEach((id, i) => {
		const x = i * 30;
		const b = f.blocks[id];
		out.push(
			`<g class="pq ${b.hash ? "cached" : ""}"><rect x="${x}" y="${y}" width="26" height="20" rx="4"/><text x="${x + 13}" y="${y + 14}">${id}</text></g>`,
		);
	});
	if (!f.free.length)
		out.push(`<text class="pb-label" x="0" y="${y + 14}">(empty)</text>`);
	y += 44;
	out.push(`<text class="pb-label" x="0" y="${y}">block tables</text>`);
	y += 6;
	f.tables.forEach((t, i) => {
		const ty = y + i * 26;
		out.push(
			`<rect class="pt-tag" x="0" y="${ty}" width="26" height="20" rx="4" style="fill:${colors[t.id] ?? "#ccc"}"/><text class="pt-name" x="13" y="${ty + 14}">${esc(t.id)}</text>`,
		);
		t.blocks.forEach((bid, k) => {
			const x = 36 + k * 30;
			out.push(
				`<g class="pq ${k < t.hit ? "hit" : ""}"><rect x="${x}" y="${ty}" width="26" height="20" rx="4"/><text x="${x + 13}" y="${ty + 14}">${bid}</text></g>`,
			);
		});
		if (t.hit)
			out.push(
				`<text class="pb-label" x="${36 + t.blocks.length * 30 + 6}" y="${ty + 14}">${t.hit} block${t.hit > 1 ? "s" : ""} from the cache</text>`,
			);
	});
	const H = y + Math.max(1, f.tables.length) * 26 + 4;
	return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="KV block pool">${out.join("")}</svg>`;
}
