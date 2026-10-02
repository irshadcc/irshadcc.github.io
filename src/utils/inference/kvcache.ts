// A step-by-step simulation of a serving engine's KV-cache memory under two allocators, used by
// PagedKvCache.astro.
//
// Requests arrive over time with a prompt and an output length the engine doesn't know in
// advance. Every step, each running request adds one token; a request leaves when its output is
// done, and waiting requests are admitted first come, first served while they fit.
//
//   contiguous  each request reserves `maxLen` consecutive slots when admitted (first fit), as
//               serving systems did before PagedAttention
//   paged       memory is cut into blocks of `block` slots, taken one at a time as a request
//               grows; when none is free, the most recently admitted request is preempted and
//               later recomputed from scratch (vLLM's default)

export interface Request {
	id: number;
	arrive: number;
	prompt: number;
	output: number;
}

export type AllocKind = "contiguous" | "paged";

/** One memory slot: which request owns it, and whether it holds a token yet. */
export interface Slot {
	req: number;
	filled: boolean;
}

export interface Frame {
	slots: (Slot | null)[];
	running: number[];
	waiting: number[];
	finished: number;
	preempted: number;
	tokens: number;
	allocated: number;
	/** Block number -> request, for the paged allocator's block tables. */
	tables?: Record<number, number[]>;
}

export interface KvConfig {
	slots: number;
	block: number;
	maxLen: number;
}

export const DEFAULT_TRACE: Request[] = [
	{ id: 0, arrive: 0, prompt: 6, output: 10 },
	{ id: 1, arrive: 0, prompt: 3, output: 20 },
	{ id: 2, arrive: 1, prompt: 10, output: 5 },
	{ id: 3, arrive: 2, prompt: 4, output: 14 },
	{ id: 4, arrive: 4, prompt: 8, output: 8 },
	{ id: 5, arrive: 6, prompt: 2, output: 22 },
	{ id: 6, arrive: 8, prompt: 12, output: 6 },
	{ id: 7, arrive: 10, prompt: 5, output: 12 },
	{ id: 8, arrive: 12, prompt: 7, output: 9 },
	{ id: 9, arrive: 14, prompt: 3, output: 16 },
];

interface Live {
	req: Request;
	/** Tokens whose KV is stored. */
	len: number;
	/** Output tokens generated so far. */
	done: number;
	/** Contiguous: first slot of the reservation. Paged: the block table. */
	start: number;
	blocks: number[];
}

export function simulateKv(
	kind: AllocKind,
	cfg: KvConfig,
	trace: Request[] = DEFAULT_TRACE,
): Frame[] {
	const { slots, block, maxLen } = cfg;
	const nBlocks = Math.floor(slots / block);
	const frames: Frame[] = [];
	const queue: { req: Request; extra: number }[] = []; // extra: tokens to recompute after preemption
	const running: Live[] = [];
	let finished = 0;
	let preempted = 0;
	let freeBlocks = Array.from({ length: nBlocks }, (_, b) => b);
	const owner: (number | null)[] = new Array(slots).fill(null); // contiguous reservations

	const firstFit = (n: number) => {
		for (let s = 0; s + n <= slots; s++) {
			let ok = true;
			for (let i = s; i < s + n; i++)
				if (owner[i] !== null) {
					ok = false;
					s = i;
					break;
				}
			if (ok) return s;
		}
		return -1;
	};

	const snapshot = (): Frame => {
		const view: (Slot | null)[] = new Array(slots).fill(null);
		let tokens = 0;
		let allocated = 0;
		const tables: Record<number, number[]> = {};
		for (const r of running) {
			tokens += r.len;
			if (kind === "contiguous") {
				for (let i = 0; i < maxLen; i++)
					view[r.start + i] = { req: r.req.id, filled: i < r.len };
				allocated += maxLen;
			} else {
				tables[r.req.id] = [...r.blocks];
				r.blocks.forEach((b, j) => {
					for (let i = 0; i < block; i++)
						view[b * block + i] = {
							req: r.req.id,
							filled: j * block + i < r.len,
						};
				});
				allocated += r.blocks.length * block;
			}
		}
		return {
			slots: view,
			running: running.map((r) => r.req.id),
			waiting: queue.map((q) => q.req.id),
			finished,
			preempted,
			tokens,
			allocated,
			tables: kind === "paged" ? tables : undefined,
		};
	};

	const arrivals = [...trace].sort((a, b) => a.arrive - b.arrive);
	for (let t = 0; t < 400; t++) {
		while (arrivals.length && arrivals[0].arrive <= t)
			queue.push({ req: arrivals.shift() as Request, extra: 0 });

		// Admit, first come first served, while the next request fits. Admission runs the prefill,
		// which stores the prompt (plus any tokens to recompute after a preemption).
		while (queue.length) {
			const { req, extra } = queue[0];
			const need = req.prompt + extra;
			if (kind === "contiguous") {
				const s = firstFit(maxLen);
				if (s < 0) break;
				for (let i = s; i < s + maxLen; i++) owner[i] = req.id;
				running.push({ req, len: need, done: extra, start: s, blocks: [] });
			} else {
				const nb = Math.ceil((need + 1) / block); // room for the first new token too
				if (freeBlocks.length < nb) break;
				const blocks = freeBlocks.slice(0, nb);
				freeBlocks = freeBlocks.slice(nb);
				running.push({ req, len: need, done: extra, start: 0, blocks });
			}
			queue.shift();
		}
		frames.push(snapshot());
		if (!running.length && !queue.length && !arrivals.length) break;

		// Decode: every running request generates one token and stores its KV.
		for (const r of [...running]) {
			if (!running.includes(r)) continue; // preempted earlier in this step
			if (kind === "paged" && r.len === r.blocks.length * block) {
				while (!freeBlocks.length) {
					const victim = running.pop() as Live; // the most recently admitted
					freeBlocks.push(...victim.blocks);
					queue.unshift({ req: victim.req, extra: victim.done });
					preempted++;
					if (victim === r) break;
				}
				if (!running.includes(r)) continue;
				r.blocks.push(freeBlocks.shift() as number);
			}
			r.len++;
			r.done++;
		}
		for (const r of [...running])
			if (r.done >= r.req.output) {
				running.splice(running.indexOf(r), 1);
				finished++;
				if (kind === "contiguous") {
					for (let i = r.start; i < r.start + maxLen; i++) owner[i] = null;
				} else {
					freeBlocks.push(...r.blocks);
				}
			}
		freeBlocks.sort((a, b) => a - b);
	}
	return frames;
}

export const REQ_COLORS = [
	"#f4a7a3",
	"#9fc8ef",
	"#a9dba6",
	"#f5d38c",
	"#c8b4ee",
	"#f3b6d6",
	"#9fe0d6",
	"#d9c7a4",
	"#b8d98a",
	"#f0b88a",
];

/** One allocator's memory at one step, as a grid of slots. */
export function drawMemory(
	f: Frame,
	kind: AllocKind,
	cfg: KvConfig,
	cols = 16,
): string {
	const C = 16;
	const G = 2;
	const BG = kind === "paged" ? 4 : 0; // extra gap between blocks
	const perRow = cols;
	const rows = Math.ceil(cfg.slots / perRow);
	const blocksPerRow = perRow / cfg.block;
	const xOf = (i: number) => {
		const c = i % perRow;
		return c * (C + G) + Math.floor(c / cfg.block) * BG;
	};
	const W = perRow * (C + G) - G + (blocksPerRow - 1) * BG;
	const H = rows * (C + G) - G;
	const out: string[] = [];
	f.slots.forEach((s, i) => {
		const x = xOf(i);
		const y = Math.floor(i / perRow) * (C + G);
		if (!s) {
			out.push(
				`<rect class="free" x="${x}" y="${y}" width="${C}" height="${C}" rx="2"><title>Free</title></rect>`,
			);
			return;
		}
		const col = REQ_COLORS[s.req % REQ_COLORS.length];
		const tip = s.filled
			? `Request ${s.req}: a stored token`
			: `Request ${s.req}: reserved, but no token yet`;
		out.push(
			`<g class="slot${s.filled ? "" : " empty"}" data-req="${s.req}"><title>${tip}</title>`,
			`<rect x="${x}" y="${y}" width="${C}" height="${C}" rx="2" style="fill:${col}"/>`,
			s.filled
				? `<text x="${x + C / 2}" y="${y + C / 2 + 3.5}">${s.req}</text>`
				: "",
			"</g>",
		);
	});
	return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="KV-cache memory, ${kind} allocation">${out.join("")}</svg>`;
}
