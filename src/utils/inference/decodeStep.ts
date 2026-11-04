// One decode step of two requests through the KV cache kernels, on a scaled-down cache (head
// dim 2, one KV head, 4-token pages), for vLLM or SGLang. Requests A and B share a 4-token
// prompt prefix (P0–P3); A has 3 more cached tokens and decodes its 8th (A7); B decodes its 5th
// (B4). vLLM: block table → slot_mapping → reshape_and_cache_flash → FlashAttention reading one
// page per KV tile with online softmax. SGLang (Triton backend): req_to_token → out_cache_loc →
// store_kvcache → _fwd_kernel_stage1 (two KV splits per request, partial output + LSE) →
// _fwd_kernel_stage2 (LSE-weighted merge). Every output is checked against dense softmax
// attention in the test script. Kernels read at vLLM 54beee8 and SGLang 5eb7018.

export type Engine = "vllm" | "sglang";
export const D = 2;
export const PAGE = 4;
export const SCALE = 1 / Math.sqrt(D);

export interface Tok {
	id: string;
	/** P: shared prefix, A or B: the request's own tokens. */
	owner: "P" | "A" | "B";
	pos: number;
	k: number[];
	v: number[];
}

// Seeded values in [-1, 1], one decimal, so the tables stay short.
function lcg(seed: number) {
	let s = seed >>> 0;
	return () => {
		s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
		return Math.round(((s / 2 ** 32) * 2 - 1) * 10) / 10;
	};
}
const rnd = lcg(11);
const vec = () => Array.from({ length: D }, rnd);
const mk = (id: string, owner: Tok["owner"], pos: number): Tok => ({
	id,
	owner,
	pos,
	k: vec(),
	v: vec(),
});

export const TOKENS: Tok[] = [
	mk("P0", "P", 0),
	mk("P1", "P", 1),
	mk("P2", "P", 2),
	mk("P3", "P", 3),
	mk("A4", "A", 4),
	mk("A5", "A", 5),
	mk("A6", "A", 6),
	mk("A7", "A", 7),
	mk("B4", "B", 4),
];
export const tok = (id: string) => {
	const t = TOKENS.find((x) => x.id === id);
	if (!t) throw new Error(id);
	return t;
};
export const QUERY: Record<"A" | "B", number[]> = { A: vec(), B: vec() };
export const SEQ: Record<"A" | "B", string[]> = {
	A: ["P0", "P1", "P2", "P3", "A4", "A5", "A6", "A7"],
	B: ["P0", "P1", "P2", "P3", "B4"],
};
export const NEW: Record<"A" | "B", string> = { A: "A7", B: "B4" };

/** Where each token's K and V live. vLLM: block * 4 + offset; SGLang: a slot id, page size 1. */
export const SLOTS: Record<Engine, Record<string, number>> = {
	vllm: {
		P0: 20,
		P1: 21,
		P2: 22,
		P3: 23,
		A4: 8,
		A5: 9,
		A6: 10,
		A7: 11,
		B4: 24,
	},
	sglang: {
		P0: 5,
		P1: 6,
		P2: 7,
		P3: 8,
		A4: 17,
		A5: 18,
		A6: 19,
		A7: 26,
		B4: 27,
	},
};
export const BLOCK_TABLE: Record<"A" | "B", number[]> = {
	A: [5, 2],
	B: [5, 6],
};
/** Next step's slots (after sampling): vLLM gives A a new block 1; SGLang the next free slots. */
export const NEXT_SLOTS: Record<Engine, Record<"A" | "B", number>> = {
	vllm: { A: 4, B: 25 },
	sglang: { A: 28, B: 29 },
};
export const NUM_SLOTS = 32;

export const dot = (a: number[], b: number[]) =>
	a.reduce((s, x, i) => s + x * b[i], 0);

export function denseAttention(req: "A" | "B") {
	const q = QUERY[req];
	const ks = SEQ[req].map(tok);
	const s = ks.map((t) => dot(q, t.k) * SCALE);
	const m = Math.max(...s);
	const p = s.map((x) => Math.exp(x - m));
	const l = p.reduce((a, b) => a + b, 0);
	const out = Array.from(
		{ length: D },
		(_, d) => ks.reduce((acc, t, i) => acc + p[i] * t.v[d], 0) / l,
	);
	return { out, lse: m + Math.log(l) };
}

/** Online softmax over one tile; returns the state after it and the per-row numbers. */
export interface Online {
	m: number;
	l: number;
	acc: number[];
}
export function onlineTile(q: number[], ids: string[], st: Online) {
	const ts = ids.map(tok);
	const s = ts.map((t) => dot(q, t.k) * SCALE);
	const mNew = Math.max(st.m, ...s);
	const alpha = Math.exp(st.m - mNew);
	const p = s.map((x) => Math.exp(x - mNew));
	const l = st.l * alpha + p.reduce((a, b) => a + b, 0);
	const acc = st.acc.map(
		(a, d) => a * alpha + ts.reduce((x, t, i) => x + p[i] * t.v[d], 0),
	);
	return { s, p, alpha, next: { m: mNew, l, acc } as Online };
}

// ---------------------------------------------------------------- steps

export type Cell = number | string;
export interface Table {
	name: string;
	sub?: string;
	cols: string[];
	rows: { head: string; cells: Cell[]; hl?: boolean }[];
}
export interface DStep {
	label: string;
	/** Index into STAGES[engine]. */
	stage: number;
	/** Code lines to highlight (indices into CODE[engine]). */
	lines: number[];
	/** Tokens present in the cache. */
	cached: string[];
	/** Tokens written in this step (outlined). */
	written: string[];
	/** Tokens read by the current tile (thick outline). */
	read: string[];
	/** Request whose index row is highlighted. */
	req?: "A" | "B";
	/** Entries of that row being used (indices into its block table / req_to_token row). */
	cols?: number[];
	/** Show the next step's slots. */
	next?: boolean;
	tables: Table[];
	note: string;
}

export const STAGES: Record<Engine, { name: string; kernel: string }[]> = {
	vllm: [
		{ name: "schedule", kernel: "KV cache manager, CPU" },
		{ name: "slot ids", kernel: "Triton slot kernel" },
		{ name: "q, k, v", kernel: "qkv_proj + RoPE" },
		{ name: "KV write", kernel: "reshape_and_cache_flash" },
		{ name: "attention", kernel: "FlashAttention, paged" },
		{ name: "sample, append", kernel: "lm_head, sampler" },
	],
	sglang: [
		{ name: "schedule", kernel: "radix tree, allocator" },
		{ name: "slot ids", kernel: "kv_indices" },
		{ name: "q, k, v", kernel: "qkv_proj + RoPE" },
		{ name: "KV write", kernel: "store_kvcache_kernel" },
		{ name: "attention", kernel: "stage 1, split KV" },
		{ name: "merge", kernel: "stage 2" },
		{ name: "sample, append", kernel: "lm_head, sampler" },
	],
};

export const CODE: Record<Engine, string[]> = {
	vllm: [
		"# _compute_slot_mappings_kernel: one program per request",
		"slot = block_table[req, pos // B] * B + pos % B",
		"# reshape_and_cache_flash: one thread block per new token",
		"blk, off = slot_mapping[t] // B, slot_mapping[t] % B",
		"k_cache[blk, off], v_cache[blk, off] = key[t], value[t]",
		"# FlashAttention, paged (condensed): per (request, KV head)",
		"m, l, acc = -inf, 0, 0",
		"for page in range(cdiv(seqused_k[req], B)):",
		"    blk = block_table[req, page]        # logical -> physical",
		"    K, V = k_cache[blk], v_cache[blk]   # masked past seqused_k",
		"    s = (q @ K.T) * softmax_scale",
		"    m_new = max(m, max(s)); p = exp(s - m_new)",
		"    l = l * exp(m - m_new) + sum(p)",
		"    acc = acc * exp(m - m_new) + p @ V; m = m_new",
		"out[req] = acc / l",
	],
	sglang: [
		"# scheduler: radix match + allocator, then",
		"req_to_token[req, seq_len - 1] = out_cache_loc[i]",
		"kv_indices = concat(req_to_token[req, :seq_len])  # kv_indptr",
		"# store_kvcache_kernel: one warp group per new token",
		"k_buffer[out_cache_loc[i]], v_buffer[...] = k[i], v[i]",
		"# _fwd_kernel_stage1: grid (batch, head, split)",
		"per = cdiv(cdiv(seq_len, splits), MIN_BLOCK_KV) * MIN_BLOCK_KV",
		"for n in range(per * split, min(per * (split + 1), seq_len), BLOCK_N):",
		"    loc = kv_indices[kv_indptr[b] + n]  # slot ids",
		"    k, v = K_Buffer[loc], V_Buffer[loc]",
		"    qk = (q · k) * sm_scale",
		"    n_e_max = max(max(qk), e_max); re = exp(e_max - n_e_max)",
		"    p = exp(qk - n_e_max); acc = acc * re + p @ v",
		"    e_sum = e_sum * re + sum(p); e_max = n_e_max",
		"Att_Out[b, h, split] = acc / e_sum; Att_Lse[...] = e_max + log(e_sum)",
		"# _fwd_kernel_stage2: grid (batch, head)",
		"for split: w = exp(lse[split] - max_lse); acc += w * Att_Out[split]",
		"O[b, h] = acc / sum(w)",
	],
};

const r3 = (x: number) => Math.round(x * 1000) / 1000;
const v2 = (a: number[]) => a.map(r3).join(", ");
const PREFIX = ["P0", "P1", "P2", "P3"];
const CACHED0 = [...PREFIX, "A4", "A5", "A6"];
const ALL = [...CACHED0, "A7", "B4"];

function qkvTable(): Table[] {
	return [
		{
			name: "new tokens",
			sub: "layer 0, after RoPE",
			cols: ["q", "k", "v"],
			rows: (["A", "B"] as const).map((r) => ({
				head: NEW[r],
				cells: [v2(QUERY[r]), v2(tok(NEW[r]).k), v2(tok(NEW[r]).v)],
			})),
		},
	];
}

function tileTable(
	engine: Engine,
	req: "A" | "B",
	ids: string[],
	before: Online,
	t: ReturnType<typeof onlineTile>,
	masked: number,
): Table[] {
	const rows = ids.map((id, i) => ({
		head: `${id} (pos ${tok(id).pos})`,
		cells: [
			SLOTS[engine][id],
			v2(tok(id).k),
			v2(tok(id).v),
			r3(t.s[i]),
			r3(t.p[i]),
		] as Cell[],
		hl: id === NEW[req],
	}));
	for (let i = 0; i < masked; i++)
		rows.push({
			head: "(masked)",
			cells: ["–", "–", "–", "−∞", 0],
			hl: false,
		});
	return [
		{
			name: `tile rows, q = (${v2(QUERY[req])})`,
			sub: "s = q·k/√2, p = exp(s − m_new)",
			cols: ["slot", "k", "v", "s", "p"],
			rows,
		},
		{
			name: "state",
			sub: "before → after",
			cols: ["before", "after"],
			rows: [
				{
					head: "m",
					cells: [
						before.m === Number.NEGATIVE_INFINITY ? "−∞" : r3(before.m),
						r3(t.next.m),
					],
				},
				{ head: "exp(m − m_new)", cells: ["", r3(t.alpha)] },
				{ head: "l", cells: [r3(before.l), r3(t.next.l)] },
				{ head: "acc", cells: [v2(before.acc), v2(t.next.acc)] },
			],
		},
	];
}

const fresh = (): Online => ({
	m: Number.NEGATIVE_INFINITY,
	l: 0,
	acc: Array(D).fill(0),
});

function vllmSteps(): DStep[] {
	const steps: DStep[] = [];
	const base = { written: [] as string[], read: [] as string[] };
	steps.push({
		...base,
		label: "schedule",
		stage: 0,
		lines: [],
		cached: CACHED0,
		tables: [
			{
				name: "requests",
				cols: ["cached", "new pos", "block table"],
				rows: [
					{ head: "A", cells: [7, 7, "[5, 2]"] },
					{ head: "B", cells: [4, 4, "[5, 6]"] },
				],
			},
		],
		note: "The scheduler gives each running request one new token. A's 7 cached tokens fill block 5 and three slots of block 2. B's prompt is the same first 4 tokens, so block hashing found block 5 already cached: B's block table starts with the same physical block (ref count 2). B's 5th token needs a fresh block, and the free queue hands out block 6. Block 0 is vLLM's reserved null block.",
	});
	steps.push({
		...base,
		label: "slot_mapping",
		stage: 1,
		lines: [1],
		cached: CACHED0,
		req: "A",
		cols: [1],
		tables: [
			{
				name: "slot_mapping",
				sub: "B = 4",
				cols: ["pos", "pos // 4", "block", "pos % 4", "slot"],
				rows: [
					{ head: "A7", cells: [7, 1, 2, 3, 11] },
					{ head: "B4", cells: [4, 1, 6, 0, 24] },
				],
			},
		],
		note: "One Triton program per request turns positions into slots: A's position 7 is entry 1 of its block table (block 2), offset 3, slot 2·4 + 3 = 11. B's position 4 is entry 1 (block 6), offset 0, slot 24. A CUDA-graph batch padded past 2 tokens would get slot −1 in the extra entries.",
	});
	steps.push({
		...base,
		label: "q, k, v",
		stage: 2,
		lines: [],
		cached: CACHED0,
		tables: qkvTable(),
		note: "Each layer's qkv_proj and RoPE produce one query, key and value per new token. Only the two new tokens are computed; the 7 + 4 earlier tokens' keys and values are read from the cache.",
	});
	steps.push({
		...base,
		label: "write",
		stage: 3,
		lines: [3, 4],
		cached: ALL,
		written: ["A7", "B4"],
		tables: [
			{
				name: "reshape_and_cache_flash",
				cols: ["slot", "block", "offset", "k", "v"],
				rows: (["A7", "B4"] as const).map((id) => ({
					head: id,
					cells: [
						SLOTS.vllm[id],
						Math.floor(SLOTS.vllm[id] / PAGE),
						SLOTS.vllm[id] % PAGE,
						v2(tok(id).k),
						v2(tok(id).v),
					],
				})),
			},
		],
		note: "One thread block per token scatters k and v into the cache at its slot (outlined). Writing happens before attention, so the new token attends to itself. The real kernel copies 16-byte vectors per head, as the write map above shows.",
	});
	for (const req of ["A", "B"] as const) {
		let st = fresh();
		const seq = SEQ[req];
		for (let page = 0; page < BLOCK_TABLE[req].length; page++) {
			const ids = seq.slice(page * PAGE, page * PAGE + PAGE);
			const t = onlineTile(QUERY[req], ids, st);
			const blk = BLOCK_TABLE[req][page];
			steps.push({
				label: `${req}: page ${page}`,
				stage: 4,
				lines: [7, 8, 9, 10, 11, 12, 13],
				cached: ALL,
				written: [],
				read: ids,
				req,
				cols: [page],
				tables: tileTable("vllm", req, ids, st, t, PAGE - ids.length),
				note:
					page === 0
						? `The program for request ${req} reads block_table[${req}][0] = ${blk} and loads that page's 4 keys and values. ${req === "B" ? "It is the same physical page A just read: the shared prefix is stored once. " : ""}Scores, then the running max m, the normaliser l and the unnormalised output acc start from this tile.`
						: `Logical page 1 is physical block ${blk}. ${ids.length < PAGE ? `seqused_k = ${seq.length}, so only ${ids.length} of the page's 4 slots are valid; the rest are masked to −∞. ` : ""}${t.alpha === 1 ? "No score in this tile beats the running max, so exp(m − m_new) = 1 and the tile's terms are simply added to l and acc." : `The new max rescales the old l and acc by exp(m − m_new) = ${r3(t.alpha)} before this tile's terms are added.`}`,
			});
			st = t.next;
		}
		const ref = denseAttention(req);
		steps.push({
			label: `${req}: output`,
			stage: 4,
			lines: [14],
			cached: ALL,
			written: [],
			read: [],
			req,
			tables: [
				{
					name: `request ${req}`,
					cols: ["value"],
					rows: [
						{ head: "out = acc / l", cells: [v2(st.acc.map((x) => x / st.l))] },
						{ head: "dense softmax", cells: [v2(ref.out)] },
					],
				},
			],
			note: `Dividing by l once at the end gives request ${req}'s attention output; it matches dense softmax attention over all ${seq.length} tokens. In FlashAttention 3 a KV tile is over 100 tokens and spans several 16-token pages; here one tile is one 4-token page.`,
		});
	}
	steps.push({
		label: "next step",
		stage: 5,
		lines: [],
		cached: ALL,
		written: [],
		read: [],
		next: true,
		tables: [
			{
				name: "next decode step",
				cols: ["pos", "block table", "slot"],
				rows: [
					{ head: "A", cells: [8, "[5, 2, 1]", 4] },
					{ head: "B", cells: [5, "[5, 6]", 25] },
				],
			},
		],
		note: "The attention output goes on through o_proj, the MLP and the remaining layers; the sampler picks one token per request. Next step A's position 8 starts a new logical page, so the manager appends free block 1 to its table (slot 4); B's position 5 goes to block 6, offset 1 (slot 25). Nothing is copied: the cache only grows.",
	});
	return steps;
}

function sglangSteps(): DStep[] {
	const steps: DStep[] = [];
	const base = { written: [] as string[], read: [] as string[] };
	const SPLITS = 2;
	const MIN_BLOCK_KV = 4;
	steps.push({
		...base,
		label: "schedule",
		stage: 0,
		lines: [0],
		cached: CACHED0,
		tables: [
			{
				name: "requests",
				cols: ["cached", "new pos", "out_cache_loc"],
				rows: [
					{ head: "A", cells: [7, 7, 26] },
					{ head: "B", cells: [4, 4, 27] },
				],
			},
		],
		note: "A and B share the radix tree node for P0–P3, whose KV lives in slots 5–8; the tree hands both requests the same slot tensor. The allocator gives each new token one free slot (page size 1): 26 and 27. Slot 0 is reserved for CUDA-graph padding.",
	});
	steps.push({
		...base,
		label: "kv_indices",
		stage: 1,
		lines: [1, 2],
		cached: CACHED0,
		req: "A",
		cols: [7],
		tables: [
			{
				name: "Triton backend metadata",
				cols: ["value"],
				rows: [
					{ head: "kv_indptr", cells: ["[0, 8, 13]"] },
					{
						head: "kv_indices",
						cells: [
							`[${[...SEQ.A, ...SEQ.B].map((id) => SLOTS.sglang[id]).join(", ")}]`,
						],
					},
				],
			},
		],
		note: "The new slots are written at the end of each request's req_to_token row. The Triton backend then gathers the rows into one flat kv_indices array (pack_read_stream), with kv_indptr = cumsum(seq_lens) marking where each request starts: A owns entries 0–7, B entries 8–12. Slots 5–8 appear twice, once per request.",
	});
	steps.push({
		...base,
		label: "q, k, v",
		stage: 2,
		lines: [],
		cached: CACHED0,
		tables: qkvTable(),
		note: "Each layer's qkv_proj and RoPE produce one query, key and value per new token, exactly as in vLLM: the model code does not know which engine runs it.",
	});
	steps.push({
		...base,
		label: "write",
		stage: 3,
		lines: [4],
		cached: ALL,
		written: ["A7", "B4"],
		tables: [
			{
				name: "store_kvcache",
				cols: ["slot", "k → k_buffer", "v → v_buffer"],
				rows: (["A7", "B4"] as const).map((id) => ({
					head: id,
					cells: [SLOTS.sglang[id], v2(tok(id).k), v2(tok(id).v)],
				})),
			},
		],
		note: "store_kvcache_kernel copies each token's k into row out_cache_loc[i] of the layer's k_buffer and v into the same row of v_buffer (outlined).",
	});
	const partial: Record<string, { o: number[]; lse: number }[]> = {};
	for (const req of ["A", "B"] as const) {
		const seq = SEQ[req];
		const per =
			Math.ceil(Math.ceil(seq.length / SPLITS) / MIN_BLOCK_KV) * MIN_BLOCK_KV;
		partial[req] = [];
		for (let sp = 0; sp < SPLITS; sp++) {
			const lo = per * sp;
			const hi = Math.min(lo + per, seq.length);
			if (hi <= lo) continue;
			const ids = seq.slice(lo, hi);
			const before = fresh();
			const t = onlineTile(QUERY[req], ids, before);
			const o = t.next.acc.map((x) => x / t.next.l);
			const lse = t.next.m + Math.log(t.next.l);
			partial[req].push({ o, lse });
			const tbl = tileTable("sglang", req, ids, before, t, PAGE - ids.length);
			tbl[1].rows.push(
				{ head: "Att_Out = acc / l", cells: ["", v2(o)] },
				{ head: "Att_Lse = m + ln l", cells: ["", r3(lse)] },
			);
			steps.push({
				label: `${req}: split ${sp}`,
				stage: 4,
				lines: [7, 8, 9, 10, 11, 12, 13, 14],
				cached: ALL,
				written: [],
				read: ids,
				req,
				cols: ids.map((_, i) => lo + i),
				tables: tbl,
				note:
					sp === 0
						? `Program (${req}, head 0, split 0) runs in parallel with split 1. With 2 splits, per = cdiv(cdiv(${seq.length}, 2), 4)·4 = ${per}, so split 0 covers entries 0–${hi - 1} of ${req}'s kv_indices${req === "B" ? ": the same slots 5–8 that A's split 0 reads" : ""}. It keeps its own m, l, acc and writes a partial output and its log-sum-exp.`
						: `Split 1 covers entries ${lo}–${hi - 1}: ${ids.map((id) => `slot ${SLOTS.sglang[id]}`).join(", ")}${ids.length < PAGE ? `; the other ${PAGE - ids.length} lanes of the BLOCK_N tile are masked` : ""}. Its state starts fresh: nothing flows between splits until stage 2.`,
			});
		}
	}
	const mergeRows: Table["rows"] = [];
	for (const req of ["A", "B"] as const) {
		const ps = partial[req];
		const M = Math.max(...ps.map((p) => p.lse));
		const w = ps.map((p) => Math.exp(p.lse - M));
		const W = w.reduce((a, b) => a + b, 0);
		const out = Array.from(
			{ length: D },
			(_, d) => ps.reduce((a, p, i) => a + w[i] * p.o[d], 0) / W,
		);
		ps.forEach((p, i) =>
			mergeRows.push({
				head: `${req} split ${i}`,
				cells: [r3(p.lse), r3(w[i]), v2(p.o), ""],
			}),
		);
		mergeRows.push({
			head: `${req} merged`,
			cells: ["", r3(W), v2(out), v2(denseAttention(req).out)],
			hl: true,
		});
	}
	steps.push({
		label: "stage 2 merge",
		stage: 5,
		lines: [16, 17],
		cached: ALL,
		written: [],
		read: [],
		tables: [
			{
				name: "_fwd_kernel_stage2",
				sub: "w = exp(lse − max lse)",
				cols: ["lse", "w", "Att_Out", "dense softmax"],
				rows: mergeRows,
			},
		],
		note: "One program per (request, head) combines the splits: each partial output is weighted by exp(lse − max lse), which is its share of the full softmax denominator. The merged outputs equal dense attention. Splitting the KV this way keeps the GPU busy when a decode batch has few requests but long contexts.",
	});
	steps.push({
		label: "next step",
		stage: 6,
		lines: [],
		cached: ALL,
		written: [],
		read: [],
		next: true,
		tables: [
			{
				name: "next decode step",
				cols: ["pos", "slot"],
				rows: [
					{ head: "A", cells: [8, 28] },
					{ head: "B", cells: [5, 29] },
				],
			},
		],
		note: "After o_proj, the MLP, the remaining layers and sampling, each request gets one more free slot (28 and 29) at the end of its req_to_token row. When a request finishes, its slots are inserted into the radix tree so a later request with the same prefix can match them.",
	});
	return steps;
}

export const buildSteps = (engine: Engine) =>
	engine === "vllm" ? vllmSteps() : sglangSteps();

// ---------------------------------------------------------------- drawing

const C = 30;
const G = 3;
const ownerOf = (engine: Engine) => {
	const m = new Map<number, string>();
	for (const [id, s] of Object.entries(SLOTS[engine])) m.set(s, id);
	return m;
};

/** SVG: the KV pool (left) and the index tables (right) for one step. */
export function drawStep(engine: Engine, s: DStep): string {
	const out: string[] = [];
	const owner = ownerOf(engine);
	const cached = new Set(s.cached);
	const written = new Set(s.written);
	const read = new Set(s.read);
	const cols = engine === "vllm" ? PAGE : 8;
	const rows = NUM_SLOTS / cols;
	const L = engine === "vllm" ? 52 : 28;
	const T = 18;
	const poolW = L + cols * (C + G);
	const ix = poolW + 26;
	const idxW = engine === "vllm" ? 190 : 330;
	const W = ix + idxW;
	const H = Math.max(T + rows * (C + G) + 18, 230);
	out.push(
		`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="KV cache slots and index tables">`,
	);
	out.push(
		`<text class="ds-h" x="${L}" y="${T - 6}">${engine === "vllm" ? "KV cache: 8 blocks × 4 slots" : "k_buffer / v_buffer: 32 slots"}</text>`,
	);
	const nextSlots = new Set(s.next ? Object.values(NEXT_SLOTS[engine]) : []);
	for (let r = 0; r < rows; r++) {
		if (engine === "vllm")
			out.push(
				`<text class="ds-row" x="${L - 6}" y="${T + r * (C + G) + C / 2 + 3}">block ${r}</text>`,
			);
		else
			out.push(
				`<text class="ds-row" x="${L - 6}" y="${T + r * (C + G) + C / 2 + 3}">${r * cols}</text>`,
			);
		for (let c = 0; c < cols; c++) {
			const slot = r * cols + c;
			const x = L + c * (C + G);
			const y = T + r * (C + G);
			const id = owner.get(slot);
			const reserved =
				(engine === "vllm" && r === 0) || (engine === "sglang" && slot === 0);
			let cls = "ds-c";
			let lab = "";
			if (id && cached.has(id)) {
				cls += ` own-${tok(id).owner}`;
				lab = id;
			} else if (reserved) {
				cls += " rsv";
				lab = engine === "vllm" ? (c === 0 ? "null" : "") : "pad";
			}
			if (id && written.has(id)) cls += " wr";
			if (id && read.has(id)) cls += " rd";
			if (nextSlots.has(slot)) {
				cls += " nx";
				lab = slot === NEXT_SLOTS[engine].A ? "A8" : "B5";
			}
			out.push(
				`<g class="${cls}"><title>slot ${slot}${id && cached.has(id) ? `: ${id}` : ""}</title><rect x="${x}" y="${y}" width="${C}" height="${C}" rx="4"/><text class="ds-lab" x="${x + C / 2}" y="${y + C / 2 + 1}">${lab}</text><text class="ds-slot" x="${x + C - 3}" y="${y + 8}">${slot}</text></g>`,
			);
		}
	}
	// Index tables.
	const reqs = ["A", "B"] as const;
	const cw = engine === "vllm" ? 34 : 26;
	out.push(
		`<text class="ds-h" x="${ix}" y="${T - 6}">${engine === "vllm" ? "block_table" : "req_to_token rows"}</text>`,
	);
	reqs.forEach((req, ri) => {
		const y = T + ri * (C + 14);
		const entries =
			engine === "vllm"
				? [...BLOCK_TABLE[req], ...(s.next && req === "A" ? [1] : [])]
				: [
						...SEQ[req].map((id) => SLOTS.sglang[id]),
						...(s.next ? [NEXT_SLOTS.sglang[req]] : []),
					];
		const known =
			engine === "vllm"
				? entries.length
				: s.stage === 0
					? SEQ[req].length - 1
					: entries.length;
		out.push(
			`<text class="ds-row" x="${ix + 12}" y="${y + C / 2 + 3}">${req}</text>`,
		);
		entries.forEach((e, i) => {
			if (i >= known) return;
			const x = ix + 18 + i * (cw + 2);
			const on = s.req === req && (s.cols ?? []).includes(i);
			const isNew =
				(s.next &&
					i === entries.length - 1 &&
					(engine === "sglang" || req === "A")) ||
				false;
			out.push(
				`<g class="ds-ix${on ? " on" : ""}${isNew ? " nx" : ""}"><rect x="${x}" y="${y}" width="${cw}" height="${C}" rx="3"/><text class="ds-lab" x="${x + cw / 2}" y="${y + C / 2 + 1}">${e}</text><text class="ds-slot" x="${x + cw - 3}" y="${y + 8}">${i}</text></g>`,
			);
		});
	});
	// Legend.
	const ly = T + 2 * (C + 14) + 16;
	const legend: [string, string][] = [
		["own-P", "shared prefix P0–P3"],
		["own-A", "request A"],
		["own-B", "request B"],
		["wr", "written this step"],
		["rd", "read by this tile"],
	];
	legend.forEach(([cls, text], i) => {
		const y = ly + i * 17;
		out.push(
			`<g class="ds-c ${cls === "wr" || cls === "rd" ? `own-P ${cls}` : cls}"><rect x="${ix + 18}" y="${y}" width="12" height="12" rx="2"/></g><text class="ds-leg" x="${ix + 36}" y="${y + 10}">${text}</text>`,
		);
	});
	out.push("</svg>");
	return out.join("");
}
