// Which thread copies which 16-byte vector when one token's key and value are written into the
// KV cache, for vLLM's reshape_and_cache_flash (csrc/cache_kernels.cu at 54beee8, default LBNHC
// layout with FlashAttention: K and V of a head side by side in one page) and SGLang's
// store_kvcache_kernel (python/sglang/kernels/jit/csrc/elementwise/kvcache.cuh at 5eb7018:
// separate K and V buffers, one row per token slot). Drawn by KvWriteMap.astro.
//
// vLLM: grid = num_tokens, block = min(H * D, 512) threads. With LBNHC the key view has
// head_stride = 2 * D != D, so the kernel takes its strided branch: warp w handles heads
// w, w + warps, ...; inside a head, lane l copies vectors l, l + 32, ... of D / 8 bf16 vectors.
// SGLang: the thread count per token comes from the heuristic in kernels/ops/kvcache/kvcache.py
// (start at 32, double while each lane would still copy >= 64 bytes of K and V together and
// both halves stay multiples of 8); warp s of a token copies bytes [s * R/S, (s+1) * R/S) of the
// row, lane l the 16 bytes at l * 16 inside that split (when the split is a multiple of 512 B).

export type Kernel = "vllm" | "sglang";

export interface KvConfig {
	/** KV heads on this GPU. */
	heads: number;
	/** Head size in elements. */
	headDim: number;
	/** Bytes per element (2 for bf16). */
	elem: number;
	/** vLLM block size (tokens per block). */
	blockSize: number;
}

export interface Vec {
	row: "K" | "V";
	head: number;
	/** Index of the 16-byte vector inside the head (0 .. D * elem / 16 - 1). */
	v: number;
	/** Byte offset in the token's input row (k or v, [H * D] contiguous). */
	src: number;
	/** Byte offset from the start of the cache tensor. */
	dst: number;
}

export interface ThreadWork {
	t: number;
	warp: number;
	lane: number;
	vecs: Vec[];
}

export interface KvPlan {
	kernel: Kernel;
	cfg: KvConfig;
	slot: number;
	threads: ThreadWork[];
	blockDim: number;
	vecBytes: number;
	/** vLLM strides in bytes: block, page (slot in block), head. SGLang: row stride. */
	strides: Record<string, number>;
	/** Base byte offset of this token's destination (page for vLLM; K row for SGLang). */
	base: number;
	/** SGLang only: base of the V row. */
	vBase?: number;
	note: string;
}

export const KV_PRESETS: Record<string, { label: string; cfg: KvConfig }> = {
	tp4: {
		label: "Llama-3.1-8B, TP = 4 (2 KV heads)",
		cfg: { heads: 2, headDim: 128, elem: 2, blockSize: 16 },
	},
	tp1: {
		label: "Llama-3.1-8B, TP = 1 (8 KV heads)",
		cfg: { heads: 8, headDim: 128, elem: 2, blockSize: 16 },
	},
};

export function sglangThreads(kRowBytes: number, vRowBytes: number): number {
	let n = 32;
	if (kRowBytes % n !== 0 || vRowBytes % n !== 0) return n;
	let k = kRowBytes / n;
	let v = vRowBytes / n;
	while (k % 8 === 0 && v % 8 === 0 && k + v >= 64) {
		n *= 2;
		k /= 2;
		v /= 2;
	}
	return n;
}

export function planWrite(kernel: Kernel, cfg: KvConfig, slot: number): KvPlan {
	const { heads: H, headDim: D, elem } = cfg;
	const VEC = 16;
	const vecsPerHead = (D * elem) / VEC;
	if (kernel === "vllm") {
		const blockDim = Math.min(H * D, 512);
		const warps = blockDim / 32;
		const head = 2 * D * elem; // head stride in bytes (K then V)
		const page = H * head;
		const block = cfg.blockSize * page;
		const b = Math.floor(slot / cfg.blockSize);
		const o = slot % cfg.blockSize;
		const base = b * block + o * page;
		const threads: ThreadWork[] = [];
		for (let t = 0; t < blockDim; t++) {
			const warp = Math.floor(t / 32);
			const lane = t % 32;
			const vecs: Vec[] = [];
			for (let h = warp; h < H; h += warps)
				for (let v = lane; v < vecsPerHead; v += 32)
					for (const row of ["K", "V"] as const)
						vecs.push({
							row,
							head: h,
							v,
							src: h * D * elem + v * VEC,
							dst: base + h * head + (row === "V" ? D * elem : 0) + v * VEC,
						});
			threads.push({ t, warp, lane, vecs });
		}
		const busy = threads.filter((x) => x.vecs.length).length;
		return {
			kernel,
			cfg,
			slot,
			threads,
			blockDim,
			vecBytes: VEC,
			strides: { block, page, head },
			base,
			note: `One thread block of min(H·D, 512) = ${blockDim} threads per token; ${busy} of them (${Math.round((100 * busy) / blockDim)}%) copy anything, because a warp takes a whole head and a head is only ${vecsPerHead} vectors of 16 B.`,
		};
	}
	const row = H * D * elem;
	const n = sglangThreads(row, row);
	const split = n / 32;
	const splitBytes = row / split;
	const threads: ThreadWork[] = [];
	const base = slot * row;
	for (let t = 0; t < n; t++) {
		const warp = Math.floor(t / 32);
		const lane = t % 32;
		const vecs: Vec[] = [];
		for (let off = lane * VEC; off < splitBytes; off += 32 * VEC) {
			const src = warp * splitBytes + off;
			const h = Math.floor(src / (D * elem));
			const v = (src % (D * elem)) / VEC;
			for (const r of ["K", "V"] as const)
				vecs.push({ row: r, head: h, v, src, dst: base + src });
		}
		threads.push({ t, warp, lane, vecs });
	}
	return {
		kernel,
		cfg,
		slot,
		threads,
		blockDim: n,
		vecBytes: VEC,
		strides: { row },
		base,
		vBase: base,
		note: `The heuristic gives ${n} threads (${split} warp${split === 1 ? "" : "s"}) per token, so every lane copies one 16 B vector of K and one of V: ${row} B of each, split ${splitBytes} B per warp.${slot === 0 ? " Slot 0 is the padding slot CUDA-graph padding tokens point at: the kernel loads these vectors but skips the stores." : ""}`,
	};
}

/** Every 16-byte vector is written exactly once and lands inside the token's destination. */
export function checkPlan(p: KvPlan): string | null {
	const { heads: H, headDim: D, elem } = p.cfg;
	const per = (D * elem) / 16;
	const seen = new Set<string>();
	for (const th of p.threads)
		for (const v of th.vecs) {
			const key = `${v.row}${v.head}.${v.v}`;
			if (seen.has(key)) return `vector ${key} written twice`;
			seen.add(key);
		}
	if (seen.size !== 2 * H * per)
		return `wrote ${seen.size} vectors, expected ${2 * H * per}`;
	return null;
}

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

const C = 11; // cell size
const G = 1; // gap

function cells(
	x0: number,
	y0: number,
	rows: number,
	cols: number,
	cls: (r: number, c: number) => string,
	attrs: (r: number, c: number) => string,
): string {
	const out: string[] = [];
	for (let r = 0; r < rows; r++)
		for (let c = 0; c < cols; c++)
			out.push(
				`<rect class="${cls(r, c)}" ${attrs(r, c)} x="${x0 + c * (C + G)}" y="${y0 + r * (C + G)}" width="${C}" height="${C}" rx="1.5"/>`,
			);
	return out.join("");
}

/** SVG, top to bottom: thread grid, the token's K and V input rows, the destination. */
export function drawWrite(p: KvPlan): string {
	const { heads: H, headDim: D, elem } = p.cfg;
	const per = (D * elem) / 16;
	const owner = new Map<string, number>();
	for (const th of p.threads)
		for (const v of th.vecs) owner.set(`${v.row}${v.head}.${v.v}`, th.t);
	const out: string[] = [];
	const tRows = Math.ceil(p.blockDim / 32);
	const L = 30;
	const Ttop = 16;
	const srcX = L;
	const srcY = Ttop + tRows * (C + G) + 26;
	const dstY = srcY + H * (C + G) + 26;
	const W = L + Math.max(32 * (C + G), 2 * per * (C + G) + 10) + 4;
	const Ht = dstY + H * (C + G) + 6;
	out.push(
		`<svg viewBox="0 0 ${W} ${Ht}" width="${W}" height="${Ht}" role="img" aria-label="Threads of ${p.kernel === "vllm" ? "reshape_and_cache_flash" : "store_kvcache_kernel"} and the 16-byte vectors they copy">`,
	);
	// Thread grid.
	out.push(
		`<text class="kw-h" x="${L}" y="${Ttop - 5}">threads of one token (rows = warps, 32 lanes each)</text>`,
	);
	for (let w = 0; w < tRows; w++)
		out.push(
			`<text class="kw-row" x="${L - 5}" y="${Ttop + w * (C + G) + C - 1}">w${w}</text>`,
		);
	for (const th of p.threads) {
		const head = th.vecs[0]?.head;
		const cls = head === undefined ? "kw-t idle" : `kw-t h${head % 8}`;
		out.push(
			`<rect class="${cls}" data-t="${th.t}" x="${L + th.lane * (C + G)}" y="${Ttop + th.warp * (C + G)}" width="${C}" height="${C}" rx="1.5"/>`,
		);
	}
	// Source rows.
	out.push(
		`<text class="kw-h" x="${srcX}" y="${srcY - 5}">input k (left) and v (right) of the token, 16 B per cell</text>`,
	);
	const vecCls = (row: "K" | "V") => (r: number, c: number) =>
		`kw-v h${r % 8}${row === "V" ? " vv" : ""}`;
	const vecAttr = (row: "K" | "V", kind: string) => (r: number, c: number) =>
		`data-t="${owner.get(`${row}${r}.${c}`) ?? -1}" data-k="${kind}${row}${r}.${c}"`;
	out.push(cells(srcX, srcY, H, per, vecCls("K"), vecAttr("K", "s")));
	out.push(
		cells(
			srcX + per * (C + G) + 10,
			srcY,
			H,
			per,
			vecCls("V"),
			vecAttr("V", "s"),
		),
	);
	for (let h = 0; h < H; h++)
		out.push(
			`<text class="kw-row" x="${srcX - 5}" y="${srcY + h * (C + G) + C - 1}">h${h}</text>`,
		);
	// Destination.
	if (p.kernel === "vllm") {
		out.push(
			`<text class="kw-h" x="${L}" y="${dstY - 6}">destination: page of slot ${p.slot} (block ${Math.floor(p.slot / p.cfg.blockSize)}, offset ${p.slot % p.cfg.blockSize}): per head, K then V</text>`,
		);
		out.push(cells(L, dstY, H, per, vecCls("K"), vecAttr("K", "d")));
		out.push(
			cells(L + per * (C + G), dstY, H, per, vecCls("V"), vecAttr("V", "d")),
		);
	} else {
		out.push(
			`<text class="kw-h" x="${L}" y="${dstY - 6}">destination: row ${p.slot} of k_buffer (left) and v_buffer (right)</text>`,
		);
		out.push(cells(L, dstY, H, per, vecCls("K"), vecAttr("K", "d")));
		out.push(
			cells(
				L + per * (C + G) + 10,
				dstY,
				H,
				per,
				vecCls("V"),
				vecAttr("V", "d"),
			),
		);
	}
	for (let h = 0; h < H; h++)
		out.push(
			`<text class="kw-row" x="${L - 5}" y="${dstY + h * (C + G) + C - 1}">h${h}</text>`,
		);
	out.push("</svg>");
	return out.join("");
}

export function describeThread(p: KvPlan, t: number | null): string {
	if (t === null || t < 0 || t >= p.threads.length)
		return `<p>${esc(p.note)}</p><p>Hover a thread or a vector.</p>`;
	const th = p.threads[t];
	const head = `<p><b>thread ${t}</b> = warp ${th.warp}, lane ${th.lane}</p>`;
	if (!th.vecs.length)
		return `${head}<p>No work: ${p.kernel === "vllm" ? `warp ${th.warp} has no head (${p.cfg.heads} heads, warps take heads w, w + ${p.blockDim / 32}, ...)` : "idle"}${p.kernel === "vllm" && th.warp < p.cfg.heads ? `, and lane ${th.lane} is past the ${(p.cfg.headDim * p.cfg.elem) / 16} vectors of a head` : ""}.</p>`;
	const lines = th.vecs.map((v) => {
		if (p.kernel === "vllm") {
			const s = p.strides;
			const b = Math.floor(p.slot / p.cfg.blockSize);
			const o = p.slot % p.cfg.blockSize;
			const extra = v.row === "V" ? ` + ${p.cfg.headDim * p.cfg.elem}` : "";
			return `<li>${v.row} head ${v.head}, bytes ${v.src}–${v.src + 15} of the input → cache byte ${b}·${s.block} + ${o}·${s.page} + ${v.head}·${s.head}${extra} + ${v.v}·16 = <b>${v.dst}</b></li>`;
		}
		return `<li>${v.row} bytes ${v.src}–${v.src + 15} → ${v.row === "K" ? "k" : "v"}_buffer byte ${p.slot}·${p.strides.row} + ${v.src} = <b>${v.dst}</b></li>`;
	});
	return `${head}<ul class="kw-list">${lines.join("")}</ul>`;
}
