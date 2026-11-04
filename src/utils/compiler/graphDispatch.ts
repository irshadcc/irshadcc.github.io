// Which CUDA graph a serving engine replays for one forward step, and which ops of a small
// decoder run inside it. Drawn by GraphDispatch.astro.
//
// Rules, read from the source (vLLM 54beee8, SGLang 5eb7018):
// vLLM (CompilationMode.VLLM_COMPILE, cudagraph_mode FULL_AND_PIECEWISE):
//   - the FX graph is split at the splitting ops (attention + unified_kv_cache_update);
//     consecutive splitting ops stay in one piece, so each layer has one eager piece.
//   - a uniform decode batch replays one FULL graph of the padded size;
//   - any other batch replays one PIECEWISE graph per compiled piece, attention eager;
//   - above max_cudagraph_capture_size (default 512) the compiled pieces run without graphs.
//   capture sizes: [1, 2, 4] + range(8, 256, 8) + range(256, max + 1, 16).
// SGLang (decode backend "full", prefill backend "breakable" on CUDA, no torch.compile):
//   - decode: one full graph per batch-size bucket up to max_bs (256 on an 80 GB GPU, TP < 4);
//     buckets [1, 2, 4, 8, 12] + range(16, 257, 8) + range(272, 512, 16) + range(512, max + 1, 32);
//   - prefill: a breakable graph, split at RadixAttention._eager_attention (eager_on_graph),
//     buckets over tokens up to 2048: range(4, 33, 4) + range(48, 257, 16) + range(288, 513, 32)
//     + range(576, 1025, 64) + range(1280, 2049, 256);
//   - beyond the largest bucket the forward runs eagerly.

export type Engine = "vllm" | "sglang";
export type BatchKind = "decode" | "mixed";

export interface Op {
	/** Short label drawn on the chip. */
	label: string;
	/** Layer index, -1 before the first layer, `layers` after the last. */
	layer: number;
	/** True for the ops that read or write the KV cache. */
	attn: boolean;
	/** Who provides the kernel when the op is not in a graph. */
	kernel: string;
}

export interface Segment {
	/** Index of the first and last op in the segment. */
	from: number;
	to: number;
	/** "graph" ops are replayed from a CUDA graph; "eager" ops are launched one by one. */
	kind: "graph" | "eager";
}

export interface Dispatch {
	mode: string;
	padded: number | null;
	segments: Segment[];
	graphReplays: number;
	eagerOps: number;
	/** One sentence explaining the choice. */
	why: string;
	/** Capture sizes the engine has graphs for. */
	sizes: number[];
}

const range = (a: number, b: number, s: number) => {
	const out: number[] = [];
	for (let x = a; x < b; x += s) out.push(x);
	return out;
};

export function vllmCaptureSizes(max = 512): number[] {
	return [
		...new Set([1, 2, 4, ...range(8, 256, 8), ...range(256, max + 1, 16)]),
	].sort((a, b) => a - b);
}

export function sglangDecodeSizes(maxBs = 256): number[] {
	const s = [
		1,
		2,
		4,
		8,
		12,
		...range(16, 257, 8),
		...range(272, 512, 16),
		...range(512, maxBs + 1, 32),
	];
	const kept = s.filter((x) => x <= maxBs);
	if (!kept.includes(maxBs)) kept.push(maxBs);
	return kept;
}

export function sglangPrefillSizes(maxBs = 2048): number[] {
	const s = [
		...range(4, 33, 4),
		...range(48, 257, 16),
		...range(288, 513, 32),
		...range(576, 1025, 64),
		...range(1280, 4097, 256),
		...range(4608, maxBs + 1, 512),
	];
	return s.filter((x) => x <= maxBs);
}

export function padTo(sizes: number[], n: number): number | null {
	for (const s of sizes) if (s >= n) return s;
	return null;
}

/** The ops of a `layers`-layer Llama-style decoder, in execution order. */
export function decoderOps(engine: Engine, layers = 3): Op[] {
	const fused = engine === "vllm" ? "Inductor" : "custom kernel";
	const ops: Op[] = [
		{ label: "embed", layer: -1, attn: false, kernel: "custom kernel" },
	];
	for (let l = 0; l < layers; l++) {
		ops.push(
			{
				label: l === 0 ? "rmsnorm" : "add+norm",
				layer: l,
				attn: false,
				kernel: fused,
			},
			{ label: "qkv_proj", layer: l, attn: false, kernel: "cuBLAS" },
			{ label: "rope", layer: l, attn: false, kernel: fused },
			{ label: "kv write", layer: l, attn: true, kernel: "custom kernel" },
			{ label: "attention", layer: l, attn: true, kernel: "FlashAttention" },
			{ label: "o_proj", layer: l, attn: false, kernel: "cuBLAS" },
			{ label: "add+norm", layer: l, attn: false, kernel: fused },
			{ label: "gate_up", layer: l, attn: false, kernel: "cuBLAS" },
			{ label: "silu·mul", layer: l, attn: false, kernel: fused },
			{ label: "down", layer: l, attn: false, kernel: "cuBLAS" },
		);
	}
	ops.push({ label: "add+norm", layer: layers, attn: false, kernel: fused });
	return ops;
}

/** Split ops into maximal runs of attention / non-attention ops. */
function splitAtAttention(ops: Op[], graphKind: "graph" | "eager"): Segment[] {
	const segs: Segment[] = [];
	let start = 0;
	for (let i = 1; i <= ops.length; i++) {
		if (i === ops.length || ops[i].attn !== ops[start].attn) {
			segs.push({
				from: start,
				to: i - 1,
				kind: ops[start].attn ? "eager" : graphKind,
			});
			start = i;
		}
	}
	return segs;
}

export function dispatch(
	engine: Engine,
	kind: BatchKind,
	tokens: number,
	layers = 3,
): Dispatch {
	const ops = decoderOps(engine, layers);
	const whole: Segment[] = [{ from: 0, to: ops.length - 1, kind: "graph" }];
	const allEager: Segment[] = [{ from: 0, to: ops.length - 1, kind: "eager" }];
	const count = (segs: Segment[]) => ({
		graphReplays: segs.filter((s) => s.kind === "graph").length,
		eagerOps: segs
			.filter((s) => s.kind === "eager")
			.reduce((a, s) => a + s.to - s.from + 1, 0),
	});
	if (engine === "vllm") {
		const sizes = vllmCaptureSizes(512);
		const padded = padTo(sizes, tokens);
		if (padded === null) {
			return {
				mode: "no CUDA graph",
				padded: null,
				segments: allEager,
				...count(allEager),
				why: `${tokens} tokens is above max_cudagraph_capture_size = 512, so the Inductor-compiled pieces are called directly, one kernel launch at a time.`,
				sizes,
			};
		}
		if (kind === "decode") {
			return {
				mode: "FULL",
				padded,
				segments: whole,
				...count(whole),
				why: `Every request has one query token, a uniform decode batch: FlashAttention 3 supports full capture, so one graph captured for ${padded} tokens replays the whole forward, attention and KV writes included.`,
				sizes,
			};
		}
		const segs = splitAtAttention(ops, "graph");
		return {
			mode: "PIECEWISE",
			padded,
			segments: segs,
			...count(segs),
			why: `A batch with prefill tokens has per-request query lengths the attention kernel reads at run time, so vLLM replays the ${segs.filter((s) => s.kind === "graph").length} piecewise graphs captured for ${padded} tokens and runs each layer's KV write and attention eagerly between them.`,
			sizes,
		};
	}
	if (kind === "decode") {
		const sizes = sglangDecodeSizes(256);
		const padded = padTo(sizes, tokens);
		if (padded === null) {
			return {
				mode: "eager",
				padded: null,
				segments: allEager,
				...count(allEager),
				why: `${tokens} requests is above the decode graph's max_bs (256 on an 80 GB GPU at TP < 4), so the forward runs eagerly.`,
				sizes,
			};
		}
		return {
			mode: "full",
			padded,
			segments: whole,
			...count(whole),
			why: `Decode uses the "full" backend: the graph captured for batch size ${padded} replays everything, with the attention backend's metadata copied into the graph's static buffers before the replay.`,
			sizes,
		};
	}
	const sizes = sglangPrefillSizes(2048);
	const padded = padTo(sizes, tokens);
	if (padded === null) {
		return {
			mode: "eager",
			padded: null,
			segments: allEager,
			...count(allEager),
			why: `${tokens} tokens is above the largest prefill bucket (2048), so the forward runs eagerly.`,
			sizes,
		};
	}
	const segs = splitAtAttention(ops, "graph");
	return {
		mode: "breakable",
		padded,
		segments: segs,
		...count(segs),
		why: `Prefill uses a breakable CUDA graph captured for ${padded} tokens: RadixAttention._eager_attention is an eager_on_graph break, so the capture is cut there into ${segs.filter((s) => s.kind === "graph").length} graph segments and attention (which also writes the KV cache) runs eagerly on the real tokens.`,
		sizes,
	};
}

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

/** SVG of the decoder's ops, one row per layer, each chip tagged with its segment. */
export function drawDispatch(engine: Engine, d: Dispatch, layers = 3): string {
	const ops = decoderOps(engine, layers);
	const segOf = new Map<number, number>();
	d.segments.forEach((s, k) => {
		for (let i = s.from; i <= s.to; i++) segOf.set(i, k);
	});
	const graphNo = new Map<number, number>();
	let g = 0;
	d.segments.forEach((s, k) => {
		if (s.kind === "graph") graphNo.set(k, ++g);
	});
	const rows: Op[][] = [];
	const rowOf = (op: Op) =>
		op.layer < 0 ? 0 : op.layer >= layers ? layers + 1 : op.layer + 1;
	for (const op of ops) {
		const r = rowOf(op);
		rows[r] ??= [];
		rows[r].push(op);
	}
	const cw = 54;
	const ch = 24;
	const gap = 4;
	const L = 50;
	const T = 13;
	const rh = ch + 12;
	const maxCols = Math.max(...rows.map((r) => r.length));
	const W = L + maxCols * (cw + gap) + 4;
	const H = T + rows.length * rh;
	const out: string[] = [
		`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Ops of a ${layers}-layer decoder coloured by the CUDA graph segment that runs them">`,
	];
	let idx = 0;
	rows.forEach((row, r) => {
		const y = T + r * rh;
		const name =
			r === 0 ? "input" : r === rows.length - 1 ? "output" : `layer ${r - 1}`;
		out.push(
			`<text class="gd-row" x="${L - 8}" y="${y + ch / 2 + 4}">${name}</text>`,
		);
		row.forEach((op, c) => {
			const x = L + c * (cw + gap);
			const k = segOf.get(idx) ?? 0;
			const seg = d.segments[k];
			const no = graphNo.get(k);
			const cls =
				seg.kind === "graph"
					? `gd-op graph g${(no ?? 1) % 2}`
					: `gd-op eager${op.attn ? " attn" : ""}`;
			out.push(
				`<g class="${cls}" data-i="${idx}"><title>${esc(`${op.label}: ${seg.kind === "graph" ? `replayed from graph ${no}` : "launched eagerly"}; kernel from ${op.kernel}`)}</title>`,
				`<rect x="${x}" y="${y}" width="${cw}" height="${ch}" rx="5"/>`,
				`<text class="gd-lab" x="${x + cw / 2}" y="${y + ch / 2 + 3.5}">${esc(op.label)}</text>`,
			);
			if (no !== undefined && idx === seg.from)
				out.push(`<text class="gd-no" x="${x + 3}" y="${y - 2}">G${no}</text>`);
			out.push("</g>");
			idx++;
		});
	});
	out.push("</svg>");
	return out.join("");
}

export function describeDispatch(tokens: number, d: Dispatch): string {
	const pad =
		d.padded === null
			? "no captured size is large enough"
			: d.padded === tokens
				? `captured size ${d.padded} matches exactly`
				: `padded to the next captured size, ${d.padded} (${d.padded - tokens} dummy tokens)`;
	return [
		`<p><b>Mode:</b> <code>${esc(d.mode)}</code> &middot; ${tokens} tokens, ${pad}.</p>`,
		`<p><b>Per step:</b> ${d.graphReplays} graph replay${d.graphReplays === 1 ? "" : "s"} + ${d.eagerOps} eager op${d.eagerOps === 1 ? "" : "s"} launched from Python.</p>`,
		`<p>${esc(d.why)}</p>`,
	].join("");
}
