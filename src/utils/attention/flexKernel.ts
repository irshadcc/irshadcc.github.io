// One program of FlexAttention's forward Triton template (flex_attention.py.jinja and
// forward_block_mn in common.py.jinja), run on a scaled-down example so every number fits on
// screen: 8 tokens, head dim 4 (so SM_SCALE = 1/sqrt(4) = 0.5), value dim 2, one head,
// sparse blocks of 4, BLOCK_M = 4, BLOCK_N = 2. score_mod is ALiBi with slope 0.5, mask_mod is
// causal. Q, K, V are the integers torch.randint(-1, 2) / randint(-2, 3) drew with seed 0.
// The program shown is q_start = 1 (query rows 4-7). Output and LSE were checked against
// math_attention in torch 2.13.0; see the post's verification notes.
import {
	type BlockMaskData,
	type Tile,
	buildBlockMask,
	visitOrder,
} from "./blockMask";

export const Q = [
	[1, -1, 1, -1],
	[0, -1, 0, 0],
	[0, -1, 1, 1],
	[-1, -1, 0, 1],
	[-1, -1, 1, -1],
	[1, 1, 1, 1],
	[1, 1, 0, 1],
	[-1, -1, 0, -1],
];
export const K = [
	[1, -1, 0, -1],
	[0, 1, 1, 0],
	[1, -1, 0, 0],
	[0, 1, 0, 0],
	[1, 0, 0, 1],
	[1, 0, 1, 1],
	[1, 0, 1, 1],
	[0, 0, -1, -1],
];
export const V = [
	[-2, -2],
	[1, 0],
	[1, 0],
	[0, -2],
	[-2, -2],
	[1, -1],
	[2, -1],
	[-1, -2],
];
export const SLOPE = 0.5;
export const SM_SCALE = 0.5;
export const SEQ = 8;
export const SPARSE = 4;
export const BLOCK_M = 4;
export const BLOCK_N = 2;
export const Q_START = 1;
// The constant the template uses for 1 / ln 2.
// biome-ignore lint/suspicious/noApproximativeNumericConstant: the kernel's own truncated value
export const RCP_LN2 = 1.44269504;

export const CODE = [
	"m_i = -inf; l_i = 0; acc = 0          # per row of the program's BLOCK_M rows",
	"q = load(Q + offs_m)                   # query rows 4-7, kept for the whole loop",
	"# partial blocks: kv_indices[1] = [1, 0], kv_num_blocks[1] = 1",
	"for start_n in range(kv_num_blocks * SPARSE_KV_MULTIPLE):",
	"    qk = tl.dot(q, k) * SM_SCALE",
	"    post_mod_scores = qk + slope * (n - m)          # score_mod, inlined",
	"    post_mod_scores = tl.where(m >= n, post_mod_scores, -inf)  # mask_mod",
	"    post_mod_scores *= RCP_LN2                       # to base 2",
	"    m_ij = maximum(m_i, max(post_mod_scores, 1))",
	"    alpha = exp2(m_i - m_ij); p = exp2(post_mod_scores - m_ij[:, None])",
	"    l_i = l_i * alpha + sum(p, 1)",
	"    acc = acc * alpha[:, None] + tl.dot(p, v)",
	"    m_i = m_ij",
	"    offs_n += get_offset_for_next_block(start_n, ...)",
	"# full blocks: full_kv_indices[1] = [0, 1], full_kv_num_blocks[1] = 1, no mask_mod",
	"out = acc / l_i[:, None]; lse = m_i + log2(l_i)",
];

export type Mat = number[][];

export interface KernelStep {
	kind: "load" | "scores" | "update" | "epilogue";
	label: string;
	lines: number[];
	note: string;
	tile?: Tile;
	/** Key indices of the tile. */
	cols?: number[];
	qk?: Mat;
	mod?: Mat;
	masked?: Mat;
	base2?: Mat;
	mPrev?: number[];
	m?: number[];
	alpha?: number[];
	p?: Mat;
	lPrev?: number[];
	l?: number[];
	accPrev?: Mat;
	acc?: Mat;
	v?: Mat;
	out?: Mat;
	lse2?: number[];
	lse?: number[];
	refOut?: Mat;
	refLse?: number[];
}

const rows = Array.from({ length: BLOCK_M }, (_, i) => Q_START * BLOCK_M + i);
const dot = (a: number[], b: number[]) =>
	a.reduce((s, x, i) => s + x * b[i], 0);
export const fmt = (x: number, d = 3) =>
	x === Number.NEGATIVE_INFINITY
		? "−∞"
		: (Math.abs(x) < 5e-4 ? 0 : x).toFixed(d).replace("-", "−");

/** Dense reference: softmax over the whole masked row, natural-log LSE. */
function reference(): { out: Mat; lse: number[] } {
	const out: Mat = [];
	const lse: number[] = [];
	for (const m of rows) {
		const s = K.map((k, n) =>
			m >= n
				? dot(Q[m], k) * SM_SCALE + SLOPE * (n - m)
				: Number.NEGATIVE_INFINITY,
		);
		const mx = Math.max(...s);
		const e = s.map((x) => Math.exp(x - mx));
		const z = e.reduce((a, b) => a + b, 0);
		out.push([0, 1].map((d) => e.reduce((a, w, n) => a + w * V[n][d], 0) / z));
		lse.push(mx + Math.log(z));
	}
	return { out, lse };
}

export function blockMask(): BlockMaskData {
	return buildBlockMask({ preset: "causal" }, SEQ, SEQ, SPARSE, SPARSE);
}

export function buildSteps(): KernelStep[] {
	const bm = blockMask();
	const tiles = visitOrder(bm, Q_START, BLOCK_N, "fwd");
	const steps: KernelStep[] = [];
	let m = rows.map(() => Number.NEGATIVE_INFINITY);
	let l = rows.map(() => 0);
	let acc: Mat = rows.map(() => [0, 0]);
	steps.push({
		kind: "load",
		label: "load q",
		lines: [0, 1, 2],
		note: `The program owns query rows ${rows[0]}–${rows[rows.length - 1]}. It loads them once and reads its row of the block mask: one partial block (block 1, on the diagonal) and one full block (block 0). With sparse blocks of ${SPARSE} and BLOCK_N = ${BLOCK_N}, each block is ${SPARSE / BLOCK_N} tiles.`,
		m: [...m],
		l: [...l],
		acc: acc.map((r) => [...r]),
	});
	tiles.forEach((t, ti) => {
		const cols = Array.from({ length: BLOCK_N }, (_, j) => t.start + j);
		const qk = rows.map((r) => cols.map((n) => dot(Q[r], K[n]) * SM_SCALE));
		const mod = qk.map((row, i) =>
			row.map((s, j) => s + SLOPE * (cols[j] - rows[i])),
		);
		const masked = t.full
			? undefined
			: mod.map((row, i) =>
					row.map((s, j) =>
						rows[i] >= cols[j] ? s : Number.NEGATIVE_INFINITY,
					),
				);
		const base2 = (masked ?? mod).map((row) => row.map((s) => s * RCP_LN2));
		const next = ti + 1 < tiles.length && tiles[ti + 1].full === t.full;
		const tag = `tile ${ti + 1}`;
		steps.push({
			kind: "scores",
			label: `${tag} scores`,
			lines: t.full ? [14, 4, 5, 7] : [3, 4, 5, 6, 7],
			tile: t,
			cols,
			qk,
			mod,
			masked,
			base2,
			note: t.full
				? `Keys ${cols[0]}–${cols[cols.length - 1]} lie in block ${t.block}, a full block: every query in rows ${rows[0]}–${rows[rows.length - 1]} may see them, so the kernel is compiled with IS_FULL_BLOCKS = True for this loop and the mask_mod lines are not emitted. Only score_mod runs.`
				: `Keys ${cols[0]}–${cols[cols.length - 1]} lie in block ${t.block}, a partial block, so both score_mod and mask_mod run. score_mod adds slope × (n − m), the ALiBi penalty; mask_mod sets future keys (n > m) to −∞.`,
		});
		const mPrev = [...m];
		const lPrev = [...l];
		const accPrev = acc.map((r) => [...r]);
		const mNew = m.map((mi, i) => Math.max(mi, ...base2[i]));
		const mSafe = mNew.map((x) => (x === Number.NEGATIVE_INFINITY ? 0 : x));
		const alpha = m.map((mi, i) => 2 ** (mi - mSafe[i]));
		const p = base2.map((row, i) => row.map((s) => 2 ** (s - mSafe[i])));
		l = l.map((li, i) => li * alpha[i] + p[i].reduce((a, b) => a + b, 0));
		const v = cols.map((n) => V[n]);
		acc = acc.map((row, i) =>
			row.map(
				(a, d) => a * alpha[i] + p[i].reduce((s, w, j) => s + w * v[j][d], 0),
			),
		);
		m = mNew;
		const off = next
			? t.jump
				? `get_offset_for_next_block returns ${t.offset}: the next tile is in block ${tiles[ti + 1].block}.`
				: `get_offset_for_next_block returns ${t.offset} = BLOCK_N: the next tile is in the same block.`
			: "This was the loop's last tile.";
		steps.push({
			kind: "update",
			label: `${tag} update`,
			lines: [8, 9, 10, 11, 12, 13],
			tile: t,
			cols,
			base2,
			mPrev,
			m: [...m],
			alpha,
			p,
			lPrev,
			l: [...l],
			accPrev,
			acc: acc.map((r) => [...r]),
			v,
			note: `The online softmax update, in base 2. m_ij is the running row maximum; alpha = 2^(m_old − m_new) rescales what was summed before. ${off}`,
		});
	});
	const out = acc.map((row, i) => row.map((a) => a / (l[i] === 0 ? 1 : l[i])));
	const lse2 = m.map((mi, i) => mi + Math.log2(l[i] === 0 ? 1 : l[i]));
	const ref = reference();
	steps.push({
		kind: "epilogue",
		label: "epilogue",
		lines: [15],
		out,
		lse2,
		lse: lse2.map((x) => x * Math.LN2),
		refOut: ref.out,
		refLse: ref.lse,
		m: [...m],
		l: [...l],
		acc: acc.map((r) => [...r]),
		note: "Divide by l_i and store. The kernel's LSE is in base 2 (m_i + log2 l_i); flex_attention multiplies it by ln 2 before returning it. Both match a dense softmax over the masked row.",
	});
	return steps;
}
