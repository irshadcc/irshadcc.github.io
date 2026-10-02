// SVG builders for the FlashAttention figures, used by OnlineSoftmax.astro and
// AttentionTiles.astro.
//
// drawOnline steps one query row through online softmax, a block of keys at a time: the scores,
// the running maximum m, the weights exp(s - m) (which shrink when m rises), and the running
// sum, accumulator and output. drawTiles draws the score matrix as tiles, with the tiles one
// thread block visits, and those a causal mask lets it skip.

export interface OnlineData {
	scores: number[];
	values: number[];
	block: number;
}

export const ONLINE_EXAMPLE: OnlineData = {
	scores: [1.0, 2.0, 0.5, 1.5, 3.0, 0.2, 2.5, 1.0, 0.8, 3.5, 1.2, 2.0],
	values: [2, -1, 0.5, 1, 3, -2, 1.5, 0, 1, 4, -1, 2],
	block: 4,
};

export interface OnlineState {
	/** Blocks processed so far. */
	done: number;
	m: number;
	l: number;
	acc: number;
	/** exp(m_prev - m) applied at the last step; 0 on the first block. */
	alpha: number;
	mPrev: number;
}

export function onlineStates(d: OnlineData): OnlineState[] {
	const states: OnlineState[] = [
		{
			done: 0,
			m: Number.NEGATIVE_INFINITY,
			l: 0,
			acc: 0,
			alpha: 1,
			mPrev: Number.NEGATIVE_INFINITY,
		},
	];
	const nb = Math.ceil(d.scores.length / d.block);
	let { m, l, acc } = states[0];
	for (let b = 0; b < nb; b++) {
		const idx = Array.from(
			{ length: d.block },
			(_, i) => b * d.block + i,
		).filter((j) => j < d.scores.length);
		const mNew = Math.max(m, ...idx.map((j) => d.scores[j]));
		const alpha = m === Number.NEGATIVE_INFINITY ? 0 : Math.exp(m - mNew);
		const p = idx.map((j) => Math.exp(d.scores[j] - mNew));
		l = alpha * l + p.reduce((a, x) => a + x, 0);
		acc = alpha * acc + p.reduce((a, x, i) => a + x * d.values[idx[i]], 0);
		states.push({ done: b + 1, m: mNew, l, acc, alpha, mPrev: m });
		m = mNew;
	}
	return states;
}

export function exactOutput(d: OnlineData): number {
	const mx = Math.max(...d.scores);
	const w = d.scores.map((s) => Math.exp(s - mx));
	return (
		w.reduce((a, x, i) => a + x * d.values[i], 0) / w.reduce((a, x) => a + x, 0)
	);
}

const f = (x: number, digits = 3) =>
	Number.isFinite(x) ? x.toFixed(digits) : "−∞";

export function drawOnline(d: OnlineData, s: OnlineState): string {
	const n = d.scores.length;
	const BW = 22;
	const BG = 4;
	const GG = 16; // between blocks
	const LEFT = 64;
	const xOf = (j: number) =>
		LEFT + j * (BW + BG) + Math.floor(j / d.block) * GG;
	const right = xOf(n - 1) + BW;
	const SMAX = 4;
	const TOP1 = 22;
	const H1 = 92;
	const y1 = (v: number) => TOP1 + H1 - (H1 * v) / SMAX;
	const TOP2 = TOP1 + H1 + 40;
	const H2 = 70;
	const y2 = (v: number) => TOP2 + H2 - H2 * v;
	const out: string[] = [];
	const processed = (j: number) => Math.floor(j / d.block) < s.done;
	const current = (j: number) => Math.floor(j / d.block) === s.done - 1;

	out.push(
		`<text class="lab" x="${LEFT - 8}" y="${TOP1 + H1 / 2}">score s</text>`,
		`<line class="axis" x1="${LEFT - 4}" x2="${right + 4}" y1="${y1(0)}" y2="${y1(0)}"/>`,
		`<text class="lab" x="${LEFT - 8}" y="${TOP2 + H2 / 2 + 4}">e^(s − m)</text>`,
		`<line class="axis" x1="${LEFT - 4}" x2="${right + 4}" y1="${y2(0)}" y2="${y2(0)}"/>`,
		`<text class="tick" x="${LEFT - 8}" y="${y2(1) + 3}">1</text>`,
	);
	for (let b = 0; b * d.block < n; b++) {
		const x0 = xOf(b * d.block) - 5;
		const x1 = xOf(Math.min(n, (b + 1) * d.block) - 1) + BW + 5;
		const cls = b < s.done - 1 ? "done" : b === s.done - 1 ? "cur" : "todo";
		out.push(
			`<rect class="blk ${cls}" x="${x0}" y="${TOP1 - 14}" width="${x1 - x0}" height="${TOP2 + H2 - TOP1 + 30}" rx="5"/>`,
			`<text class="blab" x="${(x0 + x1) / 2}" y="${TOP1 - 3}">block ${b + 1}</text>`,
		);
	}
	for (let j = 0; j < n; j++) {
		const x = xOf(j);
		const sc = d.scores[j];
		const st = processed(j) ? (current(j) ? "cur" : "done") : "todo";
		out.push(
			`<g><title>key ${j}: s = ${sc}, v = ${d.values[j]}</title><rect class="sbar ${st}" x="${x}" y="${y1(sc)}" width="${BW}" height="${y1(0) - y1(sc)}" rx="1.5"/>`,
			`<text class="val" x="${x + BW / 2}" y="${y1(0) + 11}">${sc}</text></g>`,
		);
		if (processed(j)) {
			const w = Math.exp(sc - s.m);
			if (!current(j) && s.alpha < 1) {
				const before = Math.exp(sc - s.mPrev);
				out.push(
					`<rect class="ghost" x="${x}" y="${y2(before)}" width="${BW}" height="${y2(0) - y2(before)}" rx="1.5"/>`,
				);
			}
			out.push(
				`<g><title>key ${j}: e^(${sc} − ${s.m}) = ${w.toFixed(3)}</title><rect class="wbar ${st}" x="${x}" y="${y2(w)}" width="${BW}" height="${y2(0) - y2(w)}" rx="1.5"/></g>`,
			);
		}
		out.push(
			`<text class="val v" x="${x + BW / 2}" y="${y2(0) + 11}">v=${d.values[j]}</text>`,
		);
	}
	if (s.done > 0) {
		out.push(
			`<line class="mline" x1="${LEFT - 4}" x2="${right + 4}" y1="${y1(s.m)}" y2="${y1(s.m)}"/>`,
			`<text class="mlab" x="${right + 8}" y="${y1(s.m) + 4}">m = ${f(s.m, 1)}</text>`,
		);
		if (s.alpha < 1 && s.alpha > 0)
			out.push(
				`<line class="mline old" x1="${LEFT - 4}" x2="${right + 4}" y1="${y1(s.mPrev)}" y2="${y1(s.mPrev)}"/>`,
				`<text class="mlab old" x="${right + 8}" y="${y1(s.mPrev) + 4}">was ${f(s.mPrev, 1)}</text>`,
			);
	}
	const W = right + 72;
	const H = TOP2 + H2 + 24;
	return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" style="min-width:${Math.round(W * 0.72)}px" role="img" aria-label="Online softmax after ${s.done} blocks">${out.join("")}</svg>`;
}

/** The score matrix as T x T tiles: computed, partly masked, or skipped. */
export function drawTiles(
	t: number,
	causal: boolean,
	row: number | null,
): string {
	const C = 26;
	const G = 3;
	const LEFT = 64;
	const TOP = 34;
	const out: string[] = [];
	const xOf = (j: number) => LEFT + j * (C + G);
	const yOf = (i: number) => TOP + i * (C + G);
	out.push(
		`<text class="lab" x="${xOf(0)}" y="${TOP - 18}">key / value tiles →</text>`,
		`<text class="lab end" x="${LEFT - 10}" y="${yOf(0) - 6}">query tiles</text>`,
	);
	for (let j = 0; j < t; j++)
		out.push(
			`<text class="tick" x="${xOf(j) + C / 2}" y="${TOP - 5}">${j}</text>`,
		);
	for (let i = 0; i < t; i++) {
		out.push(
			`<text class="tick end" x="${LEFT - 10}" y="${yOf(i) + C / 2 + 4}">${i}</text>`,
		);
		let order = 0;
		for (let j = 0; j < t; j++) {
			const x = xOf(j);
			const y = yOf(i);
			const skip = causal && j > i;
			const diag = causal && j === i;
			const on = row === i && !skip;
			if (on) order++;
			const cls = `tile${skip ? " skip" : ""}${diag ? " diag" : ""}${row === i ? " sel" : ""}${on ? " on" : ""}`;
			const tip = skip
				? `Query tile ${i}, key tile ${j}: every score is masked, so the tile is skipped`
				: diag
					? `Query tile ${i}, key tile ${j}: on the diagonal, so part of it is masked`
					: `Query tile ${i}, key tile ${j}: computed`;
			out.push(
				`<g class="${cls}" data-row="${i}"><title>${tip}</title><rect x="${x}" y="${y}" width="${C}" height="${C}" rx="2"/>`,
			);
			if (diag)
				out.push(
					`<path class="mask" d="M${x + C},${y} L${x + C},${y + C} L${x},${y} Z"/>`,
				);
			if (on)
				out.push(
					`<text class="ord" x="${x + C / 2}" y="${y + C / 2 + 4}">${order}</text>`,
				);
			out.push("</g>");
		}
	}
	const W = xOf(t - 1) + C + 10;
	const H = yOf(t - 1) + C + 8;
	return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Attention score tiles${causal ? " with a causal mask" : ""}">${out.join("")}</svg>`;
}
