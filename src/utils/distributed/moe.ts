// A toy mixture-of-experts router and the all-to-all traffic it causes, used by
// MoeDispatch.astro.
//
// Every rank has `tokens` tokens, and each picks its top-k experts. Expert popularity follows
// p_e ∝ 1 / (1 + r_e)^skew, where r_e is the expert's place in a fixed shuffled order, so skew 0
// is a perfectly balanced router and larger skews pile tokens onto a few experts. Each token's
// choice is a Gumbel-top-k sample from those probabilities, with the noise drawn once from a
// seeded generator, so moving the skew changes the routing smoothly.
//
// With a capacity factor, each rank may send each expert at most
// ceil(capacity · tokens · k / experts) token copies, as in GShard and Switch; the rest are
// dropped (the token skips the expert and only the residual connection carries it).

export interface MoeConfig {
	ranks: number;
	experts: number;
	/** Tokens per rank. */
	tokens: number;
	k: number;
	skew: number;
	/** Capacity factor, or null for dropless. */
	capacity: number | null;
	seed?: number;
}

export interface MoeResult {
	config: MoeConfig;
	/** Token copies each expert receives, after drops. */
	load: number[];
	/** Token copies routed to each expert but dropped. */
	dropped: number[];
	/** send[i][j]: token copies rank i sends to rank j (to its experts), after drops. */
	send: number[][];
	/** Per-rank limit for one expert, or null. */
	localCap: number | null;
	/** Expert e lives on rank owner(e). */
	owner: (e: number) => number;
}

function mulberry32(seed: number) {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

export function simulateMoe(cfg: MoeConfig): MoeResult {
	const { ranks, experts, tokens, k, skew, capacity } = cfg;
	const rand = mulberry32(cfg.seed ?? 7);
	const perRank = experts / ranks;
	const owner = (e: number) => Math.floor(e / perRank);

	// A fixed shuffled order of popularity, so the popular experts land on different ranks.
	const order = Array.from({ length: experts }, (_, e) => e);
	for (let i = experts - 1; i > 0; i--) {
		const j = Math.floor(rand() * (i + 1));
		[order[i], order[j]] = [order[j], order[i]];
	}
	const logp = order.map((r) => -skew * Math.log(1 + r));

	const localCap =
		capacity === null ? null : Math.ceil((capacity * tokens * k) / experts);
	const load = new Array(experts).fill(0);
	const dropped = new Array(experts).fill(0);
	const send = Array.from({ length: ranks }, () => new Array(ranks).fill(0));

	for (let r = 0; r < ranks; r++) {
		const sentHere = new Array(experts).fill(0);
		for (let t = 0; t < tokens; t++) {
			const scores = logp.map(
				(lp) => lp - Math.log(-Math.log(rand() || 1e-12)),
			);
			const top = scores
				.map((s, e) => [s, e] as const)
				.sort((a, b) => b[0] - a[0])
				.slice(0, k)
				.map(([, e]) => e);
			for (const e of top) {
				if (localCap !== null && sentHere[e] >= localCap) {
					dropped[e]++;
					continue;
				}
				sentHere[e]++;
				load[e]++;
				send[r][owner(e)]++;
			}
		}
	}
	return { config: cfg, load, dropped, send, localCap, owner };
}

export interface MoeStats {
	/** Token copies the busiest rank's experts process, against the average. */
	maxRecv: number;
	meanRecv: number;
	droppedFrac: number;
}

export function moeStats(r: MoeResult): MoeStats {
	const { ranks } = r.config;
	const recv = Array.from({ length: ranks }, (_, j) =>
		r.send.reduce((s, row) => s + row[j], 0),
	);
	const total = r.load.reduce((a, b) => a + b, 0);
	const drops = r.dropped.reduce((a, b) => a + b, 0);
	return {
		maxRecv: Math.max(...recv),
		meanRecv: recv.reduce((a, b) => a + b, 0) / ranks,
		droppedFrac: drops / (total + drops),
	};
}

const RANK_COLORS = [
	"#f4a7a3",
	"#9fc8ef",
	"#a9dba6",
	"#f5d38c",
	"#c8b4ee",
	"#f3b6d6",
	"#9fe0d6",
	"#d9c7a4",
];

/** Expert loads as bars, grouped by rank, beside the rank-to-rank all-to-all matrix. */
export function drawMoe(r: MoeResult): string {
	const { ranks, experts, tokens, k } = r.config;
	const perRank = experts / ranks;
	const out: string[] = [];

	// ---- Bars.
	const BW = 22;
	const BG = 6;
	const GROUP = 14;
	const PH = 150; // plot height
	const TOP = 22;
	const LEFT = 30;
	const fair = (tokens * ranks * k) / experts;
	const yMax = Math.max(fair * 2.2, ...r.load.map((l, e) => l + r.dropped[e]));
	const y = (v: number) => TOP + PH - (PH * v) / yMax;
	const barX = (e: number) => LEFT + e * (BW + BG) + owner(e) * GROUP;
	function owner(e: number) {
		return Math.floor(e / perRank);
	}
	const plotW = barX(experts - 1) + BW - LEFT;

	out.push(`<text class="h" x="${LEFT}" y="12">Token copies per expert</text>`);
	for (let rk = 0; rk < ranks; rk++) {
		const x0 = barX(rk * perRank) - 4;
		const w = perRank * (BW + BG) - BG + 8;
		out.push(
			`<rect class="group" x="${x0}" y="${TOP}" width="${w}" height="${PH}" rx="3" style="fill:${RANK_COLORS[rk]}"/>`,
			`<text class="glabel" x="${x0 + w / 2}" y="${TOP + PH + 26}">rank ${rk}</text>`,
		);
	}
	for (let e = 0; e < experts; e++) {
		const x = barX(e);
		const l = r.load[e];
		const d = r.dropped[e];
		out.push(
			`<g><title>Expert ${e} (rank ${owner(e)}): ${l} token copies processed${d ? `, ${d} dropped` : ""}</title>`,
			`<rect class="bar" x="${x}" y="${y(l)}" width="${BW}" height="${y(0) - y(l)}"/>`,
			d
				? `<rect class="drop" x="${x}" y="${y(l + d)}" width="${BW}" height="${y(l) - y(l + d)}"/>`
				: "",
			`<text class="elabel" x="${x + BW / 2}" y="${TOP + PH + 12}">E${e}</text>`,
			"</g>",
		);
	}
	out.push(
		`<line class="fair" x1="${LEFT - 4}" x2="${LEFT + plotW + 4}" y1="${y(fair)}" y2="${y(fair)}"/>`,
		`<text class="tick" x="${LEFT - 6}" y="${y(fair) + 3}">${Math.round(fair)}</text>`,
		`<text class="tick" x="${LEFT - 6}" y="${y(0) + 3}">0</text>`,
	);
	if (r.localCap !== null) {
		const cap = r.localCap * ranks;
		out.push(
			`<line class="cap" x1="${LEFT - 4}" x2="${LEFT + plotW + 4}" y1="${y(cap)}" y2="${y(cap)}"/>`,
			`<text class="tick cap-t" x="${LEFT - 6}" y="${y(cap) + 3}">${cap}</text>`,
		);
	}

	// ---- Matrix.
	const MX = LEFT + plotW + 56;
	const C = 30;
	const MY = TOP + 18;
	const maxSend = Math.max(1, ...r.send.flat());
	out.push(
		`<text class="h" x="${MX}" y="12">All-to-all: copies sent from row to column</text>`,
	);
	for (let j = 0; j < ranks; j++)
		out.push(
			`<text class="mh" x="${MX + j * C + C / 2}" y="${MY - 5}">→${j}</text>`,
		);
	for (let i = 0; i < ranks; i++) {
		out.push(
			`<text class="mh end" x="${MX - 5}" y="${MY + i * C + C / 2 + 4}">${i}</text>`,
		);
		for (let j = 0; j < ranks; j++) {
			const v = r.send[i][j];
			const a = (0.06 + (0.5 * v) / maxSend).toFixed(3);
			out.push(
				`<g><title>Rank ${i} sends ${v} token copies to rank ${j}${i === j ? " (stays local)" : ""}</title>`,
				`<rect class="cell${i === j ? " self" : ""}" x="${MX + j * C}" y="${MY + i * C}" width="${C - 2}" height="${C - 2}" rx="2" style="fill-opacity:${a}"/>`,
				`<text class="cv" x="${MX + j * C + C / 2 - 1}" y="${MY + i * C + C / 2 + 3}">${v}</text></g>`,
			);
		}
	}
	const recv = Array.from({ length: ranks }, (_, j) =>
		r.send.reduce((s, row) => s + row[j], 0),
	);
	recv.forEach((v, j) =>
		out.push(
			`<text class="sum" x="${MX + j * C + C / 2 - 1}" y="${MY + ranks * C + 12}">${v}</text>`,
		),
	);
	out.push(
		`<text class="mh end" x="${MX - 5}" y="${MY + ranks * C + 12}">Σ</text>`,
	);

	const W = MX + ranks * C + 8;
	const H = TOP + PH + 34;
	return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" style="min-width:${Math.round(W * 0.7)}px" role="img" aria-label="Expert loads and all-to-all traffic">${out.join("")}</svg>`;
}
