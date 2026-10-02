// SVG builders for the tensor-parallelism figures, used by TpMatmul.astro and TpBlock.astro.
//
// drawMatmul draws one split matrix multiply (column-parallel, row-parallel, or the MLP that
// chains the two), with every piece coloured by the rank that holds it. drawBlock draws which
// slice of each activation every rank holds as a transformer block runs, with plain tensor
// parallelism or with sequence parallelism, and where the collectives sit between them.

/** Light pastels shared with CollectiveSteps: the dark labels on them read in both themes. */
export const RANK_COLORS = [
	"#f4a7a3",
	"#9fc8ef",
	"#a9dba6",
	"#f5d38c",
	"#c8b4ee",
	"#f3b6d6",
	"#9fe0d6",
	"#d9c7a4",
];

const esc = (s: string) =>
	s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// ---- One split matrix multiply.

export type MatmulMode = "column" | "row" | "mlp";

interface Box {
	x: number;
	y: number;
	w: number;
	h: number;
}

/** A matrix, whole (replicated on every rank) or cut into one slab per rank. */
function matrix(
	b: Box,
	n: number,
	split: "none" | "cols" | "rows",
	label: string,
	sub?: string,
) {
	const out: string[] = [];
	if (split === "none") {
		out.push(
			`<rect class="m-rep" x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" rx="2"/>`,
		);
	} else {
		for (let r = 0; r < n; r++) {
			const s =
				split === "cols"
					? { x: b.x + (r * b.w) / n, y: b.y, w: b.w / n, h: b.h }
					: { x: b.x, y: b.y + (r * b.h) / n, w: b.w, h: b.h / n };
			out.push(
				`<rect class="slab" data-rank="${r}" x="${s.x}" y="${s.y}" width="${s.w}" height="${s.h}" style="fill:${RANK_COLORS[r]}"><title>Held by rank ${r}</title></rect>`,
			);
		}
		out.push(
			`<rect class="m-outline" x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" rx="2"/>`,
		);
	}
	out.push(
		`<text class="m-label" x="${b.x + b.w / 2}" y="${b.y + b.h + 15}">${esc(label)}</text>`,
	);
	if (sub)
		out.push(
			`<text class="m-sub" x="${b.x + b.w / 2}" y="${b.y + b.h + 27}">${esc(sub)}</text>`,
		);
	return out.join("");
}

/** n partial sums stacked like a fanned deck of cards, one per rank. */
/** How far each partial sum in the deck is offset from the one before. */
const deckOffset = (n: number) => Math.min(5, 18 / Math.max(1, n - 1));

function partials(b: Box, n: number, label: string) {
	const off = deckOffset(n);
	const w = b.w - off * (n - 1);
	const h = b.h - off * (n - 1);
	const out: string[] = [];
	for (let r = n - 1; r >= 0; r--)
		out.push(
			`<rect class="slab partial" data-rank="${r}" x="${b.x + r * off}" y="${b.y + r * off}" width="${w}" height="${h}" rx="2" style="fill:${RANK_COLORS[r]}"><title>Rank ${r}'s partial sum: the full shape, but only its share of the inner dimension</title></rect>`,
		);
	out.push(
		`<text class="m-label" x="${b.x + b.w / 2}" y="${b.y + b.h + 15}">${esc(label)}</text>`,
	);
	out.push(
		`<text class="m-sub" x="${b.x + b.w / 2}" y="${b.y + b.h + 27}">one per rank, same shape</text>`,
	);
	return out.join("");
}

/** The completed sum: every rank's colour, striped. */
function summed(b: Box, n: number, label: string) {
	const out: string[] = [];
	for (let r = 0; r < n; r++)
		out.push(
			`<rect x="${b.x + (r * b.w) / n}" y="${b.y}" width="${b.w / n}" height="${b.h}" style="fill:${RANK_COLORS[r]}"/>`,
		);
	out.push(
		`<rect class="m-outline" x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" rx="2"><title>The full sum, now on every rank</title></rect>`,
	);
	out.push(
		`<text class="m-label" x="${b.x + b.w / 2}" y="${b.y + b.h + 15}">${esc(label)}</text>`,
	);
	out.push(
		`<text class="m-sub" x="${b.x + b.w / 2}" y="${b.y + b.h + 27}">on every rank</text>`,
	);
	return out.join("");
}

const op = (x: number, y: number, s: string) =>
	`<text class="op" x="${x}" y="${y}">${esc(s)}</text>`;

function arrow(x1: number, x2: number, y: number, label: string, id: string) {
	return `<g class="comm"><line x1="${x1}" y1="${y}" x2="${x2 - 2}" y2="${y}" marker-end="url(#${id})"/><text x="${(x1 + x2) / 2}" y="${y - 7}">${esc(label)}</text></g>`;
}

export function drawMatmul(mode: MatmulMode, n: number, id: string): string {
	const S = 64; // rows of X: tokens
	const Hd = 64; // hidden size
	const F = mode === "mlp" ? 112 : 64; // output features of the first matrix
	const TOP = 16;
	const out: string[] = [];
	let x = 8;
	const mid = TOP + S / 2 + 5;
	const place = (w: number, h: number, gapAfter: number) => {
		const b = { x, y: TOP + (S - h) / 2, w, h };
		x += w + gapAfter;
		return b;
	};

	if (mode === "column") {
		out.push(matrix(place(Hd, S, 10), n, "none", "X", "every rank"));
		out.push(op(x - 2, mid, "×"));
		x += 10;
		out.push(matrix(place(F, Hd, 10), n, "cols", "W", "split by columns"));
		out.push(op(x - 2, mid, "="));
		x += 10;
		out.push(
			matrix(place(F, S, 0), n, "cols", "Y = [XWᵢ]", "split by columns"),
		);
	} else if (mode === "row") {
		out.push(matrix(place(Hd, S, 10), n, "cols", "X", "split by columns"));
		out.push(op(x - 2, mid, "×"));
		x += 10;
		out.push(matrix(place(Hd, Hd, 10), n, "rows", "W", "split by rows"));
		out.push(op(x - 2, mid, "="));
		x += 10;
		out.push(
			partials(
				place(F + deckOffset(n) * (n - 1), S + deckOffset(n) * (n - 1), 0),
				n,
				"XᵢWᵢ",
			),
		);
		const x1 = x + 6;
		x += 92;
		out.push(arrow(x1, x - 6, mid - 5, "all-reduce", `${id}-arrow`));
		out.push(summed(place(F, S, 0), n, "Y = Σ XᵢWᵢ"));
	} else {
		out.push(matrix(place(Hd, S, 10), n, "none", "X", "every rank"));
		out.push(op(x - 2, mid, "×"));
		x += 10;
		out.push(matrix(place(F, Hd, 10), n, "cols", "A", "split by columns"));
		out.push(op(x + 4, mid, "→ GeLU →"));
		x += 64;
		out.push(
			matrix(place(F, S, 10), n, "cols", "Z = GeLU(XA)", "split by columns"),
		);
		out.push(op(x - 2, mid, "×"));
		x += 10;
		out.push(matrix(place(Hd, F, 10), n, "rows", "B", "split by rows"));
		out.push(op(x - 2, mid, "="));
		x += 10;
		out.push(
			partials(
				place(Hd + deckOffset(n) * (n - 1), S + deckOffset(n) * (n - 1), 0),
				n,
				"ZᵢBᵢ",
			),
		);
		const x1 = x + 6;
		x += 92;
		out.push(arrow(x1, x - 6, mid - 5, "all-reduce", `${id}-arrow`));
		out.push(summed(place(Hd, S, 0), n, "Y"));
	}

	const W = x + 8;
	// Matrices are centred on X's rows; in the MLP, B is F tall, so shift everything down.
	const shift = mode === "mlp" ? (F - S) / 2 : 0;
	const H = TOP + Math.max(S, mode === "mlp" ? F : S) + 46;
	return [
		`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" style="min-width:${Math.round(W * 0.7)}px" role="img" aria-label="${mode}-parallel matrix multiply over ${n} ranks">`,
		`<defs><marker id="${id}-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path class="arrowhead" d="M0,0.5 L8,4 L0,7.5 z"/></marker></defs>`,
		`<g transform="translate(0 ${shift})">`,
		...out,
		"</g></svg>",
	].join("");
}

// ---- A whole transformer block.

/** What one rank holds of a (sequence × hidden) activation. */
type Hold = "full" | "seq" | "hidden" | "partial";

interface Stage {
	kind: "tensor";
	label: string;
	hold: Hold;
	tip: string;
}
interface Comm {
	kind: "comm";
	label: string;
	/** Communicates in the forward pass (f only does in the backward). */
	fwd: boolean;
	tip: string;
}

export function blockStages(sp: boolean): (Stage | Comm)[] {
	const t = (label: string, hold: Hold, tip: string): Stage => ({
		kind: "tensor",
		label,
		hold,
		tip,
	});
	const enter: Comm = sp
		? {
				kind: "comm",
				label: "AG",
				fwd: true,
				tip: "All-gather along the sequence (its backward is a reduce-scatter)",
			}
		: {
				kind: "comm",
				label: "f",
				fwd: false,
				tip: "f: nothing in the forward pass, an all-reduce in the backward pass",
			};
	const exit: Comm = sp
		? {
				kind: "comm",
				label: "RS",
				fwd: true,
				tip: "Reduce-scatter along the sequence (its backward is an all-gather)",
			}
		: {
				kind: "comm",
				label: "g",
				fwd: true,
				tip: "g: an all-reduce in the forward pass, nothing in the backward pass",
			};
	const outside: Hold = sp ? "seq" : "full";
	const where = sp
		? "this rank's tokens only"
		: "every token, the same on every rank";
	return [
		t("input", outside, `Block input: ${where}`),
		t("LayerNorm", outside, `Layer norm output: ${where}`),
		enter,
		t(
			"Q K V, attention",
			"hidden",
			"Q, K, V and the attention output: every token, this rank's heads only",
		),
		t(
			"out proj",
			"partial",
			"Output projection: every token and feature, but a partial sum over this rank's heads",
		),
		exit,
		t("+ residual", outside, `Residual sum: ${where}`),
		t("LayerNorm", outside, `Layer norm output: ${where}`),
		enter,
		t(
			"up, GeLU",
			"hidden",
			"First MLP layer and GeLU: every token, this rank's slice of the 4h features",
		),
		t(
			"down",
			"partial",
			"Second MLP layer: every token and feature, but a partial sum over this rank's features",
		),
		exit,
		t("+ residual", outside, `Block output: ${where}`),
	];
}

export function drawBlock(sp: boolean, n: number, id: string): string {
	const stages = blockStages(sp);
	const CW = 46; // a tensor cell: sequence across, hidden down
	const CH = 30;
	const GAP = 10;
	const COMM = 26;
	const LEFT = 52;
	const TOP = 30;
	const ROWGAP = 10;
	const out: string[] = [];
	let x = LEFT;
	const xs = stages.map((s) => {
		const at = x;
		x += (s.kind === "tensor" ? CW : COMM) + GAP;
		return at;
	});
	const W = x - GAP + 8;
	const rowY = (r: number) => TOP + r * (CH + ROWGAP);
	const H = rowY(n) - ROWGAP + 10;

	for (let r = 0; r < n; r++)
		out.push(
			`<text class="rank" x="${LEFT - 8}" y="${rowY(r) + CH / 2 + 4}">rank ${r}</text>`,
		);

	stages.forEach((s, i) => {
		const x0 = xs[i];
		if (s.kind === "comm") {
			const cx = x0 + COMM / 2;
			out.push(
				`<g class="comm-col${s.fwd ? " real" : ""}"><title>${esc(s.tip)}</title>`,
				`<rect class="comm-bar" x="${x0 + 3}" y="${TOP - 4}" width="${COMM - 6}" height="${H - TOP}" rx="4"/>`,
				`<text class="comm-label" x="${cx}" y="${TOP - 10}">${esc(s.label)}</text>`,
				"</g>",
			);
			return;
		}
		const lines = s.label.split(", ");
		lines.forEach((ln, k) =>
			out.push(
				`<text class="stage" x="${x0 + CW / 2}" y="${TOP - 10 - (lines.length - 1 - k) * 10}">${esc(ln)}</text>`,
			),
		);
		for (let r = 0; r < n; r++) {
			const y = rowY(r);
			const c = RANK_COLORS[r];
			out.push(
				`<g class="cell" data-rank="${r}"><title>${esc(`Rank ${r}. ${s.tip}`)}</title>`,
			);
			out.push(
				`<rect class="frame" x="${x0}" y="${y}" width="${CW}" height="${CH}" rx="2"/>`,
			);
			if (s.hold === "full")
				out.push(
					`<rect class="rep" x="${x0}" y="${y}" width="${CW}" height="${CH}" rx="2"/>`,
				);
			else if (s.hold === "seq")
				out.push(
					`<rect x="${x0 + (r * CW) / n}" y="${y}" width="${CW / n}" height="${CH}" style="fill:${c}"/>`,
				);
			else if (s.hold === "hidden")
				out.push(
					`<rect x="${x0}" y="${y + (r * CH) / n}" width="${CW}" height="${CH / n}" style="fill:${c}"/>`,
				);
			else
				out.push(
					`<rect class="part" x="${x0}" y="${y}" width="${CW}" height="${CH}" rx="2" style="fill:${c}"/>`,
					`<text class="sigma" x="${x0 + CW / 2}" y="${y + CH / 2 + 4}">partial</text>`,
				);
			out.push(
				`<rect class="frame-line" x="${x0}" y="${y}" width="${CW}" height="${CH}" rx="2"/></g>`,
			);
		}
	});

	return [
		`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" style="min-width:${Math.round(W * 0.75)}px" role="img" aria-label="Activations held by each of ${n} ranks through a transformer block${sp ? " with sequence parallelism" : ""}">`,
		...out,
		"</svg>",
	].join("");
}

/**
 * Activation bytes one rank stores for the backward pass of one layer, in units of s·b·h, from
 * Korthikanti et al. (2022), leaving out the attention-score term that FlashAttention avoids.
 */
export function activationUnits(sp: boolean, t: number): number {
	return sp ? 34 / t : 10 + 24 / t;
}
