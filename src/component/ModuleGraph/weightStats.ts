// Statistics of a parameter (moduleGraph.ts's WeightSpec) for ModuleGraph's hover cards: a
// histogram of its values and its singular values, with three ranks that show rank collapse.
//   - numerical rank: singular values above a tolerance (numpy.linalg.matrix_rank's rule);
//   - effective rank (Roy & Vetterli, 2007): exp of the entropy of σᵢ / Σσⱼ, between 1 (one
//     direction carries everything) and min(m, n) (all σ equal);
//   - stable rank: ‖W‖²_F / σ₁² = Σσᵢ² / σ₁², at most the rank and insensitive to tiny σ.
// Pure: it runs at build time.
import type { WeightSpec } from "./moduleGraph";

/** Machine epsilon of float32, the usual dtype of a weight. */
export const EPS32 = 2 ** -23;

export interface WeightStats {
	shape: number[];
	/** Rows and columns of the matrix the singular values are of. */
	m: number;
	n: number;
	/** Present when the spec gives values. */
	count?: number;
	mean?: number;
	std?: number;
	min?: number;
	max?: number;
	histogram?: { edges: number[]; counts: number[] };
	/** Largest first; empty when neither values nor singular values are given. */
	sv: number[];
	/** Relative tolerance of the numerical rank (σ > tol · σ₁). */
	tol: number;
	rank?: number;
	effectiveRank?: number;
	stableRank?: number;
	note?: string;
}

/**
 * Singular values of a matrix, largest first, by one-sided Jacobi: rotate pairs of columns until
 * all are orthogonal; the singular values are then the column norms.
 */
export function singularValues(a: number[][]): number[] {
	if (!a.length || !a[0].length) return [];
	// Work on the side with fewer columns: A and Aᵀ have the same singular values.
	const wide = a[0].length > a.length;
	const rows = wide
		? a[0].map((_, j) => a.map((r) => r[j]))
		: a.map((r) => [...r]);
	const n = rows[0].length;
	const col = Array.from({ length: n }, (_, j) => rows.map((r) => r[j]));
	const dot = (x: number[], y: number[]) =>
		x.reduce((s, v, i) => s + v * y[i], 0);
	for (let sweep = 0; sweep < 60; sweep++) {
		let rotated = false;
		for (let p = 0; p < n - 1; p++)
			for (let q = p + 1; q < n; q++) {
				const alpha = dot(col[p], col[p]);
				const beta = dot(col[q], col[q]);
				const gamma = dot(col[p], col[q]);
				if (Math.abs(gamma) <= 1e-15 * Math.sqrt(alpha * beta)) continue;
				rotated = true;
				const zeta = (beta - alpha) / (2 * gamma);
				const t =
					Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
				const c = 1 / Math.sqrt(1 + t * t);
				const s = c * t;
				for (let i = 0; i < col[p].length; i++) {
					const [x, y] = [col[p][i], col[q][i]];
					col[p][i] = c * x - s * y;
					col[q][i] = s * x + c * y;
				}
			}
		if (!rotated) break;
	}
	return col.map((c) => Math.sqrt(dot(c, c))).sort((x, y) => y - x);
}

/** `bins` equal-width bins from the minimum to the maximum (one bin if all values are equal). */
export function histogram(values: number[], bins: number) {
	const lo = Math.min(...values);
	const hi = Math.max(...values);
	const k = hi > lo ? bins : 1;
	const width = hi > lo ? (hi - lo) / k : 1;
	const edges = Array.from({ length: k + 1 }, (_, i) => lo + i * width);
	const counts = new Array(k).fill(0);
	for (const v of values)
		counts[Math.min(k - 1, Math.floor((v - lo) / width))]++;
	return { edges, counts };
}

export function weightStats(w: WeightSpec): WeightStats {
	const [m, n] = w.values?.length
		? [w.values.length, w.values[0].length]
		: [w.shape[0] ?? 1, w.shape.slice(1).reduce((p, d) => p * d, 1)];
	const tol = w.rank_tol ?? Math.max(m, n) * EPS32;
	const out: WeightStats = { shape: w.shape, m, n, sv: [], tol, note: w.note };
	if (w.values?.length) {
		if (w.values.some((r) => r.length !== n))
			throw new Error(
				"ModuleGraph: weight values must be a matrix (rows of equal length)",
			);
		const flat = w.values.flat();
		const mean = flat.reduce((s, v) => s + v, 0) / flat.length;
		Object.assign(out, {
			count: flat.length,
			mean,
			std: Math.sqrt(
				flat.reduce((s, v) => s + (v - mean) ** 2, 0) / flat.length,
			),
			min: Math.min(...flat),
			max: Math.max(...flat),
			histogram: histogram(
				flat,
				Math.min(24, Math.max(4, Math.ceil(Math.sqrt(flat.length)))),
			),
			sv: singularValues(w.values),
		});
	}
	if (w.histogram) out.histogram = w.histogram;
	if (w.singular_values) out.sv = [...w.singular_values].sort((x, y) => y - x);
	const sv = out.sv;
	if (sv.length && sv[0] > 0) {
		const total = sv.reduce((s, v) => s + v, 0);
		const entropy = -sv.reduce(
			(s, v) => (v > 0 ? s + (v / total) * Math.log(v / total) : s),
			0,
		);
		out.rank = sv.filter((v) => v > tol * sv[0]).length;
		out.effectiveRank = Math.exp(entropy);
		out.stableRank = sv.reduce((s, v) => s + v * v, 0) / (sv[0] * sv[0]);
	} else if (sv.length) {
		out.rank = 0;
	}
	return out;
}

/** Three significant digits, without trailing zeros; tiny or huge values in exponent form. */
export const fmt = (v: number) =>
	v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5)
		? v.toExponential(2)
		: Number(v.toPrecision(3)).toString();
const esc = (t: string) =>
	t
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;");

const PLOT = { w: 168, h: 54, label: 11 };
/** At most this many singular values are drawn, the largest. */
export const MAX_SV_BARS = 48;

/** Bars of a histogram, with the range under it and a tick at zero when 0 is inside it. */
function histogramSvg(h: { edges: number[]; counts: number[] }): string {
	const { w, h: ht, label } = PLOT;
	const top = Math.max(1, ...h.counts);
	const bw = w / h.counts.length;
	const [lo, hi] = [h.edges[0], h.edges[h.edges.length - 1]];
	const bars = h.counts
		.map((c, i) => {
			const bh = (c / top) * ht;
			return `<rect class="wt-bar" x="${(i * bw + 0.5).toFixed(1)}" y="${(ht - bh).toFixed(1)}" width="${Math.max(0.5, bw - 1).toFixed(1)}" height="${bh.toFixed(1)}"><title>[${fmt(h.edges[i])}, ${fmt(h.edges[i + 1])}): ${c}</title></rect>`;
		})
		.join("");
	const zero =
		lo < 0 && hi > 0
			? `<line class="wt-zero" x1="${((-lo / (hi - lo)) * w).toFixed(1)}" x2="${((-lo / (hi - lo)) * w).toFixed(1)}" y1="0" y2="${ht}"/>`
			: "";
	return `<svg viewBox="0 0 ${w} ${ht + label}" width="${w}" height="${ht + label}">${bars}${zero}<line class="wt-axis" x1="0" x2="${w}" y1="${ht}" y2="${ht}"/><text x="0" y="${ht + 9}">${fmt(lo)}</text><text x="${w}" y="${ht + 9}" text-anchor="end">${fmt(hi)}</text></svg>`;
}

/** One bar per singular value, σᵢ / σ₁, largest first; those under the rank tolerance muted. */
function spectrumSvg(sv: number[], tol: number): string {
	const { w, h: ht, label } = PLOT;
	const shown = sv.slice(0, MAX_SV_BARS);
	const bw = w / Math.max(shown.length, 4);
	const bars = shown
		.map((v, i) => {
			const r = sv[0] > 0 ? v / sv[0] : 0;
			const bh = Math.max(r * ht, 0.5);
			return `<rect class="wt-bar${v > tol * sv[0] ? "" : " low"}" x="${(i * bw + 0.5).toFixed(1)}" y="${(ht - bh).toFixed(1)}" width="${Math.max(0.5, bw - 1).toFixed(1)}" height="${bh.toFixed(1)}"><title>σ${i + 1} = ${fmt(v)} (${fmt(r)} σ₁)</title></rect>`;
		})
		.join("");
	return `<svg viewBox="0 0 ${w} ${ht + label}" width="${w}" height="${ht + label}">${bars}<line class="wt-axis" x1="0" x2="${w}" y1="${ht}" y2="${ht}"/><text x="0" y="${ht + 9}">σ₁</text><text x="${w}" y="${ht + 9}" text-anchor="end">σ${shown.length}</text></svg>`;
}

/** A parameter's panel in a hover card: plots, ranks against min(m, n), moments, and a flag when rank-deficient. */
export function weightHtml(name: string, s: WeightStats): string {
	const full = Math.min(s.m, s.n);
	const plots = [
		s.histogram
			? `<figure>${histogramSvg(s.histogram)}<figcaption>values${s.count ? ` (${s.count})` : ""}</figcaption></figure>`
			: "",
		s.sv.length
			? `<figure>${spectrumSvg(s.sv, s.tol)}<figcaption>singular values σᵢ / σ₁${s.sv.length > MAX_SV_BARS ? `, first ${MAX_SV_BARS} of ${s.sv.length}` : ""}</figcaption></figure>`
			: "",
	].join("");
	const meter = (v: number) =>
		`<span class="wt-meter"><span style="width:${Math.min(100, (100 * v) / full).toFixed(1)}%"></span></span>`;
	const rows: string[] = [];
	if (s.rank !== undefined)
		rows.push(
			`<tr><th>rank</th><td>${s.rank} / ${full}</td><td>${meter(s.rank)}</td></tr>`,
		);
	if (s.effectiveRank !== undefined)
		rows.push(
			`<tr><th>effective rank</th><td>${fmt(s.effectiveRank)} / ${full}</td><td>${meter(s.effectiveRank)}</td></tr>`,
		);
	if (s.stableRank !== undefined)
		rows.push(
			`<tr><th>stable rank</th><td>${fmt(s.stableRank)}</td><td></td></tr>`,
		);
	if (s.mean !== undefined && s.std !== undefined)
		rows.push(
			`<tr><th>mean ± std</th><td colspan="2">${fmt(s.mean)} ± ${fmt(s.std)}</td></tr>`,
		);
	if (s.min !== undefined && s.max !== undefined)
		rows.push(
			`<tr><th>range</th><td colspan="2">[${fmt(s.min)}, ${fmt(s.max)}]</td></tr>`,
		);
	const warn =
		s.rank !== undefined && s.rank < full
			? `<p class="wt-warn">Rank-deficient: ${s.rank} of ${full} singular values above ${fmt(s.tol)} σ₁.</p>`
			: "";
	return `<div class="wt"><p class="wt-head"><code>${esc(name)}</code> <span>(${s.shape.join(", ")})</span></p>${plots ? `<div class="wt-plots">${plots}</div>` : ""}${rows.length ? `<table class="wt-stats"><tbody>${rows.join("")}</tbody></table>` : ""}${warn}${s.note ? `<p class="card-note">${esc(s.note)}</p>` : ""}</div>`;
}
