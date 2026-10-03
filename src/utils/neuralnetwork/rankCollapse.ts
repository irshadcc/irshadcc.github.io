// The computation behind RankCollapseSteps.astro: push points on the unit circle through the same
// 2 × 2 linear layer L times and measure the batch matrix H (one point per row) at each depth.
// A 2-column matrix has a closed-form SVD (the eigenvalues of the 2 × 2 Gram matrix HᵀH), so
// everything here is exact up to float64 rounding and can be checked against NumPy.

export type Mat2 = [[number, number], [number, number]];
export type Vec2 = [number, number];

export interface CollapseStep {
	depth: number;
	/** The batch after `depth` layers, one row per input point. */
	points: Vec2[];
	/** Image of the unit circle after `depth` layers, sampled densely (the ellipse the points lie on). */
	outline: Vec2[];
	/** Singular values of the centred batch, largest first. */
	sigma: Vec2;
	/** Unit vector along the first right singular direction (the long axis). */
	major: Vec2;
	ratio: number;
	effectiveRank: number;
	stableRank: number;
	/** Rank as NumPy's matrix_rank counts it: singular values above σ₁ · max(n, 2) · ε. */
	exactRank: number;
	/** Euclidean distance between the two points of `pair`. */
	pairDistance: number;
	/** Number of distinct rows when every coordinate is rounded to `decimals` places. */
	distinct: number;
}

export const circlePoints = (
	n: number,
): { angles: number[]; points: Vec2[] } => {
	const angles = Array.from({ length: n }, (_, k) => (360 * k) / n);
	const points = angles.map((a) => {
		const t = (a * Math.PI) / 180;
		return [Math.cos(t), Math.sin(t)] as Vec2;
	});
	return { angles, points };
};

export const apply = (W: Mat2, [x, y]: Vec2): Vec2 => [
	W[0][0] * x + W[0][1] * y,
	W[1][0] * x + W[1][1] * y,
];

/** Singular values (largest first) and the leading right singular vector of an n × 2 matrix. */
export function svd2(rows: Vec2[]): { sigma: Vec2; major: Vec2 } {
	let a = 0;
	let b = 0;
	let c = 0;
	for (const [x, y] of rows) {
		a += x * x;
		b += x * y;
		c += y * y;
	}
	// Eigenvalues of [[a, b], [b, c]], written to avoid cancellation in the smaller one.
	const mid = (a + c) / 2;
	const rad = Math.hypot((a - c) / 2, b);
	const l1 = mid + rad;
	const det = a * c - b * b;
	const l2 = l1 > 0 ? Math.max(0, det / l1) : 0;
	const major: Vec2 =
		Math.abs(b) > 1e-300 ? norm([l1 - c, b]) : a >= c ? [1, 0] : [0, 1];
	return { sigma: [Math.sqrt(l1), Math.sqrt(l2)], major };
}

const norm = ([x, y]: Vec2): Vec2 => {
	const r = Math.hypot(x, y) || 1;
	return [x / r, y / r];
};

/** exp of the entropy of σᵢ / Σσⱼ: 1 when one direction carries everything, d when all are equal. */
export function effectiveRank(sigma: readonly number[]): number {
	const total = sigma.reduce((s, v) => s + v, 0);
	if (total === 0) return 0;
	let h = 0;
	for (const v of sigma) {
		const p = v / total;
		if (p > 0) h -= p * Math.log(p);
	}
	return Math.exp(h);
}

/** Σσᵢ² / σ₁². */
export function stableRank(sigma: readonly number[]): number {
	const top = Math.max(...sigma);
	return top === 0 ? 0 : sigma.reduce((s, v) => s + v * v, 0) / (top * top);
}

export function exactRank(sigma: readonly number[], rows: number): number {
	const top = Math.max(...sigma);
	const tol = top * Math.max(rows, sigma.length) * Number.EPSILON;
	return sigma.filter((v) => v > tol).length;
}

const centre = (rows: Vec2[]): Vec2[] => {
	const mx = rows.reduce((s, r) => s + r[0], 0) / rows.length;
	const my = rows.reduce((s, r) => s + r[1], 0) / rows.length;
	return rows.map(([x, y]) => [x - mx, y - my]);
};

export function countDistinct(rows: Vec2[], decimals: number): number {
	// Adding 0 turns -0 into 0, so a coordinate that rounds to zero from below matches one from above.
	const key = (v: number) =>
		(Number(v.toFixed(decimals)) + 0).toFixed(decimals);
	return new Set(rows.map(([x, y]) => `${key(x)},${key(y)}`)).size;
}

export interface CollapseOptions {
	/** The layer, applied as h ← W h at every depth. */
	matrix: Mat2;
	/** Number of input points, evenly spaced on the unit circle starting at 0°. */
	n?: number;
	/** Depths to report, e.g. [0, 1, 2, 4, 8, 16, 32]. */
	depths: number[];
	/** Indices of two points whose distance is tracked. */
	pair?: [number, number];
	/** Rounding used to count distinct points. */
	decimals?: number;
	/** Samples on the outline ellipse. */
	outlineSamples?: number;
}

export function collapseSteps(opts: CollapseOptions): {
	angles: number[];
	steps: CollapseStep[];
} {
	const {
		matrix,
		n = 12,
		depths,
		pair = [0, 3],
		decimals = 3,
		outlineSamples = 96,
	} = opts;
	const { angles, points: inputs } = circlePoints(n);
	const ring = circlePoints(outlineSamples).points;
	const maxDepth = Math.max(...depths);
	const wanted = new Set(depths);
	const steps: CollapseStep[] = [];
	let pts = inputs;
	let outline = ring;
	for (let depth = 0; depth <= maxDepth; depth++) {
		if (depth > 0) {
			pts = pts.map((p) => apply(matrix, p));
			outline = outline.map((p) => apply(matrix, p));
		}
		if (!wanted.has(depth)) continue;
		const { sigma, major } = svd2(centre(pts));
		const [i, j] = pair;
		steps.push({
			depth,
			points: pts,
			outline,
			sigma,
			major,
			ratio: sigma[0] === 0 ? 0 : sigma[1] / sigma[0],
			effectiveRank: effectiveRank(sigma),
			stableRank: stableRank(sigma),
			exactRank: exactRank(sigma, pts.length),
			pairDistance: Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]),
			distinct: countDistinct(pts, decimals),
		});
	}
	return { angles, steps };
}
