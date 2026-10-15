// Two-dimensional PCA helpers for PcaExplorer: centring, sample covariance (n - 1),
// the closed-form eigendecomposition of a symmetric 2x2 matrix, and what happens when
// the centred points are projected onto a unit direction at angle θ.

export type Point = [number, number];

export interface Pca2 {
	mean: Point;
	centred: Point[];
	/** Sample covariance [[cxx, cxy], [cxy, cyy]], divided by n - 1. */
	cov: [[number, number], [number, number]];
	/** Eigenvalues, largest first. */
	values: [number, number];
	/** Unit eigenvectors matching `values`, signed so the larger-magnitude entry is positive. */
	vectors: [Point, Point];
	/** Angle of the first principal direction in degrees, in [0, 180). */
	angle: number;
}

export interface Projection {
	/** Unit direction. */
	u: Point;
	/** Coordinate of each centred point along u. */
	scores: number[];
	/** Foot of each centred point on the line through the origin along u. */
	feet: Point[];
	/** Sample variance of the scores (n - 1). */
	variance: number;
	/** Sum of squared perpendicular distances divided by n - 1. */
	residual: number;
}

const dot = (a: Point, b: Point) => a[0] * b[0] + a[1] * b[1];

/** Same sign rule as scikit-learn's svd_flip: the largest-magnitude entry is positive. */
function flip(v: Point): Point {
	const big = Math.abs(v[0]) >= Math.abs(v[1]) ? v[0] : v[1];
	return big < 0 ? [-v[0], -v[1]] : v;
}

export function pca2(points: Point[]): Pca2 {
	const n = points.length;
	if (n < 2) throw new Error("PCA needs at least two points");
	const mean: Point = [
		points.reduce((s, p) => s + p[0], 0) / n,
		points.reduce((s, p) => s + p[1], 0) / n,
	];
	const centred = points.map((p): Point => [p[0] - mean[0], p[1] - mean[1]]);
	let sxx = 0;
	let sxy = 0;
	let syy = 0;
	for (const [x, y] of centred) {
		sxx += x * x;
		sxy += x * y;
		syy += y * y;
	}
	const cxx = sxx / (n - 1);
	const cxy = sxy / (n - 1);
	const cyy = syy / (n - 1);
	const half = (cxx + cyy) / 2;
	const r = Math.hypot((cxx - cyy) / 2, cxy);
	const values: [number, number] = [half + r, half - r];
	// Angle of the top eigenvector of [[a, b], [b, c]] is atan2(2b, a - c) / 2.
	const t = Math.atan2(2 * cxy, cxx - cyy) / 2;
	const v1 = flip([Math.cos(t), Math.sin(t)]);
	const v2 = flip([-v1[1], v1[0]]);
	let angle = (Math.atan2(v1[1], v1[0]) * 180) / Math.PI;
	angle = ((angle % 180) + 180) % 180;
	return {
		mean,
		centred,
		cov: [
			[cxx, cxy],
			[cxy, cyy],
		],
		values,
		vectors: [v1, v2],
		angle,
	};
}

/** Project centred points onto the unit direction at `degrees`. */
export function project(centred: Point[], degrees: number): Projection {
	const a = (degrees * Math.PI) / 180;
	const u: Point = [Math.cos(a), Math.sin(a)];
	const scores = centred.map((p) => dot(p, u));
	const feet = scores.map((s): Point => [s * u[0], s * u[1]]);
	const n = centred.length;
	const variance = scores.reduce((s, x) => s + x * x, 0) / (n - 1);
	const residual =
		centred.reduce(
			(s, p, i) => s + (p[0] - feet[i][0]) ** 2 + (p[1] - feet[i][1]) ** 2,
			0,
		) /
		(n - 1);
	return { u, scores, feet, variance, residual };
}

/** Variance along the direction at `degrees`: uᵀ C u. */
export function directionalVariance(cov: Pca2["cov"], degrees: number): number {
	const a = (degrees * Math.PI) / 180;
	const c = Math.cos(a);
	const s = Math.sin(a);
	return cov[0][0] * c * c + 2 * cov[0][1] * c * s + cov[1][1] * s * s;
}
