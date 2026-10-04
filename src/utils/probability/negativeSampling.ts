// Computation behind NegativeSamplingExplorer and the figures of the post on nonuniform negative
// sampling (Wang, Zhang and Wang, NeurIPS 2021, arXiv:2110.13048).
//
// Running example: one feature x ~ N(0, 1) and log odds g(x) = alpha + beta * x with beta = 1.
// In the paper's notation f(x; beta) = x and the gradient of g with respect to (alpha, beta) is
// (1, x). For this model the paper's asymptotic variances have closed forms:
//   E{e^f} = e^{1/2},  M_f = e^{1/2} [[1, 1], [1, 2]],  V_f = [[2, -1], [-1, 1]].

export type Vec2 = [number, number];
export type Mat2 = [[number, number], [number, number]];
export type Phi = (x: number) => number;

export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));

/** Small seeded generator, so a figure draws the same data on every load. */
export function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** Standard normal draw by the Box-Muller transform. */
export function normal(rng: () => number): number {
	const u = 1 - rng();
	return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

export const normalPdf = (x: number): number =>
	Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

/** E{fn(x)} for x ~ N(0, 1), by the trapezoid rule on [-12, 12]. */
export function expectNormal(fn: (x: number) => number, steps = 6000): number {
	const lo = -12;
	const h = 24 / steps;
	let sum = 0;
	for (let i = 0; i <= steps; i++) {
		const x = lo + i * h;
		const w = i === 0 || i === steps ? 0.5 : 1;
		sum += w * fn(x) * normalPdf(x);
	}
	return sum * h;
}

/** E{s(x) (1, x)(1, x)^T} for x ~ N(0, 1). */
export function expectOuter(s: (x: number) => number): Mat2 {
	const a = expectNormal((x) => s(x));
	const b = expectNormal((x) => s(x) * x);
	const d = expectNormal((x) => s(x) * x * x);
	return [
		[a, b],
		[b, d],
	];
}

export function inv2(m: Mat2): Mat2 {
	const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
	return [
		[m[1][1] / det, -m[0][1] / det],
		[-m[1][0] / det, m[0][0] / det],
	];
}

export function mul2(a: Mat2, b: Mat2): Mat2 {
	return [
		[
			a[0][0] * b[0][0] + a[0][1] * b[1][0],
			a[0][0] * b[0][1] + a[0][1] * b[1][1],
		],
		[
			a[1][0] * b[0][0] + a[1][1] * b[1][0],
			a[1][0] * b[0][1] + a[1][1] * b[1][1],
		],
	];
}

export const scale2 = (k: number, m: Mat2): Mat2 => [
	[k * m[0][0], k * m[0][1]],
	[k * m[1][0], k * m[1][1]],
];

export const add2 = (a: Mat2, b: Mat2): Mat2 => [
	[a[0][0] + b[0][0], a[0][1] + b[0][1]],
	[a[1][0] + b[1][0], a[1][1] + b[1][1]],
];

export const trace2 = (m: Mat2): number => m[0][0] + m[1][1];

// ---- Theory for the running example -------------------------------------------------------

/** E{e^f(x)} = E{e^x} = e^{1/2}. */
export const E_EF = Math.exp(0.5);
export const MF: Mat2 = expectOuter(Math.exp);
export const MF_INV: Mat2 = inv2(MF);
/** Theorem 1: V_f = E{e^f} M_f^{-1}. */
export const VF: Mat2 = scale2(E_EF, MF_INV);

/** Uniform negative sampling: phi(x) = 1. */
export const phiUniform: Phi = () => 1;

/** e^{f(x)} ||M_f^{-1} (1, x)||: the optimal t(x; theta) of Theorem 3 without the factor e^alpha. */
export const aOptimalScore = (x: number): number =>
	Math.exp(x) *
	Math.hypot(MF_INV[0][0] + MF_INV[0][1] * x, MF_INV[1][0] + MF_INV[1][1] * x);
const A_NORM = expectNormal(aOptimalScore);
/** Theorem 3: phi_os(x) = t(x) / E{t(x)}, ignoring the truncation level T. */
export const phiOptimal: Phi = (x) => aOptimalScore(x) / A_NORM;
/** Simpler choice that keeps only p(x): phi(x) proportional to e^{f(x)}. */
export const phiProbability: Phi = (x) => Math.exp(x) / E_EF;

/** Theorem 2: V_sub = c E{e^f} M_f^{-1} E{phi^{-1} e^{2f} gdot gdot^T} M_f^{-1}. */
export function varianceSub(phi: Phi, c: number): Mat2 {
	const lam = expectOuter((x) => Math.exp(2 * x) / phi(x));
	return scale2(c * E_EF, mul2(mul2(MF_INV, lam), MF_INV));
}

/** Theorem 2: V_w = V_f + V_sub. */
export const varianceIpw = (phi: Phi, c: number): Mat2 =>
	add2(VF, varianceSub(phi, c));

/** Theorem 4: V_lik = E{e^f} Lambda_lik^{-1}, Lambda_lik = E{e^f gdot gdot^T / (1 + c e^f / phi)}. */
export function varianceLik(phi: Phi, c: number): Mat2 {
	const lam = expectOuter(
		(x) => Math.exp(x) / (1 + (c * Math.exp(x)) / phi(x)),
	);
	return scale2(E_EF, inv2(lam));
}

/** c E{e^f}: the limiting number of positives per kept negative in the subsample. */
export const ratioToC = (ratio: number): number => ratio / E_EF;

// ---- Data, sampling and fitting ---------------------------------------------------------------

export interface Dataset {
	x: Float64Array;
	y: Uint8Array;
}

export function generate(
	n: number,
	alpha: number,
	beta: number,
	rng: () => number,
): Dataset {
	const x = new Float64Array(n);
	const y = new Uint8Array(n);
	for (let i = 0; i < n; i++) {
		x[i] = normal(rng);
		y[i] = rng() < sigmoid(alpha + beta * x[i]) ? 1 : 0;
	}
	return { x, y };
}

export type Scheme = "uniform" | "optimal";

/**
 * pi(x): the probability that a negative at x is kept, min(max(rho phi(x), floor * rho), 1).
 * The optimal scheme uses the true parameter (an oracle pilot).
 */
export function negativeProb(
	scheme: Scheme,
	x: number,
	rho: number,
	floor = 0.1,
): number {
	if (scheme === "uniform") return Math.min(rho, 1);
	return Math.min(Math.max(rho * phiOptimal(x), floor * rho), 1);
}

/** The sampling function the theory sees once pi(x) is capped at 1 and floored: pi(x) / rho. */
export const effectivePhi =
	(scheme: Scheme, rho: number, floor = 0.1): Phi =>
	(x) =>
		negativeProb(scheme, x, rho, floor) / rho;

/**
 * Maximizes sum_i w_i [y_i eta_i - log(1 + exp(eta_i + o_i))] with eta_i = a + b x_i by Newton's
 * method with step halving. w = 1/pi gives the IPW estimator, o = -log pi(x) the likelihood
 * estimator with log odds correction, and w = 1, o = 0 the ordinary (full data) MLE.
 */
export function fitLogistic(
	x: ArrayLike<number>,
	y: ArrayLike<number>,
	w?: ArrayLike<number>,
	o?: ArrayLike<number>,
): Vec2 {
	const n = x.length;
	const wt = (i: number) => (w ? w[i] : 1);
	const off = (i: number) => (o ? o[i] : 0);
	let pos = 0;
	let neg = 0;
	let offNeg = 0;
	for (let i = 0; i < n; i++) {
		if (y[i]) pos += wt(i);
		else {
			neg += wt(i);
			offNeg += wt(i) * off(i);
		}
	}
	let th: Vec2 = [Math.log(pos / neg) - offNeg / neg, 0];
	const objective = (t: Vec2) => {
		let s = 0;
		for (let i = 0; i < n; i++) {
			const eta = t[0] + t[1] * x[i];
			const e = eta + off(i);
			const softplus =
				e > 0 ? e + Math.log1p(Math.exp(-e)) : Math.log1p(Math.exp(e));
			s += wt(i) * (y[i] * eta - softplus);
		}
		return s;
	};
	let current = objective(th);
	for (let iter = 0; iter < 50; iter++) {
		let g0 = 0;
		let g1 = 0;
		let h00 = 0;
		let h01 = 0;
		let h11 = 0;
		for (let i = 0; i < n; i++) {
			const p = sigmoid(th[0] + th[1] * x[i] + off(i));
			const r = wt(i) * (y[i] - p);
			const v = wt(i) * p * (1 - p);
			g0 += r;
			g1 += r * x[i];
			h00 += v;
			h01 += v * x[i];
			h11 += v * x[i] * x[i];
		}
		const det = h00 * h11 - h01 * h01;
		const step: Vec2 = [
			(h11 * g0 - h01 * g1) / det,
			(h00 * g1 - h01 * g0) / det,
		];
		let s = 1;
		let next: Vec2 = [th[0] + step[0], th[1] + step[1]];
		let value = objective(next);
		while (value < current - 1e-9 && s > 1e-6) {
			s /= 2;
			next = [th[0] + s * step[0], th[1] + s * step[1]];
			value = objective(next);
		}
		th = next;
		current = value;
		if (Math.max(Math.abs(s * step[0]), Math.abs(s * step[1])) < 1e-9) break;
	}
	return th;
}

export interface SampleResult {
	/** Indices of the kept instances (all positives and the sampled negatives). */
	kept: number[];
	positives: number;
	keptNegatives: number;
	naive: Vec2;
	ipw: Vec2;
	lik: Vec2;
}

/** Algorithm 1 on a dataset, then the three estimators on the subsample. */
export function sampleAndFit(
	data: Dataset,
	scheme: Scheme,
	rho: number,
	rng: () => number,
): SampleResult {
	const kept: number[] = [];
	const pis: number[] = [];
	let positives = 0;
	for (let i = 0; i < data.x.length; i++) {
		const pi = negativeProb(scheme, data.x[i], rho);
		if (data.y[i] === 1) {
			positives++;
			kept.push(i);
			pis.push(pi);
		} else if (rng() <= pi) {
			kept.push(i);
			pis.push(pi);
		}
	}
	const xs = kept.map((i) => data.x[i]);
	const ys = kept.map((i) => data.y[i]);
	// IPW weight is 1 / pi(x, y): 1 for positives, 1 / pi(x) for negatives.
	const weights = kept.map((i, k) => (data.y[i] === 1 ? 1 : 1 / pis[k]));
	// The log odds correction l = -log pi(x) uses the negative-class probability for every kept point.
	const offsets = pis.map((pi) => -Math.log(pi));
	return {
		kept,
		positives,
		keptNegatives: kept.length - positives,
		naive: fitLogistic(xs, ys),
		ipw: fitLogistic(xs, ys, weights),
		lik: fitLogistic(xs, ys, undefined, offsets),
	};
}
