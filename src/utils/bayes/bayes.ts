// Logic for the Bayesian statistics and optimisation figures: Beta–Bernoulli updating, Gaussian
// process regression and the Bayesian optimisation loop with EI, PI, UCB and Thompson sampling.
// Everything is deterministic given a seed, so figures and the numbers quoted in the post agree.
// No imports, so the file can be run directly with `node --experimental-strip-types`.

// ---------------------------------------------------------------- special functions

/** ln Γ(x) for x > 0 (Lanczos, g = 7). */
export function logGamma(x: number): number {
	const c = [
		0.99999999999980993, 676.5203681218851, -1259.1392167224028,
		771.32342877765313, -176.61502916214059, 12.507343278686905,
		-0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
	];
	if (x < 0.5)
		return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
	x -= 1;
	let a = c[0];
	const t = x + 7.5;
	for (let i = 1; i < 9; i++) a += c[i] / (x + i);
	return (
		0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a)
	);
}

/** ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a + b). */
export const logBeta = (a: number, b: number) =>
	logGamma(a) + logGamma(b) - logGamma(a + b);

/** Beta(a, b) density at θ. */
export function betaPdf(t: number, a: number, b: number): number {
	if (t < 0 || t > 1) return 0;
	if (t === 0)
		return a < 1
			? Number.POSITIVE_INFINITY
			: a === 1
				? Math.exp(-logBeta(a, b))
				: 0;
	if (t === 1)
		return b < 1
			? Number.POSITIVE_INFINITY
			: b === 1
				? Math.exp(-logBeta(a, b))
				: 0;
	return Math.exp(
		(a - 1) * Math.log(t) + (b - 1) * Math.log(1 - t) - logBeta(a, b),
	);
}

/** Continued fraction for the incomplete beta function (modified Lentz). */
function betaContinuedFraction(x: number, a: number, b: number): number {
	const tiny = 1e-300;
	let c = 1;
	let d = 1 - ((a + b) * x) / (a + 1);
	if (Math.abs(d) < tiny) d = tiny;
	d = 1 / d;
	let h = d;
	for (let m = 1; m <= 300; m++) {
		const m2 = 2 * m;
		let aa = (m * (b - m) * x) / ((a + m2 - 1) * (a + m2));
		d = 1 + aa * d;
		if (Math.abs(d) < tiny) d = tiny;
		c = 1 + aa / c;
		if (Math.abs(c) < tiny) c = tiny;
		d = 1 / d;
		h *= d * c;
		aa = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1));
		d = 1 + aa * d;
		if (Math.abs(d) < tiny) d = tiny;
		c = 1 + aa / c;
		if (Math.abs(c) < tiny) c = tiny;
		d = 1 / d;
		const del = d * c;
		h *= del;
		if (Math.abs(del - 1) < 1e-14) break;
	}
	return h;
}

/** Beta(a, b) CDF: the regularised incomplete beta function I_x(a, b). */
export function betaCdf(x: number, a: number, b: number): number {
	if (x <= 0) return 0;
	if (x >= 1) return 1;
	const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - logBeta(a, b));
	if (x < (a + 1) / (a + b + 2))
		return (front * betaContinuedFraction(x, a, b)) / a;
	return 1 - (front * betaContinuedFraction(1 - x, b, a)) / b;
}

/** Beta(a, b) quantile by bisection on the CDF. */
export function betaQuantile(p: number, a: number, b: number): number {
	let lo = 0;
	let hi = 1;
	for (let i = 0; i < 80; i++) {
		const mid = (lo + hi) / 2;
		if (betaCdf(mid, a, b) < p) lo = mid;
		else hi = mid;
	}
	return (lo + hi) / 2;
}

export const normPdf = (z: number) =>
	Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

/** Standard normal CDF Φ(z), via erfc (Numerical Recipes Chebyshev fit, |error| < 1.2e-7). */
export function normCdf(z: number): number {
	const x = Math.abs(z) / Math.SQRT2;
	const t = 1 / (1 + 0.5 * x);
	const erfc =
		t *
		Math.exp(
			-x * x -
				1.26551223 +
				t *
					(1.00002368 +
						t *
							(0.37409196 +
								t *
									(0.09678418 +
										t *
											(-0.18628806 +
												t *
													(0.27886807 +
														t *
															(-1.13520398 +
																t *
																	(1.48851587 +
																		t * (-0.82215223 + t * 0.17087277)))))))),
		);
	return z >= 0 ? 1 - erfc / 2 : erfc / 2;
}

// ---------------------------------------------------------------- Beta–Bernoulli updating

export interface BetaState {
	/** Flips seen so far. */
	n: number;
	heads: number;
	a: number;
	b: number;
	mean: number;
	/** Posterior mode (NaN when a or b ≤ 1 and the mode sits on the boundary). */
	mode: number;
	/** Maximum-likelihood estimate heads / n (NaN before any data). */
	mle: number;
	/** Equal-tailed 95% credible interval. */
	lo: number;
	hi: number;
	/** Posterior predictive probability that the next flip is heads. */
	predictive: number;
}

/** Beta(a0, b0) prior updated by the first `n` flips of `flips` (1 = heads, 0 = tails). */
export function betaPosterior(
	a0: number,
	b0: number,
	flips: number[],
	n = flips.length,
): BetaState {
	const heads = flips.slice(0, n).reduce((s, f) => s + f, 0);
	const a = a0 + heads;
	const b = b0 + n - heads;
	return {
		n,
		heads,
		a,
		b,
		mean: a / (a + b),
		mode: a > 1 && b > 1 ? (a - 1) / (a + b - 2) : Number.NaN,
		mle: n ? heads / n : Number.NaN,
		lo: betaQuantile(0.025, a, b),
		hi: betaQuantile(0.975, a, b),
		predictive: a / (a + b),
	};
}

/** Log marginal likelihood of a specific flip sequence with h heads, t tails under Beta(a, b). */
export const logEvidenceBeta = (h: number, t: number, a = 1, b = 1) =>
	logBeta(a + h, b + t) - logBeta(a, b);

// ---------------------------------------------------------------- linear algebra

export type Matrix = number[][];

/** Lower-triangular Cholesky factor of a symmetric positive-definite matrix. */
export function cholesky(A: Matrix): Matrix {
	const n = A.length;
	const L: Matrix = Array.from({ length: n }, () => new Array(n).fill(0));
	for (let i = 0; i < n; i++) {
		for (let j = 0; j <= i; j++) {
			let s = A[i][j];
			for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
			if (i === j) L[i][i] = Math.sqrt(Math.max(s, 1e-300));
			else L[i][j] = s / L[j][j];
		}
	}
	return L;
}

/** Solve L x = b for lower-triangular L. */
export function forwardSolve(L: Matrix, b: number[]): number[] {
	const x = b.slice();
	for (let i = 0; i < x.length; i++) {
		for (let k = 0; k < i; k++) x[i] -= L[i][k] * x[k];
		x[i] /= L[i][i];
	}
	return x;
}

/** Solve Lᵀ x = b for lower-triangular L. */
export function backSolve(L: Matrix, b: number[]): number[] {
	const x = b.slice();
	for (let i = x.length - 1; i >= 0; i--) {
		for (let k = i + 1; k < x.length; k++) x[i] -= L[k][i] * x[k];
		x[i] /= L[i][i];
	}
	return x;
}

/** Inverse of a small symmetric positive-definite matrix via Cholesky. */
export function spdInverse(A: Matrix): Matrix {
	const L = cholesky(A);
	const n = A.length;
	const cols = Array.from({ length: n }, (_, j) =>
		backSolve(
			L,
			forwardSolve(
				L,
				Array.from({ length: n }, (_, i) => (i === j ? 1 : 0)),
			),
		),
	);
	return Array.from({ length: n }, (_, i) => cols.map((c) => c[i]));
}

// ---------------------------------------------------------------- Bayesian linear regression

export interface LinearPosterior {
	mean: number[];
	cov: Matrix;
	/** Predictive mean and standard deviation (including noise) at basis vector φ. */
	predict: (phi: number[]) => { mean: number; sd: number };
}

/** Posterior of w for t = wᵀφ(x) + ε, ε ~ N(0, 1/β), prior w ~ N(0, I/α). */
export function bayesLinear(
	Phi: Matrix,
	t: number[],
	alpha: number,
	beta: number,
): LinearPosterior {
	const d = Phi[0].length;
	const A: Matrix = Array.from({ length: d }, (_, i) =>
		Array.from(
			{ length: d },
			(_, j) =>
				(i === j ? alpha : 0) + beta * Phi.reduce((s, r) => s + r[i] * r[j], 0),
		),
	);
	const cov = spdInverse(A);
	const rhs = Array.from({ length: d }, (_, i) =>
		Phi.reduce((s, r, n) => s + r[i] * t[n], 0),
	);
	const mean = cov.map(
		(row) => beta * row.reduce((s, v, j) => s + v * rhs[j], 0),
	);
	const predict = (phi: number[]) => {
		const m = phi.reduce((s, v, i) => s + v * mean[i], 0);
		let q = 0;
		for (let i = 0; i < d; i++)
			for (let j = 0; j < d; j++) q += phi[i] * cov[i][j] * phi[j];
		return { mean: m, sd: Math.sqrt(1 / beta + q) };
	};
	return { mean, cov, predict };
}

// ---------------------------------------------------------------- Gaussian processes

export interface GpHyper {
	/** Length-scale ℓ of the squared-exponential kernel. */
	ell: number;
	/** Signal standard deviation σ_f (in standardised units of y). */
	sf: number;
	/** Noise standard deviation σ_n (in standardised units of y). */
	sn: number;
}

export const rbf = (a: number, b: number, h: GpHyper) =>
	h.sf * h.sf * Math.exp(-((a - b) ** 2) / (2 * h.ell * h.ell));

export interface GpFit {
	X: number[];
	y: number[];
	h: GpHyper;
	/** Mean and standard deviation used to standardise y. */
	yMean: number;
	yStd: number;
	L: Matrix;
	alpha: number[];
}

/** Fit a zero-mean GP to standardised y (y − mean) / std. */
export function gpFit(X: number[], y: number[], h: GpHyper): GpFit {
	const n = y.length;
	const yMean = y.reduce((s, v) => s + v, 0) / n;
	const v = y.reduce((s, u) => s + (u - yMean) ** 2, 0) / n;
	const yStd = n > 1 && v > 0 ? Math.sqrt(v) : 1;
	const z = y.map((u) => (u - yMean) / yStd);
	const K = X.map((a, i) =>
		X.map((b, j) => rbf(a, b, h) + (i === j ? h.sn * h.sn + 1e-10 : 0)),
	);
	const L = cholesky(K);
	const alpha = backSolve(L, forwardSolve(L, z));
	return { X, y, h, yMean, yStd, L, alpha };
}

/** Posterior mean and standard deviation of f (no noise) at x, in the units of y. */
export function gpPredict(fit: GpFit, x: number): { mu: number; sd: number } {
	const k = fit.X.map((a) => rbf(a, x, fit.h));
	const mu = k.reduce((s, v, i) => s + v * fit.alpha[i], 0);
	const w = forwardSolve(fit.L, k);
	const v2 = Math.max(rbf(x, x, fit.h) - w.reduce((s, u) => s + u * u, 0), 0);
	return { mu: fit.yMean + fit.yStd * mu, sd: fit.yStd * Math.sqrt(v2) };
}

/** Log marginal likelihood of the standardised data under the GP (eq. 2.30 of Rasmussen & Williams). */
export function gpLogEvidence(fit: GpFit): number {
	const z = fit.y.map((u) => (u - fit.yMean) / fit.yStd);
	const n = z.length;
	let logDet = 0;
	for (let i = 0; i < n; i++) logDet += 2 * Math.log(fit.L[i][i]);
	const quad = z.reduce((s, v, i) => s + v * fit.alpha[i], 0);
	return -0.5 * quad - 0.5 * logDet - 0.5 * n * Math.log(2 * Math.PI);
}

/** One joint posterior sample of f on `grid` (standard normals from `randn`). */
export function gpSample(
	fit: GpFit,
	grid: number[],
	randn: () => number,
): number[] {
	const m = grid.length;
	const W = grid.map((x) =>
		forwardSolve(
			fit.L,
			fit.X.map((a) => rbf(a, x, fit.h)),
		),
	);
	const mean = grid.map((x) =>
		fit.X.reduce((s, a, i) => s + rbf(a, x, fit.h) * fit.alpha[i], 0),
	);
	const C: Matrix = Array.from({ length: m }, (_, i) =>
		Array.from({ length: m }, (_, j) => {
			let s = rbf(grid[i], grid[j], fit.h);
			for (let k = 0; k < fit.X.length; k++) s -= W[i][k] * W[j][k];
			return s + (i === j ? 1e-8 : 0);
		}),
	);
	const L = cholesky(C);
	const z = grid.map(() => randn());
	return grid.map(
		(_, i) =>
			fit.yMean +
			fit.yStd * (mean[i] + L[i].reduce((s, v, k) => s + v * z[k], 0)),
	);
}

// ---------------------------------------------------------------- acquisition functions (maximisation)

/** Expected improvement over the incumbent `best`, with exploration margin ξ. */
export function expectedImprovement(
	mu: number,
	sd: number,
	best: number,
	xi = 0,
): number {
	const d = mu - best - xi;
	if (sd <= 1e-12) return Math.max(d, 0);
	const z = d / sd;
	return d * normCdf(z) + sd * normPdf(z);
}

/** Probability of improving on `best` by at least ξ. */
export function probabilityOfImprovement(
	mu: number,
	sd: number,
	best: number,
	xi = 0,
): number {
	const d = mu - best - xi;
	if (sd <= 1e-12) return d > 0 ? 1 : 0;
	return normCdf(d / sd);
}

/** Upper confidence bound μ + κσ. */
export const upperConfidenceBound = (mu: number, sd: number, kappa = 2) =>
	mu + kappa * sd;

// ---------------------------------------------------------------- the optimisation loop

/** Negated Forrester et al. (2008) test function: maximum ≈ 6.0207 at x ≈ 0.7572 on [0, 1]. */
export const forrester = (x: number) =>
	-((6 * x - 2) ** 2) * Math.sin(12 * x - 4);

export type Acquisition = "ei" | "pi" | "ucb" | "ts";

export interface BoStep {
	X: number[];
	y: number[];
	/** Posterior mean and sd on the grid. */
	mu: number[];
	sd: number[];
	/** Acquisition value on the grid (for Thompson sampling, the posterior sample). */
	acq: number[];
	/** Index into the grid of the next point to evaluate. */
	next: number;
	best: number;
	bestX: number;
}

export interface BoOptions {
	f?: (x: number) => number;
	init?: number[];
	iterations?: number;
	acquisition?: Acquisition;
	/** ξ for EI and PI, κ for UCB. */
	param?: number;
	h?: GpHyper;
	gridSize?: number;
	seed?: number;
}

/** Small seeded uniform generator (mulberry32). */
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

/** Standard normals from a uniform generator (Box–Muller). */
export function normals(rng: () => number): () => number {
	return () => {
		const u = 1 - rng();
		return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
	};
}

export const boGrid = (n: number) =>
	Array.from({ length: n }, (_, i) => i / (n - 1));

/** Run Bayesian optimisation; step k holds the model fitted to init + k evaluations. */
export function runBo(opt: BoOptions = {}): BoStep[] {
	const f = opt.f ?? forrester;
	const h = opt.h ?? { ell: 0.1, sf: 1, sn: 1e-3 };
	const acquisition = opt.acquisition ?? "ei";
	const param =
		opt.param ?? (acquisition === "ucb" ? 2 : acquisition === "pi" ? 0.01 : 0);
	const grid = boGrid(opt.gridSize ?? 201);
	const randn = normals(mulberry32(opt.seed ?? 1));
	const X = (opt.init ?? [0, 0.5, 1]).slice();
	const y = X.map(f);
	const steps: BoStep[] = [];
	for (let it = 0; it <= (opt.iterations ?? 10); it++) {
		const fit = gpFit(X, y, h);
		const pred = grid.map((x) => gpPredict(fit, x));
		const mu = pred.map((p) => p.mu);
		const sd = pred.map((p) => p.sd);
		const bi = y.indexOf(Math.max(...y));
		const best = y[bi];
		// The acquisition's scale is set by y's spread; ξ is given in standardised units.
		const scale = fit.yStd;
		let acq: number[];
		if (acquisition === "ei")
			acq = pred.map((p) =>
				expectedImprovement(p.mu, p.sd, best, param * scale),
			);
		else if (acquisition === "pi")
			acq = pred.map((p) =>
				probabilityOfImprovement(p.mu, p.sd, best, param * scale),
			);
		else if (acquisition === "ucb")
			acq = pred.map((p) => upperConfidenceBound(p.mu, p.sd, param));
		else acq = gpSample(fit, grid, randn);
		let next = 0;
		for (let i = 1; i < grid.length; i++) if (acq[i] > acq[next]) next = i;
		steps.push({
			X: X.slice(),
			y: y.slice(),
			mu,
			sd,
			acq,
			next,
			best,
			bestX: X[bi],
		});
		X.push(grid[next]);
		y.push(f(grid[next]));
	}
	return steps;
}
