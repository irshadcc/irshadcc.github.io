// Computations behind the figures for the Probabilities post (Bishop & Bishop, Deep Learning,
// chapter 2). No imports, so it runs directly with `node probTheory.ts` for checks.
// Everything is deterministic: random data comes from mulberry32 with a fixed seed, so the
// numbers quoted in the post equal what the figures show.

export type Rng = () => number;

/** Small fast seeded PRNG on [0, 1). */
export function mulberry32(seed: number): Rng {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** Standard normal draws by the Box–Muller method. */
export function normals(rng: Rng): () => number {
	let spare: number | null = null;
	return () => {
		if (spare !== null) {
			const s = spare;
			spare = null;
			return s;
		}
		const u = 1 - rng();
		const v = rng();
		const r = Math.sqrt(-2 * Math.log(u));
		spare = r * Math.sin(2 * Math.PI * v);
		return r * Math.cos(2 * Math.PI * v);
	};
}

export const gauss = (x: number, mu: number, s2: number) =>
	Math.exp(-((x - mu) ** 2) / (2 * s2)) / Math.sqrt(2 * Math.PI * s2);

// ---------------------------------------------------------------------------------------------
// 2.1 Medical screening (eqs 2.12–2.22)

export interface Screening {
	/** p(T = 1) by the sum rule (2.20). */
	pPos: number;
	/** p(C = 1 | T = 1) by Bayes' theorem (2.21). */
	pCancerGivenPos: number;
	/** p(C = 1 | T = 0). */
	pCancerGivenNeg: number;
	/** Joint p(C, T) as [[C0T0, C0T1], [C1T0, C1T1]]. */
	joint: [[number, number], [number, number]];
}

export function screening(
	prior: number,
	sensitivity: number,
	falsePositive: number,
): Screening {
	const c1t1 = sensitivity * prior;
	const c1t0 = (1 - sensitivity) * prior;
	const c0t1 = falsePositive * (1 - prior);
	const c0t0 = (1 - falsePositive) * (1 - prior);
	const pPos = c1t1 + c0t1;
	return {
		pPos,
		pCancerGivenPos: c1t1 / pPos,
		pCancerGivenNeg: c1t0 / (1 - pPos),
		joint: [
			[c0t0, c0t1],
			[c1t0, c1t1],
		],
	};
}

/**
 * Whole-person counts out of `n` for a population figure: [no cancer & negative, no cancer &
 * positive, cancer & negative, cancer & positive], rounded so they add up to n.
 */
export function screeningCounts(
	n: number,
	prior: number,
	sensitivity: number,
	falsePositive: number,
): [number, number, number, number] {
	const sick = Math.round(n * prior);
	const tp = Math.round(sick * sensitivity);
	const fp = Math.round((n - sick) * falsePositive);
	return [n - sick - fp, fp, sick - tp, tp];
}

// ---------------------------------------------------------------------------------------------
// 2.1.2 Sum and product rules from counts (Figures 2.4 and 2.5)

/** Draw `n` samples from a joint table p[j][i] (row j = value of Y, column i = value of X). */
export function sampleCounts(
	p: number[][],
	n: number,
	seed: number,
): number[][] {
	const rng = mulberry32(seed);
	const flat = p.flat();
	const total = flat.reduce((a, b) => a + b, 0);
	const cols = p[0].length;
	const counts = p.map((row) => row.map(() => 0));
	for (let k = 0; k < n; k++) {
		let u = rng() * total;
		let idx = 0;
		while (idx < flat.length - 1 && u >= flat[idx]) {
			u -= flat[idx];
			idx++;
		}
		counts[Math.floor(idx / cols)][idx % cols]++;
	}
	return counts;
}

/** Joint over X ∈ {1..9} (columns) and Y ∈ {1, 2} (rows), in the spirit of Figure 2.5. */
export const figure25Joint: number[][] = (() => {
	const y1 = [1, 3, 6, 8, 6, 3, 1, 0, 0]; // Y = 1: narrow bump on the left
	const y2 = [0, 1, 2, 3, 4, 5, 6, 5, 4]; // Y = 2: broad bump to the right
	return [y2, y1];
})();

// ---------------------------------------------------------------------------------------------
// 2.3 Gaussian maximum likelihood

export function gaussLogLik(xs: number[], mu: number, s2: number): number {
	let ss = 0;
	for (const x of xs) ss += (x - mu) ** 2;
	const n = xs.length;
	return (
		-ss / (2 * s2) - (n / 2) * Math.log(s2) - (n / 2) * Math.log(2 * Math.PI)
	);
}

export function gaussMl(xs: number[]): { mu: number; s2: number } {
	const n = xs.length;
	const mu = xs.reduce((a, b) => a + b, 0) / n;
	const s2 = xs.reduce((a, x) => a + (x - mu) ** 2, 0) / n;
	return { mu, s2 };
}

/** The small data set used in the likelihood figure. */
export const likelihoodData = [0.3, 1.1, 1.6, 2.2, 2.4, 3.4];

export interface BiasRun {
	/** Each data set of size n drawn from N(mu, s2). */
	sets: number[][];
	muMl: number[];
	s2Ml: number[];
}

/** `count` data sets of `n` points from N(0, 1), for the bias figure (2.59–2.60). */
export function biasRuns(n: number, count: number, seed: number): BiasRun {
	const z = normals(mulberry32(seed));
	const sets: number[][] = [];
	const muMl: number[] = [];
	const s2Ml: number[] = [];
	for (let k = 0; k < count; k++) {
		const xs = Array.from({ length: n }, () => z());
		const ml = gaussMl(xs);
		sets.push(xs);
		muMl.push(ml.mu);
		s2Ml.push(ml.s2);
	}
	return { sets, muMl, s2Ml };
}

// ---------------------------------------------------------------------------------------------
// 2.3.4 / 2.6.2 Polynomial regression: maximum likelihood and MAP

/** Solve A x = b by Gaussian elimination with partial pivoting (A is copied). */
export function solve(A: number[][], b: number[]): number[] {
	const n = b.length;
	const M = A.map((r, i) => [...r, b[i]]);
	for (let c = 0; c < n; c++) {
		let piv = c;
		for (let r = c + 1; r < n; r++)
			if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
		[M[c], M[piv]] = [M[piv], M[c]];
		for (let r = c + 1; r < n; r++) {
			const f = M[r][c] / M[c][c];
			for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
		}
	}
	const x = new Array<number>(n).fill(0);
	for (let r = n - 1; r >= 0; r--) {
		let s = M[r][n];
		for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
		x[r] = s / M[r][r];
	}
	return x;
}

export const poly = (w: number[], x: number) =>
	w.reduceRight((acc, wi) => acc * x + wi, 0);

/** The sine-curve data of Section 1.2: N points at x = n/(N−1), t = sin 2πx + N(0, 0.3²). */
export function sineData(
	n = 10,
	seed = 7,
	noise = 0.3,
): { x: number[]; t: number[] } {
	const z = normals(mulberry32(seed));
	const x = Array.from({ length: n }, (_, i) => i / (n - 1));
	const t = x.map((xi) => Math.sin(2 * Math.PI * xi) + noise * z());
	return { x, t };
}

export interface PolyFit {
	w: number[];
	/** Sum-of-squares error (2.67). */
	sse: number;
	/** σ²_ML (2.68) for this w. */
	s2: number;
	/** wᵀw */
	norm2: number;
}

/**
 * Minimise ½ Σ {y(x_n, w) − t_n}² + (λ/2) wᵀw for an order-M polynomial; λ = 0 is maximum
 * likelihood, λ = σ²/s² is MAP under the prior (2.115).
 */
export function fitPoly(
	x: number[],
	t: number[],
	M: number,
	lambda: number,
): PolyFit {
	const P = M + 1;
	const phi = x.map((xi) => Array.from({ length: P }, (_, j) => xi ** j));
	const A = Array.from({ length: P }, (_, i) =>
		Array.from({ length: P }, (_, j) => {
			let s = i === j ? lambda : 0;
			for (const row of phi) s += row[i] * row[j];
			return s;
		}),
	);
	const b = Array.from({ length: P }, (_, i) =>
		phi.reduce((s, row, n) => s + row[i] * t[n], 0),
	);
	const w = solve(A, b);
	const sse = x.reduce((s, xi, n) => s + (poly(w, xi) - t[n]) ** 2, 0);
	return {
		w,
		sse: sse / 2,
		s2: sse / x.length,
		norm2: w.reduce((s, v) => s + v * v, 0),
	};
}

// ---------------------------------------------------------------------------------------------
// 2.4 Transformation of densities (Figure 2.12)

/** x = g(y) = ln y − ln(1 − y) + 5 (2.74) and its inverse, the logistic sigmoid (2.75). */
export const gLogit = (y: number) => Math.log(y) - Math.log(1 - y) + 5;
export const sigmoid5 = (x: number) => 1 / (1 + Math.exp(-x + 5));

/** p_y(y) = p_x(g(y)) |g'(y)| with p_x = N(mu, s²) and g'(y) = 1 / (y(1 − y)). */
export const pyTransformed = (y: number, mu: number, s2: number) =>
	y <= 0 || y >= 1 ? 0 : gauss(gLogit(y), mu, s2) / (y * (1 - y));

/** Argmax on a fine grid followed by golden-section refinement. */
export function argmax(
	f: (x: number) => number,
	lo: number,
	hi: number,
): number {
	const n = 4000;
	let best = lo;
	let bv = Number.NEGATIVE_INFINITY;
	for (let i = 0; i <= n; i++) {
		const x = lo + ((hi - lo) * i) / n;
		const v = f(x);
		if (v > bv) {
			bv = v;
			best = x;
		}
	}
	let a = Math.max(lo, best - (hi - lo) / n);
	let b = Math.min(hi, best + (hi - lo) / n);
	const r = (Math.sqrt(5) - 1) / 2;
	for (let k = 0; k < 80; k++) {
		const c = b - r * (b - a);
		const d = a + r * (b - a);
		if (f(c) > f(d)) b = d;
		else a = c;
	}
	return (a + b) / 2;
}

// ---------------------------------------------------------------------------------------------
// 2.5 Information theory

/** Entropy of a discrete distribution in nats (base e) or bits (base 2); 0 ln 0 = 0. */
export function entropy(p: number[], base = Math.E): number {
	let h = 0;
	for (const v of p) if (v > 0) h -= v * Math.log(v);
	return h / Math.log(base);
}

export const normalise = (v: number[]) => {
	const s = v.reduce((a, b) => a + b, 0);
	return v.map((x) => x / s);
};

/** Huffman code words (strings of 0 and 1) for the probabilities p. */
export function huffman(p: number[]): string[] {
	if (p.length === 1) return ["0"];
	type Node = { w: number; ids: number[] };
	let nodes: Node[] = p.map((w, i) => ({ w, ids: [i] }));
	const codes = p.map(() => "");
	while (nodes.length > 1) {
		// Stable: lowest weight first, ties broken by the later symbol so that more probable
		// (earlier) states keep the shorter words, as in the book's code.
		nodes.sort((a, b) => a.w - b.w || Math.max(...b.ids) - Math.max(...a.ids));
		const [lo, hi] = [nodes[0], nodes[1]];
		for (const i of lo.ids) codes[i] = `1${codes[i]}`;
		for (const i of hi.ids) codes[i] = `0${codes[i]}`;
		nodes = [
			{ w: lo.w + hi.w, ids: [...hi.ids, ...lo.ids] },
			...nodes.slice(2),
		];
	}
	return codes;
}

/** KL(N(mu1, s1²) ‖ N(mu2, s2²)) in nats (Exercise 2.27). */
export const klGauss = (mu1: number, s1: number, mu2: number, s2: number) =>
	Math.log(s2 / s1) + (s1 * s1 + (mu1 - mu2) ** 2) / (2 * s2 * s2) - 0.5;

/** Numerical KL by the trapezoidal rule on [lo, hi], as a check of the closed form. */
export function klNumeric(
	p: (x: number) => number,
	q: (x: number) => number,
	lo: number,
	hi: number,
	n = 20000,
): number {
	let s = 0;
	const h = (hi - lo) / n;
	for (let i = 0; i <= n; i++) {
		const x = lo + i * h;
		const px = p(x);
		const v = px > 1e-300 ? px * Math.log(px / q(x)) : 0;
		s += (i === 0 || i === n ? 0.5 : 1) * v;
	}
	return s * h;
}

export interface InfoQuantities {
	hx: number;
	hy: number;
	hxy: number;
	hxGivenY: number;
	hyGivenX: number;
	mi: number;
}

/** Entropies and mutual information of a joint table p[x][y] (nats). */
export function info(p: number[][]): InfoQuantities {
	const px = p.map((r) => r.reduce((a, b) => a + b, 0));
	const py = p[0].map((_, j) => p.reduce((a, r) => a + r[j], 0));
	const hx = entropy(px);
	const hy = entropy(py);
	const hxy = entropy(p.flat());
	return {
		hx,
		hy,
		hxy,
		hxGivenY: hxy - hy,
		hyGivenX: hxy - hx,
		mi: hx + hy - hxy,
	};
}

// ---------------------------------------------------------------------------------------------
// Exercises

/** Probability that die a rolls higher than die b (faces equally likely). */
export function beats(a: number[], b: number[]): number {
	let w = 0;
	for (const x of a) for (const y of b) if (x > y) w++;
	return w / (a.length * b.length);
}

/**
 * Efron's dice (Exercise 2.2, Figure 2.16). Each beats the next with probability 2/3: A beats B,
 * B beats C, C beats D and D beats A. The book's cycle B → A → D → C → B has each die beat the
 * previous one.
 */
export const efronDice: Record<string, number[]> = {
	A: [4, 4, 4, 4, 0, 0],
	B: [3, 3, 3, 3, 3, 3],
	C: [6, 6, 2, 2, 2, 2],
	D: [5, 5, 5, 1, 1, 1],
};

/**
 * Bent coin (Exercise 2.40): the concave side lands up with probability 0.6. Prior
 * p(convex = heads) = priorConvexHeads. Returns the posterior p(concave = heads | data) and the
 * predictive p(next = heads | data).
 */
export function bentCoin(
	heads: number,
	tails: number,
	priorConvexHeads = 0.1,
	pConcave = 0.6,
): { posteriorConcaveHeads: number; nextHeads: number } {
	const likConcaveHeads = pConcave ** heads * (1 - pConcave) ** tails;
	const likConvexHeads = (1 - pConcave) ** heads * pConcave ** tails;
	const a = (1 - priorConvexHeads) * likConcaveHeads;
	const b = priorConvexHeads * likConvexHeads;
	const post = a / (a + b);
	return {
		posteriorConcaveHeads: post,
		nextHeads: post * pConcave + (1 - post) * (1 - pConcave),
	};
}

// ---------------------------------------------------------------------------------------------
// Going further: multivariate Gaussian (Section 3.2)

export interface Gauss2 {
	/** Eigenvalues λ1 ⩾ λ2 of Σ and the angle of u1 (radians). */
	l1: number;
	l2: number;
	angle: number;
	det: number;
	/** p(x2 | x1) = N(x2 | m + k (x1 − µ1), v) from (3.65)–(3.66). */
	k: number;
	condVar: number;
}

/** Geometry and conditional of a 2-D Gaussian with standard deviations s1, s2 and correlation rho. */
export function gauss2(s1: number, s2: number, rho: number): Gauss2 {
	const a = s1 * s1;
	const d = s2 * s2;
	const b = rho * s1 * s2;
	const tr = a + d;
	const disc = Math.sqrt(((a - d) / 2) ** 2 + b * b);
	const l1 = tr / 2 + disc;
	const l2 = tr / 2 - disc;
	const angle =
		b === 0 && a >= d ? 0 : b === 0 ? Math.PI / 2 : Math.atan2(l1 - a, b);
	return {
		l1,
		l2,
		angle,
		det: a * d - b * b,
		k: b / a,
		condVar: d - (b * b) / a,
	};
}

/** Density of the mean of n Uniform(0, 1) variables (Irwin–Hall, rescaled), as in Figure 3.2. */
export function meanOfUniformsPdf(x: number, n: number): number {
	const s = n * x;
	if (s < 0 || s > n) return 0;
	let f = 1;
	for (let k = 2; k < n; k++) f *= k;
	let sum = 0;
	let binom = 1;
	for (let k = 0; k <= Math.floor(s); k++) {
		sum += (k % 2 ? -1 : 1) * binom * (s - k) ** (n - 1);
		binom = (binom * (n - k)) / (k + 1);
	}
	return (n * sum) / f;
}

// ---------------------------------------------------------------------------------------------
// Going further: bias–variance trade-off (Section 4.3), scaled down

export interface BiasVariance {
	lnLambda: number;
	bias2: number;
	variance: number;
	/** Average squared error against fresh noisy targets: bias² + variance + noise. */
	test: number;
	/** First `keep` fitted curves and their average, sampled on `grid`. */
	fits: number[][];
	mean: number[];
}

export const bvGrid = Array.from({ length: 101 }, (_, i) => i / 100);

/**
 * L data sets of N points from sin 2πx + N(0, noise²); each fitted with `basis` Gaussian basis
 * functions (width s, centres evenly spread on [0, 1]) plus a constant, by regularised least
 * squares with coefficient λ. Bias² and variance are averaged over the grid as in (4.50)–(4.52).
 */
export function biasVariance(
	lnLambda: number,
	opt: {
		L?: number;
		N?: number;
		basis?: number;
		s?: number;
		noise?: number;
		seed?: number;
		keep?: number;
	} = {},
): BiasVariance {
	const {
		L = 100,
		N = 25,
		basis = 24,
		s = 0.1,
		noise = 0.3,
		seed = 2,
		keep = 20,
	} = opt;
	const z = normals(mulberry32(seed));
	const centres = Array.from({ length: basis }, (_, j) => j / (basis - 1));
	const phi = (x: number) => [
		1,
		...centres.map((c) => Math.exp(-((x - c) ** 2) / (2 * s * s))),
	];
	const P = basis + 1;
	const lam = Math.exp(lnLambda);
	const all: number[][] = [];
	for (let l = 0; l < L; l++) {
		const xs = Array.from({ length: N }, (_, i) => (i + 0.5) / N);
		const ts = xs.map((x) => Math.sin(2 * Math.PI * x) + noise * z());
		const rows = xs.map(phi);
		const A = Array.from({ length: P }, (_, i) =>
			Array.from(
				{ length: P },
				(_, j) =>
					(i === j ? lam : 0) + rows.reduce((a, r) => a + r[i] * r[j], 0),
			),
		);
		const b = Array.from({ length: P }, (_, i) =>
			rows.reduce((a, r, n) => a + r[i] * ts[n], 0),
		);
		const w = solve(A, b);
		all.push(bvGrid.map((x) => phi(x).reduce((a, v, j) => a + v * w[j], 0)));
	}
	const mean = bvGrid.map((_, g) => all.reduce((a, f) => a + f[g], 0) / L);
	let bias2 = 0;
	let variance = 0;
	bvGrid.forEach((x, g) => {
		bias2 += (mean[g] - Math.sin(2 * Math.PI * x)) ** 2;
		variance += all.reduce((a, f) => a + (f[g] - mean[g]) ** 2, 0) / L;
	});
	bias2 /= bvGrid.length;
	variance /= bvGrid.length;
	return {
		lnLambda,
		bias2,
		variance,
		test: bias2 + variance + noise * noise,
		fits: all.slice(0, keep),
		mean,
	};
}

// ---------------------------------------------------------------------------------------------
// Going further: model evidence, ELBO, normalizing flow

/** ln Γ(n + 1) = ln n! for integers. */
const lnFact = (n: number) => {
	let s = 0;
	for (let k = 2; k <= n; k++) s += Math.log(k);
	return s;
};

/** Probability of one particular sequence with h heads and t tails under each coin model. */
export function coinEvidence(
	h: number,
	t: number,
): { fair: number; uniform: number } {
	return {
		fair: 0.5 ** (h + t),
		// ∫ θ^h (1 − θ)^t dθ = h! t! / (h + t + 1)!
		uniform: Math.exp(lnFact(h) + lnFact(t) - lnFact(h + t + 1)),
	};
}

export interface ElboParts {
	logEvidence: number;
	elbo: number;
	kl: number;
	posterior1: number;
}

/**
 * Two-component mixture: p(z = 1) = pi, p(x | z) = N(x | mu_z, 1). For an observed x and any
 * q(z = 1) = q, ln p(x) = ELBO(q) + KL(q ‖ p(z | x)) as in (16.57)–(16.59).
 */
export function elboParts(
	x: number,
	q: number,
	pi = 0.3,
	mu0 = -1,
	mu1 = 2,
): ElboParts {
	const j0 = (1 - pi) * gauss(x, mu0, 1);
	const j1 = pi * gauss(x, mu1, 1);
	const px = j0 + j1;
	const post1 = j1 / px;
	const term = (qz: number, joint: number) =>
		qz > 0 ? qz * Math.log(joint / qz) : 0;
	const elbo = term(1 - q, j0) + term(q, j1);
	const klTerm = (qz: number, p: number) =>
		qz > 0 ? qz * Math.log(qz / p) : 0;
	return {
		logEvidence: Math.log(px),
		elbo,
		kl: klTerm(1 - q, 1 - post1) + klTerm(q, post1),
		posterior1: post1,
	};
}

/** A two-layer 1-D flow x = f_A(f_B(z)): f_B(z) = z + 0.8 tanh(3z), f_A(u) = 0.5 u + 1. */
export const flowB = (z: number) => z + 0.8 * Math.tanh(3 * z);
export const flowBPrime = (z: number) => 1 + 2.4 / Math.cosh(3 * z) ** 2;
export const flowA = (u: number) => 0.5 * u + 1;

/** Invert the flow by bisection; returns z and the log-density ln p_x(x) via (18.4). */
export function flowDensity(x: number): { z: number; u: number; logp: number } {
	const u = (x - 1) / 0.5; // g_A
	let lo = -20;
	let hi = 20;
	for (let k = 0; k < 100; k++) {
		const m = (lo + hi) / 2;
		if (flowB(m) < u) lo = m;
		else hi = m;
	}
	const z = (lo + hi) / 2; // g_B
	const logp =
		Math.log(gauss(z, 0, 1)) + Math.log(1 / 0.5) + Math.log(1 / flowBPrime(z));
	return { z, u, logp };
}
