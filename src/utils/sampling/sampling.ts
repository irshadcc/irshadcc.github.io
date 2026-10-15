// Sampling algorithms from Bishop & Bishop, Deep Learning: Foundations and Concepts, ch. 14.
// Everything is deterministic given a seed so figures and the numbers quoted in the post agree.
// No imports, so the file can be run directly with `node --experimental-strip-types`.

export type Rng = () => number;

/** Small, fast seeded generator of uniforms on [0, 1) (mulberry32). */
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

/**
 * Box–Muller in the polar form of eqs 14.10–14.11: draw (z1, z2) uniformly in the unit disc,
 * then scale by sqrt(-2 ln r² / r²). Returns null when the pair falls outside the disc.
 */
export function boxMuller(z1: number, z2: number): [number, number] | null {
	const r2 = z1 * z1 + z2 * z2;
	if (r2 > 1 || r2 === 0) return null;
	const s = Math.sqrt((-2 * Math.log(r2)) / r2);
	return [z1 * s, z2 * s];
}

/** Standard normal generator built from `rng` with Box–Muller; caches the second value. */
export function normalGenerator(rng: Rng): () => number {
	let spare: number | null = null;
	return () => {
		if (spare !== null) {
			const v = spare;
			spare = null;
			return v;
		}
		for (;;) {
			const y = boxMuller(2 * rng() - 1, 2 * rng() - 1);
			if (y) {
				spare = y[1];
				return y[0];
			}
		}
	};
}

// ---------------------------------------------------------------- densities

export const gaussianPdf = (x: number, mu = 0, s = 1) =>
	Math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));

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

/** Gamma density Gam(z | a, b) of eq 14.15 (shape a, rate b). */
export function gammaPdf(z: number, a: number, b: number): number {
	if (z <= 0) return 0;
	return Math.exp(
		a * Math.log(b) + (a - 1) * Math.log(z) - b * z - logGamma(a),
	);
}

/** Cauchy density with location c and scale s; c = 0, s = 1 is eq 14.8. */
export const cauchyPdf = (z: number, c = 0, s = 1) =>
	1 / (Math.PI * s * (1 + ((z - c) / s) ** 2));

// ------------------------------------------------- transformation method (14.5–14.8)

export interface Transform {
	name: string;
	/** Target density p(y). */
	pdf: (y: number) => number;
	/** Its indefinite integral h(y) of eq 14.6 (the CDF). */
	cdf: (y: number) => number;
	/** y = h⁻¹(z). */
	inv: (z: number) => number;
	range: [number, number];
}

export const transforms: Record<string, Transform> = {
	exponential: {
		name: "exponential, λ = 1",
		pdf: (y) => (y < 0 ? 0 : Math.exp(-y)),
		cdf: (y) => (y < 0 ? 0 : 1 - Math.exp(-y)),
		inv: (z) => -Math.log(1 - z),
		range: [0, 5],
	},
	cauchy: {
		name: "Cauchy",
		pdf: (y) => cauchyPdf(y),
		cdf: (y) => 0.5 + Math.atan(y) / Math.PI,
		inv: (z) => Math.tan(Math.PI * (z - 0.5)),
		range: [-6, 6],
	},
};

// ------------------------------------------------------- rejection sampling (14.13–14.16)

export interface RejectionSetup {
	/** Unnormalized target p̃(z). */
	pTilde: (z: number) => number;
	/** Normalized proposal q(z). */
	q: (z: number) => number;
	sampleQ: (rng: Rng) => number;
	/** Smallest k with k q(z) ≥ p̃(z) everywhere (found on a fine grid). */
	kMin: number;
	/** ∫ p̃(z) dz, so the acceptance rate is area / k (eq 14.14). */
	area: number;
	range: [number, number];
}

/** Largest ratio f/g on a grid over [lo, hi], refined around the best point. */
function maxRatio(
	f: (z: number) => number,
	g: (z: number) => number,
	lo: number,
	hi: number,
): number {
	let best = lo;
	let bestV = 0;
	const n = 20000;
	for (let i = 0; i <= n; i++) {
		const z = lo + ((hi - lo) * i) / n;
		const v = f(z) / g(z);
		if (v > bestV) [best, bestV] = [z, v];
	}
	let h = (hi - lo) / n;
	for (let it = 0; it < 60; it++) {
		for (const z of [best - h, best + h]) {
			const v = f(z) / g(z);
			if (v > bestV) [best, bestV] = [z, v];
		}
		h /= 2;
	}
	return bestV;
}

/**
 * The book's example: target Gam(z | a, b), proposal a Cauchy with location c = a − 1 and
 * squared scale 2a − 1, sampled by the transformation z = s tan(y) + c with y uniform.
 */
export function gammaCauchy(
	a: number,
	b: number,
): RejectionSetup & { c: number; s: number } {
	const c = a - 1;
	const s = Math.sqrt(2 * a - 1);
	const pTilde = (z: number) => gammaPdf(z, a, b);
	const q = (z: number) => cauchyPdf(z, c, s);
	const kMin = maxRatio(pTilde, q, 1e-6, c + 60 * s);
	return {
		pTilde,
		q,
		sampleQ: (rng) => s * Math.tan(Math.PI * (rng() - 0.5)) + c,
		kMin,
		area: 1,
		range: [0, 30],
		c,
		s,
	};
}

export interface RejectionDraw {
	z0: number;
	u0: number;
	accept: boolean;
}

/** One rejection step: z0 ~ q, u0 ~ U[0, k q(z0)], keep z0 if u0 ≤ p̃(z0). */
export function rejectionStep(
	rng: Rng,
	s: RejectionSetup,
	k: number,
): RejectionDraw {
	const z0 = s.sampleQ(rng);
	const u0 = rng() * k * s.q(z0);
	return { z0, u0, accept: u0 <= s.pTilde(z0) };
}

/** Acceptance rate of rejection sampling a N(0, σp²) target with a N(0, σq²) proposal in D dims. */
export const gaussianRejectionRate = (ratio: number, dims: number) =>
	ratio ** -dims;

// --------------------------------------------- adaptive rejection sampling (14.17)

export interface EnvelopePiece {
	/** Left and right end of the piece; right may be Infinity. */
	lo: number;
	hi: number;
	/** The tangent line u(z) = h + slope (z − x) at grid point x. */
	x: number;
	h: number;
	slope: number;
	/** ∫ exp(u(z)) dz over the piece. */
	mass: number;
}

export interface Envelope {
	pieces: EnvelopePiece[];
	total: number;
}

function pieceMass(p: Omit<EnvelopePiece, "mass">): number {
	const ua = p.h + p.slope * (p.lo - p.x);
	if (!Number.isFinite(p.hi))
		return p.slope < 0 ? -Math.exp(ua) / p.slope : Number.POSITIVE_INFINITY;
	const w = p.hi - p.lo;
	if (Math.abs(p.slope) < 1e-12) return Math.exp(ua) * w;
	return (Math.exp(ua) * Math.expm1(p.slope * w)) / p.slope;
}

/**
 * Piecewise-exponential envelope of a log-concave density on [lo, ∞): tangents of ln p̃ at the
 * sorted grid points, cut where neighbouring tangents intersect (Gilks and Wild, 1992).
 */
export function buildEnvelope(
	logp: (z: number) => number,
	dlogp: (z: number) => number,
	grid: number[],
	lo: number,
): Envelope {
	const xs = [...grid].sort((a, b) => a - b);
	const hs = xs.map(logp);
	const ds = xs.map(dlogp);
	const knots: number[] = [];
	for (let j = 0; j + 1 < xs.length; j++) {
		knots.push(
			(hs[j + 1] - hs[j] - xs[j + 1] * ds[j + 1] + xs[j] * ds[j]) /
				(ds[j] - ds[j + 1]),
		);
	}
	const pieces = xs.map((x, j) => {
		const p = {
			lo: j === 0 ? lo : knots[j - 1],
			hi: j === xs.length - 1 ? Number.POSITIVE_INFINITY : knots[j],
			x,
			h: hs[j],
			slope: ds[j],
		};
		return { ...p, mass: pieceMass(p) };
	});
	return { pieces, total: pieces.reduce((s, p) => s + p.mass, 0) };
}

/** The envelope's log value at z (the upper hull of the tangents). */
export function envelopeLog(env: Envelope, z: number): number {
	const p =
		env.pieces.find((q) => z <= q.hi) ?? env.pieces[env.pieces.length - 1];
	return p.h + p.slope * (z - p.x);
}

/** Draw from the normalized envelope: pick a piece by mass, then invert its exponential CDF. */
export function sampleEnvelope(env: Envelope, rng: Rng): number {
	let u = rng() * env.total;
	let p = env.pieces[env.pieces.length - 1];
	for (const q of env.pieces) {
		if (u < q.mass) {
			p = q;
			break;
		}
		u -= q.mass;
	}
	const v = rng();
	if (!Number.isFinite(p.hi)) return p.lo + Math.log1p(-v) / p.slope;
	if (Math.abs(p.slope) < 1e-12) return p.lo + v * (p.hi - p.lo);
	return p.lo + Math.log1p(v * Math.expm1(p.slope * (p.hi - p.lo))) / p.slope;
}

export interface ArsStep {
	grid: number[];
	env: Envelope;
	z: number;
	u: number;
	/** p̃(z) / envelope(z), the acceptance probability of this draw. */
	ratio: number;
	accept: boolean;
}

/** Run adaptive rejection sampling; each rejected point joins the grid for the next step. */
export function runArs(
	logp: (z: number) => number,
	dlogp: (z: number) => number,
	grid0: number[],
	lo: number,
	steps: number,
	rng: Rng,
): ArsStep[] {
	let grid = [...grid0].sort((a, b) => a - b);
	const out: ArsStep[] = [];
	for (let i = 0; i < steps; i++) {
		const env = buildEnvelope(logp, dlogp, grid, lo);
		const z = sampleEnvelope(env, rng);
		const u = rng();
		const ratio = Math.exp(logp(z) - envelopeLog(env, z));
		const accept = u <= ratio;
		out.push({ grid, env, z, u, ratio, accept });
		if (!accept) grid = [...grid, z].sort((a, b) => a - b);
	}
	return out;
}

// ----------------------------------------- importance sampling and SIR (14.19–14.26)

/** Normalized importance weights w_l of eq 14.23 from p̃ and q evaluated at the samples. */
export function importanceWeights(
	zs: number[],
	pTilde: (z: number) => number,
	q: (z: number) => number,
): number[] {
	const r = zs.map((z) => pTilde(z) / q(z));
	const s = r.reduce((a, b) => a + b, 0);
	return r.map((v) => v / s);
}

/** Effective sample size 1 / Σ w², equal to L for equal weights and 1 when one weight is 1. */
export const effectiveSampleSize = (w: number[]) =>
	1 / w.reduce((s, v) => s + v * v, 0);

/** Draw `n` indices with replacement with probabilities `w` (multinomial resampling). */
export function resample(w: number[], n: number, rng: Rng): number[] {
	const cum: number[] = [];
	let s = 0;
	for (const v of w) {
		s += v;
		cum.push(s);
	}
	const out: number[] = [];
	for (let i = 0; i < n; i++) {
		const u = rng() * s;
		let lo = 0;
		let hi = cum.length - 1;
		while (lo < hi) {
			const mid = (lo + hi) >> 1;
			if (cum[mid] > u) hi = mid;
			else lo = mid + 1;
		}
		out.push(lo);
	}
	return out;
}

// ---------------------------------------------------- discrete Markov chains (14.31–14.41)

export type Matrix = number[][];

/**
 * Transition matrix T[i][j] = p(z' = j | z = i) of the Metropolis algorithm on states 0..n−1
 * with target p̃ and a symmetric proposal that moves one state left or right with probability
 * 1/2 each. A proposal off either end is rejected, so the chain stays put.
 */
export function metropolisMatrix(pTilde: number[]): Matrix {
	const n = pTilde.length;
	const T = pTilde.map(() => new Array<number>(n).fill(0));
	for (let i = 0; i < n; i++) {
		for (const j of [i - 1, i + 1]) {
			if (j < 0 || j >= n) continue;
			T[i][j] = 0.5 * Math.min(1, pTilde[j] / pTilde[i]);
		}
		T[i][i] = 1 - T[i].reduce((s, v) => s + v, 0);
	}
	return T;
}

/** One step of eq 14.32: p(z^(m+1) = j) = Σ_i p(z^(m) = i) T[i][j]. */
export const propagate = (p: number[], T: Matrix) =>
	p.map((_, j) => p.reduce((s, pi, i) => s + pi * T[i][j], 0));

export const totalVariation = (p: number[], q: number[]) =>
	0.5 * p.reduce((s, v, i) => s + Math.abs(v - q[i]), 0);

/** Exact distribution of the lazy random walk of eqs 14.28–14.30 after `steps` steps from 0. */
export function randomWalkDistribution(steps: number): Map<number, number> {
	let d = new Map<number, number>([[0, 1]]);
	for (let t = 0; t < steps; t++) {
		const nxt = new Map<number, number>();
		for (const [z, p] of d) {
			nxt.set(z, (nxt.get(z) ?? 0) + 0.5 * p);
			nxt.set(z + 1, (nxt.get(z + 1) ?? 0) + 0.25 * p);
			nxt.set(z - 1, (nxt.get(z - 1) ?? 0) + 0.25 * p);
		}
		d = nxt;
	}
	return d;
}

// --------------------------------------------- MCMC on a 2D Gaussian (14.27, 14.40–14.46, 14.61)

export type Vec2 = [number, number];

export interface Gauss2 {
	mean: Vec2;
	cov: [Vec2, Vec2];
	/** Precision matrix Σ⁻¹. */
	prec: [Vec2, Vec2];
	/** Standard deviations along the principal axes, largest first, and the axes themselves. */
	sd: Vec2;
	axes: [Vec2, Vec2];
}

/** A 2D Gaussian with standard deviations sMax, sMin along axes rotated by `deg` degrees. */
export function gauss2(
	mean: Vec2,
	sMax: number,
	sMin: number,
	deg: number,
): Gauss2 {
	const t = (deg * Math.PI) / 180;
	const u: Vec2 = [Math.cos(t), Math.sin(t)];
	const v: Vec2 = [-Math.sin(t), Math.cos(t)];
	const a = sMax * sMax;
	const b = sMin * sMin;
	const cov: [Vec2, Vec2] = [
		[a * u[0] * u[0] + b * v[0] * v[0], a * u[0] * u[1] + b * v[0] * v[1]],
		[a * u[0] * u[1] + b * v[0] * v[1], a * u[1] * u[1] + b * v[1] * v[1]],
	];
	const det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
	const prec: [Vec2, Vec2] = [
		[cov[1][1] / det, -cov[0][1] / det],
		[-cov[1][0] / det, cov[0][0] / det],
	];
	return { mean, cov, prec, sd: [sMax, sMin], axes: [u, v] };
}

/** ln p̃(z) = −½ (z − μ)ᵀ Σ⁻¹ (z − μ); the normalizing constant is never needed. */
export function logTarget(g: Gauss2, z: Vec2): number {
	const d0 = z[0] - g.mean[0];
	const d1 = z[1] - g.mean[1];
	return (
		-0.5 *
		(d0 * (g.prec[0][0] * d0 + g.prec[0][1] * d1) +
			d1 * (g.prec[1][0] * d0 + g.prec[1][1] * d1))
	);
}

/** Score ∇ ln p(z) = −Σ⁻¹ (z − μ). */
export function score(g: Gauss2, z: Vec2): Vec2 {
	const d0 = z[0] - g.mean[0];
	const d1 = z[1] - g.mean[1];
	return [
		-(g.prec[0][0] * d0 + g.prec[0][1] * d1),
		-(g.prec[1][0] * d0 + g.prec[1][1] * d1),
	];
}

export type ChainMethod = "metropolis" | "gibbs" | "langevin";

export interface ChainMove {
	from: Vec2;
	/** The candidate z★ (Metropolis) or the new point (Gibbs, Langevin). */
	to: Vec2;
	accept: boolean;
	next: Vec2;
	/** Coordinate updated by a Gibbs move. */
	coord?: number;
}

/**
 * One move of the chosen sampler. `param` is the proposal standard deviation ρ (Metropolis),
 * the over-relaxation α of eq 14.46 (Gibbs, α = 0 is plain Gibbs) or the step size η (Langevin).
 * Gibbs alternates coordinates by move index `t`.
 */
export function chainMove(
	method: ChainMethod,
	g: Gauss2,
	z: Vec2,
	param: number,
	t: number,
	rng: Rng,
	randn: () => number,
): ChainMove {
	if (method === "metropolis") {
		const to: Vec2 = [z[0] + param * randn(), z[1] + param * randn()];
		const a = Math.exp(Math.min(0, logTarget(g, to) - logTarget(g, z)));
		const accept = a > rng();
		return { from: z, to, accept, next: accept ? to : z };
	}
	if (method === "gibbs") {
		const i = t % 2;
		const j = 1 - i;
		// Conditional of z_i given z_j: mean μ_i − (Λ_ij / Λ_ii)(z_j − μ_j), variance 1 / Λ_ii.
		const mu = g.mean[i] - (g.prec[i][j] / g.prec[i][i]) * (z[j] - g.mean[j]);
		const sd = 1 / Math.sqrt(g.prec[i][i]);
		const a = param;
		const zi = mu + a * (z[i] - mu) + sd * Math.sqrt(1 - a * a) * randn();
		const to: Vec2 = i === 0 ? [zi, z[1]] : [z[0], zi];
		return { from: z, to, accept: true, next: to, coord: i };
	}
	const s = score(g, z);
	const k = Math.sqrt(2 * param);
	const to: Vec2 = [
		z[0] + param * s[0] + k * randn(),
		z[1] + param * s[1] + k * randn(),
	];
	return { from: z, to, accept: true, next: to };
}

export function runChain(
	method: ChainMethod,
	g: Gauss2,
	z0: Vec2,
	param: number,
	steps: number,
	seed: number,
): ChainMove[] {
	const rng = mulberry32(seed);
	const randn = normalGenerator(rng);
	const out: ChainMove[] = [];
	let z = z0;
	for (let t = 0; t < steps; t++) {
		const m = chainMove(method, g, z, param, t, rng, randn);
		out.push(m);
		z = m.next;
	}
	return out;
}

/** Coordinate of z along unit vector u, measured from the mean. */
export const along = (g: Gauss2, z: Vec2, u: Vec2) =>
	(z[0] - g.mean[0]) * u[0] + (z[1] - g.mean[1]) * u[1];

/**
 * Integrated autocorrelation time τ = 1 + 2 Σ ρ_k, summed until the first negative pair of
 * autocorrelations (Geyer's initial positive sequence). Effective sample size is n / τ.
 */
export function autocorrelationTime(x: number[]): number {
	const n = x.length;
	const m = x.reduce((s, v) => s + v, 0) / n;
	const c = x.map((v) => v - m);
	const c0 = c.reduce((s, v) => s + v * v, 0) / n;
	if (c0 === 0) return Number.POSITIVE_INFINITY;
	const rho = (k: number) => {
		let s = 0;
		for (let i = 0; i + k < n; i++) s += c[i] * c[i + k];
		return s / n / c0;
	};
	let tau = -1;
	for (let k = 0; k + 1 < Math.min(n / 2, 2000); k += 2) {
		const pair = rho(k) + rho(k + 1);
		if (pair <= 0) break;
		tau += 2 * pair;
	}
	return Math.max(tau, 1);
}

/**
 * Exact stationary variance of unadjusted Langevin (eq 14.61) along a principal axis of a
 * Gaussian with variance s²: x' = (1 − η/s²) x + √(2η) ε gives s² / (1 − η / (2 s²)).
 */
export const langevinStationaryVariance = (s2: number, eta: number) =>
	s2 / (1 - eta / (2 * s2));

// ------------------------------------------------------ energy-based model (14.50–14.62)

/** Features φ(x) = (x, x², x³, x⁴); the energy is E(x, w) = w · φ(x), so ∇_w E = φ(x). */
export const features = (x: number) => [x, x * x, x * x * x, x * x * x * x];

export const energy = (w: number[], x: number) =>
	w[0] * x + w[1] * x * x + w[2] * x ** 3 + w[3] * x ** 4;

/** dE/dx; the score of eq 14.60 is its negative. */
export const energyGrad = (w: number[], x: number) =>
	w[0] + 2 * w[1] * x + 3 * w[2] * x * x + 4 * w[3] * x ** 3;

/** p(x | w) on an even grid, normalized numerically (only for drawing and checking). */
export function modelDensity(w: number[], xs: number[]): number[] {
	const e = xs.map((x) => -energy(w, x));
	const top = Math.max(...e);
	const p = e.map((v) => Math.exp(v - top));
	const dx = xs[1] - xs[0];
	const z = p.reduce((s, v) => s + v, 0) * dx;
	return p.map((v) => v / z);
}

/** E_{x∼p(x|w)}[φ(x)] by numerical integration on a grid. */
export function modelMoments(w: number[], xs: number[]): number[] {
	const p = modelDensity(w, xs);
	const dx = xs[1] - xs[0];
	const m = [0, 0, 0, 0];
	xs.forEach((x, i) => {
		features(x).forEach((f, k) => {
			m[k] += f * p[i] * dx;
		});
	});
	return m;
}

export const meanFeatures = (xs: number[]) => {
	const m = [0, 0, 0, 0];
	for (const x of xs) {
		features(x).forEach((f, k) => {
			m[k] += f / xs.length;
		});
	}
	return m;
};

export interface EbmState {
	w: number[];
	/** Persistent Langevin chains that supply the model samples. */
	particles: number[];
	iter: number;
}

export interface EbmOptions {
	lr: number;
	eta: number;
	langevinSteps: number;
	/** Start chains from the data each iteration (contrastive divergence) instead of persisting them. */
	fromData: boolean;
}

/**
 * One training step of eq 14.57 with ∇_w E = φ(x): move the particles with Langevin dynamics
 * (eq 14.61), then w ← w + lr (−mean_data φ + mean_model φ), each feature scaled by its data
 * second moment so the four coordinates move at comparable speed.
 */
export function ebmStep(
	state: EbmState,
	data: number[],
	opt: EbmOptions,
	randn: () => number,
): EbmState {
	let xs = opt.fromData
		? state.particles.map((_, i) => data[i % data.length])
		: [...state.particles];
	const k = Math.sqrt(2 * opt.eta);
	for (let t = 0; t < opt.langevinSteps; t++) {
		xs = xs.map((x) => x - opt.eta * energyGrad(state.w, x) + k * randn());
	}
	const md = meanFeatures(data);
	const mm = meanFeatures(xs);
	const m2 = md.map(
		(_, i) => data.reduce((s, x) => s + features(x)[i] ** 2, 0) / data.length,
	);
	const w = state.w.map((wk, i) => wk + (opt.lr / m2[i]) * (-md[i] + mm[i]));
	// Keep the quartic coefficient positive so exp(−E) stays normalizable.
	w[3] = Math.max(w[3], 1e-3);
	return { w, particles: xs, iter: state.iter + 1 };
}

/** The post's training set: n points from ½ N(−1.5, 0.5²) + ½ N(1.5, 0.5²). */
export function ebmData(seed: number, n: number): number[] {
	const rng = mulberry32(seed);
	const randn = normalGenerator(rng);
	return Array.from(
		{ length: n },
		() => (rng() < 0.5 ? -1.5 : 1.5) + 0.5 * randn(),
	);
}

/** Density the training set was drawn from. */
export const ebmDataDensity = (x: number) =>
	0.5 * gaussianPdf(x, -1.5, 0.5) + 0.5 * gaussianPdf(x, 1.5, 0.5);

/** Grid used to normalize p(x | w) numerically. */
export const ebmGrid = Array.from({ length: 1201 }, (_, i) => -6 + i * 0.01);

/** Average log-likelihood of the data under p(x | w), normalized on the grid. */
export function ebmLogLik(w: number[], data: number[]): number {
	const p = modelDensity(w, ebmGrid);
	return (
		data.reduce((s, x) => s + Math.log(p[Math.round((x + 6) / 0.01)]), 0) /
		data.length
	);
}

/** Exact maximum likelihood: the same update as ebmStep with grid moments instead of samples. */
export function ebmFitExact(
	data: number[],
	w0: number[],
	lr: number,
	iters: number,
): number[] {
	const md = meanFeatures(data);
	const m2 = md.map(
		(_, i) => data.reduce((s, x) => s + features(x)[i] ** 2, 0) / data.length,
	);
	let w = [...w0];
	for (let t = 0; t < iters; t++) {
		const mm = modelMoments(w, ebmGrid);
		w = w.map((wk, i) => wk + (lr / m2[i]) * (-md[i] + mm[i]));
		w[3] = Math.max(w[3], 1e-3);
	}
	return w;
}

// ----------------------------------------------- ancestral sampling on a Bayes net (14.47–14.48)

export interface BayesNode {
	name: string;
	parents: number[];
	/** P(node = 1 | parents), indexed by the parents' values read as a binary number (first parent = high bit). */
	cpt: number[];
}

/** The cloudy / sprinkler / rain / wet-grass network (Russell and Norvig's numbers). */
export const sprinkler: BayesNode[] = [
	{ name: "Cloudy", parents: [], cpt: [0.5] },
	{ name: "Sprinkler", parents: [0], cpt: [0.5, 0.1] },
	{ name: "Rain", parents: [0], cpt: [0.2, 0.8] },
	{ name: "WetGrass", parents: [1, 2], cpt: [0, 0.9, 0.9, 0.99] },
];

const parentIndex = (n: BayesNode, z: number[]) =>
	n.parents.reduce((s, p) => 2 * s + z[p], 0);

/** P(z_i = v | pa(i)) given the values of the parents in z. */
export const conditional = (n: BayesNode, z: number[], v: number) => {
	const p1 = n.cpt[parentIndex(n, z)];
	return v ? p1 : 1 - p1;
};

/** Exact joint probability by eq 14.47. */
export const joint = (net: BayesNode[], z: number[]) =>
	net.reduce((s, n, i) => s * conditional(n, z, z[i]), 1);

/** Exact P(query = 1 | evidence) by enumerating all 2^M states. */
export function exactPosterior(
	net: BayesNode[],
	query: number,
	evidence: Record<number, number>,
): number {
	let num = 0;
	let den = 0;
	for (let s = 0; s < 1 << net.length; s++) {
		const z = net.map((_, i) => (s >> i) & 1);
		if (Object.entries(evidence).some(([i, v]) => z[Number(i)] !== v)) continue;
		const p = joint(net, z);
		den += p;
		if (z[query]) num += p;
	}
	return num / den;
}

export interface NetSample {
	/** Values in node order; a node never reached (logic sampling stopped early) is −1. */
	z: number[];
	/** Logic sampling: index of the evidence node that disagreed, or −1 if the sample was kept. */
	rejectedAt: number;
	/** Likelihood weighting: weight r(z) of eq 14.48 (1 for the other methods). */
	weight: number;
	/** The uniform drawn at each node (NaN for a clamped evidence node). */
	u: number[];
}

/**
 * Sample the net in topological order. "ancestral" ignores evidence, "logic" restarts on the first
 * disagreement with an evidence node, "weighting" clamps evidence nodes and multiplies the weight
 * by P(evidence value | parents).
 */
export function sampleNet(
	net: BayesNode[],
	method: "ancestral" | "logic" | "weighting",
	evidence: Record<number, number>,
	rng: Rng,
): NetSample {
	const z = net.map(() => -1);
	const u = net.map(() => Number.NaN);
	let weight = 1;
	for (let i = 0; i < net.length; i++) {
		const ev = evidence[i];
		if (method === "weighting" && ev !== undefined) {
			z[i] = ev;
			weight *= conditional(net[i], z, ev);
			continue;
		}
		u[i] = rng();
		z[i] = u[i] < conditional(net[i], z, 1) ? 1 : 0;
		if (method === "logic" && ev !== undefined && z[i] !== ev)
			return { z, rejectedAt: i, weight: 0, u };
	}
	return { z, rejectedAt: -1, weight, u };
}
