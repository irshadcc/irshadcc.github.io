// Denoising diffusion (DDPM / DDIM) on small, exactly solvable data: a mixture of isotropic
// Gaussians in d dimensions. For such data every quantity a diffusion model learns has a closed
// form: the noised marginal q(x_t), the optimal denoiser E[x_0 | x_t], the noise prediction and
// the score. The figures run the samplers with that exact denoiser in place of a network, so
// what they show is the sampler itself, not the quality of a fit.
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

/** Standard normal generator (polar Box–Muller) built from `rng`. */
export function normalGenerator(rng: Rng): () => number {
	let spare: number | null = null;
	return () => {
		if (spare !== null) {
			const v = spare;
			spare = null;
			return v;
		}
		for (;;) {
			const u = 2 * rng() - 1;
			const v = 2 * rng() - 1;
			const r2 = u * u + v * v;
			if (r2 > 0 && r2 < 1) {
				const s = Math.sqrt((-2 * Math.log(r2)) / r2);
				spare = v * s;
				return u * s;
			}
		}
	};
}

// ---------------------------------------------------------------- noise schedules

/**
 * A noise schedule over t = 0..T. beta[t] is the variance added at step t (beta[0] = 0) and
 * alphaBar[t] = prod_{s <= t} (1 - beta[s]) is the signal fraction left after t steps.
 */
export interface Schedule {
	name: string;
	T: number;
	beta: number[];
	alphaBar: number[];
}

function fromBetas(name: string, beta: number[]): Schedule {
	const alphaBar = [1];
	for (let t = 1; t < beta.length; t++)
		alphaBar.push(alphaBar[t - 1] * (1 - beta[t]));
	return { name, T: beta.length - 1, beta, alphaBar };
}

/** DDPM's linear schedule (Ho et al. 2020): beta from b1 at t = 1 to bT at t = T. */
export function linearSchedule(T = 1000, b1 = 1e-4, bT = 0.02): Schedule {
	const beta = [0];
	for (let t = 1; t <= T; t++) beta.push(b1 + ((bT - b1) * (t - 1)) / (T - 1));
	return fromBetas("linear", beta);
}

/**
 * Nichol & Dhariwal's cosine schedule: alphaBar(t) = f(t) / f(0) with
 * f(t) = cos²(((t / T) + s) / (1 + s) · π / 2), betas clipped at 0.999.
 */
export function cosineSchedule(T = 1000, s = 0.008): Schedule {
	const f = (t: number) =>
		Math.cos(((t / T + s) / (1 + s)) * (Math.PI / 2)) ** 2;
	const beta = [0];
	for (let t = 1; t <= T; t++) beta.push(Math.min(1 - f(t) / f(t - 1), 0.999));
	return fromBetas("cosine", beta);
}

export const schedules = {
	linear: linearSchedule(),
	cosine: cosineSchedule(),
};
export type ScheduleName = keyof typeof schedules;

/** Signal-to-noise ratio alphaBar / (1 - alphaBar). */
export const snr = (ab: number) => ab / (1 - ab);

// ---------------------------------------------------------------- data

/** One isotropic Gaussian N(mu, s² I) with mixture weight w. */
export interface Component {
	w: number;
	mu: number[];
	s: number;
}
export type Mixture = Component[];

/** 1-D data used by the figures: two bumps of unequal weight and width. */
export const bimodal1d: Mixture = [
	{ w: 0.4, mu: [-1.2], s: 0.25 },
	{ w: 0.6, mu: [1.0], s: 0.3 },
];

/** 1-D worked example of the post: two equally likely points ±1 (s = 0). */
export const twoPoints: Mixture = [
	{ w: 0.5, mu: [-1], s: 0 },
	{ w: 0.5, mu: [1], s: 0 },
];

/** 2-D data: eight small Gaussians on a circle of radius 1.4. */
export const ring2d: Mixture = Array.from({ length: 8 }, (_, k) => ({
	w: 1 / 8,
	mu: [1.4 * Math.cos((k * Math.PI) / 4), 1.4 * Math.sin((k * Math.PI) / 4)],
	s: 0.1,
}));

/** 2-D data: two interleaved half moons, each a chain of 14 small Gaussians. */
export const moons2d: Mixture = (() => {
	const out: Mixture = [];
	const n = 14;
	for (let i = 0; i < n; i++) {
		const a = (Math.PI * i) / (n - 1);
		out.push({
			w: 1 / (2 * n),
			mu: [Math.cos(a) - 0.5, Math.sin(a) - 0.25],
			s: 0.08,
		});
		out.push({
			w: 1 / (2 * n),
			mu: [0.5 - Math.cos(a), 0.25 - Math.sin(a)],
			s: 0.08,
		});
	}
	// Scale to roughly unit variance per coordinate.
	for (const c of out) c.mu = [c.mu[0] * 1.3, c.mu[1] * 1.3];
	return out;
})();

export const datasets = { bimodal1d, twoPoints, ring2d, moons2d };

/** Draw n points from the mixture. */
export function sampleData(mix: Mixture, n: number, seed: number): number[][] {
	const rng = mulberry32(seed);
	const randn = normalGenerator(rng);
	const out: number[][] = [];
	for (let i = 0; i < n; i++) {
		let u = rng();
		let k = 0;
		while (k < mix.length - 1 && u >= mix[k].w) u -= mix[k++].w;
		out.push(mix[k].mu.map((m) => m + mix[k].s * randn()));
	}
	return out;
}

// ---------------------------------------------------------------- closed forms

/**
 * q(x_t) for mixture data. Each component N(mu, s²I) becomes
 * N(sqrt(ab) mu, (ab s² + 1 - ab) I) after noising, so q(x_t) is again a mixture.
 */
export function marginalPdf(mix: Mixture, ab: number, x: number[]): number {
	let p = 0;
	for (const c of mix) {
		const v = ab * c.s * c.s + 1 - ab;
		let d2 = 0;
		for (let i = 0; i < x.length; i++)
			d2 += (x[i] - Math.sqrt(ab) * c.mu[i]) ** 2;
		p += (c.w * Math.exp(-d2 / (2 * v))) / (2 * Math.PI * v) ** (x.length / 2);
	}
	return p;
}

/** Everything the optimal network would output at (x_t, t). */
export interface Denoised {
	/** E[x_0 | x_t]: the minimum-mean-squared-error estimate of the clean point. */
	x0: number[];
	/** E[eps | x_t] = (x_t - sqrt(ab) x0) / sqrt(1 - ab): the optimal noise prediction. */
	eps: number[];
	/** grad log q(x_t) = -eps / sqrt(1 - ab). */
	score: number[];
	/** Posterior probability of each mixture component given x_t. */
	resp: number[];
}

/** The exact denoiser for mixture data at signal level ab = alphaBar_t (0 < ab ≤ 1). */
export function denoise(mix: Mixture, ab: number, x: number[]): Denoised {
	const d = x.length;
	const sa = Math.sqrt(ab);
	const logs = mix.map((c) => {
		const v = ab * c.s * c.s + 1 - ab;
		let d2 = 0;
		for (let i = 0; i < d; i++) d2 += (x[i] - sa * c.mu[i]) ** 2;
		return Math.log(c.w) - d2 / (2 * v) - (d / 2) * Math.log(v);
	});
	const m = Math.max(...logs);
	const e = logs.map((l) => Math.exp(l - m));
	const z = e.reduce((a, b) => a + b, 0);
	const resp = e.map((v) => v / z);
	const x0 = new Array<number>(d).fill(0);
	const score = new Array<number>(d).fill(0);
	mix.forEach((c, k) => {
		const v = ab * c.s * c.s + 1 - ab;
		for (let i = 0; i < d; i++) {
			// Gaussian posterior of x_0 within component k.
			x0[i] +=
				resp[k] * (c.mu[i] + ((sa * c.s * c.s) / v) * (x[i] - sa * c.mu[i]));
			score[i] += (resp[k] * -(x[i] - sa * c.mu[i])) / v;
		}
	});
	const sn = Math.sqrt(1 - ab);
	return { x0, eps: score.map((g) => -sn * g), score, resp };
}

// ---------------------------------------------------------------- forward process

/** One forward step x_t = sqrt(1 - beta_t) x_{t-1} + sqrt(beta_t) eps. */
export const forwardStep = (x: number[], beta: number, randn: () => number) =>
	x.map((v) => Math.sqrt(1 - beta) * v + Math.sqrt(beta) * randn());

/** Jump straight to step t: x_t = sqrt(ab_t) x_0 + sqrt(1 - ab_t) eps. */
export const noiseTo = (x0: number[], ab: number, eps: number[]) =>
	x0.map((v, i) => Math.sqrt(ab) * v + Math.sqrt(1 - ab) * eps[i]);

/**
 * Run the forward chain one step at a time from n data points. Returns x_0, x_1, ..., x_T, each
 * a flat array of n points × d coordinates. The figures and the post use this same function, so
 * a seed gives the same particles in both.
 */
export function forwardStates(
	mix: Mixture,
	sched: Schedule,
	n: number,
	seed: number,
): Float64Array[] {
	const randn = normalGenerator(mulberry32(seed));
	let x = Float64Array.from(sampleData(mix, n, seed + 1).flat());
	const states = [x];
	for (let t = 1; t <= sched.T; t++) {
		const b = sched.beta[t];
		x = x.map((v) => Math.sqrt(1 - b) * v + Math.sqrt(b) * randn());
		states.push(x);
	}
	return states;
}

/**
 * Mean and variance of the forward posterior q(x_{t-1} | x_t, x_0) (DDPM eq. 7):
 *   mean = sqrt(ab_{t-1}) beta_t / (1 - ab_t) · x_0 + sqrt(alpha_t) (1 - ab_{t-1}) / (1 - ab_t) · x_t
 *   var  = (1 - ab_{t-1}) / (1 - ab_t) · beta_t
 */
export function forwardPosterior(sched: Schedule, t: number) {
	const b = sched.beta[t];
	const ab = sched.alphaBar[t];
	const abPrev = sched.alphaBar[t - 1];
	return {
		coefX0: (Math.sqrt(abPrev) * b) / (1 - ab),
		coefXt: (Math.sqrt(1 - b) * (1 - abPrev)) / (1 - ab),
		variance: ((1 - abPrev) / (1 - ab)) * b,
	};
}

// ---------------------------------------------------------------- reverse process

/** n sampling steps spread evenly over 1..T, in decreasing order, then 0. */
export function timesteps(T: number, n: number): number[] {
	const out: number[] = [];
	for (let i = n; i >= 1; i--)
		out.push(Math.round(1 + ((i - 1) * (T - 1)) / Math.max(1, n - 1)));
	if (n === 1) out[0] = T;
	out.push(0);
	return out;
}

/**
 * One reverse step from t to tPrev < t (DDIM eq. 12). eta = 1 is DDPM's ancestral sampler
 * (variance beta-tilde, eq. 7 of Ho et al.) on the chosen subsequence; eta = 0 is
 * deterministic DDIM. `denoised` is the network output at (x, t), here exact.
 */
export function reverseStep(
	x: number[],
	denoised: Denoised,
	ab: number,
	abPrev: number,
	eta: number,
	randn: () => number,
): number[] {
	const sigma = eta * Math.sqrt(((1 - abPrev) / (1 - ab)) * (1 - ab / abPrev));
	const dirCoef = Math.sqrt(Math.max(0, 1 - abPrev - sigma * sigma));
	return x.map(
		(_, i) =>
			Math.sqrt(abPrev) * denoised.x0[i] +
			dirCoef * denoised.eps[i] +
			sigma * (sigma > 0 ? randn() : 0),
	);
}

/**
 * Sample n points with the exact denoiser in `steps` steps: start at N(0, I) and apply
 * reverseStep down the timesteps. Returns the timesteps (T first, 0 last) and the state after
 * each step as a flat array of n points × d coordinates.
 */
export function reverseStates(
	mix: Mixture,
	sched: Schedule,
	steps: number,
	eta: number,
	n: number,
	seed: number,
): { ts: number[]; states: Float64Array[] } {
	const d = mix[0].mu.length;
	const randn = normalGenerator(mulberry32(seed));
	const ts = timesteps(sched.T, steps);
	let x = Float64Array.from({ length: n * d }, () => randn());
	const states = [x];
	for (let i = 0; i + 1 < ts.length; i++) {
		const ab = sched.alphaBar[ts[i]];
		const abPrev = sched.alphaBar[ts[i + 1]];
		const next = new Float64Array(n * d);
		for (let p = 0; p < n; p++) {
			const xp = Array.from(x.subarray(p * d, p * d + d));
			next.set(
				reverseStep(xp, denoise(mix, ab, xp), ab, abPrev, eta, randn),
				p * d,
			);
		}
		x = next;
		states.push(x);
	}
	return { ts, states };
}

/** Point p of a flat state with d coordinates per point. */
export const pointAt = (state: Float64Array, p: number, d: number) =>
	Array.from(state.subarray(p * d, p * d + d));

/** Mean and standard deviation of a list of numbers. */
export function moments(xs: number[]): { mean: number; sd: number } {
	const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
	const v = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length;
	return { mean, sd: Math.sqrt(v) };
}

/** Mixture mean and sd of a 1-D mixture after noising to level ab. */
export function mixtureMoments1d(mix: Mixture, ab: number) {
	const sa = Math.sqrt(ab);
	const mean = mix.reduce((a, c) => a + c.w * sa * c.mu[0], 0);
	const m2 = mix.reduce(
		(a, c) => a + c.w * ((sa * c.mu[0]) ** 2 + ab * c.s * c.s + 1 - ab),
		0,
	);
	return { mean, sd: Math.sqrt(m2 - mean * mean) };
}
