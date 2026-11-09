// Computations behind the quantitative-trading figures (src/utils/quant): a seeded random
// number generator, the normal distribution, gambler's ruin, Kelly betting, the dice market
// game, Cox–Ross–Rubinstein binomial trees and Black–Scholes prices and Greeks.
// No DOM code here, so the functions run under node for checking:
//   node --experimental-strip-types -e "import('./src/utils/quant/quant.ts').then(m => ...)"

/** Small seeded generator (mulberry32): the same seed gives the same sequence everywhere. */
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

// ---------------------------------------------------------------- normal distribution

export const normPdf = (x: number) =>
	Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

/** Standard normal CDF via erfc (absolute error below 1e-7, ample for prices to four decimals). */
export function normCdf(x: number): number {
	return 0.5 * erfc(-x / Math.SQRT2);
}

function erfc(x: number): number {
	// Numerical Recipes erfc with Chebyshev fit; fractional error < 1.2e-7.
	const z = Math.abs(x);
	const t = 1 / (1 + 0.5 * z);
	const r =
		t *
		Math.exp(
			-z * z -
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
	return x >= 0 ? r : 2 - r;
}

// ---------------------------------------------------------------- gambler's ruin

/** Probability that a walk from k (step +1 w.p. p, −1 otherwise) hits N before 0. */
export function ruinWinProb(k: number, N: number, p: number): number {
	if (k <= 0) return 0;
	if (k >= N) return 1;
	if (Math.abs(p - 0.5) < 1e-12) return k / N;
	const r = (1 - p) / p;
	return (1 - r ** k) / (1 - r ** N);
}

/** Expected number of steps until the walk from k hits 0 or N. */
export function ruinDuration(k: number, N: number, p: number): number {
	if (k <= 0 || k >= N) return 0;
	if (Math.abs(p - 0.5) < 1e-12) return k * (N - k);
	const q = 1 - p;
	return k / (q - p) - (N / (q - p)) * ruinWinProb(k, N, p);
}

export interface WalkPath {
	/** Positions, starting at k and ending at 0 or N (or cut at maxSteps). */
	pos: number[];
	/** 1 if it reached N, 0 if ruined, −1 if cut off. */
	won: number;
}

export function simulateWalk(
	rng: () => number,
	k: number,
	N: number,
	p: number,
	maxSteps = 100000,
): WalkPath {
	const pos = [k];
	let x = k;
	while (x > 0 && x < N && pos.length <= maxSteps) {
		x += rng() < p ? 1 : -1;
		pos.push(x);
	}
	return { pos, won: x >= N ? 1 : x <= 0 ? 0 : -1 };
}

// ---------------------------------------------------------------- Kelly betting

/** Expected log growth per bet when staking a fraction f on a bet paying b:1 with win prob p. */
export function kellyGrowth(f: number, p: number, b: number): number {
	if (f >= 1) return Number.NEGATIVE_INFINITY;
	return p * Math.log(1 + f * b) + (1 - p) * Math.log(1 - f);
}

/** Growth-optimal fraction f* = p − (1 − p) / b (0 when the bet has no edge). */
export function kellyFraction(p: number, b: number): number {
	return Math.max(0, p - (1 - p) / b);
}

/** Wealth path (starting at 1) over `n` bets of fraction f; the same rng draws give the same outcomes. */
export function wealthPath(
	outcomes: boolean[],
	f: number,
	b: number,
): number[] {
	const w = [1];
	let x = 1;
	for (const win of outcomes) {
		x *= win ? 1 + f * b : 1 - f;
		w.push(x);
	}
	return w;
}

// ---------------------------------------------------------------- dice market

/** Probability mass of the sum of m fair dice, indexed by the sum (0 … 6m). */
export function diceSumPmf(m: number): number[] {
	let pmf = [1];
	for (let i = 0; i < m; i++) {
		const next = new Array<number>(pmf.length + 6).fill(0);
		pmf.forEach((p, s) => {
			for (let f = 1; f <= 6; f++) next[s + f] += p / 6;
		});
		pmf = next;
	}
	return pmf;
}

export interface DiceMarket {
	/** pmf[s] = P(total = s) given the dice revealed so far. */
	pmf: number[];
	mean: number;
	sd: number;
	bid: number;
	ask: number;
	/** Probability the total ends up outside [bid, ask], so an informed trader would hit you. */
	pOutside: number;
	/** Expected loss per trade to a trader who knows the total: E[(S − ask)+] + E[(bid − S)+]. */
	informedLoss: number;
}

/** Fair value and quotes (mean ± halfSpread) for the sum of `total` dice after seeing `revealed`. */
export function diceMarket(
	revealed: number[],
	total: number,
	halfSpread: number,
): DiceMarket {
	const known = revealed.reduce((a, b) => a + b, 0);
	const rest = diceSumPmf(total - revealed.length);
	const pmf = new Array<number>(6 * total + 1).fill(0);
	rest.forEach((p, s) => {
		if (p) pmf[s + known] += p;
	});
	let mean = 0;
	let m2 = 0;
	pmf.forEach((p, s) => {
		mean += p * s;
		m2 += p * s * s;
	});
	const sd = Math.sqrt(Math.max(0, m2 - mean * mean));
	const bid = mean - halfSpread;
	const ask = mean + halfSpread;
	let pOutside = 0;
	let informedLoss = 0;
	pmf.forEach((p, s) => {
		if (s > ask + 1e-9) {
			pOutside += p;
			informedLoss += p * (s - ask);
		} else if (s < bid - 1e-9) {
			pOutside += p;
			informedLoss += p * (bid - s);
		}
	});
	return { pmf, mean, sd, bid, ask, pOutside, informedLoss };
}

// ---------------------------------------------------------------- options

export type OptionKind = "call" | "put";

export interface Greeks {
	price: number;
	delta: number;
	gamma: number;
	/** Per 1.00 change in volatility (divide by 100 for per vol point). */
	vega: number;
	/** Per year (divide by 365 for per day). */
	theta: number;
	d1: number;
	d2: number;
}

/** Black–Scholes price and Greeks of a European option on a non-dividend stock. */
export function blackScholes(
	S: number,
	K: number,
	T: number,
	r: number,
	sigma: number,
	kind: OptionKind,
): Greeks {
	if (T <= 0 || sigma <= 0) {
		const intrinsic = kind === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
		const delta = kind === "call" ? (S > K ? 1 : 0) : S < K ? -1 : 0;
		return {
			price: intrinsic,
			delta,
			gamma: 0,
			vega: 0,
			theta: 0,
			d1: Number.NaN,
			d2: Number.NaN,
		};
	}
	const sq = sigma * Math.sqrt(T);
	const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq;
	const d2 = d1 - sq;
	const disc = K * Math.exp(-r * T);
	const gamma = normPdf(d1) / (S * sq);
	const vega = S * normPdf(d1) * Math.sqrt(T);
	const decay = (-S * normPdf(d1) * sigma) / (2 * Math.sqrt(T));
	if (kind === "call") {
		return {
			price: S * normCdf(d1) - disc * normCdf(d2),
			delta: normCdf(d1),
			gamma,
			vega,
			theta: decay - r * disc * normCdf(d2),
			d1,
			d2,
		};
	}
	return {
		price: disc * normCdf(-d2) - S * normCdf(-d1),
		delta: normCdf(d1) - 1,
		gamma,
		vega,
		theta: decay + r * disc * normCdf(-d2),
		d1,
		d2,
	};
}

export interface TreeNode {
	/** Time step i (0 … n) and number of up moves j (0 … i). */
	i: number;
	j: number;
	S: number;
	/** Option value at the node. */
	V: number;
	/** Value of holding on (discounted risk-neutral expectation); NaN at expiry. */
	hold: number;
	/** Value of exercising now. */
	exercise: number;
	/** True when an American option is exercised here (exercise beats holding). */
	early: boolean;
}

export interface BinomialTree {
	u: number;
	d: number;
	/** Risk-neutral probability of an up move. */
	q: number;
	disc: number;
	dt: number;
	/** levels[i][j]: node after i steps with j up moves. */
	levels: TreeNode[][];
}

/** Cox–Ross–Rubinstein tree: u = e^{σ√Δt}, d = 1/u, q = (e^{rΔt} − d)/(u − d). */
export function binomialTree(
	S: number,
	K: number,
	T: number,
	r: number,
	sigma: number,
	n: number,
	kind: OptionKind,
	american: boolean,
): BinomialTree {
	const dt = T / n;
	const u = Math.exp(sigma * Math.sqrt(dt));
	const d = 1 / u;
	const disc = Math.exp(-r * dt);
	const q = (Math.exp(r * dt) - d) / (u - d);
	const payoff = (s: number) =>
		kind === "call" ? Math.max(s - K, 0) : Math.max(K - s, 0);
	const levels: TreeNode[][] = [];
	for (let i = 0; i <= n; i++) {
		levels.push(
			Array.from({ length: i + 1 }, (_, j) => {
				const s = S * u ** j * d ** (i - j);
				return {
					i,
					j,
					S: s,
					V: 0,
					hold: Number.NaN,
					exercise: payoff(s),
					early: false,
				};
			}),
		);
	}
	for (const node of levels[n]) node.V = node.exercise;
	for (let i = n - 1; i >= 0; i--) {
		for (const node of levels[i]) {
			const up = levels[i + 1][node.j + 1].V;
			const down = levels[i + 1][node.j].V;
			node.hold = disc * (q * up + (1 - q) * down);
			node.early = american && node.exercise > node.hold + 1e-12;
			node.V = node.early ? node.exercise : node.hold;
		}
	}
	return { u, d, q, disc, dt, levels };
}
