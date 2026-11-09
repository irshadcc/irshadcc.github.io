// Data for the static figures of the quantitative-trading post, computed from quant.ts so the
// text and figures agree. Numbers quoted in the post were cross-checked in Python (numpy/scipy).

import { diceSumPmf } from "./quant";

/** P(sum of three dice = s) for s = 3 … 18. */
export const threeDiceSums = diceSumPmf(3).slice(3);

/** P(first time the pattern appears ends on flip n), n = 1 … maxN, for a fair coin. */
export function patternWait(pattern: string, maxN = 20): number[] {
	// Track the distribution over "length of the longest prefix of pattern matched so far".
	const L = pattern.length;
	const next = (state: number, c: string) => {
		let t = pattern.slice(0, state) + c;
		while (t && !pattern.startsWith(t)) t = t.slice(1);
		return t.length;
	};
	let dist = new Array<number>(L).fill(0);
	dist[0] = 1;
	const out: number[] = [];
	for (let n = 1; n <= maxN; n++) {
		const nd = new Array<number>(L).fill(0);
		let done = 0;
		dist.forEach((p, s) => {
			for (const c of "HT") {
				const t = next(s, c);
				if (t === L) done += p / 2;
				else nd[t] += p / 2;
			}
		});
		out.push(done);
		dist = nd;
	}
	return out;
}

export const waitHH = patternWait("HH");
export const waitHT = patternWait("HT");

/** P(at least two of n people share a birthday), 365 equally likely days. */
export function birthday(n: number): number {
	let p = 1;
	for (let i = 0; i < n; i++) p *= (365 - i) / 365;
	return 1 - p;
}

/** Volatility of w in asset 1 (σ = 20%) and 1 − w in asset 2 (σ = 30%) with correlation rho. */
export function twoAssetVol(
	w: number,
	rho: number,
	s1 = 0.2,
	s2 = 0.3,
): number {
	const v =
		w * w * s1 * s1 + (1 - w) ** 2 * s2 * s2 + 2 * w * (1 - w) * rho * s1 * s2;
	return Math.sqrt(Math.max(v, 0));
}

/** Ten (market, stock) daily returns in percent used for the regression example. */
export const regressionData: [number, number][] = [
	[-4, -5],
	[-2, -3],
	[-1, 0],
	[0, 1],
	[1, 1],
	[1, 2],
	[2, 2],
	[3, 5],
	[-3, -4],
	[3, 3],
];

/** OLS fit of the regression example: slope, intercept and residual standard deviation. */
export const regressionFit = (() => {
	const n = regressionData.length;
	const mx = regressionData.reduce((s, [x]) => s + x, 0) / n;
	const my = regressionData.reduce((s, [, y]) => s + y, 0) / n;
	let sxx = 0;
	let sxy = 0;
	for (const [x, y] of regressionData) {
		sxx += (x - mx) ** 2;
		sxy += (x - mx) * (y - my);
	}
	const beta = sxy / sxx;
	const alpha = my - beta * mx;
	const sse = regressionData.reduce(
		(s, [x, y]) => s + (y - alpha - beta * x) ** 2,
		0,
	);
	return { beta, alpha, s: Math.sqrt(sse / (n - 2)), mx, sxx, n };
})();
