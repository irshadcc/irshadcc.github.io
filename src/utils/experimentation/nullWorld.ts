/**
 * Simulation logic for `NullWorld.astro`: many repeated two-arm conversion experiments,
 * used to show a p-value, power and A/A tests as counts of simulated experiments.
 *
 * Conversion counts are drawn with the normal approximation to the binomial
 * (n·p users convert on average, standard deviation sqrt(n·p·(1−p))), rounded and clipped.
 * That is accurate for the sizes used here (n ≥ 500 per arm, p around 10%).
 */
import { normalCdf, normalQuantile } from "./statistics";

export type Rng = () => number;

/** Small seeded generator (mulberry32) so the figure and the text show the same numbers. */
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

export function gaussian(rng: Rng): number {
	const u = 1 - rng();
	const v = rng();
	return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Number of converting users out of n when each converts with probability p. */
export function drawConversions(n: number, p: number, rng: Rng): number {
	const k = Math.round(n * p + Math.sqrt(n * p * (1 - p)) * gaussian(rng));
	return Math.min(n, Math.max(0, k));
}

/** Two-sided pooled two-proportion z-test p-value. */
export function zTestPValue(
	convC: number,
	convT: number,
	nPerArm: number,
): number {
	const pooled = (convC + convT) / (2 * nPerArm);
	const se = Math.sqrt((pooled * (1 - pooled) * 2) / nPerArm);
	if (se === 0) return 1;
	const z = (convT - convC) / nPerArm / se;
	return 2 * (1 - normalCdf(Math.abs(z)));
}

/** Standard error of the difference in rates when both arms convert at p. */
export function nullSe(p: number, nPerArm: number): number {
	return Math.sqrt((2 * p * (1 - p)) / nPerArm);
}

/**
 * Simulate `sims` experiments with control rate p0 and treatment rate p0 + lift.
 * Returns the observed differences in conversion rate (treatment − control).
 */
export function simulateDiffs(
	p0: number,
	lift: number,
	nPerArm: number,
	sims: number,
	seed: number,
): number[] {
	const rng = mulberry32(seed);
	const out = new Array<number>(sims);
	for (let i = 0; i < sims; i++) {
		const c = drawConversions(nPerArm, p0, rng);
		const t = drawConversions(nPerArm, p0 + lift, rng);
		out[i] = (t - c) / nPerArm;
	}
	return out;
}

/** Share of simulated gaps at least as far from zero as the observed one (two-sided). */
export function simulatedPValue(diffs: number[], observed: number): number {
	const a = Math.abs(observed) - 1e-12;
	let k = 0;
	for (const d of diffs) if (Math.abs(d) >= a) k++;
	return k / diffs.length;
}

/** Normal-approximation p-value for an observed gap, using the no-effect spread. */
export function formulaPValue(
	p0: number,
	observed: number,
	nPerArm: number,
): number {
	const z = Math.abs(observed) / nullSe(p0, nPerArm);
	return 2 * (1 - normalCdf(z));
}

/** The gap a 5%-level (or other alpha) test needs before it calls a result significant. */
export function cutoff(p0: number, nPerArm: number, alpha = 0.05): number {
	return normalQuantile(1 - alpha / 2) * nullSe(p0, nPerArm);
}

/** Share of simulated gaps beyond ±cutoff: the simulated power when the lift is real. */
export function shareBeyond(diffs: number[], c: number): number {
	let k = 0;
	for (const d of diffs) if (Math.abs(d) >= c) k++;
	return k / diffs.length;
}

/**
 * Simulate A/A tests (both arms convert at p) that run for `days` days with nPerArm/days new
 * users per arm per day. Returns the p-value at the end, and the smallest p-value seen if the
 * team had checked after every day.
 */
export function simulateAa(
	p: number,
	nPerArm: number,
	days: number,
	sims: number,
	seed: number,
): { final: number[]; smallestDaily: number[] } {
	const rng = mulberry32(seed);
	const perDay = Math.max(1, Math.round(nPerArm / days));
	const final = new Array<number>(sims);
	const smallestDaily = new Array<number>(sims);
	for (let i = 0; i < sims; i++) {
		let c = 0;
		let t = 0;
		let smallest = 1;
		let pv = 1;
		for (let d = 1; d <= days; d++) {
			c += drawConversions(perDay, p, rng);
			t += drawConversions(perDay, p, rng);
			pv = zTestPValue(c, t, perDay * d);
			if (pv < smallest) smallest = pv;
		}
		final[i] = pv;
		smallestDaily[i] = smallest;
	}
	return { final, smallestDaily };
}

/** Counts per bin over [lo, hi); values outside are clamped into the end bins. */
export function histogram(
	values: number[],
	lo: number,
	hi: number,
	bins: number,
): number[] {
	const counts = new Array<number>(bins).fill(0);
	const w = (hi - lo) / bins;
	for (const v of values) {
		const b = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / w)));
		counts[b]++;
	}
	return counts;
}
