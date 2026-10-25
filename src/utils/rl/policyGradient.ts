// Policy gradient on a K-armed bandit with a softmax policy π(a) = exp(θ_a) / Σ_b exp(θ_b).
// A bandit is a one-state MDP, so q_π(a) is the mean reward μ_a of arm a and the policy
// gradient theorem reduces to ∇J = E_{A∼π}[∇ ln π(A) μ_A] with ∇_θ ln π(a) = e_a − π.
// Rewards are Gaussian, R | a ∼ N(μ_a, σ²). Used for the REINFORCE and baseline examples.

import { rng } from "./gridWorld";

export function softmax(theta: number[]): number[] {
	const m = Math.max(...theta);
	const e = theta.map((t) => Math.exp(t - m));
	const z = e.reduce((a, b) => a + b, 0);
	return e.map((x) => x / z);
}

/** ∇_θ ln π(a) = e_a − π. */
export function gradLogPi(pi: number[], a: number): number[] {
	return pi.map((p, i) => (i === a ? 1 : 0) - p);
}

/** Exact gradient ∇J = Σ_a π(a) (e_a − π) μ_a. */
export function exactGradient(theta: number[], mu: number[]): number[] {
	const pi = softmax(theta);
	const g = new Array(pi.length).fill(0);
	pi.forEach((p, a) => {
		gradLogPi(pi, a).forEach((x, i) => {
			g[i] += p * x * mu[a];
		});
	});
	return g;
}

/**
 * Mean and trace of the covariance of the one-sample estimator X = ∇ ln π(A) (R − b), with
 * A ∼ π and R | A ∼ N(μ_A, σ²). The mean does not depend on b; the variance does.
 */
export function estimatorStats(
	theta: number[],
	mu: number[],
	sigma: number,
	b: number,
): { mean: number[]; traceVar: number } {
	const pi = softmax(theta);
	const mean = exactGradient(theta, mu);
	let second = 0;
	pi.forEach((p, a) => {
		const g2 = gradLogPi(pi, a).reduce((s, x) => s + x * x, 0);
		second += p * g2 * ((mu[a] - b) ** 2 + sigma ** 2);
	});
	return { mean, traceVar: second - mean.reduce((s, x) => s + x * x, 0) };
}

/** The variance-minimising baseline of the book's eq. (10.5) and the state value v = Σ π μ. */
export function baselines(
	theta: number[],
	mu: number[],
): { optimal: number; value: number } {
	const pi = softmax(theta);
	let num = 0;
	let den = 0;
	pi.forEach((p, a) => {
		const g2 = gradLogPi(pi, a).reduce((s, x) => s + x * x, 0);
		num += p * g2 * mu[a];
		den += p * g2;
	});
	return {
		optimal: num / den,
		value: pi.reduce((s, p, a) => s + p * mu[a], 0),
	};
}

function gauss(rand: () => number): number {
	const u = Math.max(rand(), 1e-12);
	return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

export interface ReinforceOptions {
	mu: number[];
	sigma: number;
	alpha: number;
	steps: number;
	runs: number;
	seed: number;
	/** "none": b = 0; "value": b = v_π(θ) = Σ π μ (the exact state value, as a perfect critic). */
	baseline: "none" | "value";
}

/** Average probability of the best arm after each step, over independent runs. */
export function reinforceCurve(o: ReinforceOptions): number[] {
	const rand = rng(o.seed);
	const best = o.mu.indexOf(Math.max(...o.mu));
	const avg = new Array(o.steps + 1).fill(0);
	for (let run = 0; run < o.runs; run++) {
		const theta = new Array(o.mu.length).fill(0);
		avg[0] += softmax(theta)[best];
		for (let t = 1; t <= o.steps; t++) {
			const pi = softmax(theta);
			let u = rand();
			let a = 0;
			while (a < pi.length - 1 && u >= pi[a]) {
				u -= pi[a];
				a++;
			}
			const r = o.mu[a] + o.sigma * gauss(rand);
			const b =
				o.baseline === "value" ? pi.reduce((s, p, i) => s + p * o.mu[i], 0) : 0;
			gradLogPi(pi, a).forEach((x, i) => {
				theta[i] += o.alpha * x * (r - b);
			});
			avg[t] += softmax(theta)[best];
		}
	}
	return avg.map((x) => x / o.runs);
}
