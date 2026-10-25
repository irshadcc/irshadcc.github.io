// Data for the figures and tables of the "Mathematical foundations of reinforcement learning"
// post, computed at build time from gridWorld.ts and policyGradient.ts. Seeds and parameters are
// fixed so that the numbers quoted in the post's text stay true; change them only together with
// the text.

import {
	argmaxAll,
	bestEpsGreedy,
	bookGrid,
	buildModel,
	epsGreedyPolicy,
	greedy,
	linearValueFit,
	maxAbsDiff,
	policyIteration,
	polyFeatures,
	rmse,
	rng,
	solveOptimal,
	stationary,
	tdControl,
	uniformPolicy,
	valueIteration,
} from "./gridWorld";
import { baselines, estimatorStats, reinforceCurve } from "./policyGradient";

const grid = bookGrid();
const model = buildModel(grid);
const optimal = solveOptimal(model, grid);

/** ‖v_k − v*‖∞ for value iteration, policy iteration and truncated policy iteration (j = 3). */
export const dpIters = 40;
const stay = new Array(model.n).fill(4);
const vi = valueIteration(model, grid, dpIters);
const pi = policyIteration(model, grid, stay, dpIters);
const tpi = policyIteration(model, grid, stay, dpIters, 3);
export const dpCurves = {
	vi: vi.map((s) => s.err),
	pi: pi.map((s) => s.err),
	tpi: tpi.map((s) => s.err),
	/** Contraction bound γ^k ‖v_0 − v*‖∞ for value iteration from v_0 = 0. */
	bound: vi.map((_, k) => grid.gamma ** k * maxAbsDiff(vi[0].v, optimal.v)),
	/** Iterations until the greedy policy is optimal in every state, and until ‖v − v*‖∞ < 0.01. */
	firstOptimal: Object.fromEntries(
		(
			[
				["vi", vi],
				["pi", pi],
				["tpi", tpi],
			] as const
		).map(([k, steps]) => {
			const opt = optimal.q.map((row) => new Set(argmaxAll(row, 1e-6)));
			const policyAt = steps.findIndex(
				(s, i) => i > 0 && s.pi.every((a, st) => opt[st].has(a)),
			);
			return [
				k,
				{
					policy: policyAt,
					value: steps.findIndex((s) => s.err < 0.01),
					sweeps: steps.map((s) => s.sweeps),
				},
			];
		}),
	),
};

/** Mean estimation of a fair die roll (mean 3.5) with w_{k+1} = w_k − a_k (w_k − x_k). */
export const rmSamples = 300;
const die = rng(3);
const rolls = Array.from(
	{ length: rmSamples },
	() => 1 + Math.floor(die() * 6),
);
function rmRun(step: (k: number) => number): number[] {
	const w = [0];
	rolls.forEach((x, i) => {
		const k = i + 1;
		w.push(w[i] - step(k) * (w[i] - x));
	});
	return w;
}
export const rmCurves = {
	rolls,
	harmonic: rmRun((k) => 1 / k),
	constant: rmRun(() => 0.1),
	slow: rmRun((k) => 1 / k ** 0.6),
};

/** Q-learning and Sarsa on the book grid: RMSE of max_a q against v* every 1000 steps. */
export const tdEvery = 1000;
export const tdSteps = 100000;
const td = (algorithm: "q-learning" | "sarsa", eps: number) =>
	tdControl(model, grid, {
		algorithm,
		steps: tdSteps,
		alpha: 0.1,
		eps,
		seed: 1,
		start: 0,
		every: tdEvery,
	});
const runs = {
	q1: td("q-learning", 1),
	q05: td("q-learning", 0.5),
	q01: td("q-learning", 0.1),
	s01: td("sarsa", 0.1),
};
export const tdCurves = Object.fromEntries(
	Object.entries(runs).map(([k, r]) => [
		k,
		{ err: r.curve.map((c) => c[1]), good: r.curve.map((c) => c[2]) },
	]),
) as Record<keyof typeof runs, { err: number[]; good: number[] }>;
export const tdMinVisits = Object.fromEntries(
	Object.entries(runs).map(([k, r]) => [k, Math.min(...r.visits)]),
) as Record<keyof typeof runs, number>;

/** The cost of exploring: the best ε-greedy policy for several ε. */
export const epsTable = [0, 0.1, 0.2, 0.5, 1].map((eps) => {
	const b = bestEpsGreedy(model, grid, eps);
	const opt = optimal.q.map((row) => new Set(argmaxAll(row, 1e-6)));
	return {
		eps,
		greedyProb: 1 - (eps * 4) / 5,
		otherProb: eps / 5,
		vStart: b.v[0],
		vTarget: b.v[17],
		rmse: rmse(b.v, optimal.v),
		greedyDiffers: b.pi.filter((a, s) => !opt[s].has(a)).length,
	};
});

/** Where an ε-greedy (ε = 0.5) version of the optimal policy spends its time. */
export const stationaryEps = stationary(
	model,
	epsGreedyPolicy(greedy(optimal.q), 0.5),
);

/** TD-Linear fixed points for the uniform policy (book §8.2.4) with polynomial features. */
const uniform = uniformPolicy(model.n);
export const linearFits = ([1, 2, 3] as const).map((deg) => {
	const f = linearValueFit(model, grid, uniform, polyFeatures(grid, deg));
	return {
		features: polyFeatures(grid, deg)[0].length,
		vTd: f.vTd,
		vBest: f.vBest,
		v: f.v,
		rmseTd: rmse(f.vTd, f.v),
		rmseBest: rmse(f.vBest, f.v),
	};
});
export const uniformStationary = stationary(model, uniform);

/** Softmax bandit: arms with mean rewards μ, σ = 1; the same arms shifted by +10. */
export const banditMu = [1, 2, 4];
export const banditMuShift = banditMu.map((m) => m + 10);
const theta0 = [0, 0, 0];
export const banditStats = [banditMu, banditMuShift].map((mu) => {
	const b = baselines(theta0, mu);
	return {
		mu,
		baseline: b,
		mean: estimatorStats(theta0, mu, 1, 0).mean,
		varNone: estimatorStats(theta0, mu, 1, 0).traceVar,
		varValue: estimatorStats(theta0, mu, 1, b.value).traceVar,
	};
});
export const reinforceSteps = 300;
const rf = (mu: number[], baseline: "none" | "value") =>
	reinforceCurve({
		mu,
		sigma: 1,
		alpha: 0.1,
		steps: reinforceSteps,
		runs: 1000,
		seed: 7,
		baseline,
	});
export const reinforceCurves = {
	plain: rf(banditMu, "none"),
	plainShift: rf(banditMuShift, "none"),
	baseShift: rf(banditMuShift, "value"),
};
