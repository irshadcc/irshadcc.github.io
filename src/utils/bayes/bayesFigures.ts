// Data for the static charts and tables in the Bayesian statistics and optimisation post,
// computed from bayes.ts at build time so the text, tables and figures agree.
import {
	type Acquisition,
	bayesLinear,
	betaPdf,
	betaQuantile,
	gpFit,
	gpLogEvidence,
	gpPredict,
	logEvidenceBeta,
	normPdf,
	runBo,
} from "./bayes";

const gauss = (x: number, m: number, v: number) =>
	normPdf((x - m) / Math.sqrt(v)) / Math.sqrt(v);

/** Normal–normal example: prior N(10, 3²), σ = 2, n = 4 readings with mean 12. */
export const normalPrior = (m: number) => gauss(m, 10, 9);
/** Likelihood of the mean as a density in μ: N(x̄, σ²/n) = N(12, 1). */
export const normalLikelihood = (m: number) => gauss(m, 12, 1);
export const normalPosterior = (m: number) => gauss(m, 11.8, 0.9);

/** Negative log prior, negative log likelihood and their sum (up to constants), same example. */
export const nlPrior = (m: number) => (m - 10) ** 2 / 18;
export const nlLik = (m: number) => (m - 12) ** 2 / 2;
export const nlPost = (m: number) => nlPrior(m) + nlLik(m);

/** Beta(9, 5) posterior and its Laplace approximation N(2/3, 1/54). */
export const betaPost = (t: number) => betaPdf(t, 9, 5);
export const betaLaplace = (t: number) => gauss(t, 2 / 3, 1 / 54);
export const betaMedian = betaQuantile(0.5, 9, 5);

/** Evidence of one specific 10-flip sequence with k heads: fair coin vs uniform prior on θ. */
export const evidenceFair = (_k: number) => 0.5 ** 10;
export const evidenceUniform = (k: number) =>
	Math.exp(logEvidenceBeta(Math.round(k), 10 - Math.round(k)));

/** Bayesian linear regression: six readings, basis φ(x) = (1, x), α = 2, β = 25. */
export const blrX = [-0.9, -0.5, -0.1, 0.2, 0.6, 0.8];
export const blrT = [-0.31, 0.18, 0.39, 0.73, 0.92, 1.24];
export const blr = bayesLinear(
	blrX.map((x) => [1, x]),
	blrT,
	2,
	25,
);
export const blrMean = (x: number) => blr.predict([1, x]).mean;
export const blrLower = (x: number) => blrMean(x) - 2 * blr.predict([1, x]).sd;
export const blrUpper = (x: number) => blrMean(x) + 2 * blr.predict([1, x]).sd;
export const blrPoints = blrX.map((x, i) => [x, blrT[i]] as [number, number]);

/** The EI run used in the text (init 0, 0.5, 1; ℓ = 0.1). */
export const eiRun = runBo({ acquisition: "ei", iterations: 10 });
export const eiTable = eiRun.slice(1).map((s, i) => ({
	n: s.X.length,
	x: s.X[s.X.length - 1],
	f: s.y[s.y.length - 1],
	best: s.best,
	i: i + 1,
}));

/** Log marginal likelihood of the GP against ℓ, fitted to the 8 evaluations after 5 EI steps. */
const at5 = eiRun[5];
export const logEvidenceEll = (ell: number) =>
	gpLogEvidence(gpFit(at5.X, at5.y, { ell, sf: 1, sn: 1e-3 }));

/** First-step quantities at a few x for the worked EI example. */
const fit0 = gpFit([0, 0.5, 1], eiRun[0].y, { ell: 0.1, sf: 1, sn: 1e-3 });
export const firstStep = {
	yMean: fit0.yMean,
	yStd: fit0.yStd,
	best: eiRun[0].best,
	at: (x: number) => gpPredict(fit0, x),
};

/** Evaluations needed to come within 0.01 of the true maximum, per acquisition (20 iterations). */
const TRUE_MAX = 6.0207;
export const acqComparison = (
	[
		["ei", 0, "EI, ξ = 0"],
		["ei", 0.1, "EI, ξ = 0.1"],
		["pi", 0.01, "PI, ξ = 0.01"],
		["pi", 0.5, "PI, ξ = 0.5"],
		["ucb", 2, "UCB, κ = 2"],
		["ucb", 0.5, "UCB, κ = 0.5"],
		["ts", 0, "Thompson (seed 1)"],
	] as [Acquisition, number, string][]
).map(([acquisition, param, name]) => {
	const run = runBo({ acquisition, param, iterations: 20 });
	const last = run[run.length - 1];
	const hit = last.y.findIndex((v) => v > TRUE_MAX - 0.01);
	return { name, evals: hit < 0 ? null : hit + 1, best: last.best };
});
