/** Pure calculations used by ImportanceSamplingExplorer. */

const SQRT_TWO_PI = Math.sqrt(2 * Math.PI);

export function normalPdf(x: number, mean = 0): number {
	return Math.exp(-0.5 * (x - mean) ** 2) / SQRT_TWO_PI;
}

/** Complementary error function approximation from Numerical Recipes. */
function erfc(x: number): number {
	const z = Math.abs(x);
	const t = 1 / (1 + z / 2);
	const coefficients = [
		1.00002368, 0.37409196, 0.09678418, -0.18628806, 0.27886807, -1.13520398,
		1.48851587, -0.82215223, 0.17087277,
	];
	let polynomial = coefficients[coefficients.length - 1];
	for (let i = coefficients.length - 2; i >= 0; i--)
		polynomial = coefficients[i] + t * polynomial;
	const answer = t * Math.exp(-z * z - 1.26551223 + t * polynomial);
	return x >= 0 ? answer : 2 - answer;
}

export function normalCdf(x: number): number {
	return 0.5 * erfc(-x / Math.SQRT2);
}

export function normalTail(threshold: number): number {
	return 0.5 * erfc(threshold / Math.SQRT2);
}

/** p(x) / q(x) for p=N(0,1) and q=N(mean,1), evaluated stably. */
export function importanceWeight(x: number, mean: number): number {
	return Math.exp(0.5 * mean * mean - mean * x);
}

/**
 * Variance of one importance-sampling contribution
 * 1[x > threshold] p(x)/q(x), where q=N(mean,1).
 */
export function contributionVariance(threshold: number, mean: number): number {
	const probability = normalTail(threshold);
	const secondMoment = Math.exp(mean * mean) * normalTail(threshold + mean);
	return Math.max(0, secondMoment - probability * probability);
}

export function varianceReduction(threshold: number, mean: number): number {
	const probability = normalTail(threshold);
	const directVariance = probability * (1 - probability);
	return directVariance / contributionVariance(threshold, mean);
}
