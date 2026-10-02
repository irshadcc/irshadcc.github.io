/** Normal-distribution helpers and two-proportion A/B-test calculations. */

export function normalCdf(x: number): number {
	const sign = x < 0 ? -1 : 1;
	const a = Math.abs(x) / Math.sqrt(2);
	const t = 1 / (1 + 0.3275911 * a);
	const polynomial =
		((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
			t +
			0.254829592) *
		t;
	const erf = sign * (1 - polynomial * Math.exp(-a * a));
	return (1 + erf) / 2;
}

export function normalQuantile(p: number): number {
	if (!(p > 0 && p < 1)) throw new RangeError("p must be between zero and one");
	let lo = -8;
	let hi = 8;
	for (let i = 0; i < 80; i++) {
		const mid = (lo + hi) / 2;
		if (normalCdf(mid) < p) lo = mid;
		else hi = mid;
	}
	return (lo + hi) / 2;
}

export function twoProportionPower(
	baseline: number,
	treatment: number,
	nPerArm: number,
	alpha = 0.05,
): number {
	const pooled = (baseline + treatment) / 2;
	const nullSe = Math.sqrt((2 * pooled * (1 - pooled)) / nPerArm);
	const altSe = Math.sqrt(
		(baseline * (1 - baseline) + treatment * (1 - treatment)) / nPerArm,
	);
	const effect = treatment - baseline;
	const critical = normalQuantile(1 - alpha / 2) * nullSe;
	return (
		1 -
		normalCdf((critical - effect) / altSe) +
		normalCdf((-critical - effect) / altSe)
	);
}

export function requiredSamplePerArm(
	baseline: number,
	treatment: number,
	alpha = 0.05,
	power = 0.8,
): number {
	const pooled = (baseline + treatment) / 2;
	const za = normalQuantile(1 - alpha / 2);
	const zb = normalQuantile(power);
	const numerator =
		za * Math.sqrt(2 * pooled * (1 - pooled)) +
		zb * Math.sqrt(baseline * (1 - baseline) + treatment * (1 - treatment));
	return Math.ceil((numerator / (treatment - baseline)) ** 2);
}
