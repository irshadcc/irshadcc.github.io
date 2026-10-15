// Classification of abc-parametrizations of an L-hidden-layer MLP (Yang & Hu, "Tensor Programs
// IV", arXiv 2011.14522, Section 3.2). Layer l (1 <= l <= L + 1) has weights W^l = n^{-a_l} w^l with
// w^l ~ N(0, n^{-2 b_l}), trained by SGD with learning rate eta n^{-c}. Hidden layers 2..L share
// one (a, b). Every size is reported as an exponent e, meaning Theta(n^e) as the width n grows.

export interface Abc {
	/** Number of hidden layers L >= 1. */
	L: number;
	a1: number;
	b1: number;
	/** a_l, b_l for 2 <= l <= L (unused when L = 1). */
	aH: number;
	bH: number;
	/** a_{L+1}, b_{L+1}: the readout layer. */
	aOut: number;
	bOut: number;
	c: number;
}

export const PRESETS: Record<string, { label: string; abc: Abc }> = {
	sp: {
		label: "SP, LR Θ(1)",
		abc: { L: 2, a1: 0, b1: 0, aH: 0, bH: 0.5, aOut: 0, bOut: 0.5, c: 0 },
	},
	spSmall: {
		label: "SP, LR Θ(1/n)",
		abc: { L: 2, a1: 0, b1: 0, aH: 0, bH: 0.5, aOut: 0, bOut: 0.5, c: 1 },
	},
	ntp: {
		label: "NTP",
		abc: { L: 2, a1: 0, b1: 0, aH: 0.5, bH: 0, aOut: 0.5, bOut: 0, c: 0 },
	},
	mfp: {
		label: "Mean field (L = 1)",
		abc: { L: 1, a1: 0, b1: 0, aH: 0, bH: 0.5, aOut: 1, bOut: 0, c: -1 },
	},
	mup: {
		label: "μP",
		abc: {
			L: 2,
			a1: -0.5,
			b1: 0.5,
			aH: 0,
			bH: 0.5,
			aOut: 0.5,
			bOut: 0.5,
			c: 0,
		},
	},
};

const EPS = 1e-9;
const eq = (x: number, y: number) => Math.abs(x - y) < EPS;
const ge = (x: number, y: number) => x > y - EPS;

export interface Layer {
	/** 1-based layer index. */
	l: number;
	a: number;
	b: number;
	/** Initial variance of the effective weight W^l is Theta(n^{initVar}). */
	initVar: number;
	/** Effective SGD learning rate on W^l is Theta(n^{lr}). */
	lr: number;
	/** r_l of Proposition 5.3 (hidden and input layers only); 0 means updated maximally. */
	rl?: number;
}

export interface Size {
	key: string;
	label: string;
	/** Exponent e of Theta(n^e); 0 is the target, > 0 blows up, < 0 vanishes. */
	exp: number;
	/** What exponent the quantity is allowed: "eq0" must be exactly 0, "le0" at most 0. */
	rule: "eq0" | "le0";
}

export interface Condition {
	text: string;
	value: string;
	ok: boolean;
}

export interface Classification {
	layers: Layer[];
	r: number;
	sizes: Size[];
	conditions: Condition[];
	stable: boolean;
	nontrivial: boolean;
	verdict: "unstable" | "trivial" | "feature learning" | "kernel regime";
	/** W^{L+1} updated maximally (2a+c = 1) and initialized maximally (a+b+r = 1). */
	outUpdated: boolean;
	outInitialized: boolean;
}

export function layersOf(p: Abc): { a: number; b: number }[] {
	const out = [{ a: p.a1, b: p.b1 }];
	for (let l = 2; l <= p.L; l++) out.push({ a: p.aH, b: p.bH });
	out.push({ a: p.aOut, b: p.bOut });
	return out;
}

/** Definition 3.2 and Proposition 5.3. */
export function rValues(p: Abc): { r: number; rl: number[] } {
	const base = Math.min(p.aOut + p.bOut, 2 * p.aOut + p.c) + p.c - 1;
	const ls = layersOf(p).slice(0, p.L);
	const rl = ls.map((x, i) => base + 2 * x.a + (i === 0 ? 1 : 0));
	return { r: Math.min(...rl), rl };
}

export function classify(p: Abc): Classification {
	const ls = layersOf(p);
	const { r, rl } = rValues(p);
	const layers: Layer[] = ls.map((x, i) => ({
		l: i + 1,
		a: x.a,
		b: x.b,
		initVar: -2 * (x.a + x.b),
		lr: -(2 * x.a + p.c),
		rl: i < p.L ? rl[i] : undefined,
	}));
	const out = ls[p.L];
	const sizes: Size[] = [
		{
			key: "h1",
			label: "h¹ at init (input layer, finite fan-in)",
			exp: -(p.a1 + p.b1),
			rule: "eq0",
		},
	];
	if (p.L >= 2) {
		sizes.push({
			key: "hl",
			label: "hˡ at init, l ≥ 2 (CLT: ½ − a − b)",
			exp: 0.5 - (p.aH + p.bH),
			rule: "eq0",
		});
	}
	sizes.push(
		{
			key: "f0",
			label: "f at init (CLT: ½ − a − b)",
			exp: 0.5 - (out.a + out.b),
			rule: "le0",
		},
		{ key: "dx", label: "Δxᴸ, feature change (−r)", exp: -r, rule: "le0" },
		{
			key: "dWx",
			label: "ΔWᴸ⁺¹ xᴸ (LLN: 1 − 2a − c)",
			exp: 1 - (2 * out.a + p.c),
			rule: "le0",
		},
		{
			key: "Wdx",
			label: "W₀ᴸ⁺¹ Δxᴸ (LLN: 1 − a − b − r)",
			exp: 1 - (out.a + out.b) - r,
			rule: "le0",
		},
	);
	const hiddenOk = p.L < 2 || eq(p.aH + p.bH, 0.5);
	const conditions: Condition[] = [
		{ text: "a₁ + b₁ = 0", value: fmt(p.a1 + p.b1), ok: eq(p.a1 + p.b1, 0) },
		{
			text: "aₗ + bₗ = ½ for 2 ≤ l ≤ L",
			value: p.L < 2 ? "no such l" : fmt(p.aH + p.bH),
			ok: hiddenOk,
		},
		{
			text: "aᴸ⁺¹ + bᴸ⁺¹ ≥ ½",
			value: fmt(out.a + out.b),
			ok: ge(out.a + out.b, 0.5),
		},
		{ text: "r ≥ 0", value: fmt(r), ok: ge(r, 0) },
		{
			text: "2aᴸ⁺¹ + c ≥ 1",
			value: fmt(2 * out.a + p.c),
			ok: ge(2 * out.a + p.c, 1),
		},
		{
			text: "aᴸ⁺¹ + bᴸ⁺¹ + r ≥ 1",
			value: fmt(out.a + out.b + r),
			ok: ge(out.a + out.b + r, 1),
		},
	];
	const stable = conditions.every((x) => x.ok);
	const outUpdated = eq(2 * out.a + p.c, 1);
	const outInitialized = eq(out.a + out.b + r, 1);
	const nontrivial = outUpdated || outInitialized;
	let verdict: Classification["verdict"] = "unstable";
	if (stable)
		verdict = !nontrivial
			? "trivial"
			: eq(r, 0)
				? "feature learning"
				: "kernel regime";
	return {
		layers,
		r,
		sizes,
		conditions,
		stable,
		nontrivial,
		verdict,
		outUpdated,
		outInitialized,
	};
}

/** The symmetry of eq. (5): a_l += t, b_l -= t for every layer and c -= 2t leaves f_t unchanged. */
export function shift(p: Abc, t: number): Abc {
	return {
		...p,
		a1: p.a1 + t,
		b1: p.b1 - t,
		aH: p.aH + t,
		bH: p.bH - t,
		aOut: p.aOut + t,
		bOut: p.bOut - t,
		c: p.c - 2 * t,
	};
}

/** Exponents as fractions: 0.5 -> "1/2", -1.5 -> "−3/2". */
export function fmt(x: number): string {
	const q = Math.round(x * 4);
	if (Math.abs(x * 4 - q) > 1e-6) return x.toFixed(2);
	const sign = q < 0 ? "−" : "";
	const m = Math.abs(q);
	if (m % 4 === 0) return `${sign}${m / 4}`;
	if (m % 2 === 0) return `${sign}${m / 2}/2`;
	return `${sign}${m}/4`;
}
