// Small dense-matrix helpers for attention, run at build time. Matrices are row-major number[][].

export type Matrix = number[][];

export const shape = (m: Matrix) => [m.length, m[0]?.length ?? 0] as const;

/** a @ b */
export function matmul(a: Matrix, b: Matrix): Matrix {
	const [n, k] = shape(a);
	const [k2, m] = shape(b);
	if (k !== k2) throw new Error(`matmul: ${n}x${k} @ ${k2}x${m} do not chain`);
	return a.map((row) =>
		Array.from({ length: m }, (_, j) =>
			row.reduce((s, v, t) => s + v * b[t][j], 0),
		),
	);
}

export const transpose = (m: Matrix): Matrix =>
	(m[0] ?? []).map((_, j) => m.map((row) => row[j]));

/** x @ w.T, i.e. applying an nn.Linear weight of shape [out, in] to rows of x. */
export const linear = (x: Matrix, w: Matrix) => matmul(x, transpose(w));

/** Row-wise softmax; -Infinity entries get weight 0. */
export function softmax(m: Matrix): Matrix {
	return m.map((row) => {
		const max = Math.max(...row);
		const e = row.map((v) => (Number.isFinite(v) ? Math.exp(v - max) : 0));
		const sum = e.reduce((a, b) => a + b, 0);
		return e.map((v) => v / sum);
	});
}

/**
 * Scaled dot-product attention weights softmax(Q Kᵀ · scale), with an optional causal mask
 * (query i only sees keys 0..i). Returns the raw scores too, masked entries as -Infinity.
 */
export function attentionWeights(
	q: Matrix,
	k: Matrix,
	{ causal = false, scale }: { causal?: boolean; scale?: number } = {},
) {
	const dk = shape(k)[1];
	if (shape(q)[1] !== dk)
		throw new Error(
			`attention: Q has width ${shape(q)[1]} but K has width ${dk}`,
		);
	const s = scale ?? 1 / Math.sqrt(dk);
	const scores = matmul(q, transpose(k)).map((row, i) =>
		row.map((v, j) => (causal && j > i ? Number.NEGATIVE_INFINITY : v * s)),
	);
	return { scores, weights: softmax(scores) };
}
