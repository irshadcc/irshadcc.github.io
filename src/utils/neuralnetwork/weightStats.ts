// Summarises weight tensors for NeuralNetworkGraph's hover card: shape, parameter count,
// moments, a histogram of the values, and the numerical rank with its singular values.
// Runs at build time only (it is imported from the component's frontmatter), so neither the
// SVD nor the raw weights reach the browser.
import { Matrix, SingularValueDecomposition } from "ml-matrix";
import type { NodeWeights, Tensor } from "./NNGraph";

export interface WeightSummary {
	name: string;
	shape: number[];
	count: number;
	mean: number;
	std: number;
	min: number;
	max: number;
	/** Counts of values in HIST_BINS equal-width bins spanning [min, max]. */
	hist: number[];
	/** Absent for scalars and vectors, where rank means nothing. */
	rank?: {
		rank: number;
		/** The rank a matrix of this size would have if it were full rank: min(rows, cols). */
		full: number;
		/** Size of the matrix the rank is taken of (the mode-0 unfolding for ≥3-D tensors). */
		rows: number;
		cols: number;
		/** Singular values, largest first. */
		singular: number[];
		/** Singular values at or below this count as zero. */
		tol: number;
	};
}

export const HIST_BINS = 32;

/** A statistic for display: 3 significant digits, e.g. "0.0508", "-1.31", "1.23e-7". */
export const formatValue = (v: number) =>
	v === 0 ? "0" : Number(v.toPrecision(3)).toString();

const isFlat = (t: Tensor): t is { shape: number[]; data: ArrayLike<number> } =>
	!Array.isArray(t);

/** Row-major flat data and shape of a tensor given either way; nested arrays must be rectangular. */
function flatten(
	t: Tensor,
	name: string,
): { shape: number[]; data: ArrayLike<number>; float32: boolean } {
	if (isFlat(t)) {
		const n = t.shape.reduce((a, b) => a * b, 1);
		if (n !== t.data.length) {
			throw new Error(
				`weights "${name}": shape ${t.shape.join("×")} needs ${n} values, got ${t.data.length}`,
			);
		}
		return {
			shape: t.shape,
			data: t.data,
			float32: t.data instanceof Float32Array,
		};
	}
	const shape: number[] = [];
	for (let a: unknown = t; Array.isArray(a); a = a[0]) shape.push(a.length);
	const data: number[] = [];
	const walk = (a: unknown, depth: number) => {
		if (depth === shape.length) {
			if (typeof a !== "number")
				throw new Error(`weights "${name}": ragged or non-numeric array`);
			data.push(a);
			return;
		}
		if (!Array.isArray(a) || a.length !== shape[depth])
			throw new Error(`weights "${name}": ragged array`);
		for (const x of a) walk(x, depth + 1);
	};
	walk(t, 0);
	return { shape, data, float32: false };
}

function summarize(name: string, t: Tensor): WeightSummary {
	const { shape, data, float32 } = flatten(t, name);
	const count = data.length;
	let min = Number.POSITIVE_INFINITY;
	let max = Number.NEGATIVE_INFINITY;
	let sum = 0;
	for (let i = 0; i < count; i++) {
		const v = data[i];
		if (v < min) min = v;
		if (v > max) max = v;
		sum += v;
	}
	const mean = sum / count;
	let sq = 0;
	for (let i = 0; i < count; i++) sq += (data[i] - mean) ** 2;
	const std = Math.sqrt(sq / count);

	const hist = new Array<number>(HIST_BINS).fill(0);
	const width = (max - min) / HIST_BINS;
	for (let i = 0; i < count; i++) {
		const b = width > 0 ? Math.floor((data[i] - min) / width) : 0;
		hist[Math.min(b, HIST_BINS - 1)]++;
	}

	return {
		name,
		shape,
		count,
		mean,
		std,
		min,
		max,
		hist,
		rank: shape.length >= 2 ? rankOf(shape, data, float32) : undefined,
	};
}

/**
 * Numerical rank, as numpy's matrix_rank computes it: the number of singular values above
 * σ_max · max(rows, cols) · ε, with ε the machine epsilon of the data's precision. A tensor of
 * three or more dimensions is unfolded along its first axis (e.g. a conv kernel
 * out × in × kh × kw becomes out × (in·kh·kw)), which is the usual rank of a layer's weights.
 */
function rankOf(
	shape: number[],
	data: ArrayLike<number>,
	float32: boolean,
): WeightSummary["rank"] {
	const rows = shape[0];
	const cols = data.length / rows;
	const m = Matrix.from1DArray(rows, cols, Array.from(data));
	const svd = new SingularValueDecomposition(m, {
		computeLeftSingularVectors: false,
		computeRightSingularVectors: false,
		autoTranspose: true,
	});
	const singular = [...svd.diagonal].sort((a, b) => b - a);
	const eps = float32 ? 2 ** -23 : Number.EPSILON;
	const tol = (singular[0] ?? 0) * Math.max(rows, cols) * eps;
	return {
		rank: singular.filter((s) => s > tol).length,
		full: Math.min(rows, cols),
		rows,
		cols,
		singular,
		tol,
	};
}

/** Summaries of a node's weights; a bare tensor is named "W". */
export function summarizeWeights(w: NodeWeights): WeightSummary[] {
	const named =
		Array.isArray(w) || isFlatTensor(w)
			? { W: w as Tensor }
			: (w as Record<string, Tensor>);
	return Object.entries(named).map(([name, t]) => summarize(name, t));
}

const isFlatTensor = (w: NodeWeights) =>
	typeof w === "object" &&
	w !== null &&
	"shape" in w &&
	"data" in w &&
	Array.isArray((w as { shape: unknown }).shape);
