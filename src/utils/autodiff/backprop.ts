// The numbers behind the backpropagation post (src/content/posts/backpropagation.mdx), following
// Bishop & Bishop, "Deep Learning: Foundations and Concepts", chapter 8. Everything is plain
// float64 arithmetic so it can be checked outside the browser (node --experimental-strip-types)
// and against PyTorch autograd.
//
// - A two-layer network (8.18)-(8.20): tanh hidden units, linear outputs, biases folded in as
//   column 0 of each weight matrix (x0 = z0 = 1). forward(), backward() and jacobian() follow
//   the book's equations one for one.
// - Finite and central differences (8.24), (8.25) of one weight's derivative.
// - The evaluation trace (8.49)-(8.56) with its tangent (8.58)-(8.64) and adjoint (8.70)-(8.76)
//   equations, plus the second output f2 = v5 * v6 of (8.65).

export type Matrix = number[][];

export interface Net {
	/** First layer, shape [M, D + 1]; column 0 is the bias. */
	w1: Matrix;
	/** Second layer, shape [K, M + 1]; column 0 is the bias. */
	w2: Matrix;
}

export interface Forward {
	/** Inputs with x0 = 1 prepended. */
	x: number[];
	/** Hidden pre-activations a_j (8.18). */
	a: number[];
	/** Hidden activations with z0 = 1 prepended (8.19). */
	z: number[];
	/** Outputs y_k = a_k (8.20). */
	y: number[];
}

export interface Backward extends Forward {
	/** Sum-of-squares error (8.17). */
	E: number;
	/** Output errors delta_k = y_k - t_k (8.21). */
	dOut: number[];
	/** Hidden errors delta_j = (1 - z_j^2) sum_k w_kj delta_k (8.22). */
	dHid: number[];
	/** dE/dw1[j][i] = delta_j x_i and dE/dw2[k][j] = delta_k z_j (8.23). */
	g1: Matrix;
	g2: Matrix;
}

const dot = (w: number[], v: number[]) =>
	w.reduce((s, wi, i) => s + wi * v[i], 0);

export function forward(net: Net, input: number[]): Forward {
	const x = [1, ...input];
	const a = net.w1.map((row) => dot(row, x));
	const z = [1, ...a.map(Math.tanh)];
	const y = net.w2.map((row) => dot(row, z));
	return { x, a, z, y };
}

export function error(net: Net, input: number[], t: number[]): number {
	const { y } = forward(net, input);
	return 0.5 * y.reduce((s, yk, k) => s + (yk - t[k]) ** 2, 0);
}

export function backward(net: Net, input: number[], t: number[]): Backward {
	const f = forward(net, input);
	const E = 0.5 * f.y.reduce((s, yk, k) => s + (yk - t[k]) ** 2, 0);
	const dOut = f.y.map((yk, k) => yk - t[k]);
	// Hidden unit j sits in column j + 1 of w2 (column 0 is the bias).
	const dHid = f.a.map((_, j) => {
		const zj = f.z[j + 1];
		return (
			(1 - zj * zj) * dOut.reduce((s, dk, k) => s + net.w2[k][j + 1] * dk, 0)
		);
	});
	const g1 = dHid.map((dj) => f.x.map((xi) => dj * xi));
	const g2 = dOut.map((dk) => f.z.map((zj) => dk * zj));
	return { ...f, E, dOut, dHid, g1, g2 };
}

/** Error with one weight shifted by eps: layer 1 or 2, row r, column c. */
export function shifted(
	net: Net,
	layer: 1 | 2,
	r: number,
	c: number,
	eps: number,
): Net {
	const copy = {
		w1: net.w1.map((row) => [...row]),
		w2: net.w2.map((row) => [...row]),
	};
	(layer === 1 ? copy.w1 : copy.w2)[r][c] += eps;
	return copy;
}

/** Forward difference (8.24). */
export function forwardDiff(
	net: Net,
	input: number[],
	t: number[],
	layer: 1 | 2,
	r: number,
	c: number,
	eps: number,
) {
	return (
		(error(shifted(net, layer, r, c, eps), input, t) - error(net, input, t)) /
		eps
	);
}

/** Central difference (8.25). */
export function centralDiff(
	net: Net,
	input: number[],
	t: number[],
	layer: 1 | 2,
	r: number,
	c: number,
	eps: number,
) {
	return (
		(error(shifted(net, layer, r, c, eps), input, t) -
			error(shifted(net, layer, r, c, -eps), input, t)) /
		(2 * eps)
	);
}

export interface JacobianRow {
	/** dy_k / da_l for the output units: the identity row for linear outputs (8.31). */
	dOut: number[];
	/** dy_k / da_j for the hidden units, by (8.30). */
	dHid: number[];
	/** J_ki = dy_k / dx_i for the real inputs (no bias), by (8.29). */
	row: number[];
}

/** One row of the Jacobian per output unit, by backpropagation (8.29)-(8.31). */
export function jacobian(net: Net, input: number[]): JacobianRow[] {
	const f = forward(net, input);
	return f.y.map((_, k) => {
		const dOut = f.y.map((_, l) => (l === k ? 1 : 0));
		const dHid = f.a.map((_, j) => {
			const zj = f.z[j + 1];
			return (
				(1 - zj * zj) * dOut.reduce((s, d, l) => s + net.w2[l][j + 1] * d, 0)
			);
		});
		const row = input.map((_, i) =>
			dHid.reduce((s, d, j) => s + net.w1[j][i + 1] * d, 0),
		);
		return { dOut, dHid, row };
	});
}

/** Central-difference Jacobian (8.35): 2D forward passes. */
export function jacobianFd(net: Net, input: number[], eps: number): Matrix {
	const K = net.w2.length;
	const cols = input.map((_, i) => {
		const up = forward(
			net,
			input.map((v, m) => (m === i ? v + eps : v)),
		).y;
		const dn = forward(
			net,
			input.map((v, m) => (m === i ? v - eps : v)),
		).y;
		return up.map((u, k) => (u - dn[k]) / (2 * eps));
	});
	return Array.from({ length: K }, (_, k) => cols.map((c) => c[k]));
}

// ---------------------------------------------------------------------------------------------
// The evaluation trace of f(x1, x2) = x1 x2 + exp(x1 x2) - sin(x2)   (8.49)
// and f2(x1, x2) = (x1 x2 - sin(x2)) exp(x1 x2)                       (8.65)

/** Primal variables v1..v8 (index 0 unused), (8.50)-(8.56) plus v8 = v5 v6. */
export function tracePrimal(x1: number, x2: number): number[] {
	const v = [Number.NaN, x1, x2];
	v[3] = v[1] * v[2];
	v[4] = Math.sin(v[2]);
	v[5] = Math.exp(v[3]);
	v[6] = v[3] - v[4];
	v[7] = v[5] + v[6];
	v[8] = v[5] * v[6];
	return v;
}

/** Tangent variables for a seed (x1dot, x2dot), (8.58)-(8.64) plus v8dot. */
export function traceTangent(v: number[], seed: [number, number]): number[] {
	const d = [Number.NaN, seed[0], seed[1]];
	d[3] = v[1] * d[2] + d[1] * v[2];
	d[4] = d[2] * Math.cos(v[2]);
	d[5] = d[3] * Math.exp(v[3]);
	d[6] = d[3] - d[4];
	d[7] = d[5] + d[6];
	d[8] = d[5] * v[6] + v[5] * d[6];
	return d;
}

/** Adjoint variables of f = v7, (8.70)-(8.76). Index 8 is unused. */
export function traceAdjoint(v: number[]): number[] {
	const b = new Array<number>(9).fill(Number.NaN);
	b[7] = 1;
	b[6] = b[7];
	b[5] = b[7];
	b[4] = -b[6];
	b[3] = b[5] * v[5] + b[6];
	b[2] = b[3] * v[1] + b[4] * Math.cos(v[2]);
	b[1] = b[3] * v[2];
	return b;
}

/** Adjoint variables of f2 = v8: the second backward pass needed for a second output. */
export function traceAdjointF2(v: number[]): number[] {
	const b = new Array<number>(9).fill(Number.NaN);
	b[8] = 1;
	b[7] = 0;
	b[6] = b[8] * v[5];
	b[5] = b[8] * v[6];
	b[4] = -b[6];
	b[3] = b[5] * v[5] + b[6];
	b[2] = b[3] * v[1] + b[4] * Math.cos(v[2]);
	b[1] = b[3] * v[2];
	return b;
}

/** Closed form of df/dx1 = x2 + x2 exp(x1 x2), for checking. */
export const dfdx1 = (x1: number, x2: number) => x2 + x2 * Math.exp(x1 * x2);
