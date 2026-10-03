// Storage of a d-way tensor with every mode of size n, in full and in three decomposed formats,
// with every rank equal to r. Ranks of different formats are not comparable: the same accuracy
// usually needs a different r in each format.

export type Format = "cp" | "tucker" | "tt";

export interface Storage {
	full: number;
	/** CP: d factor matrices of size n x r. */
	cp: number;
	/** Tucker: an r x ... x r core plus d factor matrices of size n x r. */
	tucker: number;
	/** Tensor train: two n x r end cores and d - 2 cores of size r x n x r. */
	tt: number;
}

export function storage(n: number, d: number, r: number): Storage {
	return {
		full: n ** d,
		cp: d * n * r,
		tucker: r ** d + d * n * r,
		tt: (d - 2) * n * r * r + 2 * n * r,
	};
}

/** Compact number for labels: 1234 -> "1,234", 1.2e9 -> "1.2e9". */
export function formatCount(x: number): string {
	if (x < 1e7) return Math.round(x).toLocaleString("en-US");
	const exp = Math.floor(Math.log10(x));
	const mant = x / 10 ** exp;
	return `${mant.toFixed(mant < 9.95 ? 1 : 0).replace(/\.0$/, "")}e${exp}`;
}
