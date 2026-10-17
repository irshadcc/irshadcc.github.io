// Figure data for the muTransfer post (src/content/posts/mu-transfer.mdx). The numbers live in
// muTransferFigures.json, written by the scratch script gen_figures.py from theory.py (numpy,
// float64) and mlp_sweep.py (PyTorch, CIFAR-10); do not edit the JSON by hand.
import data from "./muTransferFigures.json";

type Rows = Record<string, (number | null)[]>;
export interface Sweep {
	log2lr: number[];
	steps: number;
	sp: Rows;
	mup: Rows;
}

export const toy = data.toy;
export const toyF = data.toyF;
export const av = data.av;
export const linear = data.linear;
export const limit = data.limit;
export const sweepAdam: Sweep = data.sweepAdam;
export const coord = data.coord;

/** A LineChart series function that looks y up at sampled x values (NaN, i.e. a gap, elsewhere). */
export function lookup(
	xs: readonly number[],
	ys: readonly (number | null)[],
): (x: number) => number {
	return (x: number) => {
		const i = xs.findIndex((v) => Math.abs(v - x) < 1e-9);
		const y = i < 0 ? null : ys[i];
		return y === null || y === undefined ? Number.NaN : y;
	};
}
