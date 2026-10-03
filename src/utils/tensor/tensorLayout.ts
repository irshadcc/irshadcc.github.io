// Index arithmetic for drawing a 3-way tensor: column-major linear indices, mode-k unfoldings
// (columns in the natural order of Kolda and Ballard), fiber and slice groups, and the
// isometric positions of the cells. Indices here are 0-based; the figures label them 1-based.

export type Dims = [number, number, number];
export type Mode = 1 | 2 | 3;
export type View = "fibers" | "slices";

/** Column-major linear index of entry (i, j, k): i varies fastest, as in vec(X). */
export function linear(dims: Dims, i: number, j: number, k: number): number {
	return i + dims[0] * (j + dims[1] * k);
}

/** Inverse of `linear`. */
export function subscripts(dims: Dims, lin: number): [number, number, number] {
	const i = lin % dims[0];
	const j = Math.floor(lin / dims[0]) % dims[1];
	const k = Math.floor(lin / (dims[0] * dims[1]));
	return [i, j, k];
}

/** Size of the mode-k unfolding X_(k): n_k rows by (product of the other sizes) columns. */
export function unfoldingShape(dims: Dims, mode: Mode): [number, number] {
	const rows = dims[mode - 1];
	return [rows, (dims[0] * dims[1] * dims[2]) / rows];
}

/**
 * Where entry (i, j, k) lands in X_(k). The row is the mode-k index; the column is the linear
 * index of the remaining indices in their natural order (lowest remaining mode fastest).
 */
export function unfoldingPosition(
	dims: Dims,
	mode: Mode,
	i: number,
	j: number,
	k: number,
): [number, number] {
	if (mode === 1) return [i, j + dims[1] * k];
	if (mode === 2) return [j, i + dims[0] * k];
	return [k, i + dims[0] * j];
}

/** Linear indices of the entries of X_(k), row by row. */
export function unfoldingEntries(dims: Dims, mode: Mode): number[][] {
	const [rows, cols] = unfoldingShape(dims, mode);
	const grid = Array.from({ length: rows }, () =>
		new Array<number>(cols).fill(-1),
	);
	for (let lin = 0; lin < dims[0] * dims[1] * dims[2]; lin++) {
		const [i, j, k] = subscripts(dims, lin);
		const [r, c] = unfoldingPosition(dims, mode, i, j, k);
		grid[r][c] = lin;
	}
	return grid;
}

/**
 * Group of an entry for colouring. A mode-k fiber is one column of X_(k), so fibers are
 * numbered by that column. A mode-k slice fixes the mode-k index, which is one row of X_(k).
 */
export function group(dims: Dims, view: View, mode: Mode, lin: number): number {
	const [i, j, k] = subscripts(dims, lin);
	const [r, c] = unfoldingPosition(dims, mode, i, j, k);
	return view === "fibers" ? c : r;
}

export function groupCount(dims: Dims, view: View, mode: Mode): number {
	const [rows, cols] = unfoldingShape(dims, mode);
	return view === "fibers" ? cols : rows;
}

export interface Cube {
	lin: number;
	/** Top-left corner of the front face, in SVG units. */
	x: number;
	y: number;
}

/**
 * Isometric layout of an exploded cube of cells, in painter's order (back to front). Mode 1
 * runs down, mode 2 runs right and mode 3 runs into the page (drawn up and to the right).
 */
export function isometricCubes(
	dims: Dims,
	cell: number,
	gap: number,
	depth: [number, number],
): Cube[] {
	const step = cell + gap;
	const cubes: Cube[] = [];
	for (let k = dims[2] - 1; k >= 0; k--)
		for (let i = dims[0] - 1; i >= 0; i--)
			for (let j = 0; j < dims[1]; j++)
				cubes.push({
					lin: linear(dims, i, j, k),
					x: j * step + k * depth[0],
					y: i * step - k * depth[1],
				});
	return cubes;
}
