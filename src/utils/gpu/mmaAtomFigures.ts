// Data for the atom figures in the "Warp and Warpgroup MMA in the CuTe DSL" post
// (src/content/posts/cute-dsl-warp-and-warpgroup-mma.mdx). Each figure inverts one thread-value
// layout of the SM80 m16n8k16 atom (mmaPartition.ts, copied from CUTLASS mma_traits_sm80.hpp):
// for every (thread, value) it labels the element the layout maps to with T<thread> and V<value>.
// Each entry is spread into a <Matrix2D>; data and labels are keyed by the element's offset in the
// column-major tile, which is exactly what the TV layout returns.
import { CuteLayout } from "../../component/Cute/CuteLayout";
import { SM80_16x8x16 } from "./mmaPartition";

interface Figure {
	layout: CuteLayout;
	data: Record<number, string>;
	labels: Record<number, string>;
}

function invert(
	tv: CuteLayout,
	rows: number,
	cols: number,
	display: CuteLayout = new CuteLayout([rows, cols]),
): Figure {
	const data: Record<number, string> = {};
	const labels: Record<number, string> = {};
	const threads = tv.mode(0).size;
	const values = tv.mode(1).size;
	for (let t = 0; t < threads; t++)
		for (let v = 0; v < values; v++) {
			const off = tv.offset(t, v);
			if (data[off] !== undefined)
				throw new Error(
					`${tv} maps two (thread, value) pairs to offset ${off}`,
				);
			data[off] = `V${v}`;
			labels[off] = `T${t}`;
		}
	if (Object.keys(data).length !== rows * cols)
		throw new Error(`${tv} does not cover the ${rows}×${cols} tile`);
	return { layout: display, data, labels };
}

export const mmaAtomFigures = {
	/** C (and D): 16 × 8, (m, n) at offset m + 16·n. */
	c: invert(SM80_16x8x16.layoutC_TV, 16, 8),
	/**
	 * A: 16 × 16, (m, k) at offset m + 16·k. Drawn 8 columns wide: k = k0 + 8·k1 with k1 moved
	 * into the row mode, ((16,2),8):((1,128),16), so rows 16–31 are columns k0 + 8.
	 */
	a: invert(
		SM80_16x8x16.layoutA_TV,
		16,
		16,
		new CuteLayout([[16, 2], 8], [[1, 128], 16]),
	),
	/** B: 8 × 16 as (n, k), at offset n + 8·k. Drawn as ((8,2),8):((1,64),8), like A. */
	b: invert(
		SM80_16x8x16.layoutB_TV,
		8,
		16,
		new CuteLayout([[8, 2], 8], [[1, 64], 8]),
	),
};
