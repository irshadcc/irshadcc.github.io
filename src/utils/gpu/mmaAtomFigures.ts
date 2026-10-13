// Data for the atom figures in the "Warp and Warpgroup MMA in the CuTe DSL" post
// (src/content/posts/cute-dsl-warp-and-warpgroup-mma.mdx). Each figure inverts one thread-value
// layout of the SM80 m16n8k16 atom (mmaPartition.ts, copied from CUTLASS mma_traits_sm80.hpp):
// for every (thread, value) it labels the element the layout maps to with T<thread> and V<value>.
// Each entry is spread into a <Matrix2D>; data and labels are keyed by the element's offset in the
// column-major tile, which is exactly what the TV layout returns.
import { CuteLayout } from "../../component/Cute/CuteLayout";
import { SM80_16x8x16, gmmaCLayout } from "./mmaPartition";

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
	thrID: CuteLayout = new CuteLayout(tv.mode(0).size),
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
			labels[off] = `T${thrID.offset(t)}`;
		}
	if (Object.keys(data).length !== rows * cols)
		throw new Error(`${tv} does not cover the ${rows}×${cols} tile`);
	return { layout: display, data, labels };
}

/**
 * The 64 × N wgmma accumulator drawn one cell per 16 × 8 block: block (w, j) holds rows 16w–16w+15
 * and columns 8j–8j+7. Checks that every element of a block belongs to warp w (threads 32w–32w+31)
 * and to values 4j–4j+3, then labels the block with that warp and those values.
 */
function wgmmaBlocks(n: number): Figure {
	const tv = gmmaCLayout(n);
	const blocks = n / 8;
	for (let t = 0; t < 128; t++)
		for (let v = 0; v < n / 2; v++) {
			const off = tv.offset(t, v);
			const w = Math.floor((off % 64) / 16);
			const j = Math.floor(Math.floor(off / 64) / 8);
			if (w !== Math.floor(t / 32) || j !== Math.floor(v / 4))
				throw new Error(`${tv} puts (T${t}, V${v}) in block (${w}, ${j})`);
		}
	const data: Record<number, string> = {};
	const labels: Record<number, string> = {};
	for (let j = 0; j < blocks; j++)
		for (let w = 0; w < 4; w++) {
			data[w + 4 * j] = `d${4 * j}–d${4 * j + 3}`;
			labels[w + 4 * j] = `T${32 * w}–T${32 * w + 31}`;
		}
	return { layout: new CuteLayout([4, blocks]), data, labels };
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
	/**
	 * Volta SM70_8x8x4_F32F16F16F32_NT accumulator, 8 × 8, CLayout SM70_8x8_32b from
	 * mma_traits_sm70.hpp. Labels are warp lanes: logical thread t is lane ThrID(t) of the
	 * quadpair layout (4,2):(1,16).
	 */
	volta: invert(
		new CuteLayout(
			[
				[2, 2, 2],
				[2, 2, 2],
			],
			[
				[1, 16, 4],
				[8, 2, 32],
			],
		),
		8,
		8,
		undefined,
		new CuteLayout([4, 2], [1, 16]),
	),
	/** wgmma m64n32k16 accumulator, one cell per 16 × 8 block (rows: warp, columns: 8-column block). */
	wgmmaC: wgmmaBlocks(32),
};
