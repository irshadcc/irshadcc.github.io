// Where a tcgen05 accumulator (and optionally operand A) sits in Blackwell's Tensor Memory, for
// TmemMap.astro. TMEM is 128 lanes × 512 columns of 32-bit cells per CTA (PTX ISA §9.7.18.1); a
// TMEM address is lane << 16 | column. Warp w of a warpgroup can only tcgen05.ld / st lanes
// 32·(w % 4) … 32·(w % 4) + 31 (§9.7.18.8.1).
//
// The (m, n) -> (lane, column) maps follow UMMA::tmem_frg::make in CUTLASS
// include/cute/atom/mma_traits_sm100.hpp (commit b984769), which builds them as a layout in
// "virtual" addressing (lane stride 1, column stride 128) and tiles extra modes (MMA_N, stages)
// with tiled_product. The closed forms below were checked against that composition with a script.
//
//   const plan = tmemPlan("acc-2stage");
//   plan.cell(5, 300)   // { region: 1, m: 5, n: 44, ... }  stage 1 of the 128 × 256 accumulator

export const TMEM_LANES = 128;
export const TMEM_COLS = 512;

export type TmemPreset =
	| "acc-2stage"
	| "m64"
	| "pair-m128"
	| "pair-m256"
	| "a-tmem"
	| "epilogue";
export const TMEM_PRESETS: TmemPreset[] = [
	"acc-2stage",
	"m64",
	"pair-m128",
	"pair-m256",
	"a-tmem",
	"epilogue",
];

export interface TmemRegion {
	name: string;
	/** Colour group, 0..5. */
	group: number;
}

export interface TmemCell {
	region: number;
	/** Row and column of the logical tile stored here. */
	m: number;
	n: number;
	/** Extra text, e.g. "2 fp16 values (k = 6, 7)". */
	note?: string;
}

export interface TmemPlan {
	title: string;
	/** One sentence under the figure. */
	body: string;
	regions: TmemRegion[];
	/** Columns the kernel allocates (a power of two ≥ 32). */
	allocCols: number;
	/** The logical tile at (lane, col), or undefined for an unused cell. */
	cell: (lane: number, col: number) => TmemCell | undefined;
	/** Optional highlight: one thread's cells (epilogue preset). */
	mark?: (lane: number, col: number) => boolean;
}

/** Warp of a warpgroup that may tcgen05.ld / st this lane. */
export const laneWarp = (lane: number) => Math.floor(lane / 32);

/**
 * 1-SM accumulator, M = 128: all 128 lanes, one column per n. Stage s starts at column s·N
 * (tiled_product of the (128, N):(1, 128) atom with (1, 1, stages)).
 */
function acc128(N: number, stages: number) {
	return (lane: number, col: number): TmemCell | undefined => {
		const s = Math.floor(col / N);
		if (s >= stages) return undefined;
		return { region: s, m: lane, n: col % N };
	};
}

/**
 * 1-SM accumulator, M = 64: a "half sub-partition" atom ((16, 4), N):((1, 32), 128). Row m goes to
 * lane (m % 16) + 32·⌊m / 16⌋, so a tile uses lanes 0–15, 32–47, 64–79, 96–111. In the default
 * Interleaved mode the next tile fills the other half of each sub-partition (lanes + 16) in the
 * same columns.
 */
function acc64(N: number, tiles: number) {
	return (lane: number, col: number): TmemCell | undefined => {
		if (col >= N) return undefined;
		const half = Math.floor(lane / 16) % 2; // 0: lanes 0–15 of a sub-partition, 1: lanes 16–31
		if (half >= tiles) return undefined;
		const m = (lane % 16) + 16 * laneWarp(lane);
		return { region: half, m, n: col };
	};
}

/**
 * 2-SM instruction with M = 128, so 64 rows per CTA: the "2x2" atom (64, (N/2, 2)):(1, (128, 64)).
 * Columns n < N/2 go to lanes 0–63, the rest to lanes 64–127, both in columns 0 … N/2 − 1.
 */
function pair64(N: number) {
	return (lane: number, col: number): TmemCell | undefined => {
		if (col >= N / 2) return undefined;
		const right = lane >= 64 ? 1 : 0;
		return { region: right, m: lane % 64, n: col + right * (N / 2) };
	};
}

export function tmemPlan(preset: TmemPreset): TmemPlan {
	switch (preset) {
		case "acc-2stage":
			return {
				title: "128 × 256 FP32 accumulator, two stages (cta_group::1, M = 128)",
				body: "Row m of the tile is lane m, column n is TMEM column n. One stage fills 256 of the 512 columns, so two stages (MMA on one while the epilogue drains the other) use all of TMEM.",
				regions: [
					{ name: "tCtAcc[…, 0]: stage 0", group: 0 },
					{ name: "tCtAcc[…, 1]: stage 1", group: 1 },
				],
				allocCols: 512,
				cell: acc128(256, 2),
			};
		case "m64":
			return {
				title: "Two 64 × 128 accumulators (cta_group::1, M = 64, Interleaved)",
				body: "An M = 64 instruction uses only half of each 32-lane sub-partition: rows 0–15 go to lanes 0–15, rows 16–31 to lanes 32–47, and so on. CuTe's default Interleaved mode puts the second tile in the other halves, in the same 128 columns.",
				regions: [
					{ name: "tile 0", group: 0 },
					{ name: "tile 1", group: 2 },
				],
				allocCols: 128,
				cell: acc64(128, 2),
			};
		case "pair-m128":
			return {
				title:
					"One CTA's half of a 128 × 256 pair accumulator (cta_group::2, M = 128)",
				body: "Each CTA of the pair holds 64 rows. Its 64 × 256 tile is folded in two: columns 0–127 of the tile in lanes 0–63, columns 128–255 in lanes 64–127, so it needs only 128 TMEM columns.",
				regions: [
					{ name: "n = 0–127", group: 0 },
					{ name: "n = 128–255", group: 3 },
				],
				allocCols: 128,
				cell: pair64(256),
			};
		case "pair-m256":
			return {
				title:
					"One CTA's half of a 256 × 256 pair accumulator (cta_group::2, M = 256)",
				body: "Each CTA holds 128 rows, one per lane, exactly like a 1-SM M = 128 accumulator. The peer CTA holds rows 128–255 of the tile in its own TMEM, at the same addresses.",
				regions: [{ name: "rows 0–127 (CTA v = 0)", group: 0 }],
				allocCols: 256,
				cell: acc128(256, 1),
			};
		case "a-tmem":
			return {
				title: "Accumulator plus operand A in TMEM (OperandSource.TMEM)",
				body: "With A read from TMEM, the A tile is placed after the accumulator's columns. A cell is 32 bits, so it holds two FP16 values: a 128 × 64 FP16 K-block of A needs 32 columns.",
				regions: [
					{ name: "accumulator 128 × 256 FP32", group: 0 },
					{ name: "A 128 × 64 FP16", group: 4 },
				],
				allocCols: 512,
				cell: (lane, col) => {
					if (col < 256) return { region: 0, m: lane, n: col };
					if (col < 256 + 32) {
						const c = col - 256;
						return {
							region: 1,
							m: lane,
							n: 2 * c,
							note: `2 FP16 values of A: k = ${2 * c}, ${2 * c + 1}`,
						};
					}
					return undefined;
				},
			};
		case "epilogue":
			return {
				title: "Epilogue: tcgen05.ld 32x32b.x64, sub-tile 1 of 4",
				body: "Warp w reads only lanes 32w–32w+31, one lane per thread, so thread t gets row t. Each 32x32b.x64 load gives a thread 64 consecutive columns of its row; the highlighted cells are thread 37's load of sub-tile 1 (columns 64–127).",
				regions: [
					{ name: "warp 0", group: 0 },
					{ name: "warp 1", group: 1 },
					{ name: "warp 2", group: 2 },
					{ name: "warp 3", group: 3 },
				],
				allocCols: 512,
				cell: (lane, col) =>
					col < 256 ? { region: laneWarp(lane), m: lane, n: col } : undefined,
				mark: (lane, col) => lane === 37 && col >= 64 && col < 128,
			};
	}
}

/** One line describing a TMEM cell, for the hover readout. */
export function describeCell(
	plan: TmemPlan,
	lane: number,
	col: number,
): string {
	const addr = `0x${((lane << 16) | col).toString(16).padStart(8, "0")}`;
	const where = `lane ${lane}, column ${col} (address ${addr}; readable by warp ${laneWarp(lane)})`;
	if (col >= plan.allocCols) return `${where}: not allocated`;
	const c = plan.cell(lane, col);
	if (!c) return `${where}: allocated, unused`;
	const r = plan.regions[c.region];
	return `${where}: ${r.name}, ${c.note ?? `(m, n) = (${c.m}, ${c.n})`}`;
}
