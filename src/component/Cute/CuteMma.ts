// A CuTe TiledMMA: one MMA atom (a single multiply instruction) repeated over a grid of threads,
// and the partition_A / partition_B / partition_C / make_fragment_C that hand each thread its
// share of the A, B and C tiles. Follows TiledMMA and ThrMMA in CUTLASS
// include/cute/atom/mma_atom.hpp (commit 0b55a2f), on layouts only: a partition is the thread's
// layout plus the offset of its first element, where CuTe would move the tensor's pointer.
//
//   const mma = new MMA(UniversalFMA, new CuteLayout([16, 16]));  // 256 threads over C
//   const sA = new CuteLayout([128, 8]);                           // (bM,bK)
//   mma.partitionA(sA, 17);   // { layout: (1,8,8):(0,16,128), offset: 1 }  rows 1, 17, ..., 113
//   mma.partitionC(new CuteLayout([128, 128]), 17).layout;        // (1,8,8):(0,16,2048)
//   mma.makeFragmentC(mma.partitionC(gC, 17).layout);             // (1,8,8):(1,1,8) in registers

import {
	CuteLayout,
	type TileMode,
	complement,
	composition,
	logicalDivide,
	makeLayout,
} from "./CuteLayout";

/** One MMA instruction: its M×N×K shape and which (thread, value) holds which element. */
export interface MmaAtom {
	name: string;
	/** Logical M, N, K of one instruction. */
	shapeMNK: [number, number, number];
	/** Threads that issue one instruction together (ThrID, assumed compact n:1). */
	threads: number;
	/** (thread, value) -> (m, k) inside the atom's M×K tile of A. */
	layoutA_TV: CuteLayout;
	/** (thread, value) -> (n, k) inside the atom's N×K tile of B. */
	layoutB_TV: CuteLayout;
	/** (thread, value) -> (m, n) inside the atom's M×N tile of C. */
	layoutC_TV: CuteLayout;
}

/** cute::UniversalFMA: one thread, one scalar fused multiply-add, d = a * b + c. */
export const UniversalFMA: MmaAtom = {
	name: "UniversalFMA",
	shapeMNK: [1, 1, 1],
	threads: 1,
	layoutA_TV: new CuteLayout([1, 1]),
	layoutB_TV: new CuteLayout([1, 1]),
	layoutC_TV: new CuteLayout([1, 1]),
};

// (T32,V4) -> (M16,N8), SM80_16x8_Row in mma_traits_sm80.hpp
const SM80_16x8_Row = new CuteLayout(
	[
		[4, 8],
		[2, 2],
	],
	[
		[32, 1],
		[16, 8],
	],
);

/** cute::SM80_16x8x8_F32F16F16F32_TN: Ampere tensor-core mma.sync.m16n8k8, one warp. */
export const SM80_16x8x8_F32F16F16F32_TN: MmaAtom = {
	name: "SM80_16x8x8_F32F16F16F32_TN",
	shapeMNK: [16, 8, 8],
	threads: 32,
	layoutA_TV: SM80_16x8_Row,
	layoutB_TV: new CuteLayout([[4, 8], 2], [[16, 1], 8]), // SM80_8x8_Row
	layoutC_TV: SM80_16x8_Row,
};

/** A thread's share of a tile: its layout, and the offset of its first element. */
export interface Partition {
	layout: CuteLayout;
	offset: number;
}

/** One past the largest offset a layout reaches. */
function cosize(layout: CuteLayout): number {
	let last = 0;
	for (let i = 0; i < layout.size; i++) last = Math.max(last, layout.offset(i));
	return last + 1;
}

// Which two of the M, N, K modes each operand has.
const OPERAND_MODES = { A: [0, 2], B: [1, 2], C: [0, 1] } as const;
type Operand = keyof typeof OPERAND_MODES;

export class MMA {
	readonly atom: MmaAtom;
	/** (ThrV, ThrM, ThrN, ThrK) -> thread index. */
	readonly thrLayoutVMNK: CuteLayout;
	/** Tile applied to each of M, N, K before the atom (CuTe's PermutationMNK). */
	readonly permutationMNK: [TileMode, TileMode, TileMode];

	/**
	 * make_tiled_mma(atom, thrLayout, permutations). `thrLayout` tiles the atom over M, N (and
	 * optionally K); a rank-2 layout gets the K mode (1):(0), as CuTe appends. A permutation left
	 * undefined is the plain tile AtomShape × ThrCount in that mode.
	 */
	constructor(
		atom: MmaAtom,
		thrLayout: CuteLayout = new CuteLayout([1, 1, 1]),
		permutations: (TileMode | undefined)[] = [],
	) {
		const mnk =
			thrLayout.rank === 2
				? makeLayout([
						thrLayout.mode(0),
						thrLayout.mode(1),
						new CuteLayout(1, 0),
					])
				: thrLayout;
		if (mnk.rank !== 3)
			throw new Error(
				`MMA thread layout must have rank 2 or 3, got ${thrLayout}`,
			);
		this.atom = atom;
		// tiled_product(ThrID, AtomLayoutMNK): (ThrID, complement(ThrID, size·cosize) ∘ AtomLayoutMNK), unpacked.
		const thrID = new CuteLayout(atom.threads, 1);
		const tiled = composition(
			complement(thrID, atom.threads * cosize(mnk)),
			mnk,
		);
		this.thrLayoutVMNK = makeLayout([
			thrID,
			tiled.mode(0),
			tiled.mode(1),
			tiled.mode(2),
		]);
		this.permutationMNK = [0, 1, 2].map(
			(i) =>
				permutations[i] ??
				atom.shapeMNK[i] * this.thrLayoutVMNK.mode(i + 1).size,
		) as [TileMode, TileMode, TileMode];
	}

	/** Number of threads the tiled MMA uses. */
	get size(): number {
		return this.thrLayoutVMNK.size;
	}

	/** (v, m, n, k): this thread's position in the thread grid (TiledMMA::get_slice). */
	threadCoord(thrIdx: number): number[] {
		return this.thrLayoutVMNK.flatCoord(thrIdx);
	}

	/**
	 * thrfrg_A / thrfrg_B / thrfrg_C: reshape an (M,K) / (N,K) / (M,N) layout, plus any trailing
	 * modes, into ((ThrV,(ThrP,ThrQ)),(FrgV,(RestP,RestQ,...))).
	 */
	thrfrg(operand: Operand, layout: CuteLayout): CuteLayout {
		if (layout.rank < 2)
			throw new Error(
				`thrfrg_${operand} needs a rank >= 2 layout, got ${layout}`,
			);
		const [p, q] = OPERAND_MODES[operand];
		const atomTV = {
			A: this.atom.layoutA_TV,
			B: this.atom.layoutB_TV,
			C: this.atom.layoutC_TV,
		}[operand];

		// Reorder the tensor for the tiled atom: (PermP,PermQ,...)
		const permuted = makeLayout(
			Array.from({ length: layout.rank }, (_, i) =>
				i < 2
					? logicalDivide(layout.mode(i), this.permutationMNK[i === 0 ? p : q])
					: layout.mode(i),
			),
		);
		// Tile for the atom: ((AtomP,AtomQ),(RestP,RestQ,...))
		const atomTiled = permuted.zippedDivide([
			this.atom.shapeMNK[p],
			this.atom.shapeMNK[q],
		]);
		// Atom mode from (P,Q) to (ThrV,FrgV)
		const tv = composition(atomTiled.mode(0), atomTV);
		// Tile the rest for the threads: ((ThrP,ThrQ),(RestP',RestQ',...))
		const thrTiled = atomTiled
			.mode(1)
			.zippedDivide([
				this.thrLayoutVMNK.mode(p + 1).size,
				this.thrLayoutVMNK.mode(q + 1).size,
			]);
		return makeLayout([
			makeLayout([tv.mode(0), thrTiled.mode(0)]),
			makeLayout([tv.mode(1), thrTiled.mode(1)]),
		]);
	}

	/** Fix the thread mode of a thrfrg result and flatten the rest to (FrgV, RestP, RestQ, ...). */
	private partition(
		operand: Operand,
		layout: CuteLayout,
		thrIdx: number,
	): Partition {
		const [p, q] = OPERAND_MODES[operand];
		const vmnk = this.threadCoord(thrIdx);
		const frg = this.thrfrg(operand, layout);
		const thr = frg.mode(0);
		const offset =
			thr.mode(0).offset(vmnk[0]) +
			thr
				.mode(1)
				.mode(0)
				.offset(vmnk[p + 1]) +
			thr
				.mode(1)
				.mode(1)
				.offset(vmnk[q + 1]);
		const rest = frg.mode(1).mode(1);
		return {
			layout: makeLayout([
				frg.mode(1).mode(0),
				...Array.from({ length: rest.rank }, (_, i) => rest.mode(i)),
			]),
			offset,
		};
	}

	/** ThrMMA::partition_A: thread `thrIdx`'s (MMA, MMA_M, MMA_K, ...) view of an (M,K,...) A tile. */
	partitionA(a: CuteLayout, thrIdx: number): Partition {
		return this.partition("A", a, thrIdx);
	}

	/** ThrMMA::partition_B: thread `thrIdx`'s (MMA, MMA_N, MMA_K, ...) view of an (N,K,...) B tile. */
	partitionB(b: CuteLayout, thrIdx: number): Partition {
		return this.partition("B", b, thrIdx);
	}

	/** ThrMMA::partition_C: thread `thrIdx`'s (MMA, MMA_M, MMA_N, ...) view of an (M,N,...) C tile. */
	partitionC(c: CuteLayout, thrIdx: number): Partition {
		return this.partition("C", c, thrIdx);
	}

	/**
	 * TiledMMA::make_fragment_C: the register accumulator for a partition_C result. Only its shape
	 * is kept; the layout is compact column-major, whatever the strides of C in memory.
	 */
	makeFragmentC(partitionedC: CuteLayout): CuteLayout {
		if (partitionedC.rank < 3)
			throw new Error(
				`make_fragment_C expects a partition_C result (V,M,N), got ${partitionedC}`,
			);
		if (partitionedC.mode(0).size !== this.atom.layoutC_TV.mode(1).size) {
			throw new Error(
				`Mode 0 of ${partitionedC} is not the atom's ${this.atom.layoutC_TV.mode(1).size} C values`,
			);
		}
		return new CuteLayout(partitionedC.shape);
	}
}
