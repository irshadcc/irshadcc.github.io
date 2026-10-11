// Logic behind CuteLayoutGemm.astro, which follows one thread through the two stages of a CuTe
// GEMM mainloop:
//   copy, global -> shared (copyView):
//     Tensor gA   = local_tile(mA, cta_tiler, make_coord(blockIdx.x, blockIdx.y, _), Step<_1, X,_1>{});
//     Tensor tAgA = local_partition(gA, tA, threadIdx.x);   // (THR_M,THR_K,k)
//     Tensor tAsA = local_partition(sA, tA, threadIdx.x);   // (THR_M,THR_K)
//     copy(tAgA(_,_,k_tile), tAsA);
//   MMA, shared -> registers (mmaView):
//     ThrMMA thr_mma = mma.get_slice(threadIdx.x);
//     Tensor tCsA = thr_mma.partition_A(sA);        // (MMA,MMA_M,MMA_K)
//     Tensor tCsB = thr_mma.partition_B(sB);        // (MMA,MMA_N,MMA_K)
//     Tensor tCgC = thr_mma.partition_C(gC);        // (MMA,MMA_M,MMA_N)
//     Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // registers
//     gemm(mma, tCsA(_,_,k_block), tCsB(_,_,k_block), tCrC);
//
//   const view = partitionTile(parseLayout("(16,4):(1,64)"), parseLayout("(8,4)"));
//   view.cells[3][1];        // { thread: 11, index: 0, offset: 67 }
//   view.partition(11);      // { layout: (2,1):(8,0), offset: 67 }
//   mmaView({ atom: UniversalFMA, thr: parseLayout("(4,4)"), cta: [16, 16, 4], sA: parseLayout("(16,4)"),
//     sB: parseLayout("(16,4)"), gC: parseLayout("(16,16)"), thread: 5, kBlock: 0 }).tCsA;
//   // { layout: (1,4,4):(0,4,16), offset: 1 }
//   parseSmem("Sw<3,2,3> o (32,8)").swizzle?.apply(32);   // 36: sA(0,1) moves from bank 0 to bank 4
//   bankAccess([0, 32, 64], 4).wavefronts;                // 3: three words in bank 0, a 3-way conflict

import { CuteLayout, type IntTuple, makeLayout } from "./CuteLayout";
import {
	MMA,
	type MmaAtom,
	type Partition,
	SM80_16x8x8_F32F16F16F32_TN,
	UniversalFMA,
} from "./CuteMma";

/** Parse an integer tuple such as 16, (16,4) or ((2,2),4). */
export function parseIntTuple(text: string): IntTuple {
	const src = text.replace(/\s+/g, "");
	let pos = 0;
	const fail = (what: string): never => {
		throw new Error(`Expected ${what} at position ${pos} of "${text}"`);
	};
	const parse = (): IntTuple => {
		if (src[pos] === "(") {
			pos++;
			const items: IntTuple[] = [parse()];
			while (src[pos] === ",") {
				pos++;
				items.push(parse());
			}
			if (src[pos] !== ")") fail('"," or ")"');
			pos++;
			return items;
		}
		const digits = /^\d+/.exec(src.slice(pos))?.[0] ?? fail("a number");
		pos += digits.length;
		return Number(digits);
	};
	const result = parse();
	if (pos !== src.length) fail("end of input");
	return result;
}

/** Parse "shape:stride" (e.g. "(16,4):(1,64)"), or just "shape" for the compact column-major layout. */
export function parseLayout(text: string): CuteLayout {
	const parts = text.split(":");
	if (parts.length > 2) throw new Error(`Too many ":" in "${text}"`);
	const shape = parseIntTuple(parts[0]);
	return parts.length === 2
		? new CuteLayout(shape, parseIntTuple(parts[1]))
		: new CuteLayout(shape);
}

/**
 * cute::Swizzle<B,M,S>: XOR the B bits that start at bit M+S into the B bits that start at bit M
 * (include/cute/swizzle.hpp: offset ^ ((offset & yyy_msk) >> S)). Applied to the offset a layout
 * produces, so composition(Swizzle<3,2,5>{}, layout) maps a coordinate to swizzle(layout(coord)).
 */
export class Swizzle {
	readonly bits: number;
	readonly base: number;
	readonly shift: number;

	constructor(bits: number, base: number, shift: number) {
		if (bits < 0 || base < 0 || shift < bits) {
			throw new Error(
				`Swizzle<${bits},${base},${shift}> needs B, M >= 0 and S >= B`,
			);
		}
		this.bits = bits;
		this.base = base;
		this.shift = shift;
	}

	apply(offset: number): number {
		const yyy = ((1 << this.bits) - 1) << (this.base + this.shift);
		return offset ^ ((offset & yyy) >> this.shift);
	}

	toString(): string {
		return `Sw<${this.bits},${this.base},${this.shift}>`;
	}
}

/** Parse "Sw<3,2,5>", "Swizzle<3,2,5>" or "3,2,5"; empty text means no swizzle. */
export function parseSwizzle(text: string): Swizzle | undefined {
	const src = text.replace(/\s+/g, "");
	if (!src) return undefined;
	const m = /^(?:Sw(?:izzle)?<)?(\d+),(\d+),(\d+)>?$/.exec(src);
	if (!m)
		throw new Error(`Expected a swizzle such as Sw<3,2,5>, got "${text}"`);
	return new Swizzle(Number(m[1]), Number(m[2]), Number(m[3]));
}

/** A shared-memory layout, optionally swizzled: "Sw<3,2,3> o (32,8)" or just "(32,8):(1,32)". */
export interface SmemLayout {
	layout: CuteLayout;
	swizzle?: Swizzle;
}

export function parseSmem(text: string): SmemLayout {
	const parts = text.split(/\s+o\s+/);
	if (parts.length > 2) throw new Error(`Too many "o" in "${text}"`);
	return parts.length === 2
		? { swizzle: parseSwizzle(parts[0]), layout: parseLayout(parts[1]) }
		: { layout: parseLayout(text) };
}

/** Where a layout offset really lands in shared memory, after the swizzle. */
export function physical(swizzle: Swizzle | undefined, offset: number): number {
	return swizzle ? swizzle.apply(offset) : offset;
}

/** Shared memory has 32 banks, each 4 bytes wide: word w lives in bank w mod 32. */
export const BANKS = 32;
/** Global memory moves data in 32-byte sectors. */
export const SECTOR_BYTES = 32;

/** One lane's access: the element offset it reads or writes. */
export interface LaneAccess {
	lane: number;
	offset: number;
}

export interface BankReport {
	/** Distinct 4-byte words each bank must serve, by bank. */
	words: Map<number, Set<number>>;
	/** Passes the hardware needs: the most distinct words any one bank serves (1 = conflict-free). */
	wavefronts: number;
}

/**
 * Bank conflicts of one warp-wide shared-memory access. Lanes that touch the same word share one
 * read (a broadcast); different words in the same bank are served one after another.
 */
export function bankAccess(offsets: number[], bytes: number): BankReport {
	const words = new Map<number, Set<number>>();
	for (const off of offsets) {
		const word = Math.floor((off * bytes) / 4);
		const bank = word % BANKS;
		if (!words.has(bank)) words.set(bank, new Set());
		words.get(bank)?.add(word);
	}
	const wavefronts = Math.max(0, ...[...words.values()].map((w) => w.size));
	return { words, wavefronts };
}

export interface SectorReport {
	/** Touched sectors, ascending, with the lanes that use each. */
	sectors: { sector: number; lanes: LaneAccess[] }[];
	/** Bytes the lanes asked for. */
	used: number;
	/** Bytes the memory system moves: 32 per touched sector. */
	moved: number;
}

/** The 32-byte sectors one warp-wide global access touches; offsets are elements from a 32-byte aligned base. */
export function sectorAccess(
	access: LaneAccess[],
	bytes: number,
): SectorReport {
	const bySector = new Map<number, LaneAccess[]>();
	for (const a of access) {
		const sector = Math.floor((a.offset * bytes) / SECTOR_BYTES);
		if (!bySector.has(sector)) bySector.set(sector, []);
		bySector.get(sector)?.push(a);
	}
	const sectors = [...bySector.entries()]
		.sort((x, y) => x[0] - y[0])
		.map(([sector, lanes]) => ({ sector, lanes }));
	const distinct = new Set(access.map((a) => a.offset)).size;
	return {
		sectors,
		used: distinct * bytes,
		moved: sectors.length * SECTOR_BYTES,
	};
}

/** The 32 lanes of the warp that `thread` belongs to, capped at the block size. */
export function warpLanes(thread: number, threads: number): number[] {
	const first = Math.floor(thread / 32) * 32;
	return Array.from(
		{ length: Math.min(32, threads - first) },
		(_, i) => first + i,
	);
}

/** Who owns one element of the tile. */
export interface Cell {
	/** Thread index t passed to local_partition. */
	thread: number;
	/** Position of the element inside that thread's partition (its 1-D coordinate). */
	index: number;
	/** Offset of the element in the tile's layout. */
	offset: number;
}

export interface TileView {
	rows: number;
	cols: number;
	/** cells[row][col], rows from mode 0 and columns from mode 1 of the tile. */
	cells: Cell[][];
	/** local_partition(tile, thrLayout, t): the thread's layout and the offset of its first element. */
	partition(thread: number): { layout: CuteLayout; offset: number };
}

/**
 * Thread ownership of every element of a rank-2 tile. Ownership is found on the compact layout
 * of the same shape, whose offsets are the (row, col) coordinates, because local_partition divides
 * coordinates and only then maps them through the strides; each cell's offset is then the tile's
 * own. Throws if a thread layout mode does not divide the tile, or if two threads claim one element.
 */
export function partitionTile(
	tile: CuteLayout,
	thrLayout: CuteLayout,
): TileView {
	if (tile.rank !== 2)
		throw new Error(`The tile must have rank 2, got ${tile}`);
	if (thrLayout.rank > 2)
		throw new Error(
			`The thread layout must have rank 1 or 2, got ${thrLayout}`,
		);
	for (let i = 0; i < thrLayout.rank; i++) {
		const n = thrLayout.mode(i).size;
		if (tile.mode(i).size % n !== 0) {
			throw new Error(
				`Thread mode ${i} has ${n} threads, which does not divide the tile's ${tile.mode(i).size}`,
			);
		}
	}
	const rows = tile.mode(0).size;
	const cols = tile.mode(1).size;
	const coords = new CuteLayout(tile.shape); // offset = row + rows * col
	const cells: (Cell | undefined)[][] = Array.from({ length: rows }, () =>
		Array(cols).fill(undefined),
	);
	for (let t = 0; t < thrLayout.size; t++) {
		const part = coords.localPartition(thrLayout, t);
		for (let i = 0; i < part.layout.size; i++) {
			const flat = part.offset + part.layout.offset(i);
			const row = flat % rows;
			const col = Math.floor(flat / rows);
			const prev = cells[row][col];
			if (prev) {
				throw new Error(
					`Threads ${prev.thread} and ${t} both own element (${row},${col}) of ${tile}`,
				);
			}
			cells[row][col] = { thread: t, index: i, offset: tile.offset(row, col) };
		}
	}
	return {
		rows,
		cols,
		cells: cells.map((r, row) =>
			r.map((c, col) => {
				if (!c)
					throw new Error(`No thread owns element (${row},${col}) of ${tile}`);
				return c;
			}),
		),
		partition: (thread) => tile.localPartition(thrLayout, thread),
	};
}

/** A layout with the offset of its first element, where CuTe would move the tensor's pointer. */
export interface Placed {
	layout: CuteLayout;
	offset: number;
}

/**
 * local_tile(m, (B0,B1), (blk, _)): the blk-th row of tiles of a rank-2 matrix, as (B0,B1,k),
 * the last mode stepping over the k tiles along mode 1.
 */
export function localTile(
	m: CuteLayout,
	tile: [number, number],
	blk: number,
): Placed {
	const z = m.zippedDivide(tile); // ((B0,B1),(tiles0,tiles1))
	const rest = z.mode(1);
	return {
		layout: makeLayout([z.mode(0).mode(0), z.mode(0).mode(1), rest.mode(1)]),
		offset: rest.mode(0).offset(blk),
	};
}

/** (row, col) of every offset a rank-2 layout reaches; throws if two coordinates share one. */
function inverse(layout: CuteLayout): Map<number, [number, number]> {
	const map = new Map<number, [number, number]>();
	for (let c = 0; c < layout.mode(1).size; c++) {
		for (let r = 0; r < layout.mode(0).size; r++) {
			const off = layout.offset(r, c);
			if (map.has(off)) {
				throw new Error(`${layout} maps two elements to offset ${off}`);
			}
			map.set(off, [r, c]);
		}
	}
	return map;
}

export type Operand = "A" | "B";

/** The kernel's variable names for one operand: A is (M,K) and tiled by blockIdx.x, B is (N,K) and blockIdx.y. */
export const NAMES = {
	A: {
		m: "mA",
		g: "gA",
		s: "sA",
		t: "tA",
		tg: "tAgA",
		ts: "tAsA",
		dim: "M",
		blk: "blockIdx.x",
		step: "Step<_1, X,_1>{}",
	},
	B: {
		m: "mB",
		g: "gB",
		s: "sB",
		t: "tB",
		tg: "tBgB",
		ts: "tBsB",
		dim: "N",
		blk: "blockIdx.y",
		step: "Step< X,_1,_1>{}",
	},
} as const;

export interface CopyInputs {
	operand: Operand;
	/** The whole operand in global memory: (M,K) for A, (N,K) for B. */
	matrix: CuteLayout;
	/** cta_tiler (BM,BN,BK). */
	cta: [number, number, number];
	/** Tile index along M (blockIdx.x) for A, along N (blockIdx.y) for B. */
	block: number;
	/** Shared-memory layout of one (BM,BK) or (BN,BK) tile. */
	smem: CuteLayout;
	/** Swizzle applied to smem's offsets, if any. */
	swizzle?: Swizzle;
	/** Thread layout used by local_partition. */
	thr: CuteLayout;
	thread: number;
	kTile: number;
}

/** One element the thread copies: the i-th of its partition. */
export interface CopyElement {
	index: number;
	global: [number, number];
	globalOffset: number;
	smem: [number, number];
	/** Offset in shared memory, after the swizzle. */
	smemOffset: number;
}

export interface CopyView {
	tile: [number, number];
	/** Number of tiles along each mode of the matrix. */
	tiles: [number, number];
	/** gA = local_tile(mA, cta_tiler, cta_coord, step): (BM,BK,k). */
	g: Placed;
	/** tAgA = local_partition(gA, tA, t): (THR_M,THR_K,k). */
	tg: Placed;
	/** tAgA(_,_,k_tile). */
	tgk: Placed;
	/** tAsA = local_partition(sA, tA, t). */
	ts: Placed;
	/** Owner of every element of the shared tile (and so of the current global tile); offsets swizzled. */
	owners: TileView;
	elements: CopyElement[];
	/** The elements of every lane in the thread's warp, as [lane, elements]. */
	warp: { lane: number; elements: CopyElement[] }[];
}

/** Follow thread `thread` through copy(tAgA(_,_,k_tile), tAsA) of one operand. */
export function copyView(inp: CopyInputs): CopyView {
	const { matrix, smem, thr } = inp;
	if (matrix.rank !== 2)
		throw new Error(`The matrix must have rank 2, got ${matrix}`);
	const tile: [number, number] = [
		inp.operand === "A" ? inp.cta[0] : inp.cta[1],
		inp.cta[2],
	];
	const tiles: [number, number] = [0, 1].map((i) => {
		const n = matrix.mode(i).size;
		if (n % tile[i] !== 0) {
			throw new Error(
				`The CTA tile's ${tile[i]} does not divide the matrix's ${n} in mode ${i}`,
			);
		}
		return n / tile[i];
	}) as [number, number];
	if (
		smem.rank !== 2 ||
		smem.mode(0).size !== tile[0] ||
		smem.mode(1).size !== tile[1]
	) {
		throw new Error(`The shared tile ${smem} must be ${tile[0]}×${tile[1]}`);
	}
	if (inp.block < 0 || inp.block >= tiles[0]) {
		throw new Error(`The block index must be in [0, ${tiles[0]})`);
	}
	if (inp.kTile < 0 || inp.kTile >= tiles[1]) {
		throw new Error(`k_tile must be in [0, ${tiles[1]})`);
	}
	if (inp.thread < 0 || inp.thread >= thr.size) {
		throw new Error(`threadIdx.x must be in [0, ${thr.size})`);
	}
	const plain = partitionTile(smem, thr);
	const owners: TileView = {
		...plain,
		cells: plain.cells.map((r) =>
			r.map((c) => ({ ...c, offset: physical(inp.swizzle, c.offset) })),
		),
	};
	const g = localTile(matrix, tile, inp.block);
	const gInv = inverse(matrix);
	const sInv = inverse(smem);
	/** Thread t's elements at k_tile: where each comes from in global memory and goes in shared. */
	const laneElements = (t: number) => {
		const tgRel = g.layout.localPartition(thr, t);
		const l = tgRel.layout;
		const base = g.offset + tgRel.offset + l.mode(2).offset(inp.kTile);
		const tgkLayout = makeLayout([l.mode(0), l.mode(1)]);
		const ts = smem.localPartition(thr, t);
		return Array.from({ length: ts.layout.size }, (_, i) => {
			const globalOffset = base + tgkLayout.offset(i);
			const logical = ts.offset + ts.layout.offset(i);
			return {
				index: i,
				global: gInv.get(globalOffset) as [number, number],
				globalOffset,
				smem: sInv.get(logical) as [number, number],
				smemOffset: physical(inp.swizzle, logical),
			};
		});
	};
	const tgRel = g.layout.localPartition(thr, inp.thread);
	const tg = { layout: tgRel.layout, offset: g.offset + tgRel.offset };
	const l = tg.layout;
	const tgk = {
		layout: makeLayout([l.mode(0), l.mode(1)]),
		offset: tg.offset + l.mode(2).offset(inp.kTile),
	};
	const ts = smem.localPartition(thr, inp.thread);
	const warp = warpLanes(inp.thread, thr.size).map((lane) => ({
		lane,
		elements: laneElements(lane),
	}));
	const elements = laneElements(inp.thread);
	return { tile, tiles, g, tg, tgk, ts, owners, elements, warp };
}

// ---------- MMA: shared -> registers ----------

/** The atoms the figure offers, by name. */
export const ATOMS: Record<string, MmaAtom> = {
	UniversalFMA,
	SM80_16x8x8_F32F16F16F32_TN,
};

export interface MmaInputs {
	atom: MmaAtom;
	/** AtomLayoutMNK passed to make_tiled_mma, e.g. (16,16) for 256 threads over C. */
	thr: CuteLayout;
	/** cta_tiler (BM,BN,BK). */
	cta: [number, number, number];
	/** (BM,BK) tile of A in shared memory. */
	sA: CuteLayout;
	/** (BN,BK) tile of B in shared memory. */
	sB: CuteLayout;
	/** Swizzles applied to sA's and sB's offsets, if any. */
	swizzleA?: Swizzle;
	swizzleB?: Swizzle;
	/** (BM,BN) tile of C in global memory, offsets relative to its first element. */
	gC: CuteLayout;
	thread: number;
	kBlock: number;
}

/** One element of a thread's partition. */
export interface MmaElement {
	/** 1-D index in the partition. */
	index: number;
	/** Coordinate along the partition's three top-level modes, e.g. (v, m, k) for A. */
	mode: [number, number, number];
	/** (row, col) in the tile. */
	coord: [number, number];
	/** Offset in the tile's layout. */
	offset: number;
	/** Offset in memory: the same, after the shared-memory swizzle if there is one. */
	phys: number;
}

export interface MmaView {
	mma: MMA;
	/** (ThrV, ThrM, ThrN, ThrK) coordinate of the thread. */
	vmnk: number[];
	tCsA: Partition;
	tCsB: Partition;
	tCgC: Partition;
	tCrC: CuteLayout;
	/** MMA_K: the number of k blocks gemm() walks through. */
	kBlocks: number;
	a: MmaElement[];
	b: MmaElement[];
	c: MmaElement[];
	/** Thread that owns each C element, [row][col]. */
	cOwner: number[][];
	/** How gemm() updates each of the thread's accumulators in the current k_block. */
	equations: MmaEquation[];
	/** Every lane of the thread's warp: its A and B elements in the current k_block, and its C elements. */
	warp: { lane: number; a: MmaElement[]; b: MmaElement[]; c: MmaElement[] }[];
}

/** A register: element `index` of thread `thread`'s partition (tCsA or tCsB). */
export interface RegRef {
	thread: number;
	index: number;
}

/** tCrC(index) = C(m,n) += sum over k of A(m,k) * B(n,k), with the register each factor comes from. */
export interface MmaEquation {
	index: number;
	coord: [number, number];
	terms: { k: number; a: RegRef; b: RegRef }[];
}

function elements(
	p: Partition,
	tile: CuteLayout,
	swizzle?: Swizzle,
): MmaElement[] {
	const inv = inverse(tile);
	const s0 = p.layout.mode(0).size;
	const s1 = p.layout.mode(1).size;
	return Array.from({ length: p.layout.size }, (_, i) => {
		const offset = p.offset + p.layout.offset(i);
		const coord = inv.get(offset);
		if (!coord) throw new Error(`Offset ${offset} is outside ${tile}`);
		return {
			index: i,
			mode: [i % s0, Math.floor(i / s0) % s1, Math.floor(i / (s0 * s1))],
			coord,
			offset,
			phys: physical(swizzle, offset),
		};
	});
}

function checkTile(name: string, tile: CuteLayout, shape: [number, number]) {
	if (
		tile.rank !== 2 ||
		tile.mode(0).size !== shape[0] ||
		tile.mode(1).size !== shape[1]
	) {
		throw new Error(`${name} = ${tile} must be ${shape[0]}×${shape[1]}`);
	}
}

/** Thread `thread`'s view of one CTA tile under the tiled MMA. */
export function mmaView(inp: MmaInputs): MmaView {
	const [bm, bn, bk] = inp.cta;
	checkTile("sA", inp.sA, [bm, bk]);
	checkTile("sB", inp.sB, [bn, bk]);
	checkTile("gC", inp.gC, [bm, bn]);
	const mma = new MMA(inp.atom, inp.thr);
	inp.cta.forEach((n, i) => {
		const perm = mma.permutationMNK[i] as number;
		if (n % perm !== 0) {
			throw new Error(
				`The tiled MMA covers ${perm} in ${"MNK"[i]}, which does not divide the CTA's ${n}`,
			);
		}
	});
	if (inp.thread < 0 || inp.thread >= mma.size) {
		throw new Error(`threadIdx.x must be in [0, ${mma.size})`);
	}
	const cOwner = Array.from({ length: bm }, () => Array<number>(bn).fill(-1));
	const cInv = inverse(inp.gC);
	for (let t = 0; t < mma.size; t++) {
		const p = mma.partitionC(inp.gC, t);
		for (let i = 0; i < p.layout.size; i++) {
			const [r, c] = cInv.get(p.offset + p.layout.offset(i)) ?? [-1, -1];
			if (r < 0 || cOwner[r][c] !== -1) {
				throw new Error(
					`C element at offset ${p.offset + p.layout.offset(i)} is claimed twice`,
				);
			}
			cOwner[r][c] = t;
		}
	}
	const tCsA = mma.partitionA(inp.sA, inp.thread);
	const tCsB = mma.partitionB(inp.sB, inp.thread);
	const tCgC = mma.partitionC(inp.gC, inp.thread);
	const kBlocks = tCsA.layout.mode(2).size;
	if (inp.kBlock < 0 || inp.kBlock >= kBlocks) {
		throw new Error(`k_block must be in [0, ${kBlocks})`);
	}
	const vmnk = mma.threadCoord(inp.thread);
	const c = elements(tCgC, inp.gC);
	const now = (els: MmaElement[]) =>
		els.filter((e) => e.mode[2] === inp.kBlock);
	const warp = warpLanes(inp.thread, mma.size).map((lane) => ({
		lane,
		a: now(elements(mma.partitionA(inp.sA, lane), inp.sA, inp.swizzleA)),
		b: now(elements(mma.partitionB(inp.sB, lane), inp.sB, inp.swizzleB)),
		c: elements(mma.partitionC(inp.gC, lane), inp.gC),
	}));
	return {
		mma,
		vmnk,
		tCsA,
		tCsB,
		tCgC,
		tCrC: mma.makeFragmentC(tCgC.layout),
		kBlocks,
		a: elements(tCsA, inp.sA, inp.swizzleA),
		b: elements(tCsB, inp.sB, inp.swizzleB),
		c,
		cOwner,
		equations: equations(mma, inp, vmnk, c),
		warp,
	};
}

/**
 * The accumulator updates of one gemm() step. The A and B values come from the lanes of the
 * thread's own atom (threads with the same ThrM, ThrN, ThrK), the only ones an instruction can
 * read: for UniversalFMA that is the thread itself, for an SM80 atom its warp.
 */
function equations(
	mma: MMA,
	inp: MmaInputs,
	vmnk: number[],
	c: MmaElement[],
): MmaEquation[] {
	const holders = (operand: "A" | "B") => {
		const map = new Map<string, RegRef>();
		for (let t = 0; t < mma.size; t++) {
			const v = mma.threadCoord(t);
			if (v[1] !== vmnk[1] || v[2] !== vmnk[2] || v[3] !== vmnk[3]) continue;
			const p =
				operand === "A"
					? elements(mma.partitionA(inp.sA, t), inp.sA)
					: elements(mma.partitionB(inp.sB, t), inp.sB);
			for (const e of p) {
				const key = `${e.coord[0]},${e.coord[1]}`;
				if (e.mode[2] === inp.kBlock && !map.has(key)) {
					map.set(key, { thread: t, index: e.index });
				}
			}
		}
		return map;
	};
	const aHeld = holders("A");
	const bHeld = holders("B");
	const ks = [
		...new Set([...aHeld.keys()].map((key) => Number(key.split(",")[1]))),
	].sort((x, y) => x - y);
	return c.map((e) => {
		const [m, n] = e.coord;
		const terms = ks.flatMap((k) => {
			const a = aHeld.get(`${m},${k}`);
			const b = bHeld.get(`${n},${k}`);
			return a && b ? [{ k, a, b }] : [];
		});
		if (terms.length !== ks.length) {
			throw new Error(
				`No lane of thread ${inp.thread}'s atom holds A(${m},·) or B(${n},·)`,
			);
		}
		return { index: e.index, coord: e.coord, terms };
	});
}
