// Steps that derive every mode of a CuTe MMA partition, from the whole matrix down to one
// thread's fragment, for a warp MMA (Ampere mma.sync m16n8k16) or a warpgroup MMA (Hopper wgmma
// m64nNk16, A and B read from shared memory). MmaPartitionSteps.astro draws them.
//
// Atom thread-value layouts are copied from CUTLASS include/cute/atom/mma_traits_sm80.hpp and
// mma_traits_sm90_gmma.hpp (commit b984769); partitions come from the MMA class in
// component/Cute/CuteMma.ts, which follows thrfrg_A / thrfrg_C in mma_atom.hpp. The partition
// shapes were cross-checked against pycute.
//
//   const steps = buildSteps("warp-A", 0);   // 7 steps, mA (256,128) down to thread 0's tCsA
//   steps[6].shape                           // tCsA: (MMA, MMA_M, MMA_K, PIPE) = ((2,2,2),2,2,2)

import { CuteLayout, formatTuple } from "../../component/Cute/CuteLayout";
import { MMA, type MmaAtom } from "../../component/Cute/CuteMma";

/** cute::SM80_16x8x16_F32F16F16F32_TN: mma.sync.aligned.m16n8k16, one warp of 32 threads. */
export const SM80_16x8x16: MmaAtom = {
	name: "SM80_16x8x16_F32F16F16F32_TN",
	shapeMNK: [16, 8, 16],
	threads: 32,
	layoutA_TV: new CuteLayout(
		[
			[4, 8],
			[2, 2, 2],
		],
		[
			[32, 1],
			[16, 8, 128],
		],
	),
	layoutB_TV: new CuteLayout(
		[
			[4, 8],
			[2, 2],
		],
		[
			[16, 1],
			[8, 64],
		],
	),
	layoutC_TV: new CuteLayout(
		[
			[4, 8],
			[2, 2],
		],
		[
			[32, 1],
			[16, 8],
		],
	),
};

/** GMMA::CLayout_64xN: (T128, V) -> (M64, N), the accumulator of every 64×N wgmma. */
export function gmmaCLayout(n: number): CuteLayout {
	return new CuteLayout(
		[
			[4, 8, 4],
			[2, 2, n / 8],
		],
		[
			[128, 1, 16],
			[64, 8, 512],
		],
	);
}

/** SM90 wgmma.m64n{N}k16 with A and B from shared memory (the _SS atoms), one warpgroup. */
export function gmmaSS(n: number): MmaAtom {
	return {
		name: `SM90_64x${n}x16_F32F16F16_SS`,
		shapeMNK: [64, n, 16],
		threads: 128,
		// GMMA::ABLayout<M,K>: every thread maps to the whole tile (thread stride 0).
		layoutA_TV: new CuteLayout([128, [64, 16]], [0, [1, 64]]),
		layoutB_TV: new CuteLayout([128, [n, 16]], [0, [1, n]]),
		layoutC_TV: gmmaCLayout(n),
	};
}

export type Preset = "warp-A" | "warp-C" | "wgmma-A" | "wgmma-C";
export const PRESETS: Preset[] = ["warp-A", "warp-C", "wgmma-A", "wgmma-C"];

interface Config {
	kind: "warp" | "warpgroup";
	operand: "A" | "C";
	atom: MmaAtom;
	/** Atoms (warps or warpgroups) along M, N, K: atom_layout_mnk. */
	atomLayout: [number, number, number];
	gemm: [number, number, number];
	tile: [number, number, number];
	pipe: number;
	/** The CTA's (m, n) tile coordinate. */
	cta: [number, number];
}

const CONFIGS: Record<Preset, Config> = {
	"warp-A": {
		kind: "warp",
		operand: "A",
		atom: SM80_16x8x16,
		atomLayout: [2, 2, 1],
		gemm: [256, 256, 128],
		tile: [64, 64, 32],
		pipe: 2,
		cta: [1, 0],
	},
	"warp-C": {
		kind: "warp",
		operand: "C",
		atom: SM80_16x8x16,
		atomLayout: [2, 2, 1],
		gemm: [256, 256, 128],
		tile: [64, 64, 32],
		pipe: 2,
		cta: [1, 0],
	},
	"wgmma-A": {
		kind: "warpgroup",
		operand: "A",
		atom: gmmaSS(256),
		atomLayout: [2, 1, 1],
		gemm: [512, 768, 256],
		tile: [128, 256, 64],
		pipe: 4,
		cta: [1, 0],
	},
	"wgmma-C": {
		kind: "warpgroup",
		operand: "C",
		atom: gmmaSS(256),
		atomLayout: [2, 1, 1],
		gemm: [512, 768, 256],
		tile: [128, 256, 64],
		pipe: 4,
		cta: [1, 0],
	},
};

/** A piece of the layout line; `hl` marks the mode the step adds. */
export interface Seg {
	t: string;
	hl?: boolean;
}

export interface Box {
	r0: number;
	c0: number;
	r1: number;
	c1: number;
	label?: string;
}

/** A grid of cells, each standing for unitH × unitW matrix elements. */
export interface Grid {
	rows: number;
	cols: number;
	unitH: number;
	unitW: number;
	rowName: string;
	colName: string;
	/** Colour group per cell (row-major, r * cols + c); -1 leaves the cell blank. */
	group: Int32Array;
	/** Cells to emphasise (1) when the step follows one thread. */
	mark?: Uint8Array;
	/** Short text per cell, drawn when the cell is large enough. */
	text?: (string | undefined)[];
	boxes: Box[];
	/** One line describing cell (r, c), shown on hover. */
	describe: (r: number, c: number) => string;
}

export interface Step {
	title: string;
	code: string;
	shape: Seg[];
	body: string;
	grid: Grid;
	/** The step depends on the chosen thread. */
	usesThread: boolean;
}

/** Number of threads in the tiled MMA, the range of the thread picker. */
export function threadCount(preset: Preset): number {
	const c = CONFIGS[preset];
	return c.atom.threads * c.atomLayout[0] * c.atomLayout[1] * c.atomLayout[2];
}

function tiledMma(c: Config): MMA {
	const [m, n, k] = c.atomLayout;
	return new MMA(c.atom, new CuteLayout([m, n, k]));
}

/** For every element of an (P,Q) tile: one thread that owns it and that thread's value index. */
interface Ownership {
	thread: Int32Array;
	value: Int32Array;
	/** All threads owning the element, as a count (A is shared by warps that differ in N). */
	owners: Int32Array;
}

function ownership(mma: MMA, op: "A" | "C", P: number, Q: number): Ownership {
	const tile = new CuteLayout([P, Q]);
	const thread = new Int32Array(P * Q).fill(-1);
	const value = new Int32Array(P * Q).fill(-1);
	const owners = new Int32Array(P * Q);
	// Threads with the same partition (layout and offset) hold the same elements; walk one of them.
	const sharers = new Map<string, number[]>();
	for (let t = 0; t < mma.size; t++) {
		const part = op === "A" ? mma.partitionA(tile, t) : mma.partitionC(tile, t);
		const key = `${part.layout}+${part.offset}`;
		const same = sharers.get(key);
		if (same) {
			same.push(t);
			continue;
		}
		const group = [t];
		sharers.set(key, group);
		for (let i = 0; i < part.layout.size; i++) {
			const off = part.offset + part.layout.offset(i);
			if (thread[off] === -1) {
				thread[off] = t;
				value[off] = i;
			}
		}
	}
	const count = new Map<number, number>();
	for (const ts of sharers.values()) count.set(ts[0], ts.length);
	for (let off = 0; off < P * Q; off++)
		owners[off] = count.get(thread[off]) ?? 0;
	return { thread, value, owners };
}

/** Offsets (column-major in the (P,Q) tile) of the values one thread holds, by value index. */
function threadValues(
	mma: MMA,
	op: "A" | "C",
	P: number,
	Q: number,
	t: number,
): Map<number, number> {
	const tile = new CuteLayout([P, Q]);
	const part = op === "A" ? mma.partitionA(tile, t) : mma.partitionC(tile, t);
	const map = new Map<number, number>();
	for (let i = 0; i < part.layout.size; i++)
		map.set(part.offset + part.layout.offset(i), i);
	return map;
}

/** Natural coordinate of value i in a partition's shape, e.g. ((1,0,1),0,1). */
function coordIn(layout: CuteLayout, i: number): string {
	return formatTuple(layout.coord(i));
}

function blockGrid(
	rows: number,
	cols: number,
	unitH: number,
	unitW: number,
	rowName: string,
	colName: string,
	group: (r: number, c: number) => number,
	text: (r: number, c: number) => string | undefined,
	describe: (r: number, c: number) => string,
): Grid {
	const g = new Int32Array(rows * cols);
	const tx: (string | undefined)[] = [];
	for (let r = 0; r < rows; r++)
		for (let c = 0; c < cols; c++) {
			g[r * cols + c] = group(r, c);
			tx[r * cols + c] = text(r, c);
		}
	const boxes: Box[] = [];
	for (let r = 0; r < rows; r++)
		for (let c = 0; c < cols; c++)
			boxes.push({ r0: r, c0: c, r1: r + 1, c1: c + 1 });
	return {
		rows,
		cols,
		unitH,
		unitW,
		rowName,
		colName,
		group: g,
		text: tx,
		boxes,
		describe,
	};
}

function seg(...parts: (string | [string])[]): Seg[] {
	return parts.map((p) =>
		typeof p === "string" ? { t: p } : { t: p[0], hl: true },
	);
}

/** The steps for one preset, following thread `thr` where a step singles one out. */
export function buildSteps(preset: Preset, thr = 0): Step[] {
	const c = CONFIGS[preset];
	const mma = tiledMma(c);
	const [M, N, K] = c.gemm;
	const [BM, BN, BK] = c.tile;
	const [aM, aN, aK] = c.atom.shapeMNK;
	const [tM, tN] = c.atomLayout;
	const isA = c.operand === "A";
	const warp = c.kind === "warp";
	const unit = warp ? "warp" : "warpgroup";
	const T = c.atom.threads;
	// Operand A is (M,K), C is (M,N).
	const Q = isA ? K : N;
	const BQ = isA ? BK : BN;
	const aQ = isA ? aK : aN;
	const qName = isA ? "K" : "N";
	const name = isA ? "A" : "C";
	const tQ = isA ? 1 : tN;
	const [ctaM, ctaN] = c.cta;
	const thrCoord = mma.threadCoord(thr); // (v, m, n, k)
	const steps: Step[] = [];

	// 1. The whole matrix, cut into CTA tiles.
	const gRows = M / BM;
	const gCols = Q / BQ;
	const ctaCol = isA ? -1 : ctaN;
	const isCta = (r: number, q: number) => r === ctaM && (isA || q === ctaCol);
	steps.push({
		title: `The matrix ${name}`,
		code: `# ${isA ? "an operand" : "the output"} in global memory`,
		shape: seg(`m${name}: (M, ${qName}) = (${M}, ${Q})`),
		body: isA
			? `A is M × K = ${M} × ${Q}, drawn here in blocks of BM × BK = ${BM} × ${BK}. The CTA at tile coordinate (${ctaM}, ${ctaN}) computes rows ${ctaM * BM}–${(ctaM + 1) * BM - 1} of C, so it reads the whole highlighted strip of A: ${BM} rows by all ${K} columns.`
			: `C is M × N = ${M} × ${N}, in blocks of BM × BN = ${BM} × ${BN}. Each CTA owns one block; the CTA at tile coordinate (${ctaM}, ${ctaN}) owns the highlighted one.`,
		usesThread: false,
		grid: blockGrid(
			gRows,
			gCols,
			BM,
			BQ,
			"M",
			qName,
			(r, q) => (isCta(r, q) ? 0 : -1),
			(r, q) => `(${r},${q})`,
			(r, q) =>
				`block (${r}, ${q}): rows ${r * BM}–${(r + 1) * BM - 1}, columns ${q * BQ}–${(q + 1) * BQ - 1}${isCta(r, q) ? ", read by this CTA" : ""}`,
		),
	});

	// 2. local_tile.
	const kTiles = K / BK;
	steps.push({
		title: "local_tile: the CTA's tile",
		code: isA
			? `g${name} = cute.local_tile(m${name}, cta_tiler, (${ctaM}, ${ctaN}, None), proj=(1, None, 1))`
			: `g${name} = cute.local_tile(m${name}, cta_tiler, (${ctaM}, ${ctaN}, None), proj=(1, 1, None))`,
		shape: isA
			? seg("gA: (BM, BK, ", ["k"], `) = (${BM}, ${BK}, `, [`${kTiles}`], ")")
			: seg(`gC: (BM, BN) = (${BM}, ${BN})`),
		body: isA
			? `local_tile keeps the CTA's strip and splits it into K / BK = ${K} / ${BK} = ${kTiles} tiles of ${BM} × ${BK}. The new last mode, k, numbers them: the main loop walks along it. proj=(1, None, 1) drops N from cta_tiler, because A has no N mode.`
			: `local_tile keeps the CTA's ${BM} × ${BN} block. proj=(1, 1, None) drops K from cta_tiler, because C has no K mode, so gC has no loop mode: the CTA accumulates into the same block for every k.`,
		usesThread: false,
		grid: blockGrid(
			gRows,
			gCols,
			BM,
			BQ,
			"M",
			qName,
			(r, q) => (isCta(r, q) ? (isA ? q : 0) : -1),
			(r, q) => (isCta(r, q) ? (isA ? `k=${q}` : "gC") : undefined),
			(r, q) =>
				isCta(r, q)
					? isA
						? `gA[:, :, ${q}]: columns ${q * BK}–${(q + 1) * BK - 1}`
						: "gC, this CTA's block"
					: "another CTA's block",
		),
	});

	// 3. Shared-memory stages (operands only).
	if (isA) {
		steps.push({
			title: "Shared memory: PIPE stages",
			code: "sA = storage.a.get_tensor(sA_layout, swizzle=sA_swizzle)",
			shape: seg(
				"sA: (BM, BK, ",
				["PIPE"],
				`) = (${BM}, ${BK}, `,
				[`${c.pipe}`],
				")",
			),
			body: `Shared memory holds PIPE = ${c.pipe} tiles of ${BM} × ${BK} at once, so the copy of a later k tile overlaps the MMAs on an earlier one. Tile k lands in stage k mod ${c.pipe}. From here on the figure shows one stage; every stage is partitioned the same way.`,
			usesThread: false,
			grid: blockGrid(
				1,
				kTiles,
				BM,
				BK,
				"M",
				"K",
				(_, q) => q % c.pipe,
				(_, q) => `k=${q}→s${q % c.pipe}`,
				(_, q) => `k tile ${q} is copied into stage ${q % c.pipe}`,
			),
		});
	}

	// Element-level views of one stage (A) or of the C tile.
	const P = BM;
	const PQ = BQ;
	const own = ownership(mma, c.operand, P, PQ);
	const mine = threadValues(mma, c.operand, P, PQ, thr);
	const at = (r: number, q: number) => r + q * P; // column-major offset in the tile
	const atomBoxes: Box[] = [];
	for (let r = 0; r < P / aM; r++)
		for (let q = 0; q < PQ / aQ; q++)
			atomBoxes.push({
				r0: r * aM,
				c0: q * aQ,
				r1: (r + 1) * aM,
				c1: (q + 1) * aQ,
			});
	const cellGrid = (
		group: (r: number, q: number) => number,
		boxes: Box[],
		describe: (r: number, q: number) => string,
		mark?: (r: number, q: number) => boolean,
		text?: (r: number, q: number) => string | undefined,
	): Grid => {
		const g = new Int32Array(P * PQ);
		const mk = mark ? new Uint8Array(P * PQ) : undefined;
		const tx: (string | undefined)[] | undefined = text ? [] : undefined;
		for (let r = 0; r < P; r++)
			for (let q = 0; q < PQ; q++) {
				g[r * PQ + q] = group(r, q);
				if (mk && mark?.(r, q)) mk[r * PQ + q] = 1;
				if (tx) tx[r * PQ + q] = text?.(r, q);
			}
		return {
			rows: P,
			cols: PQ,
			unitH: 1,
			unitW: 1,
			rowName: "m",
			colName: qName.toLowerCase(),
			group: g,
			mark: mk,
			text: tx,
			boxes,
			describe,
		};
	};
	const restP = P / aM;
	const restQ = PQ / aQ;
	const atomOf = (r: number, q: number) => [
		Math.floor(r / aM),
		Math.floor(q / aQ),
	];

	// 4. Cut into atom-sized tiles.
	steps.push({
		title: `Cut into ${aM} × ${aQ} atom tiles`,
		code: `# inside partition_${name}: zipped_divide(tile, (${aM}, ${aQ}))`,
		shape: seg(
			`((Atom_M, Atom_${qName}), `,
			[`(Rest_M, Rest_${qName})`],
			`) = ((${aM}, ${aQ}), `,
			[`(${restP}, ${restQ})`],
			")",
		),
		body: `One ${warp ? "mma.sync m16n8k16" : `wgmma m64n${aN}k16`} instruction covers an ${aM} × ${aQ} piece of ${name}. partition_${name} first divides the ${P} × ${PQ} tile into ${restP} × ${restQ} such pieces. (It divides by permutation_mnk first; an integer permutation such as the default ${warp ? `(${tM * aM}, ${tN * aN}, ${aK})` : `(${tM * aM}, ${aN}, ${aK})`} leaves the order unchanged.)`,
		usesThread: false,
		grid: cellGrid(
			(r, q) => {
				const [i, j] = atomOf(r, q);
				return i * restQ + j;
			},
			atomBoxes,
			(r, q) => {
				const [i, j] = atomOf(r, q);
				return `(m, ${qName.toLowerCase()}) = (${r}, ${q}) is in atom tile (${i}, ${j})`;
			},
		),
	});

	// 5. Hand the atom tiles to warps / warpgroups.
	const unitOf = (r: number, q: number) => {
		const [i, j] = atomOf(r, q);
		return [i % tM, j % tQ];
	};
	const unitIds = (um: number, uq: number) => {
		// Thread-layout index of each warp/warpgroup with M coordinate um (and N coordinate uq for C).
		const ids: number[] = [];
		for (let n = 0; n < tN; n++)
			if (!isA ? n === uq : true) ids.push(um + tM * n);
		return ids;
	};
	const mmaP = restP / tM;
	const mmaQ = restQ / tQ;
	const myUnit = [thrCoord[1], isA ? 0 : thrCoord[2]];
	const unitBoxes: Box[] = atomBoxes.map((b) => {
		const [um, uq] = unitOf(b.r0, b.c0);
		const ids = unitIds(um, uq);
		return { ...b, label: `${warp ? "W" : "WG"}${ids.join(",")}` };
	});
	const unitText = isA
		? warp
			? `atom_layout_mnk = (${tM}, ${tN}, 1) puts ${tM * tN} warps on a ${tM} × ${tN} grid over (M, N); warp w sits at (w mod ${tM}, w div ${tM}). A has no N mode, so warps that differ only in N, here warps 0 and 2, read the same A. Along M, atom tiles alternate between the ${tM} warp rows, which leaves each warp (MMA_M, MMA_K) = (${restP} / ${tM}, ${restQ} / ${tQ}) = (${mmaP}, ${mmaQ}) atom tiles: the two new modes.`
			: `atom_layout_mnk = (${tM}, 1, 1) stacks ${tM} warpgroups along M. Atom tile rows alternate between them, so each warpgroup keeps (MMA_M, MMA_K) = (${restP} / ${tM}, ${restQ}) = (${mmaP}, ${mmaQ}) atom tiles. MMA_M is 1 because one ${aM}-row wgmma already spans a warpgroup's share of BM; MMA_K = BK / 16 = ${mmaQ} instructions run back to back along K.`
		: warp
			? `The ${tM * tN} warps sit on a ${tM} × ${tN} grid over (M, N), warp w at (w mod ${tM}, w div ${tM}). Atom tiles are dealt out round-robin in both directions, so each warp keeps (MMA_M, MMA_N) = (${restP} / ${tM}, ${restQ} / ${tN}) = (${mmaP}, ${mmaQ}) of them.`
			: `The ${tM} warpgroups split M; one wgmma with N = ${aN} already spans BN. Each warpgroup keeps (MMA_M, MMA_N) = (${mmaP}, ${mmaQ}): a single instruction-sized block of C per warpgroup, accumulated over every k.`;
	steps.push({
		title: `Hand the atom tiles to ${unit}s`,
		code: `thr_mma = tiled_mma.get_slice(${warp ? "tidx" : "warp_group_thread_layout(warp_group_idx)"})`,
		shape: seg(
			`((Atom_M, Atom_${qName}), ((Thr_M, Thr_${qName}), `,
			[`(MMA_M, MMA_${qName})`],
			`)) = ((${aM}, ${aQ}), ((${tM}, ${tQ}), `,
			[`(${mmaP}, ${mmaQ})`],
			"))",
		),
		body: unitText,
		usesThread: true,
		grid: cellGrid(
			(r, q) => {
				const [um, uq] = unitOf(r, q);
				return um + tM * uq;
			},
			unitBoxes,
			(r, q) => {
				const [um, uq] = unitOf(r, q);
				const [i, j] = atomOf(r, q);
				return `atom tile (${i}, ${j}) → ${unit}${tN > 1 && isA ? "s" : ""} ${unitIds(um, uq).join(", ")}, repeat (${Math.floor(i / tM)}, ${Math.floor(j / tQ)})`;
			},
			(r, q) => {
				const [um, uq] = unitOf(r, q);
				return um === myUnit[0] && uq === myUnit[1];
			},
		),
	});

	// 6. Inside one atom: the thread-value layout.
	const tv = isA ? c.atom.layoutA_TV : c.atom.layoutC_TV;
	const lane = (t: number) => t % 32;
	const warpIn = (t: number) => Math.floor((t % T) / 32);
	if (warp) {
		steps.push({
			title: "Inside an atom: the thread-value layout",
			code: `${name}_TV = ${tv}`,
			shape: seg(
				"((",
				["ThrV, FrgV"],
				"), ...) = ((",
				[`${tv.mode(0).size}, ${formatTuple(tv.mode(1).shape)}`],
				"), ...)",
			),
			body: isA
				? `The atom's A layout maps (lane, value) to an element of the 16 × 16 tile: lane t = (t0, t1) = (t mod 4, t div 4) holds rows t1 and t1 + 8, columns 2·t0, 2·t0 + 1, 2·t0 + 8 and 2·t0 + 9, which is 8 values. Colour = lane; hover a cell for its lane and value.`
				: `The atom's C layout maps (lane, value) to an element of the 16 × 8 tile: lane t = (t0, t1) = (t mod 4, t div 4) holds rows t1 and t1 + 8, columns 2·t0 and 2·t0 + 1, which is 4 values. Colour = lane; hover a cell for its lane and value.`,
			usesThread: false,
			grid: cellGrid(
				(r, q) => lane(own.thread[at(r, q)]),
				atomBoxes,
				(r, q) => {
					const t = own.thread[at(r, q)];
					const [i, j] = atomOf(r, q);
					const v = tv.mode(1);
					// The value index inside the atom: invert by search over the TV layout.
					let vi = -1;
					const lr = r - i * aM;
					const lq = q - j * aQ;
					for (let x = 0; x < v.size; x++)
						if (tv.offset(lane(t), x) === lr + lq * aM) vi = x;
					return `(m, ${qName.toLowerCase()}) = (${r}, ${q}): lane ${lane(t)}, atom value ${vi} ${formatTuple(v.coord(vi))}${own.owners[at(r, q)] > 1 ? `, held by ${own.owners[at(r, q)]} threads` : ""}`;
				},
			),
		});
	} else {
		steps.push({
			title: isA
				? "Inside an atom: every thread sees the whole tile"
				: "Inside an atom: four warps, one 16 × 8 pattern",
			code: `${name}_TV = ${tv}`,
			shape: isA
				? seg(
						"((",
						["ThrV, FrgV"],
						"), ...) = ((",
						["128, (64, 16)"],
						"), ...)",
					)
				: seg(
						"((",
						["ThrV, FrgV"],
						"), ...) = ((",
						[`128, ${formatTuple(tv.mode(1).shape)}`],
						"), ...)",
					),
			body: isA
				? "For a shared-memory operand, GMMA::ABLayout gives the thread mode stride 0: all 128 threads of a warpgroup map to the same 64 × 16 tile. No thread owns a piece of A; the warpgroup hands the hardware the whole tile through one shared-memory descriptor. Colour = which K block (one wgmma each) of the selected thread's warpgroup."
				: `CLayout_64xN gives thread t rows 16·(t div 32) + (t mod 32) div 4, plus 8, and columns 2·(t mod 4), 2·(t mod 4) + 1 in each of the N / 8 = ${aN / 8} column blocks: the m16n8 accumulator pattern of mma.sync, once per warp, stacked four high. Colour = warp within the warpgroup.`,
			usesThread: isA,
			grid: isA
				? cellGrid(
						(r, q) =>
							Math.floor(r / aM) % tM === myUnit[0] ? Math.floor(q / aQ) : -1,
						atomBoxes,
						(r, q) => {
							const [i, j] = atomOf(r, q);
							return `atom tile (${i}, ${j}) is read whole by all 128 threads of warpgroup ${i % tM}`;
						},
					)
				: cellGrid(
						(r, q) => warpIn(own.thread[at(r, q)]),
						atomBoxes,
						(r, q) => {
							const t = own.thread[at(r, q)];
							return `(m, n) = (${r}, ${q}): thread ${t % T} of warpgroup ${Math.floor(t / T)} (warp ${warpIn(t)}, lane ${lane(t)}), value ${own.value[at(r, q)]}`;
						},
					),
		});
	}

	// 7. One thread's fragment.
	const tile3 = isA ? new CuteLayout([P, PQ, c.pipe]) : new CuteLayout([P, PQ]);
	const part3 = isA ? mma.partitionA(tile3, thr) : mma.partitionC(tile3, thr);
	const part2 = (
		isA
			? mma.partitionA(new CuteLayout([P, PQ]), thr)
			: mma.partitionC(new CuteLayout([P, PQ]), thr)
	).layout;
	const fullShape = formatTuple(part3.layout.shape);
	const modeNames = isA ? "(MMA, MMA_M, MMA_K, PIPE)" : "(MMA, MMA_M, MMA_N)";
	const varName = isA ? "tCsA" : "tCgC";
	const partShape = part3.layout.shape as (number | number[])[];
	const mmaMode = formatTuple(partShape[0]);
	const sizes = partShape.map((s) => formatTuple(s));
	if (!(warp === false && isA)) {
		const perThread = part2.size;
		steps.push({
			title: `Thread ${thr}'s fragment`,
			code: `${varName} = thr_mma.partition_${name}(${isA ? "sA" : "gC"})`,
			shape: seg(
				`${varName}: (`,
				["MMA"],
				`, ${modeNames.slice(6, -1)}) = (`,
				[mmaMode],
				`, ${sizes.slice(1).join(", ")})`,
			),
			body: isA
				? `Fixing the thread (warp ${Math.floor(thr / 32)}, lane ${lane(thr)}) leaves its values: the atom's FrgV mode becomes MMA = ${mmaMode} (${tv.mode(1).size} values), followed by the repeats MMA_M = ${sizes[1]} and MMA_K = ${sizes[2]}, and the stage mode PIPE = ${sizes[3]} carried along from sA. That is ${perThread} elements per stage. Numbers are the value's index in tCsA[None, None, None, 0].`
				: warp
					? `Fixing the thread (warp ${Math.floor(thr / 32)}, lane ${lane(thr)}) leaves MMA = ${mmaMode} accumulator values per atom, MMA_M = ${sizes[1]} and MMA_N = ${sizes[2]} atom repeats: ${perThread} values. Check: ${BM} × ${BN} / ${mma.size} threads = ${(BM * BN) / mma.size}. make_fragment_C gives the register tensor tCrC of the same shape.`
					: `Thread ${thr} (warpgroup ${Math.floor(thr / T)}, warp ${warpIn(thr)}, lane ${lane(thr)}) holds MMA = ${mmaMode} = ${perThread} accumulators, with MMA_M = ${sizes[1]} and MMA_N = ${sizes[2]}. Check: ${BM} × ${BN} / ${mma.size} threads = ${(BM * BN) / mma.size}. The kernel allocates acc = make_rmem_tensor(tCgC.shape, Float32).`,
			usesThread: true,
			grid: cellGrid(
				(r, q) =>
					warp ? lane(own.thread[at(r, q)]) : warpIn(own.thread[at(r, q)]),
				atomBoxes,
				(r, q) => {
					const v = mine.get(at(r, q));
					return v === undefined
						? `(m, ${qName.toLowerCase()}) = (${r}, ${q}): not held by thread ${thr}`
						: `(m, ${qName.toLowerCase()}) = (${r}, ${q}): ${varName}[${coordIn(part2, v)}${isA ? ", 0" : ""}], value ${v} of thread ${thr}`;
				},
				(r, q) => mine.has(at(r, q)),
				(r, q) => {
					const v = mine.get(at(r, q));
					return v === undefined ? undefined : `${v}`;
				},
			),
		});
	} else {
		// Warpgroup A: the fragment is a tensor of shared-memory descriptors.
		steps.push({
			title: "make_fragment_A: one descriptor per K block",
			code: "tCrA = tiled_mma.make_fragment_A(tCsA)",
			shape: seg(
				"tCsA: (",
				["MMA"],
				", MMA_M, MMA_K, PIPE) = (",
				[mmaMode],
				`, ${sizes.slice(1).join(", ")})  →  tCrA: (`,
				["1"],
				`, ${sizes.slice(1).join(", ")})`,
			),
			body: `partition_A leaves the whole ${aM} × ${aK} tile in the MMA mode: tCsA = ${fullShape}. make_fragment_A replaces that mode with a single 64-bit descriptor (start address, swizzle, strides), so tCrA = (1, ${sizes.slice(1).join(", ")}): one descriptor per (MMA_M, MMA_K, PIPE) entry, ${mmaQ} per stage for warpgroup ${myUnit[0]}. cute.gemm passes them to wgmma, and the tensor cores read shared memory directly.`,
			usesThread: true,
			grid: cellGrid(
				(r, q) =>
					Math.floor(r / aM) % tM === myUnit[0] ? Math.floor(q / aQ) : -1,
				atomBoxes.map((b) => {
					const [i, j] = atomOf(b.r0, b.c0);
					return i % tM === myUnit[0]
						? { ...b, label: `desc ${j}` }
						: b;
				}),
				(r, q) => {
					const [i, j] = atomOf(r, q);
					return i % tM === myUnit[0]
						? `tCrA[0, ${Math.floor(i / tM)}, ${j}, stage]: one descriptor for rows ${i * aM}–${(i + 1) * aM - 1}, columns ${j * aK}–${(j + 1) * aK - 1}`
						: `read by warpgroup ${i % tM}, not by thread ${thr}'s`;
				},
				(r) => Math.floor(r / aM) % tM === myUnit[0],
			),
		});
	}
	return steps;
}
