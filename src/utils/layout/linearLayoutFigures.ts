// Layouts drawn in the linear-layouts post, all built with the Triton-faithful constructors in
// linearLayout.ts (checked against Triton v3.8.0's LinearLayoutConversionsTest.cpp).
import {
	type LinearLayout,
	blocked,
	makeLayout,
	mmaV2,
	mmaV2Operand,
	slice,
	swizzledShared,
} from "./linearLayout";

export interface LayoutPreset {
	key: string;
	label: string;
	/** Tensor shape [rows, cols] or [n]. */
	shape: number[];
	layout: LinearLayout;
	/** One sentence shown under the figure. */
	note: string;
}

const layoutA = (shape: number[]) =>
	blocked(
		{
			sizePerThread: [2, 2],
			threadsPerWarp: [4, 8],
			warpsPerCTA: [2, 1],
			order: [1, 0],
		},
		shape,
	);

export const explorerPresets: LayoutPreset[] = [
	{
		key: "blockedA",
		label: "Blocked (paper's layout A)",
		shape: [16, 16],
		layout: layoutA([16, 16]),
		note: "sizePerThread [2, 2], threadsPerWarp [4, 8], warpsPerCTA [2, 1], order [1, 0]: each thread owns a 2 × 2 block, 8 threads across a row.",
	},
	{
		key: "blockedColMajor",
		label: "Blocked, column-major",
		shape: [16, 16],
		layout: blocked(
			{
				sizePerThread: [2, 2],
				threadsPerWarp: [8, 4],
				warpsPerCTA: [1, 2],
				order: [0, 1],
			},
			[16, 16],
		),
		note: "The same pieces with order [0, 1]: registers and threads run down columns first, like the paper's layout B.",
	},
	{
		key: "mma",
		label: "mma.sync m16n8 accumulator",
		shape: [16, 16],
		layout: mmaV2([1, 1], [16, 16]),
		note: "NvidiaMmaEncoding version 2, one warp, two m16n8 tiles side by side: register 1 jumps 8 rows, register 2 jumps to the second tile.",
	},
	{
		key: "dotB",
		label: "mma B operand (fp16)",
		shape: [16, 8],
		layout: mmaV2Operand(1, 2, [1, 1], [16, 8]),
		note: "DotOperandEncoding opIdx 1, kWidth 2: the m16n8k16 B fragment (k down, n across). It is the A pattern transposed: each thread holds 2 consecutive k values of one column.",
	},
	{
		key: "broadcast",
		label: "Layout A on an 8 × 8 tensor",
		shape: [8, 8],
		layout: layoutA([8, 8]),
		note: "The layout's tile is 16 × 16 but the tensor is 8 × 8: the bases that would leave the tensor become zero, so several threads hold copies of each element.",
	},
	{
		key: "slice",
		label: "Slice of layout A (dim 0 reduced)",
		shape: [16],
		layout: slice(layoutA, [16], 0),
		note: "The result of tl.sum(x, axis=0): the row bits are dropped, so the lane and warp bases that moved along rows become zero (broadcast).",
	},
	{
		key: "unswizzled",
		label: "Shared memory, unswizzled",
		shape: [8, 8],
		layout: swizzledShared(1, 1, 1, [8, 8]),
		note: "Offset → element, row-major: offset 8i + j holds element (i, j).",
	},
	{
		key: "swizzled",
		label: "Shared memory, XOR swizzle",
		shape: [8, 8],
		layout: swizzledShared(1, 1, 8, [8, 8]),
		note: "vec 1, perPhase 1, maxPhase 8: row i's offset bases also XOR i into the column, so element (i, j) sits at offset 8i + (j ⊕ i).",
	},
];

/** The paper's Figure 4: 8 elements, 4 threads, 2 registers each. */
export const shuffleSrc = blocked(
	{ sizePerThread: [2], threadsPerWarp: [4], warpsPerCTA: [1], order: [0] },
	[8],
);
export const shuffleDst = makeLayout(
	[
		["register", [[4]]],
		["lane", [[1], [2]]],
		["warp", []],
	],
	["dim0"],
	[8],
);
const B = (
	spt: number[],
	tpw: number[],
	wpc: number[],
	order: number[],
	shape: number[],
) =>
	blocked(
		{ sizePerThread: spt, threadsPerWarp: tpw, warpsPerCTA: wpc, order },
		shape,
	);

export interface ConversionCase {
	label: string;
	src: LinearLayout;
	dst: LinearLayout;
	srcText: string;
	dstText: string;
}

/** Conversions on a 16 × 16 tensor with 2 warps, one per lowering. */
export const conversionCases: ConversionCase[] = [
	{
		label: "same layout",
		src: layoutA([16, 16]),
		dst: layoutA([16, 16]),
		srcText: "layout A",
		dstText: "layout A",
	},
	{
		label: "registers only",
		src: layoutA([16, 16]),
		dst: swapRegs(layoutA([16, 16])),
		srcText: "layout A",
		dstText: "layout A, registers 1 and 2 swapped",
	},
	{
		label: "within a warp",
		src: B([1, 2], [8, 4], [2, 1], [1, 0], [16, 16]),
		dst: B([1, 4], [8, 4], [2, 1], [1, 0], [16, 16]),
		srcText: "sizePerThread [1, 2], threads [8, 4], warps [2, 1]",
		dstText: "sizePerThread [1, 4], threads [8, 4], warps [2, 1]",
	},
	{
		label: "across warps",
		src: layoutA([16, 16]),
		dst: mmaV2([2, 1], [16, 16]),
		srcText: "layout A",
		dstText: "mma m16n8, warps [2, 1]",
	},
];

/** Layout A with its two register bases exchanged: the same elements, numbered differently. */
function swapRegs(L: LinearLayout): LinearLayout {
	return {
		ins: L.ins.map((d) =>
			d.name === "register"
				? { name: d.name, bases: [d.bases[1], d.bases[0], ...d.bases.slice(2)] }
				: d,
		),
		outs: L.outs,
	};
}

export interface SwizzleCase {
	key: string;
	label: string;
	shape: [number, number];
	elemBits: number;
	store: LinearLayout;
	load: LinearLayout;
	storeText: string;
	loadText: string;
	/** Triton swizzle parameters (vec, perPhase, maxPhase) to compare with. */
	swizzles: [number, number, number][];
}

export const swizzleCases: SwizzleCase[] = [
	{
		key: "transpose32",
		label: "fp32, rows → columns",
		shape: [32, 32],
		elemBits: 32,
		store: B([1, 1], [1, 32], [1, 1], [1, 0], [32, 32]),
		load: B([1, 1], [32, 1], [1, 1], [0, 1], [32, 32]),
		storeText: "lane j holds column j (threadsPerWarp [1, 32])",
		loadText: "lane i holds row i (threadsPerWarp [32, 1], order [0, 1])",
		swizzles: [
			[1, 1, 32],
			[4, 1, 8],
		],
	},
	{
		key: "fp16cm",
		label: "fp16, row-major → column-major",
		shape: [64, 32],
		elemBits: 16,
		store: B([1, 8], [8, 4], [1, 1], [1, 0], [64, 32]),
		load: B([8, 1], [8, 4], [1, 1], [0, 1], [64, 32]),
		storeText: "sizePerThread [1, 8], threadsPerWarp [8, 4], order [1, 0]",
		loadText: "sizePerThread [8, 1], threadsPerWarp [8, 4], order [0, 1]",
		swizzles: [[1, 1, 32]],
	},
];
