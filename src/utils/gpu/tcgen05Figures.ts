// Figure data for the tcgen05 post (src/content/posts/cute-dsl-tcgen05-mma.mdx): the data flow
// of a Blackwell GEMM kernel, drawn by BoxDiagram. Follows the "GMEM to MMA data flow" diagram of
// NVIDIA's tcgen05 MMA Programming Guide (CUTLASS 4.5.3) and the fp16_gemm_0.py tutorial.

import type { BoxEdge, BoxNode } from "../payments/boxDiagram";

export const tcgen05FlowNodes: BoxNode[] = [
	{
		id: "gmem",
		label: "mA, mB",
		sub: "global memory",
		col: 0,
		row: 0,
		group: 0,
		note: "The input matrices. <code>local_tile</code> and <code>partition_A</code> / <code>partition_B</code> turn them into <code>tCgA</code> and <code>tCgB</code>: views of the CTA's tile, one k-block at a time. Nothing is copied by these calls.",
	},
	{
		id: "tma",
		label: "TMA",
		sub: "copy engine",
		col: 1,
		row: 0,
		group: 1,
		note: "The Tensor Memory Accelerator copies a whole k-block of A and B (here 16 KiB + 32 KiB) from global to shared memory in one asynchronous bulk operation, swizzling it on the way, and signals an mbarrier when the bytes have landed.",
	},
	{
		id: "smem",
		label: "sA, sB",
		sub: "shared memory, stages",
		col: 2,
		row: 0,
		group: 2,
		note: "A ring of SMEM buffers (stages). TMA writes stage s while the tensor core reads stage s − 1. <code>make_fragment_A(sA)</code> turns each stage into 64-bit shared-memory descriptors, <code>tCrA</code>; the MMA reads SMEM through them.",
	},
	{
		id: "tc",
		label: "Tensor core",
		sub: "tcgen05.mma",
		col: 3,
		row: 0,
		group: 3,
		note: "Reads A and B from SMEM (A may also come from TMEM), multiplies, and adds into the accumulator in TMEM. A 128 × 256 × 16 FP16 instruction is 1,048,576 FLOPs. The issuing thread does not wait for it.",
	},
	{
		id: "tmem",
		label: "tCtAcc",
		sub: "Tensor Memory",
		col: 4,
		row: 0,
		group: 4,
		note: "256 KiB per SM: 128 lanes × 512 columns of 32-bit cells. The accumulator lives here for the whole K loop, so it does not take any registers. Allocated with <code>tcgen05.alloc</code> by one warp, in columns.",
	},
	{
		id: "tmathr",
		label: "TMA producer",
		sub: "one thread",
		col: 1,
		row: 1,
		group: 5,
		note: "Waits for an empty SMEM stage, then issues the TMA copies for the next k-block: <code>cute.copy(tma_atom_a, tAgA[k], tAsA[stage])</code>.",
	},
	{
		id: "mmathr",
		label: "MMA issuer",
		sub: "one thread",
		col: 3,
		row: 1,
		group: 5,
		note: "Waits for a full SMEM stage, then issues one <code>tcgen05.mma</code> per k-slice with <code>cute.gemm</code>. A <code>tcgen05.commit</code> to the stage's mbarrier releases the buffer once the tensor core has read it.",
	},
	{
		id: "epi",
		label: "Epilogue warps",
		sub: "registers",
		col: 4,
		row: 2,
		group: 1,
		note: "Four warps (a warpgroup). Warp w can read only TMEM lanes 32w to 32w + 31, so together they cover the 128 rows. <code>tcgen05.ld</code> moves the accumulator into registers; then they convert and store.",
	},
	{
		id: "out",
		label: "mC",
		sub: "global memory",
		col: 0,
		row: 2,
		group: 0,
		note: "The output. <code>partition_C</code> gives the view <code>tCgC</code>; the epilogue's tiled copy partitions it again to match what each thread loaded from TMEM.",
	},
];

export const tcgen05FlowEdges: BoxEdge[] = [
	{ from: "gmem", to: "tma", label: "tAgA[k]" },
	{ from: "tma", to: "smem", label: "tAsA[stage]" },
	{ from: "smem", to: "tc", label: "descriptors" },
	{ from: "tc", to: "tmem", label: "D", both: true },
	{ from: "tmathr", to: "tma", dashed: true, label: "issue" },
	{ from: "mmathr", to: "tc", dashed: true, label: "issue" },
	{ from: "tmem", to: "epi", label: "tcgen05.ld" },
	{ from: "epi", to: "out", label: "store" },
];
