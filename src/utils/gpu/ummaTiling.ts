// Steps that follow a tcgen05 (UMMA) GEMM tile from the global matrices to one MMA instruction,
// for UmmaTileSteps.astro. Each step is a row of panels (matrices or buffers) drawn at the same
// scale; a panel is a grid of cells, each standing for a block of elements.
//
// Presets follow the worked examples of NVIDIA's "tcgen05 MMA Programming Guide" (CUTLASS 4.5.3):
//   "1cta": GEMM 512 × 768 × 384, mma_tiler 128 × 256 × 64, atom m128n256k16, CtaGroup.ONE.
//   "2cta": the same GEMM, mma_tiler 256 × 256 × 64, atom m256n256k16, CtaGroup.TWO.
//   "perm": which M rows each CTA of a pair owns, without and with permutation_mnk.
// The partition shapes and row ownership were checked by running thrfrg_A / thrfrg_C (the MMA class
// in component/Cute/CuteMma.ts) on the atoms' layouts from mma_traits_sm100.hpp.
//
//   const steps = buildUmmaSteps("1cta");
//   steps[2].shape   // tCgA: ((128,16),1,4,6) ...

export type UmmaPreset = "1cta" | "2cta" | "perm";
export const UMMA_PRESETS: UmmaPreset[] = ["1cta", "2cta", "perm"];

export interface Panel {
	title: string;
	rows: number;
	cols: number;
	/** Elements per cell, down and across. */
	cellH: number;
	cellW: number;
	/** Colour group 0..5 per cell; -1 is a plain cell, -2 a dimmed one. */
	group: (r: number, c: number) => number;
	label?: (r: number, c: number) => string | undefined;
	describe: (r: number, c: number) => string;
	/** Axis names, e.g. ["M", "K"]. */
	axes: [string, string];
	/** Row of the figure the panel is drawn in (default 0). */
	row?: number;
}

export interface UmmaStep {
	title: string;
	code: string;
	/** Shapes after the step, one per line. */
	shape: string[];
	body: string;
	panels: Panel[];
	/** Draw each element this many times wider than tall (for very narrow panels). */
	xStretch?: number;
}

const span = (start: number, size: number) => `${start}–${start + size - 1}`;

/** A panel of a matrix (or a slice of one) cut into blocks of cellH × cellW elements. */
function blocks(
	title: string,
	axes: [string, string],
	rows: number,
	cols: number,
	cellH: number,
	cellW: number,
	group: Panel["group"],
	label: Panel["label"],
	what: (r: number, c: number) => string,
	origin: [number, number] = [0, 0],
): Panel {
	return {
		title,
		axes,
		rows,
		cols,
		cellH,
		cellW,
		group,
		label,
		describe: (r, c) =>
			`${title}: ${axes[0]} ${span(origin[0] + r * cellH, cellH)}, ${axes[1]} ${span(origin[1] + c * cellW, cellW)}. ${what(r, c)}`,
	};
}

function oneCta(): UmmaStep[] {
	const [M, N, K] = [512, 768, 384];
	const [BM, BN, BK] = [128, 256, 64];
	const [iM, iN, iK] = [128, 256, 16];
	const [cm, cn] = [0, 1];
	const kt = K / BK; // 6
	const mk = BK / iK; // 4
	const stages = 3;
	const flops = 2 * iM * iN * iK;
	const steps: UmmaStep[] = [];

	steps.push({
		title: "The GEMM in global memory",
		code: "mA, mB, mC  # A is M × K, B is N × K (both K-major), C is M × N",
		shape: [
			"mA: (M, K) = (512, 384)",
			"mB: (N, K) = (768, 384)",
			"mC: (M, N) = (512, 768)",
		],
		body: `The mma_tiler (BM, BN, BK) = (128, 256, 64) cuts A into ${M / BM} × ${kt} blocks, B into ${N / BN} × ${kt} and C into ${M / BM} × ${N / BN}. Each CTA owns one block of C; the one at mma_coord (0, 1) is highlighted, with the row of A blocks and the row of B blocks it reads.`,
		panels: [
			blocks(
				"mA",
				["m", "k"],
				M / BM,
				kt,
				BM,
				BK,
				(r) => (r === cm ? 0 : -1),
				undefined,
				(r, c) => `Block (${r}, ${c}).`,
			),
			blocks(
				"mB",
				["n", "k"],
				N / BN,
				kt,
				BN,
				BK,
				(r) => (r === cn ? 1 : -1),
				undefined,
				(r, c) => `Block (${r}, ${c}).`,
			),
			blocks(
				"mC",
				["m", "n"],
				M / BM,
				N / BN,
				BM,
				BN,
				(r, c) => (r === cm && c === cn ? 2 : -1),
				(r, c) => `(${r},${c})`,
				(r, c) => `Output tile mma_coord (${r}, ${c}), one CTA.`,
			),
		],
	});

	steps.push({
		title: "local_tile: the CTA's tiles",
		code: "gA = cute.local_tile(mA, mma_tiler_mnk, mma_coord, proj=(1, None, 1))",
		shape: [
			"gA: (BM, BK, k) = (128, 64, 6)",
			"gB: (BN, BK, k) = (256, 64, 6)",
			"gC: (BM, BN) = (128, 256)",
		],
		body: "local_tile keeps the CTA's blocks and adds a mode k that walks the K dimension: gA[:, :, k] is the k-th 128 × 64 block of A's strip. proj drops the mode a tensor does not have (N for A, M for B, K for C). The main loop runs once per k, six times here.",
		panels: [
			blocks(
				"gA",
				["m", "k"],
				1,
				kt,
				BM,
				BK,
				() => 0,
				(_, c) => `k${c}`,
				(_, c) => `gA[:, :, ${c}]`,
				[cm * BM, 0],
			),
			blocks(
				"gB",
				["n", "k"],
				1,
				kt,
				BN,
				BK,
				() => 1,
				(_, c) => `k${c}`,
				(_, c) => `gB[:, :, ${c}]`,
				[cn * BN, 0],
			),
			blocks(
				"gC",
				["m", "n"],
				1,
				1,
				BM,
				BN,
				() => 2,
				() => "gC",
				() => "The CTA's whole output tile.",
				[cm * BM, cn * BN],
			),
		],
	});

	steps.push({
		title: "partition_A / B / C: per-instruction pieces",
		code: "thr_mma = tiled_mma.get_slice(0); tCgA = thr_mma.partition_A(gA)",
		shape: [
			"tCgA: (MMA, MMA_M, MMA_K, k) = ((128,16), 1, 4, 6)",
			"tCgB: (MMA, MMA_N, MMA_K, k) = ((256,16), 1, 4, 6)",
			"tCgC: (MMA, MMA_M, MMA_N)    = ((128,256), 1, 1)",
		],
		body: `One CTA owns the whole tile (get_slice(0)), so partitioning only regroups it by instruction. Mode MMA is one instruction's operand: 128 × 16 of A, 256 × 16 of B, 128 × 256 of C. MMA_M = BM / 128 = 1, MMA_N = BN / 256 = 1 and MMA_K = BK / 16 = ${mk}: four instructions per k-block. Shown: block k = 0.`,
		panels: [
			blocks(
				"gA[:, :, 0]",
				["m", "k"],
				1,
				mk,
				BM,
				iK,
				(_, c) => c,
				(_, c) => `${c}`,
				(_, c) => `tCgA[None, 0, ${c}, 0]: MMA_K = ${c}.`,
				[cm * BM, 0],
			),
			blocks(
				"gB[:, :, 0]",
				["n", "k"],
				1,
				mk,
				BN,
				iK,
				(_, c) => c,
				(_, c) => `${c}`,
				(_, c) => `tCgB[None, 0, ${c}, 0]: MMA_K = ${c}.`,
				[cn * BN, 0],
			),
			blocks(
				"gC",
				["m", "n"],
				1,
				1,
				BM,
				BN,
				() => 5,
				() => "MMA",
				() => "tCgC[None, 0, 0]: one instruction's whole output.",
				[cm * BM, cn * BN],
			),
		],
	});

	steps.push({
		title: "make_fragment_A / B: SMEM descriptors",
		code: "tCrA = tiled_mma.make_fragment_A(sA)  # sA: staged SMEM written by TMA",
		shape: [
			"tCrA: (MMA, MMA_M, MMA_K, STAGE) = (1, 1, 4, 3)",
			"tCrB: (MMA, MMA_N, MMA_K, STAGE) = (1, 1, 4, 3)",
		],
		body: `TMA copies each k-block into one of ${stages} SMEM stages. make_fragment_A does not copy anything: it turns sA into a tensor of 64-bit shared-memory descriptors, one per (MMA_K, STAGE) cell. Its MMA mode has size 1 because a single descriptor stands for the whole 128 × 16 operand the instruction reads.`,
		panels: [
			blocks(
				"sA",
				["m", "k"],
				1,
				stages * mk,
				BM,
				iK,
				(_, c) => Math.floor(c / mk),
				(_, c) => `s${Math.floor(c / mk)}·${c % mk}`,
				(_, c) =>
					`Stage ${Math.floor(c / mk)}, MMA_K ${c % mk}: descriptor tCrA[0, 0, ${c % mk}, ${Math.floor(c / mk)}].`,
			),
			blocks(
				"sB",
				["n", "k"],
				1,
				stages * mk,
				BN,
				iK,
				(_, c) => Math.floor(c / mk),
				(_, c) => `s${Math.floor(c / mk)}·${c % mk}`,
				(_, c) =>
					`Stage ${Math.floor(c / mk)}, MMA_K ${c % mk}: descriptor tCrB[0, 0, ${c % mk}, ${Math.floor(c / mk)}].`,
			),
		],
	});

	for (let j = 0; j < mk; j++) {
		steps.push({
			title: `Instruction ${j + 1} of ${mk} for this k-block`,
			code: `cute.gemm(tiled_mma, tCtAcc, tCrA[None, None, ${j}, s], tCrB[None, None, ${j}, s], tCtAcc)`,
			shape: ["tCtAcc: (MMA, MMA_M, MMA_N) = ((128,256), 1, 1) in TMEM"],
			body:
				j === 0
					? `One thread issues tcgen05.mma with two descriptors (A and B in SMEM stage s) and a TMEM address (D). On the first k-block of the tile ACCUMULATE is False, so D = A₀ B₀ᵀ overwrites whatever TMEM held. One instruction is 128 × 256 × 16 multiply-adds, ${flops.toLocaleString("en-US")} FLOPs.`
					: `Same thread, next descriptor pair: D += A${"₀₁₂₃"[j]} B${"₀₁₂₃"[j]}ᵀ. ACCUMULATE is now True. The thread does not wait for the previous instruction: tcgen05.mma is asynchronous, and PTX orders MMAs from one warp into the same accumulator, so this one reads the D the last one wrote.`,
			panels: [
				blocks(
					"sA[stage s]",
					["m", "k"],
					1,
					mk,
					BM,
					iK,
					(_, c) => (c === j ? 0 : -2),
					(_, c) => `${c}`,
					(_, c) =>
						c === j
							? "Read by this instruction."
							: "Another instruction's slice.",
				),
				blocks(
					"sB[stage s]",
					["n", "k"],
					1,
					mk,
					BN,
					iK,
					(_, c) => (c === j ? 1 : -2),
					(_, c) => `${c}`,
					(_, c) =>
						c === j
							? "Read by this instruction."
							: "Another instruction's slice.",
				),
				blocks(
					"tCtAcc (TMEM)",
					["m", "n"],
					1,
					1,
					BM,
					BN,
					() => 2,
					() => (j === 0 ? "D =" : "D +="),
					() => "The accumulator: lane m, column n of TMEM.",
				),
			],
		});
	}
	return steps;
}

function twoCta(): UmmaStep[] {
	const [M, N, K] = [512, 768, 384];
	const [BM, BN, BK] = [256, 256, 64];
	const iK = 16;
	const kt = K / BK;
	const mk = BK / iK;
	const vColor = (v: number) => (v === 0 ? 0 : 3);
	const steps: UmmaStep[] = [];

	steps.push({
		title: "The GEMM, tiled for a CTA pair",
		code: "mma_tiler_mnk = (256, 256, 64)  # per CTA pair",
		shape: [
			"mA: (M, K) = (512, 384)",
			"mB: (N, K) = (768, 384)",
			"mC: (M, N) = (512, 768)",
		],
		body: "With CtaGroup.TWO the mma_tiler is the tile of a pair of CTAs, so BM doubles to 256: C has 2 × 3 tiles, each computed by two CTAs together. The pair at mma_coord (0, 0) is highlighted.",
		panels: [
			blocks(
				"mA",
				["m", "k"],
				M / BM,
				kt,
				BM,
				BK,
				(r) => (r === 0 ? 0 : -1),
				undefined,
				(r, c) => `Block (${r}, ${c}).`,
			),
			blocks(
				"mB",
				["n", "k"],
				N / BN,
				kt,
				BN,
				BK,
				(r) => (r === 0 ? 1 : -1),
				undefined,
				(r, c) => `Block (${r}, ${c}).`,
			),
			blocks(
				"mC",
				["m", "n"],
				M / BM,
				N / BN,
				BM,
				BN,
				(r, c) => (r === 0 && c === 0 ? 2 : -1),
				(r, c) => `(${r},${c})`,
				(r, c) => `Output tile (${r}, ${c}), one CTA pair.`,
			),
		],
	});

	steps.push({
		title: "get_slice(v): each CTA's half",
		code: "thr_mma = tiled_mma.get_slice(bidx % 2)  # v = 0 or 1",
		shape: ["per CTA: A rows 128, B rows 128 (to load), C rows 128"],
		body: "The atom's thread mode is the CTA: ThrID = 2. CTA v gets rows 128v … 128v + 127 of A and of C. B is split by N for loading only: each CTA's TMA brings in half of B's rows, but the instruction reads both halves, so each CTA multiplies its 128 rows of A by all 256 rows of B.",
		panels: [
			blocks(
				"gA[:, :, 0]",
				["m", "k"],
				2,
				mk,
				128,
				iK,
				(r) => vColor(r),
				(r, c) => `v${r}·${c}`,
				(r) => `Loaded and owned by CTA ${r}.`,
			),
			blocks(
				"gB[:, :, 0]",
				["n", "k"],
				2,
				mk,
				128,
				iK,
				(r) => vColor(r),
				(r, c) => `v${r}·${c}`,
				(r) => `Loaded by CTA ${r}; read by the MMA for both CTAs.`,
			),
			blocks(
				"gC",
				["m", "n"],
				2,
				1,
				128,
				BN,
				(r) => vColor(r),
				(r) => `CTA ${r}`,
				(r) => `Accumulated in CTA ${r}'s TMEM.`,
			),
		],
	});

	steps.push({
		title: "partition_A / B / C, per CTA",
		code: "tCgA = thr_mma.partition_A(gA)",
		shape: [
			"tCgA: (MMA, MMA_M, MMA_K, k) = ((128,16), 1, 4, 6)",
			"tCgB: (MMA, MMA_N, MMA_K, k) = ((128,16), 1, 4, 6)",
			"tCgC: (MMA, MMA_M, MMA_N)    = ((128,256), 1, 1)",
		],
		body: "Per CTA, the shapes match the single-CTA example: MMA_M = (BM / 2) / 128 = 1 and MMA_K = 64 / 16 = 4. The MMA mode of B is (128, 16), the half this CTA loads, even though the instruction consumes 256 rows of B. Shown: CTA 0's pieces of block k = 0.",
		panels: [
			blocks(
				"CTA 0: tCgA",
				["m", "k"],
				1,
				mk,
				128,
				iK,
				(_, c) => c,
				(_, c) => `${c}`,
				(_, c) => `MMA_K = ${c}.`,
			),
			blocks(
				"CTA 0: tCgB",
				["n", "k"],
				1,
				mk,
				128,
				iK,
				(_, c) => c,
				(_, c) => `${c}`,
				(_, c) => `MMA_K = ${c}.`,
			),
			blocks(
				"CTA 0: tCgC",
				["m", "n"],
				1,
				1,
				128,
				BN,
				() => 5,
				() => "MMA",
				() => "This CTA's 128 × 256 of the output.",
			),
		],
	});

	steps.push({
		title: "One pair instruction",
		code: "if is_leader_cta: cute.gemm(tiled_mma, tCtAcc, tCrA[..., j, s], tCrB[..., j, s], tCtAcc)",
		shape: [
			"tcgen05.mma.cta_group::2 … m256n256k16",
			"each CTA: tCtAcc ((128,256), 1, 1) in its own TMEM",
		],
		body: "One thread in the leader CTA (v = 0) issues the instruction for both. The tensor cores of both SMs work on it: each multiplies its own 128 rows of A, from its own SMEM, by both halves of B, one from each CTA's SMEM, and writes 128 × 256 accumulators into its own TMEM at the same address. The pair computes a 256 × 256 × 16 product while each CTA stores only half of B.",
		panels: [
			blocks(
				"CTA 0 sA",
				["m", "k"],
				1,
				mk,
				128,
				iK,
				(_, c) => (c === 0 ? 0 : -2),
				(_, c) => `${c}`,
				(_, c) =>
					c === 0
						? "A rows 0–127, read by CTA 0's tensor core."
						: "Later instruction.",
			),
			blocks(
				"CTA 0 sB",
				["n", "k"],
				1,
				mk,
				128,
				iK,
				(_, c) => (c === 0 ? 0 : -2),
				(_, c) => `${c}`,
				(_, c) =>
					c === 0 ? "B rows 0–127, read for both CTAs." : "Later instruction.",
			),
			blocks(
				"CTA 0 TMEM",
				["m", "n"],
				1,
				1,
				128,
				BN,
				() => 0,
				() => "D rows 0–127",
				() => "Accumulator for C rows 0–127.",
			),
			{
				...blocks(
					"CTA 1 sA",
					["m", "k"],
					1,
					mk,
					128,
					iK,
					(_, c) => (c === 0 ? 3 : -2),
					(_, c) => `${c}`,
					(_, c) =>
						c === 0
							? "A rows 128–255, read by CTA 1's tensor core."
							: "Later instruction.",
					[128, 0],
				),
				row: 1,
			},
			{
				...blocks(
					"CTA 1 sB",
					["n", "k"],
					1,
					mk,
					128,
					iK,
					(_, c) => (c === 0 ? 3 : -2),
					(_, c) => `${c}`,
					(_, c) =>
						c === 0
							? "B rows 128–255, read for both CTAs."
							: "Later instruction.",
					[128, 0],
				),
				row: 1,
			},
			{
				...blocks(
					"CTA 1 TMEM",
					["m", "n"],
					1,
					1,
					128,
					BN,
					() => 3,
					() => "D rows 128–255",
					() => "Accumulator for C rows 128–255.",
					[128, 0],
				),
				row: 1,
			},
		],
	});
	return steps;
}

function perm(): UmmaStep[] {
	// Atom m256 (2 CTAs × 128 rows), two atom tiles in M: 512 rows. Cells are 128 rows × 16 k.
	// Row ownership from thrfrg_A: default CTA 0 = {0–127, 256–383}; permuted CTA 0 = {0–255}.
	const owner = (permuted: boolean, r: number) => {
		// r = 128-row block of physical rows 0–511.
		if (!permuted) return { cta: r % 2, tile: Math.floor(r / 2) };
		return { cta: Math.floor(r / 2), tile: r % 2 };
	};
	const panel = (permuted: boolean): Panel =>
		blocks(
			permuted ? "with permutation_mnk" : "default",
			["m", "k"],
			4,
			1,
			128,
			16,
			(r) => (owner(permuted, r).cta === 0 ? 0 : 3),
			(r) => `CTA ${owner(permuted, r).cta} · MMA_M ${owner(permuted, r).tile}`,
			(r) => {
				const o = owner(permuted, r);
				return `CTA ${o.cta}, tCgA[None, ${o.tile}, 0]: instruction tile ${o.tile}.`;
			},
		);
	return [
		{
			title: "Two m256 tiles in M, default order",
			code: "tCgA = thr_mma.partition_A(gA)  # gA: 512 × 16, atom M = 256",
			shape: [
				"each CTA's tCgA: ((128,16), MMA_M = 2, 1)",
				"CTA 0: rows 0–127 and 256–383",
			],
			body: "A 2-CTA atom with M = 256 covers 256 rows: CTA 0 takes the top 128, CTA 1 the bottom 128. Repeating it twice in M (MMA_M = 2) without a permutation stacks the tiles, so each CTA owns two separated row ranges.",
			panels: [panel(false)],
			xStretch: 14,
		},
		{
			title: "Two m256 tiles in M, interleaved by permutation_mnk",
			code: "m_layout = cute.make_layout((128, 2, 2), stride=(1, 256, 128))",
			shape: [
				"each CTA's tCgA: ((128,16), MMA_M = 2, 1)",
				"CTA 0: rows 0–255, CTA 1: rows 256–511",
			],
			body: "The permutation maps logical row a + 128b + 256c to physical row a + 256b + 128c. Instruction tile 0 now reads physical rows 0–127 and 256–383, so CTA 0 owns the contiguous rows 0–255 and CTA 1 rows 256–511.",
			panels: [panel(false), panel(true)],
			xStretch: 14,
		},
	];
}

export function buildUmmaSteps(preset: UmmaPreset): UmmaStep[] {
	if (preset === "1cta") return oneCta();
	if (preset === "2cta") return twoCta();
	return perm();
}

const esc = (s: string) =>
	s
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;");

/**
 * One step as an SVG string. Panels share one scale (pixels per element), chosen so the widest
 * row fits `width` and all rows fit `maxH`. Cells carry their hover text in data-d and a class
 * ug0..ug5 (colour group), ugp (plain) or ugd (dimmed).
 */
export function stepSvg(step: UmmaStep, width = 680, maxH = 330): string {
	const gap = 22;
	const titleH = 16;
	const rowGap = 14;
	const rows = new Map<number, Panel[]>();
	for (const p of step.panels) {
		const r = p.row ?? 0;
		rows.set(r, [...(rows.get(r) ?? []), p]);
	}
	const ordered = [...rows.keys()]
		.sort((a, b) => a - b)
		.map((k) => rows.get(k) as Panel[]);
	const xs = step.xStretch ?? 1;
	const elemW = (p: Panel) => p.cols * p.cellW * xs;
	const elemH = (p: Panel) => p.rows * p.cellH;
	let scale = Number.POSITIVE_INFINITY;
	for (const r of ordered) {
		const free = width - gap * (r.length - 1);
		scale = Math.min(scale, free / r.reduce((s, p) => s + elemW(p), 0));
	}
	const fixedH = ordered.length * titleH + (ordered.length - 1) * rowGap;
	const totalElemH = ordered.reduce((s, r) => s + Math.max(...r.map(elemH)), 0);
	scale = Math.min(scale, (maxH - fixedH) / totalElemH);
	const parts: string[] = [];
	let y = 0;
	let usedW = 0;
	for (const r of ordered) {
		const rowW =
			r.reduce((s, p) => s + elemW(p) * scale, 0) + gap * (r.length - 1);
		usedW = Math.max(usedW, rowW);
		let x = 0;
		for (const p of r) {
			const cw = p.cellW * xs * scale;
			const ch = p.cellH * scale;
			parts.push(
				`<text class="ut-title" x="${x.toFixed(1)}" y="${(y + 11).toFixed(1)}">${esc(p.title)}</text>`,
			);
			for (let i = 0; i < p.rows; i++)
				for (let j = 0; j < p.cols; j++) {
					const g = p.group(i, j);
					const cls = g === -1 ? "ugp" : g === -2 ? "ugd" : `ug${g % 6}`;
					const cx = x + j * cw;
					const cy = y + titleH + i * ch;
					parts.push(
						`<rect class="${cls}" x="${cx.toFixed(1)}" y="${cy.toFixed(1)}" width="${cw.toFixed(1)}" height="${ch.toFixed(1)}" data-d="${esc(p.describe(i, j))}"/>`,
					);
					const label = p.label?.(i, j);
					if (label) {
						const size = Math.min(
							12,
							ch * 0.45,
							(cw / Math.max(1, label.length)) * 1.6,
						);
						if (size >= 7)
							parts.push(
								`<text class="ut-label" x="${(cx + cw / 2).toFixed(1)}" y="${(cy + ch / 2).toFixed(1)}" font-size="${size.toFixed(1)}">${esc(label)}</text>`,
							);
					}
				}
			x += elemW(p) * scale + gap;
		}
		y += titleH + Math.max(...r.map(elemH)) * scale + rowGap;
	}
	const h = y - rowGap;
	return `<svg viewBox="0 0 ${Math.ceil(usedW) + 1} ${Math.ceil(h) + 1}" role="img" aria-label="${esc(step.title)}">${parts.join("")}</svg>`;
}
