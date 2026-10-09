// The step-by-step schedule of one CTA of a tiled GEMM, C = A·B, on a GPU SM, the way a CuTe
// kernel runs it: copy a (BM×BK) tile of A and a (BK×BN) tile of B from global to shared
// memory, copy per-warp fragments from shared memory to registers, issue tensor-core MMAs into
// per-warp accumulators, and finally run the epilogue on the CUDA cores and store C.
//
// The global → shared copy runs either as per-thread cp.async (Ampere) or as TMA bulk tensor
// copies issued by one thread and tracked by an mbarrier (Hopper).
//
// Every step carries the full register state and, for one chosen thread, the elements that
// thread copies, holds or stores. It is pure computation, so it runs at build time (to check
// the configuration) and again in the browser whenever the reader picks another blockIdx or
// threadIdx. Layouts are CuTe layouts (see ../matrix/CuteLayout.ts).

import type { CuteLayout } from "../../component/Cute/CuteLayout";

export const WARP = 32;

/** CuTe's Swizzle<B, M, S>: XOR bits [M+S, M+S+B) of an offset into bits [M, M+B). */
export interface Swizzle {
	bits: number;
	base: number;
	shift: number;
}

export function swizzle(off: number, sw?: Swizzle): number {
	if (!sw) return off;
	const mask = ((1 << sw.bits) - 1) << (sw.base + sw.shift);
	return off ^ ((off & mask) >> sw.shift);
}

/** How tiles move from global to shared memory. */
export type CopyEngine = "cp.async" | "tma";

export interface GemmShape {
	M: number;
	N: number;
	K: number;
	/** CTA tile. */
	BM: number;
	BN: number;
	BK: number;
	/** Warps per CTA along M and N; warp w owns a (BM/warpsM)×(BN/warpsN) block of the C tile. */
	warpsM: number;
	warpsN: number;
	/** MMA atom: the shape one tensor-core instruction computes. */
	mmaM: number;
	mmaN: number;
	mmaK: number;
	/** The CTA followed: blockIdx.x walks tiles along M, blockIdx.y along N. */
	blockM: number;
	blockN: number;
	/** Elements per cp.async: 8 halves = 16 bytes. */
	vec: number;
	/** Global → shared copy engine; cp.async by default. */
	copy?: CopyEngine;
}

export interface GemmLayouts {
	/** Global memory: A is M×K, B is K×N, C is M×N. */
	gA: CuteLayout;
	gB: CuteLayout;
	gC: CuteLayout;
	/** Shared memory: sA is BM×BK, sB is BK×BN. */
	sA: CuteLayout;
	sB: CuteLayout;
	/** Applied to sA and sB offsets. */
	swizzle?: Swizzle;
}

export type CellGroup = "gA" | "gB" | "gC" | "sA" | "sB" | "sC" | "rA" | "rB" | "acc";

/** A rectangle of cells [r0, r1) × [c0, c1) in one group's logical coordinates. */
export interface Highlight {
	g: CellGroup;
	/** Only this warp's registers; all warps when omitted. */
	w?: number;
	r: [number, number];
	c: [number, number];
}

/** r2g stores C straight from registers; with TMA it goes r2s (registers → sC) then s2g (TMA store). */
export type StepKind = "init" | "g2s" | "sync" | "s2r" | "mma" | "epilogue" | "r2g" | "r2s" | "s2g";

export interface GemmStep {
	kind: StepKind;
	/** The instruction, for badges: "cp.async", "TMA", "ldmatrix", … */
	op: string;
	/** Lines of CODE this step executes. */
	lines: number[];
	loop: string;
	title: string;
	text: string;
	/** Hardware units busy in this step, e.g. "smem", "tc2" (tensor core of warp 2). */
	units: string[];
	/** Data paths in use, e.g. "g2l2", "s2r0". */
	flows: string[];
	/** Cells the whole CTA touches. */
	hl: Highlight[];
	/** Cells the chosen thread touches, and what it does. */
	mine: Highlight[];
	mineText: string;
	/** The k-tile resident in shared memory, or -1 before the first copy. */
	smemTile: number;
	/** Shared-memory caption, e.g. "sA ← A[8:16, 0:8] · sB ← B[0:8, 0:8]". */
	smemLabel: string;
	/** Per warp: rA, rB and acc flattened row-major, in that order; null = not loaded yet. */
	regs: (number | null)[][];
	/** C has been written back to global memory. */
	stored: boolean;
	/** The finished C tile sits in shared memory (sC), waiting for the TMA store. */
	cStaged: boolean;
}

/** The kernel's main loop for a copy engine, and the lines each kind of step executes. */
export function kernelCode(copy: CopyEngine = "cp.async"): { code: string[]; lines: Record<StepKind, number[]> } {
	if (copy === "tma") {
		return {
			code: [
				"clear(acc);                              // acc ← 0",
				"for (int kt = 0; kt < K/BK; ++kt) {",
				"  if (threadIdx.x == 0) {                // one elected thread",
				"    arrive_expect_tx(bar, bytes);        // arm the mbarrier",
				"    copy(tma_a.with(bar), gA(_, _, kt), sA);  // TMA",
				"    copy(tma_b.with(bar), gB(_, _, kt), sB);",
				"  }",
				"  wait_barrier(bar, kt % 2);             // every thread",
				"  for (int kk = 0; kk < BK/MMA_K; ++kk) {",
				"    copy(s2r, sA(_, kk), rA);            // ldmatrix",
				"    copy(s2r, sB(kk, _), rB);",
				"    gemm(mma, rA, rB, acc);              // mma.sync",
				"  }",
				"  __syncthreads();",
				"}",
				"axpby(alpha, acc, beta, rC);             // CUDA cores",
				"copy(r2s, rC, sC);                       // stmatrix",
				"fence_view_async_shared(); __syncthreads();",
				"if (threadIdx.x == 0) {",
				"  copy(tma_c, sC, gC);                   // TMA store",
				"  tma_store_arrive(); tma_store_wait<0>();",
				"}",
			],
			lines: {
				init: [0],
				g2s: [2, 3, 4, 5],
				sync: [7],
				s2r: [9, 10],
				mma: [11],
				epilogue: [15],
				r2g: [],
				r2s: [16, 17],
				s2g: [18, 19, 20],
			},
		};
	}
	return {
		code: [
			"clear(acc);                              // acc ← 0",
			"for (int kt = 0; kt < K/BK; ++kt) {",
			"  copy(g2s, gA(_, _, kt), sA);           // cp.async",
			"  copy(g2s, gB(_, _, kt), sB);",
			"  cp_async_wait<0>(); __syncthreads();",
			"  for (int kk = 0; kk < BK/MMA_K; ++kk) {",
			"    copy(s2r, sA(_, kk), rA);            // ldmatrix",
			"    copy(s2r, sB(kk, _), rB);",
			"    gemm(mma, rA, rB, acc);              // mma.sync",
			"  }",
			"  __syncthreads();",
			"}",
			"axpby(alpha, acc, beta, rC);             // CUDA cores",
			"copy(r2g, rC, gC);                       // st.global",
		],
		lines: { init: [0], g2s: [2, 3], sync: [4], s2r: [6, 7], mma: [8], epilogue: [12], r2g: [13], r2s: [], s2g: [] },
	};
}

/** Bytes per element: FP16 operands. */
const ELEM_BYTES = 2;

/** The m16n8k8 / m16n8k16 atoms, whose fragment layouts PTX specifies. */
export const isPtxAtom = (s: Pick<GemmShape, "mmaM" | "mmaN" | "mmaK">) =>
	s.mmaM === 16 && s.mmaN === 8 && (s.mmaK === 8 || s.mmaK === 16);

/**
 * The lane holding element (r, c) of a warp's A (rows × K), B (K × cols) or C fragment. For the
 * PTX atoms, lane = 4·groupID + threadID_in_group, where a group of 4 lanes shares a row of A/C
 * (a column of B) and each lane holds a pair of adjacent elements. Other atoms deal elements
 * round-robin.
 */
export function laneOf(frag: "A" | "B" | "C", r: number, c: number, cols: number, shape: GemmShape): number {
	if (!isPtxAtom(shape)) return (r * cols + c) % WARP;
	if (frag === "B") return (c % 8) * 4 + ((r % 8) >> 1);
	return (r % 8) * 4 + ((c % 8) >> 1);
}

/** Where each tile element sits in shared memory: sA first, then sB. */
export interface SmemMap {
	sA: number[][];
	sB: number[][];
	/** The C tile staged for the TMA store (TMA only), row-major after sB. */
	sC?: number[][];
	size: number;
}

export function smemMap(shape: GemmShape, layouts: GemmLayouts): SmemMap {
	const { BM, BN, BK } = shape;
	const place = (rows: number, cols: number, layout: CuteLayout, base: number) =>
		Array.from({ length: rows }, (_, i) =>
			Array.from({ length: cols }, (_, j) => base + swizzle(layout.offset(i, j), layouts.swizzle)),
		);
	const sA = place(BM, BK, layouts.sA, 0);
	const sB = place(BK, BN, layouts.sB, Math.max(...sA.flat()) + 1);
	const sBEnd = Math.max(...sB.flat()) + 1;
	const sC =
		shape.copy === "tma"
			? Array.from({ length: BM }, (_, i) => Array.from({ length: BN }, (_, j) => sBEnd + i * BN + j))
			: undefined;
	const all = [...sA.flat(), ...sB.flat(), ...(sC?.flat() ?? [])];
	if (new Set(all).size !== all.length) {
		throw new Error("Shared-memory layouts map two tile elements to the same offset");
	}
	return { sA, sB, sC, size: Math.max(...all) + 1 };
}

type Coord = [number, number];

/** Number of contiguous runs the given offsets form: 1 means one coalesced block. */
function runs(offsets: number[]): number {
	const o = [...offsets].sort((a, b) => a - b);
	return o.reduce((n, v, i) => (i > 0 && v !== o[i - 1] + 1 ? n + 1 : n), 1);
}

/** A tile's elements in address order, cut into cp.async vectors of up to `vec` contiguous elements. */
function vectors(rows: number, cols: number, r0: number, c0: number, layout: CuteLayout, vec: number): Coord[][] {
	const cells: { at: Coord; off: number }[] = [];
	for (let i = 0; i < rows; i++)
		for (let j = 0; j < cols; j++) cells.push({ at: [r0 + i, c0 + j], off: layout.offset(r0 + i, c0 + j) });
	cells.sort((a, b) => a.off - b.off);
	const out: Coord[][] = [];
	cells.forEach((cell, n) => {
		const last = out[out.length - 1];
		if (n > 0 && last.length < vec && cell.off === cells[n - 1].off + 1) last.push(cell.at);
		else out.push([cell.at]);
	});
	return out;
}

const span = (start: number, len: number) => `${start}:${start + len}`;

/** "A(16, 0:8), A(17, 0:2)": runs along rows or columns, whichever is shorter. */
export function formatCells(name: string, cells: Coord[], max = 6): string {
	if (!cells.length) return "nothing";
	const group = (major: 0 | 1) => {
		const sorted = [...cells].sort((a, b) => a[major] - b[major] || a[1 - major] - b[1 - major]);
		const parts: { fixed: number; from: number; to: number }[] = [];
		for (const c of sorted) {
			const last = parts[parts.length - 1];
			if (last && last.fixed === c[major] && last.to === c[1 - major]) last.to++;
			else parts.push({ fixed: c[major], from: c[1 - major], to: c[1 - major] + 1 });
		}
		return parts.map(({ fixed, from, to }) => {
			const run = to - from > 1 ? `${from}:${to}` : `${from}`;
			return major === 0 ? `${name}(${fixed}, ${run})` : `${name}(${run}, ${fixed})`;
		});
	};
	const byRow = group(0);
	const byCol = group(1);
	const parts = byCol.length < byRow.length ? byCol : byRow;
	return parts.length > max ? `${parts.slice(0, max).join(", ")}, … (${cells.length} in all)` : parts.join(", ");
}

export function buildSchedule(shape: GemmShape, layouts: GemmLayouts, A: number[][], B: number[][], thread = 0) {
	const { M, N, K, BM, BN, BK, warpsM, warpsN, mmaM, mmaN, mmaK, blockM, blockN, vec, copy = "cp.async" } = shape;
	const tma = copy === "tma";
	const { lines } = kernelCode(copy);
	const check = (ok: boolean, msg: string) => {
		if (!ok) throw new Error(msg);
	};
	check(M % BM === 0 && N % BN === 0 && K % BK === 0, "M, N, K must be multiples of BM, BN, BK");
	check(BM % warpsM === 0 && BN % warpsN === 0, "BM, BN must be multiples of warpsM, warpsN");
	const WM = BM / warpsM;
	const WN = BN / warpsN;
	check(WM % mmaM === 0 && WN % mmaN === 0 && BK % mmaK === 0, "Warp tile and BK must be multiples of the MMA atom");
	check(blockM >= 0 && blockM * BM < M && blockN >= 0 && blockN * BN < N, "blockIdx out of range");

	const warps = warpsM * warpsN;
	const threads = warps * WARP;
	check(thread >= 0 && thread < threads, `threadIdx.x must be in [0, ${threads})`);
	const kTiles = K / BK;
	const kSteps = BK / mmaK;
	const atoms = (WM / mmaM) * (WN / mmaN);
	const row0 = blockM * BM;
	const col0 = blockN * BN;
	const warpPos = (w: number) => ({ wm: Math.floor(w / warpsN), wn: w % warpsN });
	const smem = smemMap(shape, layouts);

	// ---- The chosen thread: its warp, lane and fragment elements (warp-local coordinates).
	const warp = Math.floor(thread / WARP);
	const lane = thread % WARP;
	const { wm: twm, wn: twn } = warpPos(warp);
	const owned = (frag: "A" | "B" | "C", rows: number, cols: number) => {
		const out: Coord[] = [];
		for (let r = 0; r < rows; r++)
			for (let c = 0; c < cols; c++) if (laneOf(frag, r, c, cols, shape) === lane) out.push([r, c]);
		return out;
	};
	const myA = owned("A", WM, mmaK);
	const myB = owned("B", mmaK, WN);
	const myC = owned("C", WM, WN);
	const cellsOf = (g: CellGroup, cells: Coord[], w?: number): Highlight[] =>
		cells.map(([r, c]) => ({ g, w, r: [r, r + 1], c: [c, c + 1] }));
	const myCGlobal = myC.map(([r, c]): Coord => [row0 + twm * WM + r, col0 + twn * WN + c]);
	const who = `Thread ${thread} is lane ${lane} of warp ${warp}`;

	const rA: (number | null)[][] = Array.from({ length: warps }, () => Array(WM * mmaK).fill(null));
	const rB: (number | null)[][] = Array.from({ length: warps }, () => Array(mmaK * WN).fill(null));
	const acc: (number | null)[][] = Array.from({ length: warps }, () => Array(WM * WN).fill(null));
	const snapshot = () => rA.map((a, w) => [...a, ...rB[w], ...acc[w]]);

	const all = (prefix: string) => Array.from({ length: warps }, (_, w) => `${prefix}${w}`);
	const steps: GemmStep[] = [];
	let smemTile = -1;
	let smemLabel = "empty";
	let stored = false;
	let cStaged = false;
	const push = (s: Omit<GemmStep, "lines" | "regs" | "smemTile" | "smemLabel" | "stored" | "cStaged">) =>
		steps.push({ ...s, lines: lines[s.kind], regs: snapshot(), smemTile, smemLabel, stored, cStaged });

	// ---- Prologue.
	for (let w = 0; w < warps; w++) acc[w].fill(0);
	push({
		kind: "init",
		op: "Setup",
		loop: `blockIdx (${blockM}, ${blockN})`,
		title: "Zero the accumulators",
		text:
			`This CTA computes C[${span(row0, BM)}, ${span(col0, BN)}], so it needs the whole ${BM}-row panel of A and ` +
			`${BN}-column panel of B. Its ${warps} warps (${threads} threads) split the tile ${warpsM}×${warpsN}; each ` +
			`warp owns a ${WM}×${WN} block of C and keeps it in registers as FP32 accumulators, zeroed here.` +
			(tma ? " Thread 0 has also initialised an mbarrier in shared memory for the TMA copies to signal." : ""),
		units: all("rf"),
		flows: [],
		hl: [
			{ g: "gA", r: [row0, row0 + BM], c: [0, K] },
			{ g: "gB", r: [0, K], c: [col0, col0 + BN] },
			{ g: "gC", r: [row0, row0 + BM], c: [col0, col0 + BN] },
			{ g: "acc", r: [0, WM], c: [0, WN] },
		],
		mine: cellsOf("acc", myC, warp),
		mineText:
			`${who}. It holds ${myC.length} of the warp's ${WM * WN} accumulators, the ones that become ` +
			`${formatCells("C", myCGlobal)}.`,
	});

	// ---- Main loop.
	for (let kt = 0; kt < kTiles; kt++) {
		const k0 = kt * BK;
		smemTile = kt;
		smemLabel = `sA ← A[${span(row0, BM)}, ${span(k0, BK)}] · sB ← B[${span(k0, BK)}, ${span(col0, BN)}]`;
		const vA = vectors(BM, BK, row0, k0, layouts.gA, vec);
		const vB = vectors(BK, BN, k0, col0, layouts.gB, vec);
		const vAll = [...vA.map((v) => ({ g: "A" as const, v })), ...vB.map((v) => ({ g: "B" as const, v }))];
		const mineV = vAll.filter((_, n) => n % threads === thread);
		const mA = mineV.filter((x) => x.g === "A").flatMap((x) => x.v);
		const mB = mineV.filter((x) => x.g === "B").flatMap((x) => x.v);
		const aOff = vA.flat().map(([r, c]) => layouts.gA.offset(r, c));
		const bOff = vB.flat().map(([r, c]) => layouts.gB.offset(r, c));
		const runText = (n: number, total: number) =>
			n === 1 ? "one contiguous block" : `${n} contiguous runs of ${total / n}`;
		const tiles = `A[${span(row0, BM)}, ${span(k0, BK)}] and B[${span(k0, BK)}, ${span(col0, BN)}]`;
		const layoutText =
			`Under gA = ${layouts.gA} the A tile is ${runText(runs(aOff), BM * BK)}; under gB = ${layouts.gB} the B tile ` +
			`is ${runText(runs(bOff), BK * BN)}.`;
		const landText =
			`Each element (i, j) lands at sA(i, j) = ${layouts.sA} or sB(i, j) = ${layouts.sB}` +
			`${layouts.swizzle ? (tma ? ", swizzled by the TMA as it writes" : ", then swizzled") : ""}.`;
		const reuse = kt > 0 ? " The barrier at the end of the previous iteration made sure no warp still reads the old tile." : "";
		const tileHl: Highlight[] = [
			{ g: "gA", r: [row0, row0 + BM], c: [k0, k0 + BK] },
			{ g: "gB", r: [k0, k0 + BK], c: [col0, col0 + BN] },
			{ g: "sA", r: [0, BM], c: [0, BK] },
			{ g: "sB", r: [0, BK], c: [0, BN] },
		];
		const bytes = (BM * BK + BK * BN) * ELEM_BYTES;
		if (tma) {
			push({
				kind: "g2s",
				op: "TMA",
				loop: `kt = ${kt}`,
				title: "Global → shared memory with TMA",
				text:
					`Thread 0 alone arms the mbarrier with the ${bytes} bytes it expects, then issues two cp.async.bulk.tensor ` +
					`copies, one for ${tiles}. Each is described by a tensor map (base address, gA/gB shape and strides, box size) ` +
					`built on the host. The Tensor Memory Accelerator generates every address itself and streams the tiles through ` +
					`L2 into shared memory; no other thread spends instructions or registers on the copy, and the mbarrier counts ` +
					`bytes as they land. ${layoutText} ${landText}${reuse}`,
				units: ["gmem", "l2", "tma", "smem", "mbar"],
				flows: ["issue", "g2l2", "l22t", "t2s"],
				hl: tileHl,
				mine: [],
				mineText:
					thread === 0
						? `Thread ${thread} is the elected thread: it arms the mbarrier and issues both TMA copies, ` +
							`${BM * BK + BK * BN} elements in two instructions, then moves on.`
						: `Thread ${thread} issues nothing: one thread starts the whole copy, where cp.async would have taken ` +
							`${vAll.length} instructions spread over the CTA's threads.`,
			});
			push({
				kind: "sync",
				op: "mbarrier",
				loop: `kt = ${kt}`,
				title: "Wait on the mbarrier",
				text:
					`Every thread waits on the mbarrier until its transaction count reaches the expected ${bytes} bytes, i.e. ` +
					`both tiles have landed in shared memory. The phase bit (kt % 2 = ${kt % 2}) tells this wait apart from the ` +
					"next k-tile's, so the same barrier is reused every iteration.",
				units: ["smem", "mbar"],
				flows: [],
				hl: [],
				mine: [],
				mineText: `Thread ${thread} waits on the mbarrier, phase ${kt % 2}.`,
			});
		} else {
			push({
				kind: "g2s",
				op: "cp.async",
				loop: `kt = ${kt}`,
				title: "Global → shared memory",
				text:
					`cp.async: the CTA's threads copy ${tiles} from global memory, through L2, into shared memory without ` +
					`passing through registers. ${layoutText} That makes ${vAll.length} copies of up to ${vec} elements ` +
					`(16 B), dealt to consecutive threads so a warp reads consecutive addresses. ${landText}${reuse}`,
				units: ["gmem", "l2", "smem"],
				flows: ["g2l2", "l22s"],
				hl: tileHl,
				mine: [
					...cellsOf("gA", mA),
					...cellsOf("gB", mB),
					...cellsOf(
						"sA",
						mA.map(([r, c]) => [r - row0, c - k0]),
					),
					...cellsOf(
						"sB",
						mB.map(([r, c]) => [r - k0, c - col0]),
					),
				],
				mineText: mineV.length
					? `Thread ${thread} issues ${mineV.length} cp.async: ${[formatCells("A", mA), formatCells("B", mB)].filter((s) => s !== "nothing").join(" and ")}.`
					: `Thread ${thread} has nothing to copy: ${vAll.length} copies for ${threads} threads.`,
			});
			push({
				kind: "sync",
				op: "Barrier",
				loop: `kt = ${kt}`,
				title: "Wait and barrier",
				text:
					"cp.async is asynchronous: each thread waits for its own copies to land, then __syncthreads() makes every " +
					"thread's copies visible to every warp, because a warp is about to read elements other threads copied.",
				units: ["smem"],
				flows: [],
				hl: [],
				mine: [],
				mineText: `Thread ${thread} waits for its own copies, then at the barrier for all ${threads} threads.`,
			});
		}

		for (let kk = 0; kk < kSteps; kk++) {
			const kk0 = kk * mmaK;
			const kg = k0 + kk0;
			for (let w = 0; w < warps; w++) {
				const { wm, wn } = warpPos(w);
				for (let i = 0; i < WM; i++)
					for (let k = 0; k < mmaK; k++) rA[w][i * mmaK + k] = A[row0 + wm * WM + i][kg + k];
				for (let k = 0; k < mmaK; k++)
					for (let j = 0; j < WN; j++) rB[w][k * WN + j] = B[kg + k][col0 + wn * WN + j];
			}
			const srcA = myA.map(([r, c]): Coord => [twm * WM + r, kk0 + c]);
			const srcB = myB.map(([r, c]): Coord => [kk0 + r, twn * WN + c]);
			push({
				kind: "s2r",
				op: "ldmatrix",
				loop: `kt = ${kt} · kk = ${kk}`,
				title: "Shared memory → registers",
				text:
					`ldmatrix: each warp loads the slice it needs for this MMA: ${WM}×${mmaK} of sA (columns ${span(kk0, mmaK)}) ` +
					`for its rows of C, and ${mmaK}×${WN} of sB for its columns, spread across its 32 lanes' registers. Warps in ` +
					`the same row of the warp grid read the same A slice, so shared memory serves each element ${warpsN}× (A) ` +
					`or ${warpsM}× (B) while global memory delivered it once.`,
				units: ["smem", ...all("rf")],
				flows: all("s2r"),
				hl: [
					{ g: "sA", r: [0, BM], c: [kk0, kk0 + mmaK] },
					{ g: "sB", r: [kk0, kk0 + mmaK], c: [0, BN] },
					{ g: "rA", r: [0, WM], c: [0, mmaK] },
					{ g: "rB", r: [0, mmaK], c: [0, WN] },
				],
				mine: [...cellsOf("sA", srcA), ...cellsOf("sB", srcB), ...cellsOf("rA", myA, warp), ...cellsOf("rB", myB, warp)],
				mineText:
					`${who}. It receives ${myA.length} A values, ${formatCells("sA", srcA)}, and ${myB.length} B values, ` +
					`${formatCells("sB", srcB)}${isPtxAtom(shape) ? `, in the m16n8k${mmaK} fragment layout PTX defines` : ""}.`,
			});

			for (let w = 0; w < warps; w++) {
				for (let i = 0; i < WM; i++)
					for (let j = 0; j < WN; j++) {
						let s = acc[w][i * WN + j] ?? 0;
						for (let k = 0; k < mmaK; k++) s += (rA[w][i * mmaK + k] ?? 0) * (rB[w][k * WN + j] ?? 0);
						acc[w][i * WN + j] = s;
					}
			}
			push({
				kind: "mma",
				op: "mma.sync",
				loop: `kt = ${kt} · kk = ${kk}`,
				title: "Tensor-core MMA",
				text:
					`mma.sync: every warp's tensor core computes acc += rA·rB, a ${WM}×${mmaK} by ${mmaK}×${WN} product ` +
					`(${atoms} m${mmaM}n${mmaN}k${mmaK} instruction${atoms > 1 ? "s" : ""}, ${WM * WN * mmaK} multiply-adds per warp). ` +
					`The ${warps} warps run on ${warps} SM sub-partitions at the same time, each with its own tensor core. ` +
					`Operands and accumulators stay in registers; the CUDA cores are idle.`,
				units: [...all("tc"), ...all("rf")],
				flows: all("r2tc"),
				hl: [
					{ g: "rA", r: [0, WM], c: [0, mmaK] },
					{ g: "rB", r: [0, mmaK], c: [0, WN] },
					{ g: "acc", r: [0, WM], c: [0, WN] },
				],
				mine: [...cellsOf("rA", myA, warp), ...cellsOf("rB", myB, warp), ...cellsOf("acc", myC, warp)],
				mineText:
					`Thread ${thread} supplies its ${myA.length} A and ${myB.length} B values and gets back its ${myC.length} ` +
					`updated accumulators. The tensor core works on all 32 lanes' registers at once: no thread computes a dot ` +
					`product on its own.`,
			});
		}
	}

	// ---- Epilogue.
	push({
		kind: "epilogue",
		op: "Epilogue",
		loop: "epilogue",
		title: "Epilogue on the CUDA cores",
		text:
			"The K loop is done and every accumulator holds a finished dot product. The elementwise epilogue, " +
			"D = α·acc + β·C (here α = 1, β = 0), plus any bias, activation or conversion to the output type, runs on the " +
			"ordinary FP32 CUDA cores: tensor cores only do matrix multiply-accumulate." +
			(tma ? " With TMA, the result then leaves through shared memory instead of straight from registers." : ""),
		units: [...all("cc"), ...all("rf")],
		flows: all("r2cc"),
		hl: [{ g: "acc", r: [0, WM], c: [0, WN] }],
		mine: cellsOf("acc", myC, warp),
		mineText: `Thread ${thread} runs the epilogue on its own ${myC.length} accumulators, one CUDA-core lane each instruction.`,
	});
	const cOff: number[] = [];
	for (let i = 0; i < BM; i++) for (let j = 0; j < BN; j++) cOff.push(layouts.gC.offset(row0 + i, col0 + j));
	const cTile = `C[${span(row0, BM)}, ${span(col0, BN)}]`;
	const reuseText =
		`Each element of A and B was read from global memory once per CTA and reused ${BN}× or ${BM}× out of shared ` +
		"memory and registers; that reuse is the point of tiling.";
	if (tma) {
		// Local C-tile coordinates of the chosen thread's accumulators.
		const myCTile = myC.map(([r, c]): Coord => [twm * WM + r, twn * WN + c]);
		cStaged = true;
		smemLabel = `sA, sB: last k-tile · sC ← acc, ${cTile}`;
		push({
			kind: "r2s",
			op: "stmatrix",
			loop: "epilogue",
			title: "Registers → shared memory",
			text:
				`stmatrix: every thread writes its accumulators into sC, a ${BM}×${BN} tile in shared memory ` +
				`((${BM},${BN}):(${BN},1)), so the whole C tile sits in one place for a single TMA store. Then ` +
				"fence.proxy.async makes these ordinary shared-memory writes visible to the TMA, which reads shared memory " +
				"through the separate async proxy, and __syncthreads() waits until every warp has written its part.",
			units: ["smem", ...all("rf")],
			flows: all("r2s"),
			hl: [
				{ g: "acc", r: [0, WM], c: [0, WN] },
				{ g: "sC", r: [0, BM], c: [0, BN] },
			],
			mine: [...cellsOf("acc", myC, warp), ...cellsOf("sC", myCTile)],
			mineText: `Thread ${thread} writes its ${myC.length} accumulators to ${formatCells("sC", myCTile)}, then fences and waits at the barrier.`,
		});
		stored = true;
		push({
			kind: "s2g",
			op: "TMA store",
			loop: "epilogue",
			title: "Shared memory → global memory with TMA",
			text:
				`Thread 0 issues one cp.async.bulk.tensor from sC to ${cTile} through the C tensor map (gC = ${layouts.gC}), ` +
				"then commits the bulk group and waits for it before the CTA exits, so shared memory isn't released while the " +
				`TMA still reads it. The TMA writes the ${BM * BN} elements (${runs(cOff)} contiguous runs) through L2; the ` +
				`other threads are already done. ${reuseText}`,
			units: ["smem", "tma", "l2", "gmem"],
			flows: ["issue", "s2t", "t2l", "l2g"],
			hl: [
				{ g: "sC", r: [0, BM], c: [0, BN] },
				{ g: "gC", r: [row0, row0 + BM], c: [col0, col0 + BN] },
			],
			mine: [],
			mineText:
				thread === 0
					? `Thread ${thread} is the elected thread: it issues the TMA store of all ${BM * BN} elements and waits for it to finish.`
					: `Thread ${thread} issues nothing: its accumulators already went to sC, and thread 0 stores the whole tile.`,
		});
	} else {
		stored = true;
		push({
			kind: "r2g",
			op: "st.global",
			loop: "epilogue",
			title: "Registers → global memory",
			text:
				`st.global: each thread writes its accumulator elements to ${cTile} ` +
				`(gC = ${layouts.gC}, ${runs(cOff)} contiguous runs). ${reuseText}`,
			units: ["gmem", ...all("cc")],
			flows: ["cc2g"],
			hl: [{ g: "gC", r: [row0, row0 + BM], c: [col0, col0 + BN] }],
			mine: [...cellsOf("acc", myC, warp), ...cellsOf("gC", myCGlobal)],
			mineText: `Thread ${thread} stores ${formatCells("C", myCGlobal)}.`,
		});
	}

	// The C tile the kernel produced, checked against a direct A·B.
	const C = Array.from({ length: BM }, (_, i) =>
		Array.from({ length: BN }, (_, j) => {
			const w = Math.floor(i / WM) * warpsN + Math.floor(j / WN);
			return acc[w][(i % WM) * WN + (j % WN)] ?? 0;
		}),
	);
	for (let i = 0; i < BM; i++)
		for (let j = 0; j < BN; j++) {
			let s = 0;
			for (let k = 0; k < K; k++) s += A[row0 + i][k] * B[k][col0 + j];
			check(s === C[i][j], `Schedule produced C(${row0 + i}, ${col0 + j}) = ${C[i][j]}, expected ${s}`);
		}

	return { steps, smem, C, WM, WN, warps, threads, row0, col0, thread: { warp, lane } };
}
