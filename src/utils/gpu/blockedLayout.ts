// Logic for BlockedLayout.astro: which thread of which warp holds each element of a 1-D tensor
// under a Triton #ttg.blocked layout, and how wide the loads of a masked tl.load can be.
//
// A 1-D blocked layout {sizePerThread = [s], threadsPerWarp = [32], warpsPerCTA = [w]} tiles the
// tensor with blocks of s * 32 * w elements: thread t = 32 * warp + lane holds the s contiguous
// elements starting at s * t of every tile. A tensor longer than one tile repeats the pattern, so a
// thread holds BLOCK / tile groups; a tensor shorter than one tile is broadcast, so several threads
// hold the same element.
//
// The vector width follows LoadOpConversion in
// third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp at Triton 3.8.0:
//   vec = min(128 / elementBits, contiguity(ptr), maskAlignment(mask))
// where contiguity(ptr) = min(sizePerThread, alignment of the pointer in elements) and the mask
// alignment is its constancy from the axis analysis. For offsets < n_elements with offsets a run of
// BLOCK contiguous, BLOCK-aligned integers, the constancy is gcd(BLOCK, divisibility(n)), and Triton
// only knows divisibility(n) = 16 (when n % 16 == 0 at launch) or 1.

export interface BlockedConfig {
	/** Elements in the tensor (BLOCK_SIZE). */
	block: number;
	sizePerThread: number;
	threadsPerWarp: number;
	warps: number;
	/** Bits per element (32 for fp32). */
	elementBits: number;
}

export interface Group {
	/** First element of the group. */
	start: number;
	/** Elements in the group (= sizePerThread). */
	size: number;
}

export interface ThreadInfo {
	thread: number;
	warp: number;
	lane: number;
	groups: Group[];
}

/**
 * The sizePerThread TritonGPUCoalesce picks for a contiguous fp32 load: as wide as a 128-bit access,
 * but no wider than the tensor allows without broadcasting. Checked against Triton 3.8.0 for BLOCK
 * 128 to 2048 and 1 to 8 warps. Before Coalesce (ConvertTritonToTritonGPU) it is 1.
 */
export function coalescedSizePerThread(
	block: number,
	warps: number,
	elementBits: number,
	threadsPerWarp = 32,
): number {
	return Math.max(
		1,
		Math.min(128 / elementBits, Math.floor(block / (threadsPerWarp * warps))),
	);
}

export const tileSize = (c: BlockedConfig) =>
	c.sizePerThread * c.threadsPerWarp * c.warps;

export const threadCount = (c: BlockedConfig) => c.threadsPerWarp * c.warps;

/** The groups of contiguous elements thread t holds, in register order. */
export function threadInfo(c: BlockedConfig, thread: number): ThreadInfo {
	const tile = tileSize(c);
	const reps = Math.max(1, c.block / tile);
	const groups: Group[] = [];
	for (let r = 0; r < reps; r++) {
		const start = (r * tile + thread * c.sizePerThread) % c.block;
		groups.push({ start, size: c.sizePerThread });
	}
	return {
		thread,
		warp: Math.floor(thread / c.threadsPerWarp),
		lane: thread % c.threadsPerWarp,
		groups,
	};
}

/** For every element, the threads that hold it (one unless the layout is broadcast). */
export function owners(c: BlockedConfig): number[][] {
	const out: number[][] = Array.from({ length: c.block }, () => []);
	for (let t = 0; t < threadCount(c); t++) {
		for (const g of threadInfo(c, t).groups) {
			for (let v = 0; v < g.size; v++) out[g.start + v].push(t);
		}
	}
	return out;
}

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));

export interface VectorWidth {
	/** 128 / elementBits: the widest global access is 16 bytes. */
	maxVec: number;
	/** min(sizePerThread, pointer alignment in elements). */
	contiguity: number;
	/** Constancy of offsets < n over the contiguous dimension. */
	maskAlignment: number;
	vec: number;
}

export function vectorWidth(
	c: BlockedConfig,
	nElements: number,
	ptrAligned16: boolean,
): VectorWidth {
	const elemBytes = c.elementBits / 8;
	const maxVec = Math.max(1, 128 / c.elementBits);
	const ptrAlign = ptrAligned16 ? Math.max(1, 16 / elemBytes) : 1;
	const contiguity = Math.min(c.sizePerThread, ptrAlign);
	const nDivisibility = nElements % 16 === 0 ? 16 : 1;
	const maskAlignment = gcd(c.block, nDivisibility);
	return {
		maxVec,
		contiguity,
		maskAlignment,
		vec: Math.min(maxVec, contiguity, maskAlignment),
	};
}

export interface Access {
	/** First element read. */
	start: number;
	width: number;
	/** False when the predicate turns the access off (start >= n). */
	active: boolean;
}

/** The global loads thread t issues for one tensor, each `vec` elements wide. */
export function threadAccesses(
	c: BlockedConfig,
	thread: number,
	vec: number,
	nElements: number,
): Access[] {
	const out: Access[] = [];
	for (const g of threadInfo(c, thread).groups) {
		for (let v = 0; v < g.size; v += vec) {
			const start = g.start + v;
			out.push({ start, width: vec, active: start < nElements });
		}
	}
	return out;
}

/** PTX-like lines for thread t's load of x, in the shape Triton emits (see tritonJitData.ts). */
export function ptxLines(
	c: BlockedConfig,
	thread: number,
	vec: number,
	nElements: number,
): string[] {
	const bytes = c.elementBits / 8;
	let reg = 1;
	return threadAccesses(c, thread, vec, nElements).map((a, i) => {
		const regs = Array.from({ length: a.width }, () => `%r${reg++}`).join(", ");
		const op =
			a.width > 1
				? `ld.global.v${a.width}.b${c.elementBits}`
				: `ld.global.b${c.elementBits}`;
		const off = a.start * bytes;
		return `@%p${i + 1} ${op} { ${regs} }, [ x + ${off} ];${a.active ? "" : "   // off: offset ≥ n"}`;
	});
}

/** The layout attribute as Triton prints it. */
export function layoutString(c: BlockedConfig): string {
	return `#ttg.blocked<{sizePerThread = [${c.sizePerThread}], threadsPerWarp = [${c.threadsPerWarp}], warpsPerCTA = [${c.warps}], order = [0]}>`;
}

/** Grid of the tensor's elements, 32 per row, one <rect> per element with its owner in data attributes. */
export function drawGrid(c: BlockedConfig, nElements: number): string {
	const cols = 32;
	const cell = 14;
	const left = 34;
	const top = 16;
	const rows = Math.ceil(c.block / cols);
	const own = owners(c);
	const w = left + cols * cell + 2;
	const h = top + rows * cell + 2;
	const parts: string[] = [];
	for (let col = 0; col < cols; col += 4) {
		parts.push(
			`<text class="blay-axis" x="${left + col * cell + cell / 2}" y="${top - 5}">${col}</text>`,
		);
	}
	for (let r = 0; r < rows; r++) {
		parts.push(
			`<text class="blay-axis blay-row" x="${left - 5}" y="${top + r * cell + cell / 2 + 3}">${r * cols}</text>`,
		);
	}
	for (let i = 0; i < c.block; i++) {
		const t = own[i][0];
		const warp = Math.floor(t / c.threadsPerWarp);
		const lane = t % c.threadsPerWarp;
		const x = left + (i % cols) * cell;
		const y = top + Math.floor(i / cols) * cell;
		const cls = [
			`w${warp % 8}`,
			lane % 2 ? "odd" : "",
			i >= nElements ? "off" : "",
		]
			.filter(Boolean)
			.join(" ");
		parts.push(
			`<rect class="${cls}" x="${x + 0.5}" y="${y + 0.5}" width="${cell - 1}" height="${cell - 1}" rx="2" data-i="${i}" data-t="${own[i].join(" ")}"><title>element ${i}: thread ${own[i].join(", ")}</title></rect>`,
		);
	}
	return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="Blocked layout of ${c.block} elements over ${c.warps} warps">${parts.join("")}</svg>`;
}
