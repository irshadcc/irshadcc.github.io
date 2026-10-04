// What one warp's memory request costs, for WarpAccess.astro. Two models, both per request
// (one warp instruction, 32 threads):
//
//  - Global memory: the L1 cache is accessed in 32-byte sectors, four to a 128-byte cache line.
//    A request costs one sector per distinct 32-byte block it touches. Nsight Compute reports
//    the average as "Sectors/Req" in the L1/TEX memory table.
//  - Shared memory: 32 banks of 4-byte words, word w in bank w % 32. Each bank serves one word
//    per wavefront, so a request takes as many wavefronts as the busiest bank has distinct
//    words (threads reading the same word are served by one broadcast). Nsight Compute reports
//    "Wavefronts" and "Bank Conflicts" in the shared-memory table.
//
// The functions are pure so they can be tested with `node --experimental-strip-types`.

export const WARP = 32;
export const SECTOR_BYTES = 32;
export const LINE_BYTES = 128;
export const BANKS = 32;
export const WORD_BYTES = 4;

/** Byte address of each thread's element: thread i reads `bytes` bytes at offset + i * stride * bytes. */
export function threadAddresses(
	stride: number,
	bytes: number,
	offset = 0,
): number[] {
	return Array.from({ length: WARP }, (_, i) => offset + i * stride * bytes);
}

export interface GlobalAccess {
	/** Distinct 32-byte sector indices touched, ascending. */
	sectors: number[];
	/** Distinct 128-byte cache-line indices touched, ascending. */
	lines: number[];
	/** Sectors per request. */
	sectorsPerRequest: number;
	/** Fewest sectors a warp could need for the same number of bytes. */
	ideal: number;
}

export function globalAccess(
	addresses: readonly number[],
	bytes: number,
): GlobalAccess {
	const sectors = new Set<number>();
	for (const a of addresses) {
		for (
			let s = Math.floor(a / SECTOR_BYTES);
			s <= Math.floor((a + bytes - 1) / SECTOR_BYTES);
			s++
		)
			sectors.add(s);
	}
	const sorted = [...sectors].sort((x, y) => x - y);
	const lines = [
		...new Set(sorted.map((s) => Math.floor((s * SECTOR_BYTES) / LINE_BYTES))),
	];
	return {
		sectors: sorted,
		lines,
		sectorsPerRequest: sorted.length,
		ideal: Math.ceil((addresses.length * bytes) / SECTOR_BYTES),
	};
}

export interface SharedAccess {
	/** For each bank, the distinct word indices requested from it. */
	perBank: number[][];
	wavefronts: number;
	/** Wavefronts needed even with no conflicts: bytes / 4, since a warp moves 32 * bytes bytes. */
	ideal: number;
	/** Extra wavefronts beyond the ideal. */
	conflicts: number;
}

export function sharedAccess(
	addresses: readonly number[],
	bytes: number,
): SharedAccess {
	const words = Math.max(1, Math.ceil(bytes / WORD_BYTES));
	const perBank: Set<number>[] = Array.from(
		{ length: BANKS },
		() => new Set<number>(),
	);
	for (const a of addresses) {
		const first = Math.floor(a / WORD_BYTES);
		for (let j = 0; j < words; j++) perBank[(first + j) % BANKS].add(first + j);
	}
	const wavefronts = Math.max(...perBank.map((s) => s.size));
	const ideal = Math.ceil((addresses.length * words) / BANKS);
	return {
		perBank: perBank.map((s) => [...s].sort((x, y) => x - y)),
		wavefronts,
		ideal,
		conflicts: wavefronts - ideal,
	};
}
