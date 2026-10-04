// Theoretical occupancy of a kernel on one SM: how many blocks, hence warps, fit at once.
// Follows the rules of the CUDA occupancy calculator: registers are allocated per warp in
// units of 256 registers, warps are allocated in groups of 4, and each block also reserves
// 1 KB of shared memory on compute capability 8.x. Limits for compute capability 8.0 (A100)
// are from the CUDA Programming Guide, "Compute Capabilities"; the allocation units are those
// of cuda_occupancy.h. Nsight Compute reports the same quantities as
// launch__occupancy_limit_{blocks,registers,shared_mem,warps} and
// sm__maximum_warps_per_active_cycle_pct.
// The functions are pure so they can be tested with `node --experimental-strip-types`.

export interface SmLimits {
	name: string;
	maxWarps: number;
	maxBlocks: number;
	regsPerSm: number;
	maxRegsPerThread: number;
	regAllocUnit: number;
	warpAllocGranularity: number;
	smemPerSm: number;
	smemPerBlockMax: number;
	smemAllocUnit: number;
	smemReservedPerBlock: number;
}

/** Compute capability 8.0 (A100), shared memory carved out at its maximum of 164 KB per SM. */
export const CC80: SmLimits = {
	name: "A100 (cc 8.0)",
	maxWarps: 64,
	maxBlocks: 32,
	regsPerSm: 65536,
	maxRegsPerThread: 255,
	regAllocUnit: 256,
	warpAllocGranularity: 4,
	smemPerSm: 164 * 1024,
	smemPerBlockMax: 163 * 1024,
	smemAllocUnit: 128,
	smemReservedPerBlock: 1024,
};

export type Limiter = "warps" | "blocks" | "registers" | "shared_mem";

export interface Occupancy {
	blocksPerSm: number;
	warpsPerSm: number;
	/** warpsPerSm / maxWarps, in percent. */
	pct: number;
	/** Blocks per SM allowed by each resource alone. */
	limits: Record<Limiter, number>;
	/** The resource with the smallest limit (the first in the order warps, blocks, registers, shared_mem on a tie). */
	limiter: Limiter;
}

const ceilTo = (x: number, unit: number) => Math.ceil(x / unit) * unit;
const floorTo = (x: number, unit: number) => Math.floor(x / unit) * unit;

export function occupancy(
	sm: SmLimits,
	blockThreads: number,
	regsPerThread: number,
	smemPerBlock = 0,
): Occupancy {
	const warpsPerBlock = Math.ceil(blockThreads / 32);
	const limits: Record<Limiter, number> = {
		warps: Math.floor(sm.maxWarps / warpsPerBlock),
		blocks: sm.maxBlocks,
		registers: sm.maxBlocks,
		shared_mem: sm.maxBlocks,
	};
	if (regsPerThread > sm.maxRegsPerThread) {
		limits.registers = 0;
	} else if (regsPerThread > 0) {
		const regsPerWarp = ceilTo(regsPerThread * 32, sm.regAllocUnit);
		const warpsByRegs = floorTo(
			Math.floor(sm.regsPerSm / regsPerWarp),
			sm.warpAllocGranularity,
		);
		limits.registers = Math.floor(warpsByRegs / warpsPerBlock);
	}
	if (smemPerBlock > sm.smemPerBlockMax) {
		limits.shared_mem = 0;
	} else if (smemPerBlock > 0) {
		const perBlock = ceilTo(
			smemPerBlock + sm.smemReservedPerBlock,
			sm.smemAllocUnit,
		);
		limits.shared_mem = Math.floor(sm.smemPerSm / perBlock);
	}
	const order: Limiter[] = ["warps", "blocks", "registers", "shared_mem"];
	const blocksPerSm = Math.min(...order.map((k) => limits[k]));
	const limiter = order.find((k) => limits[k] === blocksPerSm) as Limiter;
	const warpsPerSm = blocksPerSm * warpsPerBlock;
	return {
		blocksPerSm,
		warpsPerSm,
		pct: (100 * warpsPerSm) / sm.maxWarps,
		limits,
		limiter,
	};
}

/** Waves = grid size / (SMs * blocks per SM); the fractional part is the tail wave. */
export function waves(
	gridBlocks: number,
	smCount: number,
	blocksPerSm: number,
): number {
	return gridBlocks / (smCount * blocksPerSm);
}
