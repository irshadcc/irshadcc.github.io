// Where a strided access pattern lands in a set-associative cache, for CacheSets.astro.
// An address splits into offset (low log2(line) bits), set index (next log2(sets) bits) and
// tag (the rest). A set holds `ways` lines; when a loop cycles through more lines of one set
// than it has ways, LRU replacement evicts each line just before it is needed again, so every
// access to that set misses.

export interface CacheGeom {
	name: string;
	sizeKiB: number;
	ways: number;
	line: number;
}

export const CACHES: CacheGeom[] = [
	{ name: "Zen 4 L1D (32 KiB, 8-way)", sizeKiB: 32, ways: 8, line: 64 },
	{ name: "Golden Cove L1D (48 KiB, 12-way)", sizeKiB: 48, ways: 12, line: 64 },
	{ name: "Zen 4 L2 (1 MiB, 8-way)", sizeKiB: 1024, ways: 8, line: 64 },
];

export function numSets(g: CacheGeom): number {
	return (g.sizeKiB * 1024) / (g.ways * g.line);
}

export function split(addr: number, g: CacheGeom) {
	const sets = numSets(g);
	const offBits = Math.log2(g.line);
	const setBits = Math.log2(sets);
	return {
		offset: addr % g.line,
		set: Math.floor(addr / g.line) % sets,
		tag: Math.floor(addr / (g.line * sets)),
		offBits,
		setBits,
	};
}

export interface Placement {
	sets: number;
	/** Lines mapped to each set. */
	perSet: number[];
	/** Lines that miss on every pass (they sit in a set with more lines than ways). */
	missing: number;
	/** Sets that are touched at all. */
	used: number;
}

/** Simulate `passes` passes over `count` lines at `stride` bytes with true LRU; report steady state. */
export function place(g: CacheGeom, stride: number, count: number): Placement {
	const sets = numSets(g);
	const perSet = new Array<number>(sets).fill(0);
	const lru: number[][] = Array.from({ length: sets }, () => []);
	let missing = 0;
	for (let pass = 0; pass < 3; pass++) {
		for (let i = 0; i < count; i++) {
			const { set, tag } = split(i * stride, g);
			const l = lru[set];
			const k = l.indexOf(tag);
			if (k >= 0) l.splice(k, 1);
			else if (pass === 2) missing++;
			l.push(tag);
			if (l.length > g.ways) l.shift();
			if (pass === 0) perSet[set]++;
		}
	}
	return { sets, perSet, missing, used: perSet.filter((x) => x > 0).length };
}
