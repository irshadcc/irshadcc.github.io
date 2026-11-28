// Data-race and barrier-divergence checking for a shared-memory tree reduction, the way
// GPUVerify reasons about it: pick two distinct threads t1, t2 and compare only their accesses.
// Used by RaceCheck.astro.
//
// Kernel (blockDim = n threads, n a power of two):
//   for (s = n / 2; s > 0; s >>= 1) {
//     if (tid < s) sdata[tid] += sdata[tid + s];   // read tid, read tid + s, write tid
//     __syncthreads();                             // variant "barrier"
//   }
// Variant "none" drops the barrier; variant "inside" moves it into the if (only tid < s reach it).
//
// Two accesses race when they come from different threads, touch the same address, at least one
// is a write, and no barrier separates them. With "barrier", accesses in different rounds are
// separated; with "none", nothing is. With "inside", the barrier is divergent whenever one thread
// has tid < s and the other does not, which CUDA leaves undefined, so it is reported as an error
// before races are considered.

export type Variant = "barrier" | "none" | "inside";

export interface Access {
	t: number;
	/** Round stride s (n/2, n/4, ..., 1). */
	s: number;
	round: number;
	kind: "R" | "W";
	addr: number;
}

export interface Conflict {
	a: Access;
	b: Access;
}

export interface PairResult {
	conflicts: Conflict[];
	/** Rounds (indices) where exactly one of the two threads reaches the barrier. */
	divergent: number[];
}

export function strides(n: number): number[] {
	if (n < 2 || (n & (n - 1)) !== 0)
		throw new Error(`raceCheck: n must be a power of two >= 2, got ${n}`);
	const out: number[] = [];
	for (let s = n / 2; s > 0; s >>= 1) out.push(s);
	return out;
}

export function accesses(n: number, t: number): Access[] {
	return strides(n).flatMap((s, round) =>
		t < s
			? [
					{ t, s, round, kind: "R" as const, addr: t },
					{ t, s, round, kind: "R" as const, addr: t + s },
					{ t, s, round, kind: "W" as const, addr: t },
				]
			: [],
	);
}

export function checkPair(
	n: number,
	variant: Variant,
	t1: number,
	t2: number,
): PairResult {
	if (t1 === t2) return { conflicts: [], divergent: [] };
	const divergent =
		variant === "inside"
			? strides(n)
					.map((s, r) => (t1 < s !== t2 < s ? r : -1))
					.filter((r) => r >= 0)
			: [];
	const separated = (a: Access, b: Access) =>
		variant !== "none" && a.round !== b.round;
	const conflicts: Conflict[] = [];
	for (const a of accesses(n, t1))
		for (const b of accesses(n, t2))
			if (
				a.addr === b.addr &&
				(a.kind === "W" || b.kind === "W") &&
				!separated(a, b)
			)
				conflicts.push({ a, b });
	return { conflicts, divergent };
}

/** Every unordered pair {t1, t2} that has a race or a divergent barrier. */
export function badPairs(n: number, variant: Variant): [number, number][] {
	const out: [number, number][] = [];
	for (let a = 0; a < n; a++)
		for (let b = a + 1; b < n; b++) {
			const r = checkPair(n, variant, a, b);
			if (r.conflicts.length > 0 || r.divergent.length > 0) out.push([a, b]);
		}
	return out;
}

export function describe(c: Conflict): string {
	const verb = (k: "R" | "W") => (k === "W" ? "writes" : "reads");
	return `t${c.a.t} ${verb(c.a.kind)} sdata[${c.a.addr}] in round s = ${c.a.s}, t${c.b.t} ${verb(c.b.kind)} it in round s = ${c.b.s}`;
}
