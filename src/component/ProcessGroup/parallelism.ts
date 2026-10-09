// Rank <-> parallel coordinate mapping shared by the distributed-training components.
// A job with degrees tp × cp × dp × pp numbers its ranks mixed-radix: `order[0]` varies fastest.
// With Megatron-LM's default order (tp, cp, dp, pp), rank = tp + TP·(cp + CP·(dp + DP·pp)).
//
// Expert parallelism adds no ranks. As in Megatron-Core, MoE layers refactor the CP × DP ranks
// of each (tp, pp) position into EP × expert-DP: flatten (cp, dp) in rank order into
// k = 0 .. CP·DP - 1, and an EP group is `ep` consecutive values of k. So EP must divide CP·DP,
// and an EP group spans context chunks first, then data-parallel replicas. (Expert tensor
// parallelism is taken equal to TP.)

export type Dim = "tp" | "cp" | "ep" | "dp" | "pp";
export const DIMS: readonly Dim[] = ["tp", "cp", "ep", "dp", "pp"];
export const DIM_NAMES: Record<Dim, string> = {
	tp: "Tensor parallel",
	cp: "Context parallel",
	ep: "Expert parallel",
	dp: "Data parallel",
	pp: "Pipeline parallel",
};

/** The dimensions that index ranks; EP is carved out of CP × DP instead. */
export type Axis = Exclude<Dim, "ep">;
export const AXES: readonly Axis[] = ["tp", "cp", "dp", "pp"];

export type Coord = Record<Axis, number>;
/** Parallel degrees; cp and ep default to 1. */
export type Degrees = Record<"tp" | "dp" | "pp", number> & Partial<Record<"cp" | "ep", number>>;

export class ParallelLayout {
	readonly size: number;
	readonly degrees: Record<Dim, number>;
	readonly order: readonly Axis[];
	/** cp and dp in the order they vary in rank numbering; EP flattens them in this order. */
	private readonly inner: readonly Axis[];

	/** `order` lists tp, dp and pp, and optionally cp; a missing cp goes right after tp. */
	constructor(degrees: Degrees, order: readonly Axis[] = ["tp", "cp", "dp", "pp"]) {
		const full = [...order];
		if (!full.includes("cp")) full.splice(full.indexOf("tp") + 1, 0, "cp");
		if (full.length !== AXES.length || new Set(full).size !== AXES.length || !full.every((d) => AXES.includes(d))) {
			throw new Error(`ParallelLayout: order must list tp, dp, pp (and optionally cp) once each, got [${order}]`);
		}
		this.order = full;
		this.inner = full.filter((d) => d === "cp" || d === "dp");
		this.degrees = { cp: 1, ep: 1, ...degrees };
		for (const d of DIMS) {
			const v = this.degrees[d];
			if (!Number.isInteger(v) || v < 1) throw new Error(`ParallelLayout: ${d} must be a positive integer, got ${v}`);
		}
		const { tp, cp, ep, dp, pp } = this.degrees;
		if ((cp * dp) % ep !== 0) throw new Error(`ParallelLayout: ep (${ep}) must divide cp × dp (${cp * dp})`);
		this.size = tp * cp * dp * pp;
	}

	coordOf(rank: number): Coord {
		const c = { tp: 0, cp: 0, dp: 0, pp: 0 };
		let rest = rank;
		for (const d of this.order) {
			c[d] = rest % this.degrees[d];
			rest = Math.floor(rest / this.degrees[d]);
		}
		return c;
	}

	rankOf(c: Coord): number {
		return this.order.reduceRight((r, d) => r * this.degrees[d] + c[d], 0);
	}

	/** Position of a rank among the CP × DP ranks sharing its (tp, pp); EP groups are runs of it. */
	private flat(c: Coord): number {
		return this.inner.reduceRight((k, d) => k * this.degrees[d] + c[d], 0);
	}

	private unflat(k: number, base: Coord): Coord {
		const c = { ...base };
		for (const d of this.inner) {
			c[d] = k % this.degrees[d];
			k = Math.floor(k / this.degrees[d]);
		}
		return c;
	}

	/** Every group along `dim`: the ranks that differ only in that coordinate, in coordinate order. */
	groups(dim: Dim): number[][] {
		if (this.degrees[dim] < 2) return [];
		const out: number[][] = [];
		for (let r = 0; r < this.size; r++) {
			const c = this.coordOf(r);
			const first = dim === "ep" ? this.flat(c) % this.degrees.ep === 0 : c[dim] === 0;
			if (first) out.push(this.groupOf(dim, r));
		}
		return out;
	}

	/** The group along `dim` that contains `rank`. */
	groupOf(dim: Dim, rank: number): number[] {
		const c = this.coordOf(rank);
		if (dim === "ep") {
			const E = this.degrees.ep;
			const k0 = this.flat(c) - (this.flat(c) % E);
			return Array.from({ length: E }, (_, j) => this.rankOf(this.unflat(k0 + j, c)));
		}
		return Array.from({ length: this.degrees[dim] }, (_, k) => this.rankOf({ ...c, [dim]: k }));
	}
}
