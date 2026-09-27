// Rank <-> 3D-parallel coordinate mapping shared by the distributed-training components.
// A job with degrees tp × dp × pp numbers its ranks mixed-radix: `order[0]` varies fastest.
// With Megatron-LM's default order (tp, dp, pp), rank = tp + TP·(dp + DP·pp).

export type Dim = "tp" | "dp" | "pp";
export const DIMS: readonly Dim[] = ["tp", "dp", "pp"];
export const DIM_NAMES: Record<Dim, string> = { tp: "Tensor parallel", dp: "Data parallel", pp: "Pipeline parallel" };

export type Coord = Record<Dim, number>;

export class ParallelLayout {
	readonly size: number;

	constructor(
		readonly degrees: Record<Dim, number>,
		readonly order: readonly [Dim, Dim, Dim] = ["tp", "dp", "pp"],
	) {
		if (new Set(order).size !== 3 || !order.every((d) => DIMS.includes(d))) {
			throw new Error(`ParallelLayout: order must list tp, dp and pp once each, got [${order}]`);
		}
		for (const d of DIMS) {
			if (!Number.isInteger(degrees[d]) || degrees[d] < 1) {
				throw new Error(`ParallelLayout: ${d} must be a positive integer, got ${degrees[d]}`);
			}
		}
		this.size = degrees.tp * degrees.dp * degrees.pp;
	}

	coordOf(rank: number): Coord {
		const c = { tp: 0, dp: 0, pp: 0 };
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

	/** Every group along `dim`: the ranks that differ only in that coordinate, in coordinate order. */
	groups(dim: Dim): number[][] {
		if (this.degrees[dim] < 2) return [];
		const out: number[][] = [];
		for (let r = 0; r < this.size; r++) {
			const c = this.coordOf(r);
			if (c[dim] !== 0) continue;
			out.push(Array.from({ length: this.degrees[dim] }, (_, k) => this.rankOf({ ...c, [dim]: k })));
		}
		return out;
	}

	/** The group along `dim` that contains `rank`. */
	groupOf(dim: Dim, rank: number): number[] {
		const c = this.coordOf(rank);
		return Array.from({ length: this.degrees[dim] }, (_, k) => this.rankOf({ ...c, [dim]: k }));
	}
}
