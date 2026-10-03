export type Matrix = number[][];

export interface SinkhornState {
	iteration: number;
	plan: Matrix;
	rowSums: number[];
	columnSums: number[];
	transportCost: number;
}

export const OT_SOURCE = [0.5, 0.5];
export const OT_TARGET = [0.25, 0.75];
export const OT_COST: Matrix = [
	[1, 4],
	[2, 1],
];

export function feasiblePlan(x: number): Matrix {
	if (x < 0 || x > 0.25) throw new RangeError("x must be between 0 and 0.25");
	return [
		[x, 0.5 - x],
		[0.25 - x, 0.25 + x],
	];
}

export function matrixCost(plan: Matrix, cost: Matrix = OT_COST): number {
	return plan.reduce(
		(total, row, i) =>
			total + row.reduce((sum, mass, j) => sum + mass * cost[i][j], 0),
		0,
	);
}

function sums(plan: Matrix) {
	return {
		rowSums: plan.map((row) => row.reduce((sum, value) => sum + value, 0)),
		columnSums: plan[0].map((_, j) =>
			plan.reduce((sum, row) => sum + row[j], 0),
		),
	};
}

export function sinkhornTrace(
	iterations: number,
	epsilon = 1,
	source = OT_SOURCE,
	target = OT_TARGET,
	cost = OT_COST,
): SinkhornState[] {
	if (iterations < 0 || !Number.isInteger(iterations))
		throw new RangeError("iterations must be a non-negative integer");
	if (epsilon <= 0) throw new RangeError("epsilon must be positive");
	const kernel = cost.map((row) =>
		row.map((value) => Math.exp(-value / epsilon)),
	);
	let u = source.map(() => 1);
	let v = target.map(() => 1);
	const states: SinkhornState[] = [];

	for (let iteration = 1; iteration <= iterations; iteration += 1) {
		u = source.map(
			(mass, i) =>
				mass / kernel[i].reduce((sum, value, j) => sum + value * v[j], 0),
		);
		v = target.map(
			(mass, j) =>
				mass / kernel.reduce((sum, row, i) => sum + row[j] * u[i], 0),
		);
		const plan = kernel.map((row, i) =>
			row.map((value, j) => u[i] * value * v[j]),
		);
		states.push({
			iteration,
			plan,
			...sums(plan),
			transportCost: matrixCost(plan, cost),
		});
	}
	return states;
}
