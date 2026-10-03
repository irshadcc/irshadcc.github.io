export type DecompositionKind = "lu" | "qr" | "eigen" | "svd";

export interface MatrixFactor {
	label: string;
	values: number[][];
	role: string;
}

export interface DecompositionExample {
	kind: DecompositionKind;
	name: string;
	formula: string;
	question: string;
	answer: string;
	matrix: number[][];
	factors: MatrixFactor[];
}

const s = 1 / Math.sqrt(2);
const r = 1 / Math.sqrt(5);

export const decompositionExamples: DecompositionExample[] = [
	{
		kind: "lu",
		name: "LU",
		formula: "A = LU",
		question: "How can I solve Ax = b repeatedly?",
		answer: "Eliminate once; then use two triangular solves for each new b.",
		matrix: [
			[4, 2],
			[2, 3],
		],
		factors: [
			{
				label: "L",
				values: [
					[1, 0],
					[0.5, 1],
				],
				role: "records elimination",
			},
			{
				label: "U",
				values: [
					[4, 2],
					[0, 2],
				],
				role: "upper-triangular system",
			},
		],
	},
	{
		kind: "qr",
		name: "QR",
		formula: "A = QR",
		question: "Which vector best fits an overdetermined system?",
		answer:
			"Rotate into orthogonal coordinates, then solve a triangular system.",
		matrix: [
			[3, 1],
			[4, 2],
		],
		factors: [
			{
				label: "Q",
				values: [
					[0.6, -0.8],
					[0.8, 0.6],
				],
				role: "orthonormal directions",
			},
			{
				label: "R",
				values: [
					[5, 2.2],
					[0, 0.4],
				],
				role: "coordinates and scale",
			},
		],
	},
	{
		kind: "eigen",
		name: "Eigen",
		formula: "A = QΛQᵀ",
		question: "Which directions does a symmetric map preserve?",
		answer:
			"Change to eigenvector coordinates, scale each axis, then change back.",
		matrix: [
			[3, 1],
			[1, 3],
		],
		factors: [
			{
				label: "Q",
				values: [
					[s, s],
					[s, -s],
				],
				role: "eigenvector basis",
			},
			{
				label: "Λ",
				values: [
					[4, 0],
					[0, 2],
				],
				role: "scale each eigenvector",
			},
			{
				label: "Qᵀ",
				values: [
					[s, s],
					[s, -s],
				],
				role: "return to original basis",
			},
		],
	},
	{
		kind: "svd",
		name: "SVD",
		formula: "A = UΣVᵀ",
		question: "What are the matrix's strongest input-output directions?",
		answer:
			"Rotate the input, stretch independent axes, then rotate the output.",
		matrix: [
			[1, 2],
			[2, 4],
		],
		factors: [
			{
				label: "U",
				values: [
					[r, -2 * r],
					[2 * r, r],
				],
				role: "output directions",
			},
			{
				label: "Σ",
				values: [
					[5, 0],
					[0, 0],
				],
				role: "singular values",
			},
			{
				label: "Vᵀ",
				values: [
					[r, 2 * r],
					[-2 * r, r],
				],
				role: "input directions",
			},
		],
	},
];

export function multiply(a: number[][], b: number[][]): number[][] {
	if (a[0].length !== b.length)
		throw new Error("incompatible matrix dimensions");
	return a.map((row) =>
		b[0].map((_, j) => row.reduce((sum, value, k) => sum + value * b[k][j], 0)),
	);
}

export function reconstruct(example: DecompositionExample): number[][] {
	return example.factors.map((factor) => factor.values).reduce(multiply);
}
