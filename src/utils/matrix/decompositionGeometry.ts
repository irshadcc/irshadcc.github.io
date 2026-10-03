export type GeometryKind = "lu" | "qr" | "eigen" | "svd";

export interface Point {
	x: number;
	y: number;
}

export const apply2 = (matrix: number[][], point: Point): Point => ({
	x: matrix[0][0] * point.x + matrix[0][1] * point.y,
	y: matrix[1][0] * point.x + matrix[1][1] * point.y,
});

export const circle = (count = 97): Point[] =>
	Array.from({ length: count }, (_, i) => {
		const angle = (2 * Math.PI * i) / (count - 1);
		return { x: Math.cos(angle), y: Math.sin(angle) };
	});

export const transform = (points: Point[], matrix: number[][]): Point[] =>
	points.map((point) => apply2(matrix, point));

export const project = (point: Point, unit: Point): Point => {
	const amount = point.x * unit.x + point.y * unit.y;
	return { x: amount * unit.x, y: amount * unit.y };
};

export const GEOMETRY = {
	lu: {
		matrix: [
			[1, 0],
			[0.5, 1],
		],
	},
	qr: {
		b: { x: 2, y: 1 },
		q: { x: 0.8, y: 0.6 },
	},
	eigen: {
		matrix: [
			[3, 1],
			[1, 3],
		],
		q1: { x: 1 / Math.sqrt(2), y: 1 / Math.sqrt(2) },
		q2: { x: 1 / Math.sqrt(2), y: -1 / Math.sqrt(2) },
	},
	svd: {
		matrix: [
			[1, 2],
			[2, 4],
		],
		v1: { x: 1 / Math.sqrt(5), y: 2 / Math.sqrt(5) },
		v2: { x: -2 / Math.sqrt(5), y: 1 / Math.sqrt(5) },
	},
} as const;
