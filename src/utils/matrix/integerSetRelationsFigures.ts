// Verified figure data for the "Tensor Layouts as Integer Set Relations" post.
// Values are generated from the CuTe and F_2 formulas in arXiv:2511.10374 rather
// than copied into the MDX. Run with `node --experimental-strip-types` to check.
import type { BoxEdge, BoxNode } from "../payments/boxDiagram";
import { CuteLayout } from "./CuteLayout.ts";

const cuteLayout = new CuteLayout([4, [2, 2]], [2, [1, 8]]);
export const cuteLayoutValues: Record<number, number> = Object.fromEntries(
	Array.from({ length: cuteLayout.size }, (_, c) => {
		const offset = cuteLayout.offset(c);
		return [offset, offset];
	}),
);

const complementLayout = new CuteLayout([[4, 2], 4], [[1, 16], 4]);
export const complementValues: Record<number, number> = Object.fromEntries(
	Array.from({ length: complementLayout.size }, (_, c) => {
		const offset = complementLayout.offset(c);
		return [offset, offset];
	}),
);

export const frameworkNodes: BoxNode[] = [
	{
		id: "cute",
		label: "CuTe layout",
		sub: "shape + strides",
		col: 0,
		row: 0,
		group: 1,
	},
	{
		id: "cuteRel",
		label: "coordinate + index",
		sub: "quasi-affine relations",
		col: 1,
		row: 0,
		group: 1,
	},
	{
		id: "isl",
		label: "ISL relation",
		sub: "one common language",
		col: 2,
		row: 0,
		h: 2,
		group: 3,
	},
	{
		id: "linear",
		label: "Triton layout",
		sub: "F₂ basis vectors",
		col: 0,
		row: 1,
		group: 4,
	},
	{
		id: "binaryRel",
		label: "binary transform",
		sub: "modulo-2 relation",
		col: 1,
		row: 1,
		group: 4,
	},
];

export const frameworkEdges: BoxEdge[] = [
	{ from: "cute", to: "cuteRel" },
	{ from: "cuteRel", to: "isl" },
	{ from: "linear", to: "binaryRel" },
	{ from: "binaryRel", to: "isl" },
];

export const cutePipelineNodes: BoxNode[] = [
	{
		id: "integral",
		label: "c",
		sub: "integral coordinate",
		col: 0,
		row: 0,
		group: 0,
	},
	{
		id: "natural",
		label: "(c₀,c₁,c₂)",
		sub: "natural coordinate",
		col: 1,
		row: 0,
		group: 1,
	},
	{ id: "index", label: "i", sub: "linear index", col: 2, row: 0, group: 3 },
];

export const cutePipelineEdges: BoxEdge[] = [
	{ from: "integral", to: "natural", label: "Mᶜ: mod + floor" },
	{ from: "natural", to: "index", label: "Mⁱ: dot strides" },
];

export const linearPipelineNodes: BoxNode[] = [
	{
		id: "naturalC",
		label: "(t,w)",
		sub: "natural coordinate",
		col: 0,
		row: 0,
		group: 0,
	},
	{
		id: "linearC",
		label: "c",
		sub: "linear coordinate",
		col: 1,
		row: 0,
		group: 0,
	},
	{
		id: "bitsC",
		label: "(c₀…c₃)",
		sub: "coordinate bits",
		col: 2,
		row: 0,
		group: 4,
	},
	{
		id: "bitsI",
		label: "(i₀…i₃)",
		sub: "index bits",
		col: 3,
		row: 0,
		group: 4,
	},
	{ id: "linearI", label: "i", sub: "linear index", col: 4, row: 0, group: 0 },
	{
		id: "naturalI",
		label: "(x,y)",
		sub: "natural index",
		col: 5,
		row: 0,
		group: 3,
	},
];

export const linearPipelineEdges: BoxEdge[] = [
	{ from: "naturalC", to: "linearC", label: "Mⁱᶜ" },
	{ from: "linearC", to: "bitsC", label: "Mᵇᶜ" },
	{ from: "bitsC", to: "bitsI", label: "Mᵇᵛ" },
	{ from: "bitsI", to: "linearI", label: "Mˡⁱ" },
	{ from: "linearI", to: "naturalI", label: "Mⁿⁱ" },
];

/** CuTe Swizzle<1,2,1>: c XOR ((c & 0b1000) >> 1). */
export const swizzleValues: Record<number, number> = Object.fromEntries(
	Array.from({ length: 16 }, (_, c) => [c, c ^ ((c & 0b1000) >> 1)]),
);

export const swizzleBits: Record<number, string> = Object.fromEntries(
	Array.from({ length: 16 }, (_, c) => [
		c,
		`${c.toString(2).padStart(4, "0")}→${swizzleValues[c].toString(2).padStart(4, "0")}`,
	]),
);

// Paper example: basis inputs (1,0), (2,0), (0,1), (0,2) map to
// (1,1), (2,2), (0,1), (0,2). Linearity fills the grid by XOR.
export const linearLayoutValues: Record<number, string> = {};
for (let w = 0; w < 4; w++) {
	for (let t = 0; t < 4; t++) {
		linearLayoutValues[t + 4 * w] = `(${t},${t ^ w})`;
	}
}

// Independent invariants used by the verification command and discussed in the post.
export function verifyIntegerSetFigures(): void {
	for (let c = 0; c < 16; c++) {
		const y = swizzleValues[c];
		if (swizzleValues[y] !== c)
			throw new Error(`swizzle is not an involution at ${c}`);
	}
	if (linearLayoutValues[3 + 4 * 3] !== "(3,0)")
		throw new Error("linear-layout check failed at (3,3)");
	const complementRange = new Set<number>();
	for (let c0 = 0; c0 < 4; c0++)
		for (let c1 = 0; c1 < 2; c1++)
			for (let k = 0; k < 4; k++) complementRange.add(c0 + 16 * c1 + 4 * k);
	if (
		complementRange.size !== 32 ||
		[...complementRange].some((x) => x < 0 || x >= 32)
	)
		throw new Error("complement does not fill [0,32)");
	if (cuteLayout.offset(13) !== 11)
		throw new Error("CuTe check failed at c=13");
}
