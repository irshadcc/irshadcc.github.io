import type { BoxEdge, BoxNode } from "../payments/boxDiagram";
// Figure data for the backpropagation post (src/content/posts/backpropagation.mdx): the
// step-through figures (backpropagation on a 2-3-2 network, the Jacobian by backpropagation,
// forward- and reverse-mode autodiff on the evaluation trace of (8.49)) and the chart series.
// Every number is computed by backprop.ts; a scratch script checked those against PyTorch
// autograd (gradients, Jacobian, both trace outputs) to within 2e-16. Notes are written by hand.
import {
	type Net,
	backward,
	centralDiff,
	dfdx1,
	forward,
	forwardDiff,
	jacobian,
	jacobianFd,
	traceAdjoint,
	tracePrimal,
	traceTangent,
} from "./backprop";
import {
	type GraphEdge,
	type GraphNode,
	type GraphStep,
	type GraphStepsData,
	fmt,
} from "./graphSteps";

// ---------------------------------------------------------------------------------------------
// The worked example: D = 2 inputs, M = 3 tanh hidden units, K = 2 linear outputs.

export const net: Net = {
	w1: [
		[0.1, 0.4, -0.3],
		[-0.2, 0.6, 0.5],
		[0.0, -0.5, 0.2],
	],
	w2: [
		[0.2, 0.7, -0.4, 0.3],
		[-0.1, 0.5, 0.6, -0.8],
	],
};
export const input = [0.8, -0.5];
export const target = [0.5, -0.5];
export const bp = backward(net, input, target);
export const jac = jacobian(net, input);
export const jacFd = jacobianFd(net, input, 1e-5);

const sub = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈"];
const D = input.length;
const M = net.w1.length;
const K = net.w2.length;

const netNodes: GraphNode[] = [
	{ id: "x0", label: "x₀", col: 0, row: 0.35, bias: true },
	...input.map((_, i) => ({
		id: `x${i + 1}`,
		label: `x${sub[i + 1]}`,
		sub: "input",
		col: 0,
		row: 1.25 + i,
	})),
	{ id: "z0", label: "z₀", col: 1, row: 0, bias: true },
	...net.w1.map((_, j) => ({
		id: `z${j + 1}`,
		label: `z${sub[j + 1]}`,
		sub: "tanh",
		col: 1,
		row: 0.75 + j,
	})),
	...net.w2.map((_, k) => ({
		id: `y${k + 1}`,
		label: `y${sub[k + 1]}`,
		sub: "linear",
		col: 2,
		row: 1.25 + k,
	})),
];
const netEdges: GraphEdge[] = [
	...net.w1.flatMap((_, j) =>
		["x0", ...input.map((_, i) => `x${i + 1}`)].map((from) => ({
			from,
			to: `z${j + 1}`,
		})),
	),
	...net.w2.flatMap((_, k) =>
		["z0", ...net.w1.map((_, j) => `z${j + 1}`)].map((from) => ({
			from,
			to: `y${k + 1}`,
		})),
	),
];
const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const intoHidden = Object.fromEntries(
	range(M).flatMap((j) =>
		range(D + 1).map((i) => [
			`${i === 0 ? "x0" : `x${i}`}>z${j + 1}`,
			{ hi: "fwd" as const },
		]),
	),
);
const intoOutput = (hi: "fwd" | "bwd", withBias = true) =>
	Object.fromEntries(
		range(K).flatMap((k) =>
			range(M + 1)
				.filter((j) => withBias || j > 0)
				.map((j) => [`z${j}>y${k + 1}`, { hi }]),
		),
	);

const xLines = () =>
	Object.fromEntries(
		input.map((v, i) => [`x${i + 1}`, { lines: [`x = ${fmt(v)}`] }]),
	);
const hidLines = (upTo: "a" | "z" | "d", hi?: "fwd" | "bwd") =>
	Object.fromEntries(
		range(M).map((j) => {
			const lines = [`a = ${fmt(bp.a[j])}`];
			if (upTo !== "a") lines.push(`z = ${fmt(bp.z[j + 1])}`);
			if (upTo === "d") lines.push(`δ = ${fmt(bp.dHid[j])}`);
			return [`z${j + 1}`, { lines, hi }];
		}),
	);
const outLines = (withY: boolean, withD: boolean, hi?: "fwd" | "bwd") =>
	Object.fromEntries(
		range(K).map((k) => {
			const lines: string[] = [];
			if (withY) lines.push(`y = ${fmt(bp.y[k])}`);
			lines.push(`t = ${fmt(target[k])}`);
			if (withD) lines.push(`δ = ${fmt(bp.dOut[k])}`);
			return [`y${k + 1}`, { lines, hi }];
		}),
	);
const t = (x: number, d = 3) => fmt(x, d).replace("−", "-");

const weightTable = {
	caption: "Weights; column 0 multiplies the bias unit x₀ = 1 or z₀ = 1",
	columns: ["weight", "col 0 (bias)", "col 1", "col 2", "col 3"],
	rows: [
		...net.w1.map((r, j) => [
			`w⁽¹⁾ row ${j + 1}`,
			...r.map((v) => fmt(v, 1)),
			"",
		]),
		...net.w2.map((r, k) => [`w⁽²⁾ row ${k + 1}`, ...r.map((v) => fmt(v, 1))]),
	],
};

const z1 = 0; // the hidden unit the "one unit" step follows
const ex = {
	a1: `${t(net.w1[z1][0], 1)} + ${t(net.w1[z1][1], 1)}\\cdot${t(input[0], 1)} + (${t(net.w1[z1][2], 1)})(${t(input[1], 1)})`,
};

const networkStepList: GraphStep[] = [
	{
		label: "Setup",
		note: `The input $\\mathbf{x} = (${t(input[0], 1)}, ${t(input[1], 1)})$ and target $\\mathbf{t} = (${t(target[0], 1)}, ${t(target[1], 1)})$ are known; nothing else is. The round nodes are the bias units $x_0 = z_0 = 1$, so each bias is just the weight in column 0.`,
		nodes: { ...xLines(), ...outLines(false, false) },
		table: weightTable,
	},
	{
		label: "a, z",
		note: `Forward propagation, first layer: $a_j = \\sum_{i=0}^{D} w^{(1)}_{ji} x_i$ (8.18) and $z_j = \\tanh(a_j)$ (8.19). For unit 1: $a_1 = ${ex.a1} = ${t(bp.a[0])}$, so $z_1 = \\tanh(${t(bp.a[0])}) = ${t(bp.z[1])}$.`,
		nodes: { ...xLines(), ...hidLines("z", "fwd"), ...outLines(false, false) },
		edges: intoHidden,
	},
	{
		label: "y",
		note: `Second layer, with linear outputs: $y_k = \\sum_{j=0}^{M} w^{(2)}_{kj} z_j$ (8.20). The error (8.17) is $E_n = \\tfrac12[(${t(bp.y[0])} - ${t(target[0], 1)})^2 + (${t(bp.y[1])} + ${t(-target[1], 1)})^2] = ${t(bp.E, 4)}$.`,
		nodes: { ...xLines(), ...hidLines("z"), ...outLines(true, false, "fwd") },
		edges: intoOutput("fwd"),
	},
	{
		label: "δ out",
		note: `The backward pass starts at the outputs. With linear outputs and a sum-of-squares error, $\\delta_k = \\partial E_n/\\partial a_k = y_k - t_k$ (8.21): $\\delta_1 = ${t(bp.dOut[0])}$ (output 1 is too low), $\\delta_2 = ${t(bp.dOut[1])}$ (output 2 is too high).`,
		nodes: { ...xLines(), ...hidLines("z"), ...outLines(true, true, "bwd") },
	},
	{
		label: "δ for z₁",
		note: `Unit 1 sends its activation to both outputs, so both errors flow back to it, each scaled by the weight it travelled through (the red edge labels show $w_{k1}\\delta_k$), then by $h'(a_1) = 1 - z_1^2$ (8.22). For hidden unit $j = 1$: $\\delta_j = (1 -${t(bp.z[1])}^2)\\,[(${t(net.w2[0][1], 1)})(${t(bp.dOut[0])}) + (${t(net.w2[1][1], 1)})(${t(bp.dOut[1])})] = ${t(bp.dHid[0])}$.`,
		nodes: {
			...xLines(),
			...hidLines("z"),
			z1: {
				lines: [
					`a = ${fmt(bp.a[0])}`,
					`z = ${fmt(bp.z[1])}`,
					`δ = ${fmt(bp.dHid[0])}`,
				],
				hi: "bwd",
			},
			...outLines(true, true),
		},
		edges: Object.fromEntries(
			range(K).map((k) => [
				`z1>y${k + 1}`,
				{
					hi: "bwd" as const,
					label: `${fmt(net.w2[k][1] * bp.dOut[k])}`,
				},
			]),
		),
	},
	{
		label: "δ hidden",
		note: "The same rule for every hidden unit: $\\delta_j = h'(a_j)\\sum_k w_{kj}\\delta_k$ (8.13). The sum runs over the <em>first</em> index of $w_{kj}$, the units $j$ feeds, whereas the forward sum (8.5) runs over the second. The bias unit $z_0$ gets no $\\delta$: nothing upstream depends on it.",
		nodes: { ...xLines(), ...hidLines("d", "bwd"), ...outLines(true, true) },
		edges: intoOutput("bwd", false),
		table: {
			columns: ["unit j", "Σₖ wₖⱼ δₖ", "1 − zⱼ²", "δⱼ"],
			rows: range(M).map((j) => {
				const s = range(K).reduce(
					(acc, k) => acc + net.w2[k][j + 1] * bp.dOut[k],
					0,
				);
				return [
					`${j + 1}`,
					fmt(s, 4),
					fmt(1 - bp.z[j + 1] ** 2, 4),
					fmt(bp.dHid[j], 4),
				];
			}),
		},
	},
	{
		label: "∇w⁽²⁾",
		note: `Each derivative is the $\\delta$ at the output end of the weight times the $z$ at its input end (8.23): $\\partial E_n/\\partial w^{(2)}_{kj} = \\delta_k z_j$. Check one: $\\partial E_n/\\partial w^{(2)}_{21} = ${t(bp.dOut[1])} \\times ${t(bp.z[1])} = ${t(bp.g2[1][1], 4)}$. Column 0 (the bias) is just $\\delta_k$.`,
		nodes: { ...xLines(), ...hidLines("d"), ...outLines(true, true, "bwd") },
		edges: intoOutput("bwd"),
		table: {
			columns: ["∂E/∂w⁽²⁾", "z₀ = 1", "z₁", "z₂", "z₃"],
			rows: range(K).map((k) => [
				`row k = ${k + 1}`,
				...bp.g2[k].map((v) => fmt(v, 4)),
			]),
		},
	},
	{
		label: "∇w⁽¹⁾",
		note: `The first layer works the same way with the hidden $\\delta$s: $\\partial E_n/\\partial w^{(1)}_{ji} = \\delta_j x_i$. Check one: $\\partial E_n/\\partial w^{(1)}_{12} = ${t(bp.dHid[0])} \\times (${t(input[1], 1)}) = ${t(bp.g1[0][2], 4)}$. All ${(D + 1) * M + (M + 1) * K} derivatives came from one forward and one backward sweep.`,
		nodes: { ...xLines(), ...hidLines("d", "bwd"), ...outLines(true, true) },
		edges: Object.fromEntries(
			Object.keys(intoHidden).map((key) => [key, { hi: "bwd" as const }]),
		),
		table: {
			columns: ["∂E/∂w⁽¹⁾", "x₀ = 1", "x₁", "x₂"],
			rows: range(M).map((j) => [
				`row j = ${j + 1}`,
				...bp.g1[j].map((v) => fmt(v, 4)),
			]),
		},
	},
];

export const networkSteps: GraphStepsData = {
	nodes: netNodes,
	edges: netEdges,
	steps: networkStepList,
};

// ---------------------------------------------------------------------------------------------
// Checking the gradient with finite differences (8.24), (8.25).

/** The weights checked in the post's table: layer, row, column, name. */
const checked: [1 | 2, number, number, string][] = [
	[1, 0, 1, "w⁽¹⁾₁₁"],
	[1, 2, 2, "w⁽¹⁾₃₂"],
	[2, 0, 0, "w⁽²⁾₁₀"],
	[2, 1, 3, "w⁽²⁾₂₃"],
];
export const gradCheckRows = checked.map(([layer, r, c, name]) => {
	const g = (layer === 1 ? bp.g1 : bp.g2)[r][c];
	const fd = forwardDiff(net, input, target, layer, r, c, 1e-6);
	const cd = centralDiff(net, input, target, layer, r, c, 1e-6);
	return { name, g, fd, cd, fdErr: Math.abs(fd - g), cdErr: Math.abs(cd - g) };
});

/** log10 eps from -14 to -1 in quarter decades. */
export const epsGrid = () => Array.from({ length: 53 }, (_, i) => -14 + i / 4);
const fdCache = new Map<number, [number, number]>();
function fdErrors(logEps: number): [number, number] {
	let hit = fdCache.get(logEps);
	if (!hit) {
		const eps = 10 ** logEps;
		const g = bp.g1[0][1];
		hit = [
			Math.abs(forwardDiff(net, input, target, 1, 0, 1, eps) - g),
			Math.abs(centralDiff(net, input, target, 1, 0, 1, eps) - g),
		];
		fdCache.set(logEps, hit);
	}
	return hit;
}
export const fdSeries = [
	{
		name: "finite differences (8.24)",
		fn: (x: number) => Math.log10(fdErrors(x)[0]),
	},
	{
		name: "central differences (8.25)",
		fn: (x: number) => Math.log10(fdErrors(x)[1]),
	},
];

// ---------------------------------------------------------------------------------------------
// The Jacobian by backpropagation (8.29)-(8.31).

const jacNodes = (k: number, stage: 0 | 1 | 2 | 3) => {
	const f = forward(net, input);
	const r = jac[k];
	const name = `∂y${sub[k + 1]}/∂`;
	const nodes: GraphStep["nodes"] = {};
	input.forEach((v, i) => {
		const lines = [`x = ${fmt(v)}`];
		if (stage >= 3)
			lines.push(`J${sub[k + 1]}${sub[i + 1]} = ${fmt(r.row[i])}`);
		nodes[`x${i + 1}`] = { lines, hi: stage === 3 ? "bwd" : undefined };
	});
	f.a.forEach((a, j) => {
		const lines = [`a = ${fmt(a)}`, `z = ${fmt(f.z[j + 1])}`];
		if (stage >= 2) lines.push(`${name}a = ${fmt(r.dHid[j])}`);
		nodes[`z${j + 1}`] = {
			lines,
			hi: stage === 2 ? "bwd" : stage === 0 ? "fwd" : undefined,
		};
	});
	f.y.forEach((y, l) => {
		const lines = [`y = ${fmt(y)}`];
		if (stage >= 1) lines.push(`${name}a = ${fmt(r.dOut[l], 0)}`);
		nodes[`y${l + 1}`] = {
			lines,
			hi: stage === 1 ? "bwd" : stage === 0 ? "fwd" : undefined,
		};
	});
	return nodes;
};
const hiddenToInputs = Object.fromEntries(
	range(M).flatMap((j) =>
		range(D).map((i) => [`x${i + 1}>z${j + 1}`, { hi: "bwd" as const }]),
	),
);
const jacRow = (k: number): GraphStep[] => {
	const r = jac[k];
	const K1 = k + 1;
	return [
		{
			label: `${K1}: out`,
			note: `Row ${K1} of the Jacobian holds the derivatives of $y_${K1}$. Start at the outputs: for linear outputs $\\partial y_${K1}/\\partial a_l = \\delta_{${K1}l}$ (8.31), the Kronecker delta, so the seed is $(${r.dOut.join(", ")})$.`,
			nodes: jacNodes(k, 1),
		},
		{
			label: `${K1}: hidden`,
			note: `Backpropagate with (8.30): $\\partial y_${K1}/\\partial a_j = h'(a_j)\\sum_l w_{lj}\\,\\partial y_${K1}/\\partial a_l$. Only output ${K1} is seeded, so the sum keeps one term: $\\partial y_${K1}/\\partial a_1 = (1 - ${t(bp.z[1])}^2)(${t(net.w2[k][1], 1)}) = ${t(r.dHid[0])}$. This is the $\\delta$ recursion with $\\delta_k$ replaced by a one-hot seed.`,
			nodes: jacNodes(k, 2),
			edges: Object.fromEntries(
				range(M).map((j) => [`z${j + 1}>y${K1}`, { hi: "bwd" as const }]),
			),
		},
		{
			label: `${K1}: in`,
			note: `Finally (8.29): $J_{${K1}i} = \\sum_j w_{ji}\\,\\partial y_${K1}/\\partial a_j$. For $i = 1$: $${range(
				M,
			)
				.map((j) => `(${t(net.w1[j][1], 1)})(${t(r.dHid[j])})`)
				.join(" + ")} = ${t(r.row[0], 4)}$.`,
			nodes: jacNodes(k, 3),
			edges: hiddenToInputs,
		},
	];
};
export const jacobianSteps: GraphStepsData = {
	nodes: netNodes,
	edges: netEdges,
	steps: [
		{
			label: "Forward",
			note: "Forward-propagate the input once, as for the gradient. The Jacobian depends on the input through the $h'(a_j)$ factors, so it is only valid at this $\\mathbf{x}$.",
			nodes: jacNodes(0, 0),
			edges: { ...intoHidden, ...intoOutput("fwd") },
		},
		...jacRow(0),
		...jacRow(1),
		{
			label: "Check",
			note: `Two backward passes gave the two rows. The central-difference estimate (8.35), with $\\varepsilon = 10^{-5}$, needs $2D = ${2 * D}$ forward passes and agrees to $${(Math.max(...range(K).flatMap((k) => range(D).map((i) => Math.abs(jacFd[k][i] - jac[k].row[i])))) * 1e11).toFixed(1)} \\times 10^{-11}$.`,
			nodes: {
				...jacNodes(1, 3),
				...Object.fromEntries(
					input.map((v, i) => [
						`x${i + 1}`,
						{
							lines: [
								`x = ${fmt(v)}`,
								`J₁${sub[i + 1]} = ${fmt(jac[0].row[i])}`,
								`J₂${sub[i + 1]} = ${fmt(jac[1].row[i])}`,
							],
						},
					]),
				),
			},
			table: {
				columns: [
					"",
					"backprop ∂y/∂x₁",
					"backprop ∂y/∂x₂",
					"central diff ∂y/∂x₁",
					"central diff ∂y/∂x₂",
				],
				rows: range(K).map((k) => [
					`y${sub[k + 1]}`,
					...jac[k].row.map((v) => fmt(v, 6)),
					...jacFd[k].map((v) => fmt(v, 6)),
				]),
			},
		},
	],
};

/** y2 along x1, true change against the linear prediction J21 * dx (8.28). */
export const linearSeries = [
	{
		name: "true Δy₂",
		fn: (s: number) => forward(net, [input[0] + s, input[1]]).y[1] - bp.y[1],
	},
	{ name: "J₂₁ Δx₁ (8.28)", fn: (s: number) => jac[1].row[0] * s },
];

// ---------------------------------------------------------------------------------------------
// The evaluation trace of (8.49) at x1 = 1, x2 = 2.

export const X1 = 1;
export const X2 = 2;
export const v = tracePrimal(X1, X2);
export const vdot = traceTangent(v, [1, 0]);
export const vdot2 = traceTangent(v, [0, 1]);
export const vbar = traceAdjoint(v);
export const exactDfdx1 = dfdx1(X1, X2);

const traceNodes = (withF2: boolean): GraphNode[] => [
	{ id: "v1", label: "v₁", sub: "x₁", col: 0, row: 0 },
	{ id: "v2", label: "v₂", sub: "x₂", col: 0, row: 1 },
	{ id: "v3", label: "v₃", sub: "v₁·v₂", col: 1, row: 0 },
	{ id: "v4", label: "v₄", sub: "sin v₂", col: 1, row: 1 },
	{ id: "v5", label: "v₅", sub: "exp v₃", col: 2, row: 0 },
	{ id: "v6", label: "v₆", sub: "v₃−v₄", col: 2, row: 1 },
	{ id: "v7", label: "v₇ = f₁", sub: "v₅+v₆", col: 3, row: withF2 ? 0 : 0.5 },
	...(withF2
		? [{ id: "v8", label: "v₈ = f₂", sub: "v₅·v₆", col: 3, row: 1 }]
		: []),
];
const traceEdges = (withF2: boolean): GraphEdge[] => [
	{ from: "v1", to: "v3" },
	{ from: "v2", to: "v3" },
	{ from: "v2", to: "v4" },
	{ from: "v3", to: "v5" },
	{ from: "v3", to: "v6" },
	{ from: "v4", to: "v6" },
	{ from: "v5", to: "v7" },
	{ from: "v6", to: "v7" },
	...(withF2
		? [
				{ from: "v5", to: "v8" },
				{ from: "v6", to: "v8" },
			]
		: []),
];
const parents: Record<number, number[]> = {
	3: [1, 2],
	4: [2],
	5: [3],
	6: [3, 4],
	7: [5, 6],
	8: [5, 6],
};

/** Forward mode: node i shows (v_i, vdot_i) once it has been reached. */
const fwdNodes = (upTo: number, last: number) =>
	Object.fromEntries(
		range(last).map((m) => {
			const i = m + 1;
			const lines = [`primal  ${fmt(v[i])}`];
			if (i <= upTo) lines.push(`tangent ${fmt(vdot[i])}`);
			return [
				`v${i}`,
				{
					lines: i <= upTo ? lines : [],
					hi:
						i === upTo || (upTo === 2 && i === 1)
							? ("fwd" as const)
							: undefined,
				},
			];
		}),
	);
const tangentNotes: Record<number, string> = {
	3: `(8.60) product rule: $\\dot v_3 = v_1\\dot v_2 + \\dot v_1 v_2 = ${X1}\\cdot 0 + 1\\cdot ${X2} = ${t(vdot[3])}$.`,
	4: "(8.61): $\\dot v_4 = \\dot v_2\\cos v_2 = 0$. $x_2$ is held fixed, so nothing that depends only on it changes.",
	5: `(8.62): $\\dot v_5 = \\dot v_3 \\exp(v_3) = ${t(vdot[3])}\\, e^{${t(v[3], 0)}} = ${t(vdot[5])}$.`,
	6: `(8.63): $\\dot v_6 = \\dot v_3 - \\dot v_4 = ${t(vdot[6])}$.`,
	7: `(8.64): $\\dot v_7 = \\dot v_5 + \\dot v_6 = ${t(vdot[7])}$. This is $\\partial f_1/\\partial x_1$; the closed form $x_2 + x_2 e^{x_1x_2} = ${t(exactDfdx1)}$ agrees.`,
	8: `The second output costs one more line: $\\dot v_8 = \\dot v_5 v_6 + v_5\\dot v_6 = ${t(vdot[8])} = \\partial f_2/\\partial x_1$, from the same pass. For $\\partial/\\partial x_2$ we would need a second pass seeded with $(0, 1)$.`,
};
export const forwardModeSteps: GraphStepsData = {
	nodes: traceNodes(true),
	edges: traceEdges(true),
	steps: [
		{
			label: "Seed",
			note: `Inputs $x_1 = ${X1}$, $x_2 = ${X2}$. To get $\\partial/\\partial x_1$, seed the tangents with $\\dot{\\mathbf{x}} = \\mathbf{e}_1$: $\\dot v_1 = 1$ (8.58), $\\dot v_2 = 0$ (8.59). Each node now carries the pair $(v_i, \\dot v_i)$.`,
			nodes: fwdNodes(2, 8),
		},
		...range(6).map((m) => {
			const i = m + 3;
			return {
				label: `v${sub[i]}`,
				note: `$v_${i} = ${t(v[i], 4)}$. ${tangentNotes[i]}`,
				nodes: fwdNodes(i, 8),
				edges: Object.fromEntries(
					parents[i].map((p) => [`v${p}>v${i}`, { hi: "fwd" as const }]),
				),
			} satisfies GraphStep;
		}),
	],
};

/** Reverse mode: all primals are known; adjoints appear from v7 down to `upTo`. */
const revNodes = (upTo: number, hi: number[]) =>
	Object.fromEntries(
		range(7).map((m) => {
			const i = m + 1;
			const lines = [`primal  ${fmt(v[i])}`];
			if (i >= upTo) lines.push(`adjoint ${fmt(vbar[i])}`);
			return [
				`v${i}`,
				{ lines, hi: hi.includes(i) ? ("bwd" as const) : undefined },
			];
		}),
	);
const children: Record<number, number[]> = {
	1: [3],
	2: [3, 4],
	3: [5, 6],
	4: [6],
	5: [7],
	6: [7],
};
/** Local derivative dv_c / dv_p on the edge p -> c. */
const local = (p: number, c: number): number => {
	if (c === 3) return p === 1 ? v[2] : v[1];
	if (c === 4) return Math.cos(v[2]);
	if (c === 5) return v[5];
	if (c === 6) return p === 3 ? 1 : -1;
	return 1;
};
const adjNotes: Record<number, string> = {
	6: "(8.71): $v_7 = v_5 + v_6$, so $\\partial v_7/\\partial v_6 = 1$ and $\\bar v_6 = \\bar v_7 = 1$.",
	5: "(8.72): likewise $\\bar v_5 = \\bar v_7 = 1$.",
	4: "(8.73): $v_6 = v_3 - v_4$, so $\\bar v_4 = -\\bar v_6 = -1$.",
	3: `(8.74): $v_3$ has two children, $v_5$ and $v_6$, so two terms are summed: $\\bar v_3 = \\bar v_5 v_5 + \\bar v_6 = ${t(v[5])} + 1 = ${t(vbar[3])}$ (using $\\partial v_5/\\partial v_3 = \\exp v_3 = v_5$).`,
	2: `(8.75): $\\bar v_2 = \\bar v_3 v_1 + \\bar v_4\\cos v_2 = ${t(vbar[3])}\\cdot ${X1} + (-1)(${t(Math.cos(v[2]))}) = ${t(vbar[2])} = \\partial f/\\partial x_2$.`,
	1: `(8.76): $\\bar v_1 = \\bar v_3 v_2 = ${t(vbar[3])}\\cdot ${X2} = ${t(vbar[1])} = \\partial f/\\partial x_1$, the same value forward mode found. One backward pass gave both partial derivatives.`,
};
export const reverseModeSteps: GraphStepsData = {
	nodes: traceNodes(false),
	edges: traceEdges(false),
	steps: [
		{
			label: "Forward",
			note: `Reverse mode first runs the primal trace (8.50)-(8.56) and <em>keeps</em> every $v_i$: the backward pass needs $v_1$, $v_2$ and $v_5$ again. $f = v_7 = ${t(v[7], 4)}$.`,
			nodes: revNodes(8, []),
			edges: Object.fromEntries(
				traceEdges(false).map((e) => [
					`${e.from}>${e.to}`,
					{ hi: "fwd" as const },
				]),
			),
		},
		{
			label: "v̄₇",
			note: "Seed the output: $\\bar v_7 = \\partial f/\\partial v_7 = 1$ (8.70).",
			nodes: revNodes(7, [7]),
		},
		...[6, 5, 4, 3, 2, 1].map((i) => ({
			label: `v̄${sub[i]}`,
			note: `${adjNotes[i]} In general $\\bar v_i = \\sum_{j\\in\\mathrm{ch}(i)} \\bar v_j\\,\\partial v_j/\\partial v_i$ (8.69).`,
			nodes: revNodes(i, [i]),
			edges: Object.fromEntries(
				children[i].map((c) => [
					`v${i}>v${c}`,
					{
						hi: "bwd" as const,
						label: `${fmt(vbar[c], 2)}×${fmt(local(i, c), 2)}`,
					},
				]),
			),
		})),
	],
};

// ---------------------------------------------------------------------------------------------
// Cost of the three ways to get a gradient, for a fully connected D-M-K network.

export function costRow(Dn: number, Mn: number, Kn: number) {
	const W = (Dn + 1) * Mn + (Mn + 1) * Kn;
	// Multiplications: forward = W; backward = K*M for the delta sums, M for h', W for delta*z.
	const fwd = W;
	const bwd = Kn * Mn + Mn + W;
	return {
		D: Dn,
		M: Mn,
		K: Kn,
		W,
		fwd,
		backprop: fwd + bwd,
		central: 2 * W * fwd,
	};
}

// ---------------------------------------------------------------------------------------------
// Figure 8.1: one hidden unit j between the units i that feed it and the units k it feeds.

export const deltaFlowSteps: GraphStepsData = {
	nodes: [
		{ id: "i", label: "unit i", col: 0, row: 1 },
		{ id: "j", label: "unit j", sub: "h(·)", col: 1, row: 1 },
		{ id: "k1", label: "unit k = 1", col: 2, row: 0 },
		{ id: "k2", label: "unit k = 2", col: 2, row: 1 },
		{ id: "k3", label: "unit k = 3", col: 2, row: 2 },
	],
	edges: [
		{ from: "i", to: "j" },
		{ from: "j", to: "k1" },
		{ from: "j", to: "k2" },
		{ from: "j", to: "k3" },
	],
	steps: [
		{
			label: "Forward",
			note: "Forward: unit $j$ adds up its weighted inputs, $a_j = \\sum_i w_{ji} z_i$ (8.5), and passes $z_j = h(a_j)$ (8.6) to every unit $k$ it connects to.",
			nodes: {
				i: { lines: ["zᵢ"], hi: "fwd" },
				j: { lines: ["aⱼ = Σ wⱼᵢ zᵢ", "zⱼ = h(aⱼ)"], hi: "fwd" },
				k1: { lines: ["a₁ = Σ w₁ⱼ zⱼ"] },
				k2: { lines: ["a₂ = Σ w₂ⱼ zⱼ"] },
				k3: { lines: ["a₃ = Σ w₃ⱼ zⱼ"] },
			},
			edges: {
				"i>j": { hi: "fwd", label: "wⱼᵢ" },
				"j>k1": { hi: "fwd", label: "w₁ⱼ" },
				"j>k2": { hi: "fwd", label: "w₂ⱼ" },
				"j>k3": { hi: "fwd", label: "w₃ⱼ" },
			},
		},
		{
			label: "Backward",
			note: "Backward: each unit $k$ already knows its $\\delta_k$. Unit $j$ collects them through the same weights, $\\delta_j = h'(a_j)\\sum_k w_{kj}\\delta_k$ (8.13), and the derivative for the weight into $j$ is $\\delta_j z_i$ (8.10): the error at the weight's output end times the activation at its input end.",
			nodes: {
				i: { lines: ["zᵢ"] },
				j: { lines: ["δⱼ = h′(aⱼ)", "  × Σₖ wₖⱼ δₖ"], hi: "bwd" },
				k1: { lines: ["δ₁"], hi: "bwd" },
				k2: { lines: ["δ₂"], hi: "bwd" },
				k3: { lines: ["δ₃"], hi: "bwd" },
			},
			edges: {
				"i>j": { hi: "bwd", label: "∂E/∂wⱼᵢ = δⱼ zᵢ" },
				"j>k1": { hi: "bwd", label: "w₁ⱼ δ₁" },
				"j>k2": { hi: "bwd", label: "w₂ⱼ δ₂" },
				"j>k3": { hi: "bwd", label: "w₃ⱼ δ₃" },
			},
		},
	],
};

// ---------------------------------------------------------------------------------------------
// Figure 8.3: a system of modules; the Jacobian of a module carries error signals through it.

export const moduleNodes: BoxNode[] = [
	{
		id: "x",
		label: "input x",
		col: 0,
		row: 0,
		group: 5,
		note: "The data entering the system.",
	},
	{
		id: "m1",
		label: "module 1",
		sub: "z = g(x; w)",
		col: 1,
		row: 0,
		group: 0,
		note: "A learnable module with parameter <em>w</em>, e.g. a feature extractor.",
	},
	{
		id: "m2",
		label: "module 2",
		sub: "y = f(z)",
		col: 2,
		row: 0,
		group: 1,
		note: "Any differentiable module, fixed or learnable. Only its Jacobian ∂yₖ/∂zⱼ is needed to send errors through it.",
	},
	{
		id: "err",
		label: "error",
		sub: "E(y)",
		col: 3,
		row: 0,
		group: 2,
		note: "The training error.",
	},
	{
		id: "dy",
		label: "∂E/∂yₖ",
		col: 3,
		row: 1,
		group: 3,
		note: "Computed directly from the error function.",
	},
	{
		id: "dz",
		label: "∂E/∂zⱼ",
		sub: "Σₖ ∂E/∂yₖ · ∂yₖ/∂zⱼ",
		col: 2,
		row: 1,
		group: 3,
		note: "The error signal after multiplying by module 2's Jacobian: the middle factor of (8.27).",
	},
	{
		id: "dw",
		label: "∂E/∂w",
		sub: "Σⱼ ∂E/∂zⱼ · ∂zⱼ/∂w",
		col: 1,
		row: 1,
		group: 3,
		note: "The gradient for module 1's parameter, (8.27).",
	},
];
export const moduleEdges: BoxEdge[] = [
	{ from: "x", to: "m1" },
	{ from: "m1", to: "m2" },
	{ from: "m2", to: "err" },
	{ from: "err", to: "dy", dashed: true },
	{ from: "dy", to: "dz", dashed: true },
	{ from: "dz", to: "dw", dashed: true },
];

// ---------------------------------------------------------------------------------------------
// The full 2 x 2 Jacobian of (f1, f2) and a Jacobian-vector product (8.67).

export const traceJacobian = [
	[vdot[7], vdot2[7]],
	[vdot[8], vdot2[8]],
];
export const jvpR: [number, number] = [1, -1];
export const jvp = traceTangent(v, jvpR);
