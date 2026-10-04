export type FlowKind = "derivative" | "forward" | "reverse" | "matrix";

export interface FlowNode {
	id: string;
	title: string;
	shape: string;
	detail: string;
	x: number;
	y: number;
	tone: "input" | "operation" | "output" | "gradient";
}

export interface FlowEdge {
	from: string;
	to: string;
	label: string;
	direction?: "forward" | "reverse";
}

export interface FlowScene {
	title: string;
	intro: string;
	nodes: FlowNode[];
	edges: FlowEdge[];
	formula: string;
	note: string;
}

export const FLOW_SCENES: Record<FlowKind, FlowScene> = {
	derivative: {
		title: "Derivative as a linear map",
		intro: "A small input change goes through the local linear approximation.",
		nodes: [
			{
				id: "dx",
				title: "input change",
				shape: "dx: n",
				detail: "a direction and size",
				x: 35,
				y: 92,
				tone: "input",
			},
			{
				id: "J",
				title: "derivative",
				shape: "J: m × n",
				detail: "the local linear map",
				x: 285,
				y: 92,
				tone: "operation",
			},
			{
				id: "dy",
				title: "output change",
				shape: "dy: m",
				detail: "first-order response",
				x: 535,
				y: 92,
				tone: "output",
			},
		],
		edges: [
			{ from: "dx", to: "J", label: "feed direction" },
			{ from: "J", to: "dy", label: "apply J" },
		],
		formula: String.raw`dy = J\,dx`,
		note: "The Jacobian is one representation of the derivative. The linear map is the underlying object.",
	},
	forward: {
		title: "Forward mode: push a tangent",
		intro: "Carry one chosen input perturbation through every operation.",
		nodes: [
			{
				id: "x",
				title: "x, dx",
				shape: "x, dx: n",
				detail: "value plus tangent",
				x: 20,
				y: 92,
				tone: "input",
			},
			{
				id: "g",
				title: "g",
				shape: "J_g dx",
				detail: "first local JVP",
				x: 230,
				y: 92,
				tone: "operation",
			},
			{
				id: "h",
				title: "h",
				shape: "J_h J_g dx",
				detail: "second local JVP",
				x: 440,
				y: 92,
				tone: "operation",
			},
			{
				id: "y",
				title: "y, dy",
				shape: "y, dy: m",
				detail: "output tangent",
				x: 650,
				y: 92,
				tone: "output",
			},
		],
		edges: [
			{ from: "x", to: "g", label: "dx →" },
			{ from: "g", to: "h", label: "Jg dx →" },
			{ from: "h", to: "y", label: "dy →" },
		],
		formula: String.raw`dy = J_h\left(J_g\,dx\right)`,
		note: "One Jacobian–vector product (JVP) gives the response to one input direction.",
	},
	reverse: {
		title: "Reverse mode: pull an adjoint",
		intro:
			"Run the values forward, then send one scalar loss sensitivity backward.",
		nodes: [
			{
				id: "x",
				title: "x",
				shape: "x: n",
				detail: "many inputs",
				x: 20,
				y: 92,
				tone: "input",
			},
			{
				id: "g",
				title: "g",
				shape: "u",
				detail: "saved forward value",
				x: 230,
				y: 92,
				tone: "operation",
			},
			{
				id: "h",
				title: "h",
				shape: "y",
				detail: "saved forward value",
				x: 440,
				y: 92,
				tone: "operation",
			},
			{
				id: "L",
				title: "loss",
				shape: "L ∈ ℝ",
				detail: "one output",
				x: 650,
				y: 92,
				tone: "output",
			},
		],
		edges: [
			{ from: "x", to: "g", label: "∇xL ←", direction: "reverse" },
			{ from: "g", to: "h", label: "∇uL ←", direction: "reverse" },
			{ from: "h", to: "L", label: "1 ←", direction: "reverse" },
		],
		formula: String.raw`\nabla_x L^T = \nabla_y L^T J_h J_g`,
		note: "A vector–Jacobian product (VJP) propagates the one needed row without materializing either Jacobian.",
	},
	matrix: {
		title: "A matrix gradient keeps the input shape",
		intro:
			"The Frobenius inner product turns a matrix change into the scalar loss change.",
		nodes: [
			{
				id: "dX",
				title: "perturbation",
				shape: "dX: p × q",
				detail: "an entrywise change",
				x: 35,
				y: 92,
				tone: "input",
			},
			{
				id: "G",
				title: "gradient",
				shape: "∇ₓL: p × q",
				detail: "same shape as X",
				x: 285,
				y: 92,
				tone: "gradient",
			},
			{
				id: "dL",
				title: "loss change",
				shape: "dL ∈ ℝ",
				detail: "entrywise sum",
				x: 535,
				y: 92,
				tone: "output",
			},
		],
		edges: [
			{ from: "dX", to: "G", label: "pair entries" },
			{ from: "G", to: "dL", label: "sum" },
		],
		formula: String.raw`dL = \langle \nabla_X L, dX \rangle_F = \operatorname{tr}\!\left((\nabla_X L)^T dX\right)`,
		note: "This identity is the bridge from a differential expression to the gradient you put in code.",
	},
};

export function validateFlowScenes(
	scenes: Record<FlowKind, FlowScene> = FLOW_SCENES,
): void {
	for (const [kind, scene] of Object.entries(scenes)) {
		const ids = new Set(scene.nodes.map((node) => node.id));
		if (ids.size !== scene.nodes.length)
			throw new Error(`${kind}: duplicate node id`);
		for (const edge of scene.edges) {
			if (!ids.has(edge.from) || !ids.has(edge.to))
				throw new Error(`${kind}: edge references a missing node`);
		}
	}
}

validateFlowScenes();
