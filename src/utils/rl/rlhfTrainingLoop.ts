export type RlhfNodeId =
	| "prompt"
	| "policy"
	| "rollout"
	| "reward"
	| "critic"
	| "ppo";

export interface RlhfGraphNode {
	id: RlhfNodeId;
	label: string;
	detail: string;
	x: number;
	y: number;
	w: number;
	h: number;
	tone: "data" | "model" | "sample" | "score" | "update";
}

export interface RlhfGraphEdge {
	id: string;
	from: RlhfNodeId;
	to: RlhfNodeId;
	path: string;
}

export interface RlhfStep {
	title: string;
	body: string;
	activeNodes: RlhfNodeId[];
	activeEdges: string[];
	tokens: number;
	metric: string;
}

export const rlhfNodes: RlhfGraphNode[] = [
	{
		id: "prompt",
		label: "Prompt batch",
		detail: "x",
		x: 72,
		y: 76,
		w: 112,
		h: 54,
		tone: "data",
	},
	{
		id: "policy",
		label: "Policy πθ",
		detail: "trainable LM",
		x: 238,
		y: 76,
		w: 122,
		h: 62,
		tone: "model",
	},
	{
		id: "rollout",
		label: "Rollout",
		detail: "sample y token by token",
		x: 422,
		y: 76,
		w: 142,
		h: 62,
		tone: "sample",
	},
	{
		id: "reward",
		label: "Reward + KL",
		detail: "rφ(x,y) − β KL",
		x: 606,
		y: 76,
		w: 142,
		h: 62,
		tone: "score",
	},
	{
		id: "critic",
		label: "Critic Vψ",
		detail: "estimate advantages Ât",
		x: 606,
		y: 210,
		w: 142,
		h: 62,
		tone: "model",
	},
	{
		id: "ppo",
		label: "PPO update",
		detail: "clipped policy gradient",
		x: 334,
		y: 210,
		w: 150,
		h: 62,
		tone: "update",
	},
];

export const rlhfEdges: RlhfGraphEdge[] = [
	{ id: "prompt-policy", from: "prompt", to: "policy", path: "M128 76 H173" },
	{ id: "policy-rollout", from: "policy", to: "rollout", path: "M299 76 H351" },
	{ id: "rollout-reward", from: "rollout", to: "reward", path: "M493 76 H535" },
	{ id: "reward-critic", from: "reward", to: "critic", path: "M606 107 V179" },
	{ id: "critic-ppo", from: "critic", to: "ppo", path: "M535 210 H409" },
	{
		id: "rollout-ppo",
		from: "rollout",
		to: "ppo",
		path: "M422 107 V145 Q422 166 401 175 L380 185",
	},
	{
		id: "ppo-policy",
		from: "ppo",
		to: "policy",
		path: "M259 210 H205 Q184 210 184 189 V120 Q184 107 197 107",
	},
];

export const rlhfSteps: RlhfStep[] = [
	{
		title: "1 · Load prompts",
		body: "Draw a batch of prompts x. No response or reward exists yet.",
		activeNodes: ["prompt"],
		activeEdges: [],
		tokens: 0,
		metric: "batch: x₁ … xᴮ",
	},
	{
		title: "2 · Run the policy",
		body: "The trainable policy receives each prompt and produces a distribution over the next token.",
		activeNodes: ["prompt", "policy"],
		activeEdges: ["prompt-policy"],
		tokens: 0,
		metric: "πθ(· | x)",
	},
	{
		title: "3 · Sample a token",
		body: "Sample one token, append it to the state, and run the policy again.",
		activeNodes: ["policy", "rollout"],
		activeEdges: ["policy-rollout"],
		tokens: 1,
		metric: "y₁ ∼ πθ(· | x)",
	},
	{
		title: "4 · Finish the rollout",
		body: "Repeat autoregressive sampling. Save every token and its old log-probability for PPO.",
		activeNodes: ["policy", "rollout"],
		activeEdges: ["policy-rollout"],
		tokens: 5,
		metric: "log πold(yt | x, y<t)",
	},
	{
		title: "5 · Score the response",
		body: "The reward model scores the response. A KL penalty discourages drift from the reference policy.",
		activeNodes: ["rollout", "reward"],
		activeEdges: ["rollout-reward"],
		tokens: 5,
		metric: "R = rφ(x,y) − β KL",
	},
	{
		title: "6 · Estimate advantages",
		body: "The critic predicts each token state's value. Returns minus values produce token-level advantages.",
		activeNodes: ["reward", "critic"],
		activeEdges: ["reward-critic"],
		tokens: 5,
		metric: "Ât ≈ Rt − Vψ(st)",
	},
	{
		title: "7 · Form a PPO minibatch",
		body: "PPO combines tokens, old log-probabilities, advantages and returns from the rollout buffer.",
		activeNodes: ["rollout", "critic", "ppo"],
		activeEdges: ["rollout-ppo", "critic-ppo"],
		tokens: 5,
		metric: "{yt, log πold, Ât, Rt}",
	},
	{
		title: "8 · Update and repeat",
		body: "Clipped-gradient epochs update θ. The new policy generates the next rollout, closing the online loop.",
		activeNodes: ["ppo", "policy"],
		activeEdges: ["ppo-policy"],
		tokens: 5,
		metric: "θ ← θ + η ∇θ LPPO",
	},
];

export function validateRlhfGraph(): string[] {
	const errors: string[] = [];
	const ids = new Set(rlhfNodes.map((node) => node.id));
	if (ids.size !== rlhfNodes.length) errors.push("node ids must be unique");
	for (const edge of rlhfEdges) {
		if (!ids.has(edge.from) || !ids.has(edge.to))
			errors.push(`edge ${edge.id} has a missing endpoint`);
	}
	for (const [index, step] of rlhfSteps.entries()) {
		for (const id of step.activeNodes)
			if (!ids.has(id)) errors.push(`step ${index} references node ${id}`);
		for (const id of step.activeEdges) {
			if (!rlhfEdges.some((edge) => edge.id === id))
				errors.push(`step ${index} references edge ${id}`);
		}
	}
	return errors;
}
