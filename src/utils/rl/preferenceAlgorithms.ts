export type PreferenceMethod = "rlhf" | "dpo" | "grpo" | "drpo";

export interface PreferenceMethodSpec {
	id: PreferenceMethod;
	label: string;
	subtitle: string;
	input: string;
	nodes: { label: string; note: string; kind: "data" | "model" | "score" }[];
	loopFrom?: number;
	output: string;
	takeaway: string;
}

export const preferenceMethods: PreferenceMethodSpec[] = [
	{
		id: "rlhf",
		label: "RLHF + PPO",
		subtitle: "Learn a scalar reward, then optimize it online",
		input: "ranked response pairs",
		nodes: [
			{ label: "Reward model", note: "fits human comparisons", kind: "model" },
			{
				label: "Sample responses",
				note: "from the current policy",
				kind: "data",
			},
			{ label: "Reward + KL", note: "score each response", kind: "score" },
			{ label: "PPO update", note: "policy and value model", kind: "model" },
		],
		loopFrom: 1,
		output: "updated policy",
		takeaway:
			"Online and flexible, but it trains a reward model and usually a critic.",
	},
	{
		id: "dpo",
		label: "DPO",
		subtitle: "Turn pairwise preferences into a classification loss",
		input: "chosen / rejected pairs",
		nodes: [
			{
				label: "Policy log-ratios",
				note: "chosen versus rejected",
				kind: "model",
			},
			{
				label: "Reference log-ratios",
				note: "frozen SFT model",
				kind: "model",
			},
			{
				label: "Logistic loss",
				note: "increase the relative margin",
				kind: "score",
			},
		],
		output: "updated policy",
		takeaway:
			"Offline and simple: no explicit reward model, rollouts, or critic.",
	},
	{
		id: "grpo",
		label: "GRPO",
		subtitle: "Use other responses to the same prompt as the baseline",
		input: "prompts + verifier or reward",
		nodes: [
			{ label: "Sample a group", note: "G responses per prompt", kind: "data" },
			{ label: "Score the group", note: "reward or verifier", kind: "score" },
			{
				label: "Relative advantages",
				note: "center within the group",
				kind: "score",
			},
			{
				label: "Clipped update",
				note: "plus KL regularization",
				kind: "model",
			},
		],
		loopFrom: 0,
		output: "updated policy",
		takeaway:
			"Online and critic-free, but each prompt needs several sampled responses.",
	},
	{
		id: "drpo",
		label: "DRPO",
		subtitle: "Doubly robust preference optimization",
		input: "preference pairs",
		nodes: [
			{ label: "Preference model", note: "may be misspecified", kind: "model" },
			{ label: "Reference policy", note: "may be misspecified", kind: "model" },
			{
				label: "Doubly robust score",
				note: "combines both estimates",
				kind: "score",
			},
			{
				label: "Policy update",
				note: "robust if either is correct",
				kind: "model",
			},
		],
		output: "updated policy",
		takeaway:
			"A newer method aimed at model misspecification, not a universal name for one algorithm.",
	},
];
