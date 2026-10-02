// Figure data for the online-learning post. The boxes show the order of one online round;
// clicking a box explains what is and is not known at that point.

import type { BoxEdge, BoxNode } from "../payments/boxDiagram";

export const onlineRoundNodes: BoxNode[] = [
	{
		id: "context",
		label: "Observe context",
		sub: "x_t",
		col: 0,
		row: 0,
		group: 0,
		note: "The learner sees the information available before the outcome: for example, the query, ad, device, and time. It has not seen whether this impression will be clicked.",
	},
	{
		id: "predict",
		label: "Predict",
		sub: "w_t → p_t",
		col: 1,
		row: 0,
		group: 1,
		note: "The current parameters produce a prediction. For logistic regression, <code>p_t = sigmoid(w_t · x_t)</code>.",
	},
	{
		id: "outcome",
		label: "Reveal outcome",
		sub: "y_t",
		col: 2,
		row: 0,
		group: 2,
		note: "Only after the prediction does the environment reveal the label. This ordering prevents the learner from using the answer in its prediction.",
	},
	{
		id: "loss",
		label: "Measure loss",
		sub: "ℓ_t(w_t)",
		col: 2,
		row: 1,
		group: 3,
		note: "The loss scores the prediction against the revealed outcome. CTR models commonly use logistic, or log, loss.",
	},
	{
		id: "update",
		label: "Update state",
		sub: "w_t → w_{t+1}",
		col: 1,
		row: 1,
		group: 4,
		note: "The learner incorporates the new gradient. FTRL-Proximal updates two numbers per active coordinate, <code>z_i</code> and <code>n_i</code>, then derives the next weight lazily.",
	},
	{
		id: "repeat",
		label: "Next event",
		sub: "t ← t + 1",
		col: 0,
		row: 1,
		group: 5,
		note: "The process repeats without retraining from scratch. The data distribution may change between rounds; the regret definition does not require independent, identically distributed examples.",
	},
];

export const onlineRoundEdges: BoxEdge[] = [
	{ from: "context", to: "predict" },
	{ from: "predict", to: "outcome" },
	{ from: "outcome", to: "loss" },
	{ from: "loss", to: "update" },
	{ from: "update", to: "repeat" },
	{ from: "repeat", to: "context", label: "repeat" },
];
