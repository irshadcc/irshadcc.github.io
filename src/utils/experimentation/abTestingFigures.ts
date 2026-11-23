/** Figure data for the A/B testing post (src/content/posts/ab-testing.mdx). */
import type { BoxEdge, BoxNode } from "../payments/boxDiagram";

/** The four parts of an experimentation platform and how data flows between them. */
export const platformNodes: BoxNode[] = [
	{
		id: "ui",
		label: "Owner UI",
		sub: "define, start, ramp",
		col: 0,
		row: 0,
		group: 0,
		note: "Where a team writes down the hypothesis, the variants, the audience, the traffic share and the metrics, then starts, ramps or stops the experiment. Every change is versioned so the analysis can tell who was in which variant on which day.",
	},
	{
		id: "config",
		label: "Experiment config",
		sub: "layers, buckets, %",
		col: 1,
		row: 0,
		group: 0,
		note: "The stored definition of every running experiment: which layer it lives in, which buckets it owns and how those buckets map to variants.",
	},
	{
		id: "assign",
		label: "Assignment",
		sub: "hash → bucket",
		col: 2,
		row: 0,
		group: 1,
		note: "Turns a user ID into a variant for every active experiment, using a stable hash so the same user always gets the same answer.",
	},
	{
		id: "product",
		label: "Product code",
		sub: "if variant == T …",
		col: 2,
		row: 1,
		group: 1,
		note: "The website, app or server reads the assignment and shows the matching experience. On a phone app the new code must already be in the installed version, switched on by a flag.",
	},
	{
		id: "logs",
		label: "Instrumentation",
		sub: "events + variant IDs",
		col: 1,
		row: 1,
		group: 2,
		note: "Client and server log what users saw and did, tagged with the experiment and variant IDs so events can be joined back to the assignment.",
	},
	{
		id: "pipeline",
		label: "Analysis pipeline",
		sub: "per-user metrics, tests",
		col: 0,
		row: 1,
		group: 3,
		note: "Cleans and joins the logs, computes one row per user, then the difference, confidence interval and p-value for every metric. It also runs trust checks such as the sample ratio test.",
	},
	{
		id: "card",
		label: "Scorecard",
		sub: "OEC, guardrails, SRM",
		col: 0,
		row: 2,
		group: 3,
		note: "What the team reads: the OEC first, guardrails and trust checks next, then diagnostic metrics and segments.",
	},
];

export const platformEdges: BoxEdge[] = [
	{ from: "ui", to: "config" },
	{ from: "config", to: "assign" },
	{ from: "assign", to: "product", label: "variant" },
	{ from: "product", to: "logs" },
	{ from: "logs", to: "pipeline" },
	{ from: "pipeline", to: "card" },
	{ from: "config", to: "pipeline", dashed: true },
];
