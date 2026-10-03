// Figure data for the recommendation-retrieval post. The diagrams use BoxDiagram.astro;
// keeping the nodes here makes the MDX describe the ideas rather than pixel positions.

import type { BoxEdge, BoxNode } from "../payments/boxDiagram";

export const funnelNodes: BoxNode[] = [
	{
		id: "catalog",
		label: "Catalog",
		sub: "10M items",
		col: 0,
		row: 1,
		group: 0,
		note: "Every item that is eligible in principle. A request cannot afford to run the most expensive model over this whole set.",
	},
	{
		id: "retrieve",
		label: "Retrieval",
		sub: "~10³ candidates",
		col: 1,
		row: 1,
		group: 1,
		note: "Several cheap, high-recall channels run in parallel. Their outputs are merged, deduplicated and filtered.",
	},
	{
		id: "pre",
		label: "Pre-rank",
		sub: "~10² items",
		col: 2,
		row: 1,
		group: 2,
		note: "A medium-cost model removes obvious losers before the full ranker. Not every stack has a distinct pre-rank stage.",
	},
	{
		id: "rank",
		label: "Rank",
		sub: "tens",
		col: 3,
		row: 1,
		group: 3,
		note: "A richer model can now use cross-features, sequence attention and business objectives because it scores only hundreds of items.",
	},
	{
		id: "policy",
		label: "Re-rank",
		sub: "feed",
		col: 4,
		row: 1,
		group: 4,
		note: "Policy constraints enforce diversity, freshness, creator limits, safety and already-seen suppression before display.",
	},
];

export const funnelEdges: BoxEdge[] = [
	{ from: "catalog", to: "retrieve" },
	{ from: "retrieve", to: "pre" },
	{ from: "pre", to: "rank" },
	{ from: "rank", to: "policy" },
];

export const channelNodes: BoxNode[] = [
	{
		id: "req",
		label: "Request",
		sub: "user + context",
		col: 0,
		row: 2,
		group: 0,
	},
	{
		id: "popular",
		label: "Popular / fresh",
		sub: "rules + counters",
		col: 1,
		row: 0,
		group: 1,
		note: "Non-personalized lists provide robust fallbacks, activate new inventory and cover users with little history.",
	},
	{
		id: "itemcf",
		label: "Item-to-item",
		sub: "co-occurrence",
		col: 1,
		row: 1,
		group: 2,
		note: "Look up neighbors of the user's recent items in a precomputed inverted index, then aggregate their scores.",
	},
	{
		id: "twotower",
		label: "Two-tower",
		sub: "vector ANN",
		col: 1,
		row: 2,
		group: 3,
		note: "Encode the request and items separately. A dot product makes the learned match compatible with a nearest-neighbor index.",
	},
	{
		id: "graph",
		label: "Graph walk",
		sub: "live interactions",
		col: 1,
		row: 3,
		group: 4,
		note: "Walk user-item edges to find items reached by similar users. This channel can react quickly when its graph is updated online.",
	},
	{
		id: "content",
		label: "Content",
		sub: "text / image / tags",
		col: 1,
		row: 4,
		group: 5,
		note: "Content similarity works before an item has interactions, so it is a standard answer to item cold start.",
	},
	{
		id: "merge",
		label: "Merge + dedupe",
		sub: "quotas / calibration",
		col: 2,
		row: 2,
		group: 0,
		note: "The union is usually quota-controlled. Otherwise the largest or most prolific channel can crowd out complementary candidates.",
	},
	{
		id: "filter",
		label: "Eligibility",
		sub: "policy filters",
		col: 3,
		row: 2,
		group: 0,
		note: "Remove unavailable, unsafe, blocked, repeated or geographically invalid items. Filtering after ANN may require over-fetching.",
	},
];

export const channelEdges: BoxEdge[] = [
	{ from: "req", to: "popular" },
	{ from: "req", to: "itemcf" },
	{ from: "req", to: "twotower" },
	{ from: "req", to: "graph" },
	{ from: "req", to: "content" },
	{ from: "popular", to: "merge" },
	{ from: "itemcf", to: "merge" },
	{ from: "twotower", to: "merge" },
	{ from: "graph", to: "merge" },
	{ from: "content", to: "merge" },
	{ from: "merge", to: "filter" },
];
