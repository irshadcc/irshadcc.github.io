// Small stroke icons for NeuralNetworkGraph nodes (GraphNode.icon), drawn on a 24 × 24 grid with
// round caps, no fill, in the node's colour. Keys are what a node passes as `icon`.
/** A small neural network: 3–2–3 nodes (circles) joined by links that stop at their rims. */
function networkIcon(): string {
	const r = 2.2;
	const layers = [
		[
			[3.5, 5],
			[3.5, 12],
			[3.5, 19],
		],
		[
			[12, 8.5],
			[12, 15.5],
		],
		[
			[20.5, 5],
			[20.5, 12],
			[20.5, 19],
		],
	];
	const circle = ([x, y]: number[]) =>
		`M${x - r} ${y}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`;
	const link = ([x1, y1]: number[], [x2, y2]: number[]) => {
		const len = Math.hypot(x2 - x1, y2 - y1);
		const [ux, uy] = [(x2 - x1) / len, (y2 - y1) / len];
		const f = (v: number) => Math.round(v * 100) / 100;
		return `M${f(x1 + ux * r)} ${f(y1 + uy * r)}L${f(x2 - ux * r)} ${f(y2 - uy * r)}`;
	};
	const links = layers
		.slice(1)
		.flatMap((to, i) => layers[i].flatMap((a) => to.map((b) => link(a, b))));
	return [...layers.flat().map(circle), ...links].join("");
}

/** Four GPUs (small chips) on a ring, joined by arcs: every rank exchanges with every other. */
function ringIcon(): string {
	const chip = (x: number, y: number) =>
		`M${x - 2.6} ${y - 2.6}h5.2v5.2h-5.2zM${x - 1} ${y - 1}h2v2h-2z`;
	const arc = (a: number[], b: number[]) =>
		`M${a[0]} ${a[1]}A8 8 0 0 1 ${b[0]} ${b[1]}`;
	return [
		chip(12, 4),
		chip(20, 12),
		chip(12, 20),
		chip(4, 12),
		arc([15.4, 4.9], [19.1, 8.6]),
		arc([19.1, 15.4], [15.4, 19.1]),
		arc([8.6, 19.1], [4.9, 15.4]),
		arc([4.9, 8.6], [8.6, 4.9]),
	].join("");
}

export const NN_ICONS = {
	/** A neural network: routers and experts. */
	network: networkIcon(),
	/** GPUs on a ring: all-to-all. */
	ring: ringIcon(),
	/** Rows of hidden states. */
	tokens: "M4 5h16v3.5H4zM4 10.25h16v3.5H4zM4 15.5h16V19H4z",
	/** One input splitting into several. */
	router: "M12 21v-7M12 14 5 7M12 14l7-7M5 7V3M19 7V3M3 5l2-2 2 2M17 5l2-2 2 2",
	/** A funnel: selection with a capacity. */
	gate: "M3 4h18l-7 8.5V19l-4 2v-8.5z",
	/** Counting. */
	count: "M5 9h14M5 15h14M10 4 8 20M16 4l-2 16",
	/** Reordering rows. */
	permute: "M3 7h4l10 10h4M3 17h4L17 7h4M18 4l3 3-3 3M18 14l3 3-3 3",
	/** A buffer of rows. */
	buffer: "M3 5h18v14H3zM3 9.7h18M3 14.3h18M8 5v14",
	/** An expert MLP: a chip. */
	experts:
		"M7 7h10v10H7zM10 10h4v4h-4zM9.5 3v4M14.5 3v4M9.5 17v4M14.5 17v4M3 9.5h4M3 14.5h4M17 9.5h4M17 14.5h4",
	/** Every rank to every rank. */
	alltoall:
		"M6 6l12 12M18 6 6 18M6 12h12M12 6v12M4 6a2 2 0 1 0 4 0 2 2 0 1 0-4 0M16 6a2 2 0 1 0 4 0 2 2 0 1 0-4 0M4 18a2 2 0 1 0 4 0 2 2 0 1 0-4 0M16 18a2 2 0 1 0 4 0 2 2 0 1 0-4 0",
	/** Pieces coming together. */
	allgather:
		"M10 10h4v4h-4zM4 4l5 5M20 4l-5 5M4 20l5-5M20 20l-5-5M9 9H6M9 9V6M15 9h3M15 9V6M9 15H6M9 15v3M15 15h3M15 15v3",
	/** A sum split back out. */
	reducescatter:
		"M10 10h4v4h-4zM9 9 4 4M15 9l5-5M9 15l-5 5M15 15l5 5M4 4h3M4 4v3M20 4h-3M20 4v3M4 20h3M4 20v-3M20 20h-3M20 20v-3",
	/** Summation. */
	allreduce: "M16.5 7H8l5 5-5 5h8.5M12 2.5a9.5 9.5 0 1 0 .01 0",
	/** Between nodes: the network. */
	rdma: "M3 12h18M12 3c3 2.5 4.5 5.5 4.5 9S15 18.5 12 21M12 3c-3 2.5-4.5 5.5-4.5 9S9 18.5 12 21M12 3a9 9 0 1 0 .01 0",
	/** Inside a node: a link. */
	nvlink:
		"M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1",
	/** Partial results stacked. */
	partial: "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5M3 17.5l9 5 9-5",
	/** Done. */
	output: "M12 3a9 9 0 1 0 .01 0M8 12.5l2.8 2.8L16.5 9",
} as const;

export type NNIconName = keyof typeof NN_ICONS;
