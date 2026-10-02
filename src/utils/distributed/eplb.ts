// A port of DeepSeek's expert-parallelism load balancer (eplb.py, github.com/deepseek-ai/EPLB)
// for one MoE layer, used by EplbPlacement.astro.
//
// Given each logical expert's load, EPLB decides how many copies (physical experts) each one
// gets and which GPU each copy goes on:
//
//   hierarchical  pack expert groups onto nodes, replicate the hottest experts within each node,
//                 then pack the copies onto that node's GPUs (keeps a group on one node)
//   global        one node, one group: replicate globally, then pack onto every GPU
//
// Both packings are greedy: items in descending order of load, each to the lightest pack that
// still has room, so every pack gets the same number of items.

/** Assigns each item to a pack; returns [pack, rank within pack] per item. */
export function balancedPacking(
	weight: number[],
	numPacks: number,
): [number[], number[]] {
	const n = weight.length;
	const perPack = n / numPacks;
	const pack = new Array(n).fill(-1);
	const rank = new Array(n).fill(-1);
	if (perPack === 1) return [weight.map((_, i) => i), weight.map(() => 0)];
	const order = weight
		.map((w, i) => [w, i] as const)
		.sort((a, b) => b[0] - a[0] || a[1] - b[1]);
	const load = new Array(numPacks).fill(0);
	const count = new Array(numPacks).fill(0);
	for (const [w, i] of order) {
		let best = -1;
		for (let p = 0; p < numPacks; p++)
			if (count[p] < perPack && (best < 0 || load[p] < load[best])) best = p;
		pack[i] = best;
		rank[i] = count[best];
		load[best] += w;
		count[best]++;
	}
	return [pack, rank];
}

/** Gives the hottest expert (by load per copy) one more copy until there are numPhy copies. */
export function replicateExperts(weight: number[], numPhy: number) {
	const numLog = weight.length;
	const phy2log = Array.from({ length: numLog }, (_, i) => i);
	const phyRank = new Array(numLog).fill(0);
	const logCnt = new Array(numLog).fill(1);
	for (let k = numLog; k < numPhy; k++) {
		let best = 0;
		for (let e = 1; e < numLog; e++)
			if (weight[e] / logCnt[e] > weight[best] / logCnt[best]) best = e;
		phy2log.push(best);
		phyRank.push(logCnt[best]);
		logCnt[best]++;
	}
	return { phy2log, phyRank, logCnt };
}

export interface Placement {
	/** For each GPU, the logical expert of each physical expert it holds. */
	gpus: number[][];
	/** Copies of each logical expert. */
	logCnt: number[];
	/** Load on each GPU, splitting an expert's load evenly over its copies. */
	gpuLoad: number[];
}

export function rebalance(
	weight: number[],
	numPhy: number,
	numGroups: number,
	numNodes: number,
	numGpus: number,
): Placement {
	const hierarchical = numGroups % numNodes === 0;
	const groups = hierarchical ? numGroups : 1;
	const nodes = hierarchical ? numNodes : 1;
	const numLog = weight.length;
	const groupSize = numLog / groups;
	const groupsPerNode = groups / nodes;
	const logPerNode = numLog / nodes;
	const phyPerNode = numPhy / nodes;
	const gpusPerNode = numGpus / nodes;
	const phyPerGpu = numPhy / numGpus;

	// Step 1: pack whole groups onto nodes.
	const groupLoad = Array.from({ length: groups }, (_, g) =>
		weight.slice(g * groupSize, (g + 1) * groupSize).reduce((a, b) => a + b, 0),
	);
	const [gPack, gRank] = balancedPacking(groupLoad, nodes);
	// Node-major order of the logical experts ("mlog"): mlog2log[m] is a logical expert.
	const mlog2log = new Array(numLog);
	for (let g = 0; g < groups; g++)
		for (let j = 0; j < groupSize; j++)
			mlog2log[(gPack[g] * groupsPerNode + gRank[g]) * groupSize + j] =
				g * groupSize + j;

	const gpus: number[][] = Array.from({ length: numGpus }, () => []);
	const logCnt = new Array(numLog).fill(0);
	for (let node = 0; node < nodes; node++) {
		// Step 2: replicate within the node.
		const local = mlog2log.slice(node * logPerNode, (node + 1) * logPerNode);
		const w = local.map((e) => weight[e]);
		const { phy2log, logCnt: cnt } = replicateExperts(w, phyPerNode);
		local.forEach((e, i) => {
			logCnt[e] = cnt[i];
		});
		// Step 3: pack the copies onto the node's GPUs.
		const perCopy = phy2log.map((i) => w[i] / cnt[i]);
		const [pack, rank] = balancedPacking(perCopy, gpusPerNode);
		const slots: number[][] = Array.from(
			{ length: gpusPerNode },
			() => new Array(phyPerGpu),
		);
		phy2log.forEach((i, p) => {
			slots[pack[p]][rank[p]] = local[i];
		});
		slots.forEach((s, k) => {
			gpus[node * gpusPerNode + k] = s;
		});
	}
	const gpuLoad = gpus.map((s) =>
		s.reduce((a, e) => a + weight[e] / logCnt[e], 0),
	);
	return { gpus, logCnt, gpuLoad };
}

/** Without EPLB: each GPU holds its own contiguous block of experts, one copy each. */
export function naivePlacement(weight: number[], numGpus: number): Placement {
	const per = weight.length / numGpus;
	const gpus = Array.from({ length: numGpus }, (_, g) =>
		Array.from({ length: per }, (_, j) => g * per + j),
	);
	return {
		gpus,
		logCnt: weight.map(() => 1),
		gpuLoad: gpus.map((s) => s.reduce((a, e) => a + weight[e], 0)),
	};
}

/** The load statistics from EPLB's README example: two MoE layers of 12 experts. */
export const EPLB_EXAMPLE = [
	[90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
	[20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27],
];

const GROUP_COLORS = [
	"#f4a7a3",
	"#9fc8ef",
	"#a9dba6",
	"#f5d38c",
	"#c8b4ee",
	"#f3b6d6",
	"#9fe0d6",
	"#d9c7a4",
];

/** Logical loads on the left; on the right, each GPU's copies stacked by load, grouped by node. */
export function drawEplb(
	weight: number[],
	p: Placement,
	cfg: { numGroups: number; numNodes: number },
): string {
	const numLog = weight.length;
	const groupSize = numLog / cfg.numGroups;
	const color = (e: number) =>
		GROUP_COLORS[Math.floor(e / groupSize) % GROUP_COLORS.length];
	const numGpus = p.gpus.length;
	const gpusPerNode = numGpus / cfg.numNodes;
	const maxW = Math.max(...weight);
	const maxG = Math.max(...p.gpuLoad, maxW);
	const PH = 170;
	const TOP = 30;
	const base = TOP + PH;
	const y = (v: number) => base - (PH * v) / maxG;
	const out: string[] = [];

	// Logical experts.
	const LB = 16;
	const LG = 4;
	const L0 = 30;
	out.push(`<text class="h" x="${L0}" y="14">Load per logical expert</text>`);
	weight.forEach((w, e) => {
		const x = L0 + e * (LB + LG);
		const c = p.logCnt[e];
		out.push(
			`<g><title>Expert ${e} (group ${Math.floor(e / groupSize)}): load ${w}, ${c} ${c > 1 ? "copies" : "copy"}</title>`,
			`<rect class="lbar" x="${x}" y="${y(w)}" width="${LB}" height="${base - y(w)}" rx="1.5" style="fill:${color(e)}"/>`,
			`<text class="el" x="${x + LB / 2}" y="${base + 11}">${e}</text>`,
			c > 1
				? `<text class="cnt" x="${x + LB / 2}" y="${y(w) - 4}">×${c}</text>`
				: "",
			"</g>",
		);
	});
	const LW = numLog * (LB + LG) - LG;

	// GPUs.
	const GW = 30;
	const GG = 6;
	const NG = 16;
	const G0 = L0 + LW + 50;
	const gx = (g: number) =>
		G0 + g * (GW + GG) + Math.floor(g / gpusPerNode) * NG;
	const mean = p.gpuLoad.reduce((a, b) => a + b, 0) / numGpus;
	out.push(
		`<text class="h" x="${G0}" y="14">Load per GPU, by the copies it holds</text>`,
	);
	for (let n = 0; n < cfg.numNodes; n++) {
		const x0 = gx(n * gpusPerNode) - 4;
		const x1 = gx((n + 1) * gpusPerNode - 1) + GW + 4;
		out.push(
			`<rect class="node" x="${x0}" y="${TOP - 8}" width="${x1 - x0}" height="${PH + 38}" rx="4"/>`,
			`<text class="nl" x="${(x0 + x1) / 2}" y="${base + 25}">${cfg.numNodes > 1 ? `node ${n}` : "one flat group"}</text>`,
		);
	}
	p.gpus.forEach((slots, g) => {
		const x = gx(g);
		let acc = 0;
		for (const e of slots) {
			const share = weight[e] / p.logCnt[e];
			const top = y(acc + share);
			out.push(
				`<g><title>GPU ${g}: a copy of expert ${e}, load ${+share.toFixed(1)}${p.logCnt[e] > 1 ? ` (1/${p.logCnt[e]} of ${weight[e]})` : ""}</title>`,
				`<rect class="slot" x="${x}" y="${top}" width="${GW}" height="${y(acc) - top}" style="fill:${color(e)}"/>`,
				y(acc) - top > 11
					? `<text class="sl" x="${x + GW / 2}" y="${(top + y(acc)) / 2 + 3.5}">E${e}</text>`
					: "",
				"</g>",
			);
			acc += share;
		}
		out.push(`<text class="el" x="${x + GW / 2}" y="${base + 11}">${g}</text>`);
	});
	const gEnd = gx(numGpus - 1) + GW;
	out.push(
		`<line class="mean" x1="${G0 - 4}" x2="${gEnd + 4}" y1="${y(mean)}" y2="${y(mean)}"/>`,
		`<text class="ml" x="${gEnd + 8}" y="${y(mean) + 3}">mean</text>`,
		`<line class="axis" x1="${L0 - 4}" x2="${L0 + LW + 4}" y1="${base}" y2="${base}"/>`,
		`<line class="axis" x1="${G0 - 4}" x2="${gEnd + 4}" y1="${base}" y2="${base}"/>`,
	);
	const W = gEnd + 40;
	const H = base + 32;
	return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" style="min-width:${Math.round(W * 0.7)}px" role="img" aria-label="EPLB expert placement">${out.join("")}</svg>`;
}
