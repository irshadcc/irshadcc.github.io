// Rank layout of a Megatron-LM style job, for the TopologyExplorer component.
//
// Megatron-LM (megatron/core/parallel_state.py) numbers ranks with an order string such as
// "tp-cp-ep-dp-pp": a rank is a mixed-radix number whose first digit (fastest varying) is the
// first name in the order. It builds two generators over the same world and the same order:
//
//   dense  (attention, dense MLP): sizes tp, cp, dp, pp and ep = 1
//   expert (MoE layers):           sizes tp (expert TP = TP here), ep, expert dp, pp and cp = 1,
//                                  where expert dp = world / (tp · ep · pp)
//
// TP, CP, DP and PP groups come from the dense generator, EP and expert-DP groups from the expert
// one. A group along one dimension fixes every other digit, so its ranks are an arithmetic
// progression base + k · stride: nothing per rank is stored, which keeps jobs with hundreds of
// thousands of GPUs cheap.
//
// Ranks are placed on nodes in order: rank r runs on GPU r % gpusPerNode of node
// floor(r / gpusPerNode), which is what torchrun does with one process per GPU.
//
// Also here: the geometry of the two views (LogicalLayout, PhysicalLayout), with hit testing, so
// it can be tested without a browser.

import { ClusterNetwork, HOP_CLASSES, type HopClass } from "./network";

export type Dim = "tp" | "cp" | "ep" | "dp" | "pp";
export const DIMS: readonly Dim[] = ["tp", "cp", "ep", "dp", "pp"];
/** A group kind: a dimension, or expert data parallel ("edp"). */
export type GroupDim = Dim | "edp";
export const GROUP_NAMES: Record<GroupDim, string> = {
	tp: "Tensor parallel",
	cp: "Context parallel",
	ep: "Expert parallel",
	dp: "Data parallel",
	pp: "Pipeline parallel",
	edp: "Expert data parallel",
};
export const DIM_NAMES: Record<Dim, string> = GROUP_NAMES;
/** Jobs larger than this are refused (the per-dimension table walks every rank). */
export const MAX_WORLD = 10_000_000;

export interface TopologyConfig {
	/** Group size of each dimension; a disabled dimension counts as 1. */
	sizes: Record<Dim, number>;
	enabled: Record<Dim, boolean>;
	/** Megatron order string as a list, fastest-varying first; lists all five dimensions. */
	order: Dim[];
	gpusPerNode: number;
	nodes: number;
	/** Nodes under one set of ToR switches (a pod), and spine switches per pod (= core switches). */
	nodesPerPod: number;
	spinesPerPod: number;
}

/** Network hops from nearest to farthest: NVLink, ToR, spine, core. */
export const HOP_ORDER: HopClass[] = HOP_CLASSES.map((h) => h.id);
export const HOP_NAMES = Object.fromEntries(
	HOP_CLASSES.map((h) => [h.id, h.name]),
) as Record<HopClass, string>;

/** Mixed-radix rank numbering over `order`, as Megatron's RankGenerator. */
export class RankGenerator {
	readonly world: number;
	readonly stride: Record<Dim, number>;
	readonly size: Record<Dim, number>;
	readonly order: readonly Dim[];

	constructor(size: Record<Dim, number>, order: readonly Dim[]) {
		this.size = size;
		this.order = order;
		let s = 1;
		const stride = {} as Record<Dim, number>;
		for (const d of order) {
			stride[d] = s;
			s *= size[d];
		}
		this.stride = stride;
		this.world = s;
	}

	digit(rank: number, d: Dim): number {
		return Math.floor(rank / this.stride[d]) % this.size[d];
	}

	coordOf(rank: number): Record<Dim, number> {
		const c = {} as Record<Dim, number>;
		for (const d of this.order) c[d] = this.digit(rank, d);
		return c;
	}

	rankOf(c: Partial<Record<Dim, number>>): number {
		let r = 0;
		for (const d of this.order) r += (c[d] ?? 0) * this.stride[d];
		return r;
	}

	/** All groups for a token such as "dp" or "tp-pp", in Megatron's order (for small worlds). */
	getRanks(token: string): number[][] {
		const masked = token.split("-") as Dim[];
		const inGroup = this.order.filter((d) => masked.includes(d));
		const outside = this.order.filter((d) => !masked.includes(d));
		const count = (ds: Dim[]) => ds.reduce((n, d) => n * this.size[d], 1);
		const rankOf = (ds: Dim[], i: number) => {
			let r = 0;
			for (const d of ds) {
				r += (i % this.size[d]) * this.stride[d];
				i = Math.floor(i / this.size[d]);
			}
			return r;
		};
		const groups: number[][] = [];
		for (let g = 0; g < count(outside); g++) {
			const base = rankOf(outside, g);
			groups.push(
				Array.from(
					{ length: count(inGroup) },
					(_, i) => base + rankOf(inGroup, i),
				),
			);
		}
		return groups;
	}
}

/** The ranks base, base + stride, ..., base + (size - 1) · stride. */
export interface Group {
	base: number;
	stride: number;
	size: number;
}
export const members = (g: Group) =>
	Array.from({ length: g.size }, (_, k) => g.base + k * g.stride);
export const lastOf = (g: Group) => g.base + (g.size - 1) * g.stride;

/** "0–3", "0, 4, 8, 12", or "0, 4, …, 49996 (12500 ranks, stride 4)" for long groups. */
export function formatGroup(g: Group, max = 8): string {
	if (g.size === 1) return `${g.base}`;
	if (g.stride === 1)
		return g.size === 2 ? `${g.base}, ${g.base + 1}` : `${g.base}–${lastOf(g)}`;
	if (g.size <= max) return members(g).join(", ");
	return `${g.base}, ${g.base + g.stride}, …, ${lastOf(g)} (${g.size} ranks, stride ${g.stride})`;
}

export interface DimSummary {
	dim: GroupDim;
	size: number;
	groups: number;
	stride: number;
	/** Fewest and most nodes one group spans. */
	minNodes: number;
	maxNodes: number;
	/** Farthest hop between two ranks of any one group. */
	maxHop: HopClass;
}

export class Topology {
	readonly world: number;
	readonly expertDp: number;
	readonly dense: RankGenerator;
	readonly expert: RankGenerator;
	/** Dimensions with size > 1, in DIMS order. */
	readonly active: Dim[];
	/** Group kinds shown: the active dimensions, plus expert DP when EP is on. */
	readonly groupDims: GroupDim[];
	readonly sizes: Record<Dim, number>;
	readonly order: Dim[];
	readonly gpusPerNode: number;
	readonly nodes: number;
	readonly net: ClusterNetwork;

	constructor(
		sizes: Record<Dim, number>,
		order: Dim[],
		gpusPerNode: number,
		nodes: number,
		nodesPerPod = nodes,
		spinesPerPod = 1,
	) {
		this.net = new ClusterNetwork({
			gpusPerServer: gpusPerNode,
			serversPerPod: nodesPerPod,
			spinesPerPod,
		});
		this.sizes = sizes;
		this.order = order;
		this.gpusPerNode = gpusPerNode;
		this.nodes = nodes;
		this.world = sizes.tp * sizes.cp * sizes.dp * sizes.pp;
		this.expertDp = this.world / (sizes.tp * sizes.ep * sizes.pp);
		this.dense = new RankGenerator({ ...sizes, ep: 1 }, order);
		this.expert = new RankGenerator(
			{ ...sizes, cp: 1, dp: this.expertDp },
			order,
		);
		this.active = DIMS.filter((d) => sizes[d] > 1);
		this.groupDims = [
			...this.active,
			...(sizes.ep > 1 && this.expertDp > 1 ? (["edp"] as const) : []),
		];
	}

	nodeOf(rank: number): number {
		return Math.floor(rank / this.gpusPerNode);
	}

	/** The generator and digit a group kind varies. */
	private axis(dim: GroupDim): [RankGenerator, Dim] {
		if (dim === "ep") return [this.expert, "ep"];
		if (dim === "edp") return [this.expert, "dp"];
		return [this.dense, dim];
	}

	groupOf(dim: GroupDim, rank: number): Group {
		const [g, d] = this.axis(dim);
		const stride = g.stride[d];
		return { base: rank - g.digit(rank, d) * stride, stride, size: g.size[d] };
	}

	/** Whether `rank` is in the group along `dim` that contains `sel`. */
	sameGroup(dim: GroupDim, sel: number, rank: number): boolean {
		const [g, d] = this.axis(dim);
		const st = g.stride[d];
		return sel - g.digit(sel, d) * st === rank - g.digit(rank, d) * st;
	}

	/** The switches traffic between two ranks crosses (ids from network.ts), and its hop class. */
	route(a: number, b: number) {
		return this.net.route(a, b);
	}

	/**
	 * Farthest hop between any two ranks of a group: core if it spans pods; spine if it spans
	 * nodes and rails (two such ranks then differ in both); ToR if it spans nodes on one rail.
	 */
	groupHop(g: Group): HopClass {
		const first = this.net.place(g.base);
		let nodes = false;
		let rails = false;
		for (let k = 1; k < g.size; k++) {
			const p = this.net.place(g.base + k * g.stride);
			if (p.pod !== first.pod) return "core";
			if (p.server !== first.server) nodes = true;
			if (p.rail !== first.rail) rails = true;
		}
		return nodes ? (rails ? "spine" : "tor") : "nvlink";
	}

	/** Distinct nodes a group touches (its ranks increase, so count node changes). */
	nodesOf(g: Group): number[] {
		const out: number[] = [];
		for (let k = 0; k < g.size; k++) {
			const n = this.nodeOf(g.base + k * g.stride);
			if (out[out.length - 1] !== n) out.push(n);
		}
		return out;
	}

	private span(g: Group): number {
		let count = 0;
		let last = -1;
		for (let k = 0; k < g.size; k++) {
			const n = this.nodeOf(g.base + k * g.stride);
			if (n !== last) count++;
			last = n;
		}
		return count;
	}

	/** Per group kind: size, number of groups, stride and nodes spanned. Walks every rank once. */
	summary(): DimSummary[] {
		return this.groupDims.map((dim) => {
			const [g, d] = this.axis(dim);
			let minNodes = Number.POSITIVE_INFINITY;
			let maxNodes = 0;
			let maxHop = 0;
			for (let r = 0; r < this.world; r++) {
				if (g.digit(r, d) !== 0) continue;
				const grp = { base: r, stride: g.stride[d], size: g.size[d] };
				const n = this.span(grp);
				if (n < minNodes) minNodes = n;
				if (n > maxNodes) maxNodes = n;
				if (maxHop < 3)
					maxHop = Math.max(maxHop, HOP_ORDER.indexOf(this.groupHop(grp)));
			}
			return {
				dim,
				size: g.size[d],
				groups: this.world / g.size[d],
				stride: g.stride[d],
				minNodes,
				maxNodes,
				maxHop: HOP_ORDER[maxHop],
			};
		});
	}
}

export type BuildResult =
	| { ok: true; topo: Topology }
	| { ok: false; error: string };

/** Build both generators, or say why Megatron would reject the configuration. */
export function buildTopology(cfg: TopologyConfig): BuildResult {
	const sizes = {} as Record<Dim, number>;
	for (const d of DIMS) {
		const v = cfg.enabled[d] ? cfg.sizes[d] : 1;
		if (!Number.isInteger(v) || v < 1)
			return {
				ok: false,
				error: `${d.toUpperCase()} must be a positive integer.`,
			};
		sizes[d] = v;
	}
	const gpus = cfg.gpusPerNode * cfg.nodes;
	const world = sizes.tp * sizes.cp * sizes.dp * sizes.pp;
	if (world !== gpus) {
		return {
			ok: false,
			error: `TP × CP × DP × PP = ${world.toLocaleString()} ranks, but ${cfg.nodes.toLocaleString()} nodes × ${cfg.gpusPerNode} GPUs = ${gpus.toLocaleString()} GPUs. They must match.`,
		};
	}
	if (world > MAX_WORLD)
		return {
			ok: false,
			error: `Keep the job to ${MAX_WORLD.toLocaleString()} GPUs or fewer.`,
		};
	if ((sizes.cp * sizes.dp) % sizes.ep !== 0) {
		return {
			ok: false,
			error: `EP (${sizes.ep}) must divide CP × DP (${sizes.cp * sizes.dp}): MoE layers split the CP × DP ranks into EP × expert DP.`,
		};
	}
	for (const [k, v] of [
		["Nodes per pod", cfg.nodesPerPod],
		["Spines per pod", cfg.spinesPerPod],
	] as const) {
		if (!Number.isInteger(v) || v < 1)
			return { ok: false, error: `${k} must be a positive integer.` };
	}
	const topo = new Topology(
		sizes,
		cfg.order,
		cfg.gpusPerNode,
		cfg.nodes,
		cfg.nodesPerPod,
		cfg.spinesPerPod,
	);
	if (
		!(
			cfg.order[cfg.order.length - 1] === "pp" ||
			sizes.pp === 1 ||
			topo.expertDp === sizes.dp
		)
	) {
		return {
			ok: false,
			error:
				"With PP > 1 and pp not last in the order, Megatron needs expert DP = DP (that is, EP = CP).",
		};
	}
	// PP groups are progressions of PP ranks, enumerated by base in both generators, so Megatron's
	// check that both generators give the same PP groups reduces to equal PP strides.
	if (sizes.pp > 1 && topo.dense.stride.pp !== topo.expert.stride.pp) {
		return {
			ok: false,
			error:
				"This order gives MoE layers different pipeline groups from dense layers, which Megatron rejects.",
		};
	}
	return { ok: true, topo };
}

/**
 * Compute time of a rank for one step: a base time with ±2% deterministic noise, plus `slowBy`
 * on stragglers. Busy time, not wall time: in synchronous training every rank's wall time equals
 * the slowest one's, so only busy time points at the straggler.
 */
export function timeOf(
	rank: number,
	slow: boolean,
	base = 1.0,
	slowBy = 0.3,
): number {
	let h = Math.imul(rank + 1, 0x9e3779b1);
	h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
	const noise = (((h ^ (h >>> 13)) >>> 0) % 1000) / 1000; // 0 .. 1
	return base * (0.98 + 0.04 * noise) + (slow ? slowBy : 0);
}

// ---- Geometry, in content pixels at zoom 1.

export const SQ = 26; // one rank
const PAD = 6; // inside a cell or node box

/** Which stages, replicas or nodes a view keeps; null keeps all. */
export interface ViewFilter {
	stages: number[] | null;
	replicas: number[] | null;
	nodes: number[] | null;
}

/**
 * Slowest compute time per stage, per replica and per node, from one pass over every rank (as
 * LogicalDistributedTopology's per-stage and per-replica strips), and the ones above a threshold.
 */
export function slowParts(topo: Topology, timeOfRank: (r: number) => number) {
	const { dp: D, pp: P } = topo.sizes;
	const stage = new Float64Array(P);
	const replica = new Float64Array(D);
	const node = new Float64Array(topo.nodes);
	let lo = Number.POSITIVE_INFINITY;
	let hi = 0;
	for (let r = 0; r < topo.world; r++) {
		const v = timeOfRank(r);
		const p = topo.dense.digit(r, "pp");
		const d = topo.dense.digit(r, "dp");
		const n = topo.nodeOf(r);
		if (v > stage[p]) stage[p] = v;
		if (v > replica[d]) replica[d] = v;
		if (v > node[n]) node[n] = v;
		if (v < lo) lo = v;
		if (v > hi) hi = v;
	}
	const above = (a: Float64Array, t: number) => {
		const out: number[] = [];
		for (let i = 0; i < a.length; i++) if (a[i] > t) out.push(i);
		return out;
	};
	return {
		stage,
		replica,
		node,
		min: lo,
		max: hi,
		/** The parts whose slowest rank takes longer than `t`. */
		filter: (t: number): ViewFilter => ({
			stages: above(stage, t),
			replicas: above(replica, t),
			nodes: above(node, t),
		}),
	};
}

/** Index of each kept id in a sorted list, for laying out a subset. */
const slots = (ids: number[] | null) =>
	ids ? new Map(ids.map((v, i) => [v, i])) : null;

/**
 * Columns = pipeline stages, rows = replicas, cells = CP chunks of TP squares. With `stages` or
 * `replicas`, only those columns and rows are laid out, packed together.
 */
export class LogicalLayout {
	readonly tpCols: number;
	readonly blockH: number;
	readonly cellW: number;
	readonly cellH: number;
	readonly colGap = 30;
	readonly rowGap = 14;
	readonly blockGap = 10;
	readonly margin = 8;
	readonly width: number;
	readonly height: number;
	readonly topo: Topology;
	/** Number of columns and rows laid out. */
	readonly nCols: number;
	readonly nRows: number;
	private readonly stages: number[] | null;
	private readonly replicas: number[] | null;
	private readonly stageSlot: Map<number, number> | null;
	private readonly replicaSlot: Map<number, number> | null;

	constructor(
		topo: Topology,
		stages: number[] | null = null,
		replicas: number[] | null = null,
	) {
		this.topo = topo;
		const { tp: T, cp: C, dp: D, pp: P } = topo.sizes;
		this.stages = stages;
		this.replicas = replicas;
		this.stageSlot = slots(stages);
		this.replicaSlot = slots(replicas);
		this.nCols = stages ? stages.length : P;
		this.nRows = replicas ? replicas.length : D;
		this.tpCols = Math.min(T, 8);
		this.blockH = Math.ceil(T / this.tpCols) * SQ;
		this.cellW = this.tpCols * SQ + 2 * PAD;
		this.cellH = C * this.blockH + (C - 1) * this.blockGap + 2 * PAD;
		this.width =
			2 * this.margin +
			this.nCols * this.cellW +
			Math.max(0, this.nCols - 1) * this.colGap;
		this.height =
			2 * this.margin +
			this.nRows * this.cellH +
			Math.max(0, this.nRows - 1) * this.rowGap;
	}

	/** The stage in column i and the replica in row j. */
	stageAt(i: number) {
		return this.stages ? this.stages[i] : i;
	}
	replicaAt(j: number) {
		return this.replicas ? this.replicas[j] : j;
	}
	private col(p: number) {
		return this.stageSlot ? (this.stageSlot.get(p) ?? -1) : p;
	}
	private row(d: number) {
		return this.replicaSlot ? (this.replicaSlot.get(d) ?? -1) : d;
	}
	/** Whether a rank's stage and replica are both laid out. */
	shown(rank: number) {
		return (
			this.col(this.topo.dense.digit(rank, "pp")) >= 0 &&
			this.row(this.topo.dense.digit(rank, "dp")) >= 0
		);
	}
	cellX(p: number) {
		return this.margin + this.col(p) * (this.cellW + this.colGap);
	}
	cellY(d: number) {
		return this.margin + this.row(d) * (this.cellH + this.rowGap);
	}
	blockY(d: number, c: number) {
		return this.cellY(d) + PAD + c * (this.blockH + this.blockGap);
	}
	squareXY(c: Record<Dim, number>) {
		return {
			x: this.cellX(c.pp) + PAD + (c.tp % this.tpCols) * SQ,
			y: this.blockY(c.dp, c.cp) + Math.floor(c.tp / this.tpCols) * SQ,
		};
	}
	pos(rank: number) {
		return this.squareXY(this.topo.dense.coordOf(rank));
	}
	/** Columns and rows (as slots; map with stageAt / replicaAt) overlapping [x0, x1] × [y0, y1]. */
	visible(x0: number, y0: number, x1: number, y1: number) {
		const cw = this.cellW + this.colGap;
		const ch = this.cellH + this.rowGap;
		return {
			i0: Math.max(0, Math.floor((x0 - this.margin) / cw)),
			i1: Math.min(this.nCols - 1, Math.floor((x1 - this.margin) / cw)),
			j0: Math.max(0, Math.floor((y0 - this.margin) / ch)),
			j1: Math.min(this.nRows - 1, Math.floor((y1 - this.margin) / ch)),
		};
	}
	hit(x: number, y: number): number | null {
		const { tp: T, cp: C } = this.topo.sizes;
		const i = Math.floor((x - this.margin) / (this.cellW + this.colGap));
		const j = Math.floor((y - this.margin) / (this.cellH + this.rowGap));
		if (i < 0 || i >= this.nCols || j < 0 || j >= this.nRows) return null;
		const p = this.stageAt(i);
		const d = this.replicaAt(j);
		const ix = x - this.cellX(p) - PAD;
		const iy = y - this.cellY(d) - PAD;
		const c = Math.floor(iy / (this.blockH + this.blockGap));
		if (
			ix < 0 ||
			iy < 0 ||
			c >= C ||
			iy - c * (this.blockH + this.blockGap) >= this.blockH
		)
			return null;
		const col = Math.floor(ix / SQ);
		const row = Math.floor((iy - c * (this.blockH + this.blockGap)) / SQ);
		const t = row * this.tpCols + col;
		if (col >= this.tpCols || t >= T) return null;
		return this.topo.dense.rankOf({ tp: t, cp: c, dp: d, pp: p });
	}
}

/**
 * The physical cluster as network.ts wires it: pods of `nodesPerPod` nodes, each pod with a row
 * of spine switches above a row of ToR switches (one per rail, i.e. per GPU index), and its nodes
 * below. Core switches are not placed here: the explorer draws them in a band pinned to the top
 * of the view, so they stay visible however far the reader scrolls. Pods sit on a grid that is
 * about as wide as tall overall. With `nodes`, only those nodes (and the pods holding them) are
 * laid out, packed together.
 */
export class PhysicalLayout {
	readonly topo: Topology;
	readonly cols: number;
	readonly nodeW: number;
	readonly nodeH: number;
	readonly label = 14;
	readonly gap = 10;
	readonly margin = 8;
	/** Switch boxes. */
	readonly swW = 40;
	readonly swH = 18;
	readonly swGap = 6;
	readonly podPad = 10;
	readonly podGap = 22;
	readonly nodesPerPod: number;
	readonly spinesPerPod: number;
	/** Number of pods laid out. */
	readonly pods: number;
	readonly podCols: number;
	readonly podW: number;
	readonly podH: number;
	readonly spineY: number;
	readonly torY: number;
	readonly nodesY: number;
	readonly podsPerRow: number;
	readonly podRows: number;
	readonly width: number;
	readonly height: number;
	/** With a node subset: kept pods in order, their kept nodes, and each kept node's slot. */
	private readonly podIds: number[] | null = null;
	private readonly podSlot: Map<number, number> | null = null;
	private readonly podNodes: Map<number, number[]> | null = null;
	private readonly nodeSlot: Map<number, number> | null = null;

	constructor(topo: Topology, nodes: number[] | null = null) {
		this.topo = topo;
		const g = topo.gpusPerNode;
		const { serversPerPod, spinesPerPod } = topo.net.shape;
		this.nodesPerPod = serversPerPod;
		this.spinesPerPod = spinesPerPod;
		this.cols = Math.min(g, 8);
		this.nodeW = this.cols * SQ + 2 * PAD;
		this.nodeH = Math.ceil(g / this.cols) * SQ + 2 * PAD + this.label;
		let inPod: number;
		if (nodes) {
			const byPod = new Map<number, number[]>();
			for (const n of nodes) {
				const p = Math.floor(n / serversPerPod);
				const list = byPod.get(p) ?? [];
				list.push(n);
				byPod.set(p, list);
			}
			this.podIds = [...byPod.keys()].sort((a, b) => a - b);
			this.podSlot = slots(this.podIds);
			this.podNodes = byPod;
			this.nodeSlot = new Map();
			for (const list of byPod.values())
				list.forEach((n, j) => this.nodeSlot?.set(n, j));
			this.pods = this.podIds.length;
			inPod = Math.max(1, ...[...byPod.values()].map((l) => l.length));
		} else {
			this.pods = Math.ceil(topo.nodes / serversPerPod);
			inPod = Math.min(serversPerPod, topo.nodes);
		}
		this.podCols = Math.min(inPod, 4);
		const nodeRows = Math.ceil(inPod / this.podCols);
		const row = (n: number) => n * this.swW + (n - 1) * this.swGap;
		const inner = Math.max(
			this.podCols * this.nodeW + (this.podCols - 1) * this.gap,
			row(g),
			row(spinesPerPod),
		);
		this.podW = inner + 2 * this.podPad;
		// pod label, spine row, wires, ToR row, wires, nodes.
		this.spineY = this.podPad + 14;
		this.torY = this.spineY + this.swH + 28;
		this.nodesY = this.torY + this.swH + 24;
		this.podH =
			this.nodesY +
			nodeRows * this.nodeH +
			(nodeRows - 1) * this.gap +
			this.podPad;
		const fit = Math.max(1, Math.floor(900 / (this.podW + this.podGap)));
		const square = Math.ceil(
			Math.sqrt(
				(this.pods * (this.podH + this.podGap)) / (this.podW + this.podGap),
			),
		);
		const want = Math.max(1, Math.min(this.pods, Math.max(fit, square)));
		this.podRows = Math.ceil(this.pods / want);
		this.podsPerRow = Math.max(
			1,
			Math.ceil(this.pods / Math.max(1, this.podRows)),
		);
		this.width =
			2 * this.margin +
			this.podsPerRow * this.podW +
			(this.podsPerRow - 1) * this.podGap;
		this.height =
			2 * this.margin +
			this.podRows * this.podH +
			Math.max(0, this.podRows - 1) * this.podGap;
	}

	/** The pod in slot i (pods are laid out in slot order). */
	podAt(i: number) {
		return this.podIds ? this.podIds[i] : i;
	}
	/** The nodes laid out in pod p. */
	nodesIn(p: number): number[] {
		if (this.podNodes) return this.podNodes.get(p) ?? [];
		const first = p * this.nodesPerPod;
		const last = Math.min(this.topo.nodes, first + this.nodesPerPod);
		return Array.from(
			{ length: Math.max(0, last - first) },
			(_, i) => first + i,
		);
	}
	/** Whether a rank's node is laid out. */
	shown(rank: number) {
		return !this.nodeSlot || this.nodeSlot.has(this.topo.nodeOf(rank));
	}
	podXY(p: number) {
		const i = this.podSlot ? (this.podSlot.get(p) ?? 0) : p;
		return {
			x: this.margin + (i % this.podsPerRow) * (this.podW + this.podGap),
			y:
				this.margin +
				Math.floor(i / this.podsPerRow) * (this.podH + this.podGap),
		};
	}
	/** Top-left of switch i in a row of n, centred in the pod. */
	private swX(p: number, i: number, n: number) {
		const w = n * this.swW + (n - 1) * this.swGap;
		return this.podXY(p).x + (this.podW - w) / 2 + i * (this.swW + this.swGap);
	}
	spineXY(p: number, s: number) {
		return {
			x: this.swX(p, s, this.spinesPerPod),
			y: this.podXY(p).y + this.spineY,
		};
	}
	torXY(p: number, rail: number) {
		return {
			x: this.swX(p, rail, this.topo.gpusPerNode),
			y: this.podXY(p).y + this.torY,
		};
	}
	nodeXY(n: number) {
		const p = Math.floor(n / this.nodesPerPod);
		const i = this.nodeSlot
			? (this.nodeSlot.get(n) ?? 0)
			: n % this.nodesPerPod;
		const o = this.podXY(p);
		const w = this.podCols * this.nodeW + (this.podCols - 1) * this.gap;
		return {
			x:
				o.x +
				(this.podW - w) / 2 +
				(i % this.podCols) * (this.nodeW + this.gap),
			y:
				o.y +
				this.nodesY +
				Math.floor(i / this.podCols) * (this.nodeH + this.gap),
		};
	}
	pos(rank: number) {
		const g = this.topo.gpusPerNode;
		const { x, y } = this.nodeXY(Math.floor(rank / g));
		const i = rank % g;
		return {
			x: x + PAD + (i % this.cols) * SQ,
			y: y + this.label + PAD + Math.floor(i / this.cols) * SQ,
		};
	}
	/** Pods overlapping [x0, x1] × [y0, y1]. */
	visiblePods(x0: number, y0: number, x1: number, y1: number): number[] {
		const cw = this.podW + this.podGap;
		const ch = this.podH + this.podGap;
		const c0 = Math.max(0, Math.floor((x0 - this.margin) / cw));
		const c1 = Math.min(
			this.podsPerRow - 1,
			Math.floor((x1 - this.margin) / cw),
		);
		const r0 = Math.max(0, Math.floor((y0 - this.margin) / ch));
		const r1 = Math.min(this.podRows - 1, Math.floor((y1 - this.margin) / ch));
		const out: number[] = [];
		for (let r = r0; r <= r1; r++) {
			for (let c = c0; c <= c1; c++) {
				const i = r * this.podsPerRow + c;
				if (i < this.pods) out.push(this.podAt(i));
			}
		}
		return out;
	}
	hit(x: number, y: number): number | null {
		const g = this.topo.gpusPerNode;
		const col = Math.floor((x - this.margin) / (this.podW + this.podGap));
		const row = Math.floor((y - this.margin) / (this.podH + this.podGap));
		const slot = row * this.podsPerRow + col;
		if (col < 0 || col >= this.podsPerRow || row < 0 || slot >= this.pods)
			return null;
		const list = this.nodesIn(this.podAt(slot));
		if (!list.length) return null;
		// The pod's first node is in slot 0, so it marks the top-left of the node grid.
		const o = this.nodeXY(list[0]);
		const ni = Math.floor((x - o.x) / (this.nodeW + this.gap));
		const nj = Math.floor((y - o.y) / (this.nodeH + this.gap));
		if (x < o.x || y < o.y || ni >= this.podCols) return null;
		const n = list[nj * this.podCols + ni];
		if (n === undefined) return null;
		const q = this.nodeXY(n);
		const gx = Math.floor((x - q.x - PAD) / SQ);
		const gy = Math.floor((y - q.y - this.label - PAD) / SQ);
		const i = gy * this.cols + gx;
		if (
			x - q.x - PAD < 0 ||
			y - q.y - this.label - PAD < 0 ||
			gx >= this.cols ||
			i >= g
		)
			return null;
		return n * g + i;
	}
}

/** A box in content pixels. */
export interface Rect {
	x: number;
	y: number;
	w: number;
	h: number;
}
/** A link drawn in the focused view: ends are "gpu:<rank>" or switch ids from network.ts. */
export interface FocusEdge {
	a: string;
	b: string;
	/** The closest group the two ranks of a route using this link share, if any. */
	dim: Dim | null;
}

/** Most ranks the focused view takes. */
export const MAX_SELECTION = 64;
/** Up to this many ranks every pair is routed; beyond it, only the focus rank to each other one. */
export const ALL_PAIRS_LIMIT = 8;

/**
 * The physical view for a handful of selected ranks: only their nodes, the pods holding them and
 * the switches on the routes between every pair of them. Cores sit in one row on top; below it,
 * one column per pod with its spines, its ToRs and its nodes (all GPUs, so the rail is visible).
 */
export class FocusLayout {
	readonly topo: Topology;
	readonly ranks: number[];
	readonly nodes: number[];
	readonly pods: number[];
	/** Every route's links, each once. */
	readonly edges: FocusEdge[] = [];
	/** Pairs on the same node, which talk over NVLink. */
	readonly nvlink: { a: number; b: number; dim: Dim | null }[] = [];
	/** Whether only routes from `focus` to the other ranks are drawn (large selections). */
	readonly star: boolean;
	readonly focus: number;
	/** Number of routed pairs per hop class. */
	readonly hops: Record<HopClass, number> = {
		nvlink: 0,
		tor: 0,
		spine: 0,
		core: 0,
	};
	/** Boxes by id: "core-c", "spine-p-s", "tor-p-r", "pod-p", "node-n" and "gpu:r". */
	readonly boxes = new Map<string, Rect>();
	readonly width: number;
	readonly height: number;
	readonly swW = 48;
	readonly swH = 20;

	constructor(topo: Topology, ranks: number[], focus?: number) {
		this.topo = topo;
		this.ranks = [...new Set(ranks)]
			.filter((r) => r >= 0 && r < topo.world)
			.sort((a, b) => a - b);
		this.star = this.ranks.length > ALL_PAIRS_LIMIT;
		this.focus =
			focus !== undefined && this.ranks.includes(focus) ? focus : this.ranks[0];
		const shared = (a: number, b: number): Dim | null => {
			for (const d of ["tp", "cp", "ep", "dp", "pp"] as Dim[]) {
				if (topo.sizes[d] > 1 && topo.sameGroup(d, a, b)) return d;
			}
			return null;
		};
		const edges = new Map<string, FocusEdge>();
		const used = new Set<string>();
		const rank = (d: Dim | null) => (d === null ? 9 : DIMS.indexOf(d));
		const pairs: [number, number][] = [];
		for (let i = 0; i < this.ranks.length; i++) {
			for (let j = i + 1; j < this.ranks.length; j++) {
				const [a, b] = [this.ranks[i], this.ranks[j]];
				if (!this.star || a === this.focus || b === this.focus)
					pairs.push([a, b]);
			}
		}
		for (const [a, b] of pairs) {
			const rt = topo.route(a, b);
			const dim = shared(a, b);
			this.hops[rt.cls]++;
			if (rt.cls === "nvlink") {
				this.nvlink.push({ a, b, dim });
				continue;
			}
			const chain = [`gpu:${a}`, ...rt.switches, `gpu:${b}`];
			for (const id of rt.switches) used.add(id);
			for (let k = 1; k < chain.length; k++) {
				const [u, v] = [chain[k - 1], chain[k]].sort();
				const key = `${u}|${v}`;
				const old = edges.get(key);
				if (!old || rank(dim) < rank(old.dim))
					edges.set(key, { a: u, b: v, dim });
			}
		}
		this.edges = [...edges.values()];
		this.nodes = [...new Set(this.ranks.map((r) => topo.nodeOf(r)))].sort(
			(a, b) => a - b,
		);
		const npp = topo.net.shape.serversPerPod;
		this.pods = [...new Set(this.nodes.map((n) => Math.floor(n / npp)))].sort(
			(a, b) => a - b,
		);

		// Switches by tier, as [pod, index] pairs from their ids.
		const ids = [...used];
		const cores = ids
			.filter((id) => id.startsWith("core-"))
			.map((id) => Number(id.split("-")[1]))
			.sort((a, b) => a - b);
		const inPod = (kind: string, p: number) =>
			ids
				.filter((id) => id.startsWith(`${kind}-${p}-`))
				.map((id) => Number(id.split("-")[2]))
				.sort((a, b) => a - b);

		// Geometry.
		const M = 8;
		const pad = 10;
		const gap = 10;
		const g = topo.gpusPerNode;
		const cols = Math.min(g, 8);
		const nodeW = cols * SQ + 2 * PAD;
		const nodeH = Math.ceil(g / cols) * SQ + 2 * PAD + 14;
		const row = (n: number) => (n ? n * this.swW + (n - 1) * gap : 0);
		const columns = this.pods.map((p) => {
			const nodes = this.nodes.filter((n) => Math.floor(n / npp) === p);
			const spines = inPod("spine", p);
			const tors = inPod("tor", p);
			const nc = Math.min(nodes.length, 4);
			const w =
				Math.max(
					nc * nodeW + (nc - 1) * gap,
					row(spines.length),
					row(tors.length),
				) +
				2 * pad;
			let h = pad + 14;
			const spineY = h;
			if (spines.length) h += this.swH + 34;
			const torY = h;
			if (tors.length) h += this.swH + 34;
			const nodesY = h;
			const nr = Math.ceil(nodes.length / nc);
			h += nr * nodeH + (nr - 1) * gap + pad;
			return { p, nodes, spines, tors, nc, w, h, spineY, torY, nodesY };
		});
		const perRow = Math.min(columns.length, 4);
		const coreH = cores.length ? this.swH + 40 : 0;
		let y = M + coreH;
		let width = row(cores.length) + 2 * M;
		for (let i = 0; i < columns.length; i += perRow) {
			const line = columns.slice(i, i + perRow);
			let x = M;
			for (const c of line) {
				this.boxes.set(`pod-${c.p}`, { x, y, w: c.w, h: c.h });
				const place = (kind: string, list: number[], yy: number) => {
					const x0 = x + (c.w - row(list.length)) / 2;
					list.forEach((k, j) => {
						this.boxes.set(`${kind}-${c.p}-${k}`, {
							x: x0 + j * (this.swW + gap),
							y: y + yy,
							w: this.swW,
							h: this.swH,
						});
					});
				};
				place("spine", c.spines, c.spineY);
				place("tor", c.tors, c.torY);
				const nx0 = x + (c.w - (c.nc * nodeW + (c.nc - 1) * gap)) / 2;
				c.nodes.forEach((n, j) => {
					const nx = nx0 + (j % c.nc) * (nodeW + gap);
					const ny = y + c.nodesY + Math.floor(j / c.nc) * (nodeH + gap);
					this.boxes.set(`node-${n}`, { x: nx, y: ny, w: nodeW, h: nodeH });
					for (let k = 0; k < g; k++) {
						this.boxes.set(`gpu:${n * g + k}`, {
							x: nx + PAD + (k % cols) * SQ,
							y: ny + 14 + PAD + Math.floor(k / cols) * SQ,
							w: SQ,
							h: SQ,
						});
					}
				});
				x += c.w + 22;
			}
			width = Math.max(width, x - 22 + M);
			y += Math.max(...line.map((c) => c.h)) + 22;
		}
		const x0 = (width - row(cores.length)) / 2;
		cores.forEach((c, j) => {
			this.boxes.set(`core-${c}`, {
				x: x0 + j * (this.swW + gap),
				y: M,
				w: this.swW,
				h: this.swH,
			});
		});
		this.width = width;
		this.height = y - 22 + M;
	}

	/** The selected or unselected GPU under (x, y), as a rank. */
	hit(x: number, y: number): number | null {
		const g = this.topo.gpusPerNode;
		for (const n of this.nodes) {
			for (let k = 0; k < g; k++) {
				const b = this.boxes.get(`gpu:${n * g + k}`);
				if (b && x >= b.x && x < b.x + b.w && y >= b.y && y < b.y + b.h)
					return n * g + k;
			}
		}
		return null;
	}
}
