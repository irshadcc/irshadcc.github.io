// Parses the SelectionDAG dumps that `llc -debug-only=isel` prints, and lays them out for
// SelectionDag.astro. A dump has one node per line:
//
//   t18: f32 = fma contract t17, t11, t15
//   t11: f32,ch = load<(load (s32) from %ir.15, addrspace 1)> t0, t8, undef:i64
//
// `tN` names a node, the types before `=` are its results (`ch` is the chain that orders memory
// operations), and the operands are either other nodes (`t11`, or `t11:1` for result 1) or
// inline leaves (`Constant:i32<2>`, `Register:i32 %2`, `undef:i64`, `TargetConstant:i32<0>`).
// Node ids survive from one stage of the DAG to the next, which is how a stage marks the nodes
// that are new or changed.
import { graphlib, layout } from "@dagrejs/dagre";

export type DagKind = "generic" | "target" | "machine" | "leaf" | "entry";

export interface DagNode {
	id: string;
	opcode: string;
	/** What the node shows: opcode plus a folded register, constant or symbol. */
	label: string;
	/** Result types, e.g. ["f32", "ch"]. */
	types: string[];
	flags: string[];
	kind: DagKind;
	/** The dump line, shown when the reader hovers the node. */
	raw: string;
}

export interface DagEdge {
	from: string;
	to: string;
	/** Data, or chain/glue (ordering only). */
	chain: boolean;
}

export interface Dag {
	nodes: DagNode[];
	edges: DagEdge[];
}

const FLAGS = new Set([
	"contract",
	"nsw",
	"nuw",
	"exact",
	"disjoint",
	"nneg",
	"reassoc",
	"nnan",
	"ninf",
	"nsz",
	"arcp",
	"afn",
	"fast",
	"samesign",
]);
// Opcodes that start with a capital letter but are generic SelectionDAG nodes, not NVPTX instructions.
const GENERIC_CAPS = new Set([
	"EntryToken",
	"TokenFactor",
	"CopyFromReg",
	"CopyToReg",
	"Constant",
	"ConstantFP",
	"Register",
	"BasicBlock",
	"ExternalSymbol",
]);

/** Split on commas that aren't inside <...> or (...). */
function splitTop(s: string): string[] {
	const out: string[] = [];
	let depth = 0;
	let cur = "";
	for (const ch of s) {
		if (ch === "<" || ch === "(") depth++;
		if (ch === ">" || ch === ")") depth--;
		if (ch === "," && depth === 0) {
			out.push(cur.trim());
			cur = "";
		} else cur += ch;
	}
	if (cur.trim()) out.push(cur.trim());
	return out;
}

/** The opcode, and the index just past its `<...>` detail if it has one. */
function readOpcode(s: string): { opcode: string; end: number } {
	const m = /^[A-Za-z_:0-9]+/.exec(s);
	const opcode = m ? m[0] : s;
	let end = opcode.length;
	if (s[end] === "<") {
		let depth = 0;
		for (; end < s.length; end++) {
			if (s[end] === "<") depth++;
			if (s[end] === ">" && --depth === 0) {
				end++;
				break;
			}
		}
	}
	return { opcode, end };
}

function kindOf(opcode: string): DagKind {
	if (opcode === "EntryToken") return "entry";
	if (
		opcode === "CopyFromReg" ||
		opcode === "Constant" ||
		opcode === "ConstantFP"
	)
		return "leaf";
	if (opcode.startsWith("NVPTXISD::")) return "target";
	if (GENERIC_CAPS.has(opcode) || /^[a-z]/.test(opcode)) return "generic";
	return "machine";
}

/** A short label for an inline leaf operand, or undefined to drop it from the drawing. */
function leafLabel(op: string): { label: string; fold: boolean } | undefined {
	let m = /^Register:\S+ (%\S+)$/.exec(op);
	if (m) return { label: m[1], fold: true };
	m =
		/^ExternalSymbol:\S+'(.*)'$/.exec(op) ??
		/^TargetExternalSymbol:\S+'(.*)'$/.exec(op);
	if (m) return { label: m[1], fold: true };
	m = /^Constant:\S+<(-?\d+)>$/.exec(op);
	if (m) return { label: `Constant ${m[1]}`, fold: false };
	// undef offsets and TargetConstant operands (instruction flags such as address space and
	// width) are part of the instruction, not data the reader needs to follow.
	return undefined;
}

export function parseDag(dump: string): Dag {
	const nodes: DagNode[] = [];
	const edges: DagEdge[] = [];
	const types = new Map<string, string[]>();
	const pending: { node: DagNode; operands: string[] }[] = [];

	for (const raw of dump.split("\n")) {
		const m = /^\s*(t\d+): ([^=]+?) = (.*)$/.exec(raw);
		if (!m) continue;
		const [, id, typeList, rest] = m;
		const { opcode, end } = readOpcode(rest);
		const parts = splitTop(rest.slice(end).trim());
		const flags: string[] = [];
		if (parts.length) {
			const words = parts[0].split(" ");
			while (words.length > 1 && FLAGS.has(words[0]))
				flags.push(words.shift() as string);
			if (words.length === 1 && FLAGS.has(words[0])) {
				flags.push(words[0]);
				parts.shift();
			} else parts[0] = words.join(" ");
		}
		const node: DagNode = {
			id,
			opcode,
			label: opcode,
			types: typeList.split(","),
			flags,
			kind: kindOf(opcode),
			raw: raw.trim(),
		};
		types.set(id, node.types);
		nodes.push(node);
		pending.push({ node, operands: parts });
	}

	// Operands, once every node's result types are known (an operand can appear above its node).
	let leaves = 0;
	for (const { node, operands } of pending) {
		for (const op of operands) {
			const ref = /^(t\d+)(?::(\d+))?$/.exec(op);
			if (ref) {
				const ty = types.get(ref[1])?.[Number(ref[2] ?? 0)];
				edges.push({
					from: ref[1],
					to: node.id,
					chain: ty === "ch" || ty === "glue",
				});
				continue;
			}
			const leaf = leafLabel(op);
			if (!leaf) continue;
			if (leaf.fold) {
				node.label = `${node.label} ${leaf.label}`;
				continue;
			}
			const id = `c${leaves++}`;
			nodes.push({
				id,
				opcode: "Constant",
				label: leaf.label,
				types: [],
				flags: [],
				kind: "leaf",
				raw: op,
			});
			edges.push({ from: id, to: node.id, chain: false });
		}
	}
	return { nodes, edges };
}

/** Ids of nodes that are new in `cur`, or whose label changed, relative to `prev`. */
export function changedNodes(prev: Dag | undefined, cur: Dag): Set<string> {
	if (!prev) return new Set();
	const before = new Map(prev.nodes.map((n) => [n.id, n.label]));
	// Constant leaves get fresh ids (c0, c1, ...) at every parse, so they never count as changed.
	return new Set(
		cur.nodes
			.filter((n) => !n.id.startsWith("c") && before.get(n.id) !== n.label)
			.map((n) => n.id),
	);
}

export interface PlacedDagNode extends DagNode {
	x: number;
	y: number;
	w: number;
	h: number;
	changed: boolean;
}

export interface DagLayout {
	nodes: PlacedDagNode[];
	edges: (DagEdge & { d: string })[];
	width: number;
	height: number;
}

// SVG text can't be measured at build time; estimate from character counts at the figure's
// 11px monospace font (generous, so labels never overflow).
const CHAR = 6.9;
const NODE_H = 34;

/** Lays a DAG out top to bottom: operands above the nodes that use them. */
export function layoutDag(
	dag: Dag,
	changed: Set<string> = new Set(),
): DagLayout {
	const g = new graphlib.Graph({ multigraph: true });
	g.setGraph({
		rankdir: "TB",
		nodesep: 14,
		ranksep: 26,
		marginx: 4,
		marginy: 4,
	});
	g.setDefaultEdgeLabel(() => ({}));
	for (const n of dag.nodes) {
		const head = n.flags.length ? `${n.label} ${n.flags.join(" ")}` : n.label;
		const text = Math.max(
			head.length,
			`${n.id} ${n.types.join(",")}`.length * 0.85,
		);
		g.setNode(n.id, {
			width: Math.ceil(text * CHAR + 16),
			height: n.kind === "leaf" && n.id.startsWith("c") ? 22 : NODE_H,
		});
	}
	dag.edges.forEach((e, i) => {
		g.setEdge(e.from, e.to, { weight: e.chain ? 1 : 2 }, `e${i}`);
	});
	layout(g);

	const nodes = dag.nodes.map((n) => {
		const p = g.node(n.id);
		return {
			...n,
			x: p.x,
			y: p.y,
			w: p.width,
			h: p.height,
			changed: changed.has(n.id),
		};
	});
	const edges = dag.edges.map((e, i) => {
		const pts = g.edge({ v: e.from, w: e.to, name: `e${i}` }).points;
		const d = pts
			.map((p, k) => `${k ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
			.join(" ");
		return { ...e, d };
	});
	const gr = g.graph();
	const size = (v: number | undefined) =>
		v !== undefined && Number.isFinite(v) ? Math.ceil(v) : 0;
	return { nodes, edges, width: size(gr.width), height: size(gr.height) };
}
