// Parses one LLVM IR function into its control-flow graph for IrCfg.astro: basic blocks with
// their instructions, branch edges (labelled true/false for conditional branches), and for every
// SSA value the block that defines it and the lines that use it. Then lays the blocks out with
// dagre, top to bottom. Self-loops (a block that branches to itself) are left out of the dagre
// graph and drawn by the component as an arc on the block's right side.
import { graphlib, layout } from "@dagrejs/dagre";

/** A piece of an instruction line: plain text, or a reference to a value or a block. */
export interface Segment {
	text: string;
	/** "%8": a value. */
	value?: string;
	/** "7": a block (a branch target or a phi's incoming block). */
	block?: string;
	/** This segment is the value's definition (the left side of `%8 = ...`). */
	def?: boolean;
}

export interface Block {
	name: string;
	lines: Segment[][];
}

export interface CfgEdge {
	from: string;
	to: string;
	label?: "true" | "false";
}

export interface ValueInfo {
	/** Block that defines it, or "argument". */
	defBlock: string;
	/** The defining instruction, e.g. "phi float ...", or the argument's type. */
	def: string;
	/** Blocks of each use, in order. */
	uses: string[];
}

export interface Cfg {
	/** `define float @dot(ptr %0, ptr %1, i32 %2)`, with attributes dropped. */
	signature: string;
	blocks: Block[];
	edges: CfgEdge[];
	values: Record<string, ValueInfo>;
}

/** Split on commas at parenthesis depth 0. */
function splitArgs(s: string): string[] {
	const out: string[] = [];
	let depth = 0;
	let cur = "";
	for (const ch of s) {
		if (ch === "(" || ch === "[" || ch === "<" || ch === "{") depth++;
		if (ch === ")" || ch === "]" || ch === ">" || ch === "}") depth--;
		if (ch === "," && depth === 0) {
			out.push(cur.trim());
			cur = "";
		} else cur += ch;
	}
	if (cur.trim()) out.push(cur.trim());
	return out;
}

function parseSignature(define: string): {
	signature: string;
	args: { name: string; type: string }[];
} {
	const at = define.indexOf("@");
	const open = define.indexOf("(", at);
	let depth = 0;
	let close = open;
	for (; close < define.length; close++) {
		if (define[close] === "(") depth++;
		if (define[close] === ")" && --depth === 0) break;
	}
	const name = define.slice(at, open);
	const ret = define.slice(0, at).trim().split(/\s+/).pop() ?? "void";
	const args = splitArgs(define.slice(open + 1, close)).map((a) => {
		const words = a.split(/\s+/);
		return { type: words[0], name: words[words.length - 1] };
	});
	return {
		signature: `define ${ret} ${name}(${args.map((a) => `${a.type} ${a.name}`).join(", ")})`,
		args,
	};
}

/** Break a line into segments, marking value and block references. */
function segments(line: string): Segment[] {
	const out: Segment[] = [];
	const phiBlocks = new Set<number>();
	// In `[ %16, %7 ]` the second name is the incoming block.
	for (const m of line.matchAll(/\[ [^,\]]+, (%[\w.]+) \]/g))
		phiBlocks.add((m.index ?? 0) + m[0].lastIndexOf("%"));
	const def = /^(%[\w.]+) = /.exec(line);
	let last = 0;
	for (const m of line.matchAll(/%[\w.]+/g)) {
		const i = m.index ?? 0;
		if (i > last) out.push({ text: line.slice(last, i) });
		const isBlock =
			phiBlocks.has(i) || line.slice(Math.max(0, i - 6), i) === "label ";
		if (isBlock) out.push({ text: m[0], block: m[0].slice(1) });
		else
			out.push({ text: m[0], value: m[0], def: i === 0 && def?.[1] === m[0] });
		last = i + m[0].length;
	}
	if (last < line.length) out.push({ text: line.slice(last) });
	return out;
}

export function parseCfg(ir: string): Cfg {
	const lines = ir.split("\n");
	const define = lines.find((l) => l.startsWith("define")) ?? "";
	const { signature, args } = parseSignature(define);
	const labels: string[] = [];
	const raw: { name: string; lines: string[] }[] = [];
	for (const l of lines.slice(lines.indexOf(define) + 1)) {
		if (l.startsWith("}")) break;
		const label = /^([\w.]+):/.exec(l);
		if (label) {
			labels.push(label[1]);
			raw.push({ name: label[1], lines: [] });
		} else if (l.trim()) {
			if (!raw.length) raw.push({ name: "", lines: [] });
			raw[raw.length - 1].lines.push(l.trim());
		}
	}
	// The entry block has no label line; its implicit number is the one branch targets and
	// `; preds =` comments refer to that no label line defines.
	if (raw[0]?.name === "") {
		const refs = new Set([
			...[...ir.matchAll(/label %([\w.]+)/g)].map((m) => m[1]),
			...[...ir.matchAll(/; preds = (.*)$/gm)].flatMap((m) =>
				m[1].split(", ").map((r) => r.replace("%", "")),
			),
		]);
		const implicit = [...refs].filter((r) => !labels.includes(r));
		raw[0].name = implicit.length === 1 ? implicit[0] : "entry";
	}

	const blocks: Block[] = raw.map((b) => ({
		name: b.name,
		lines: b.lines.map(segments),
	}));
	const edges: CfgEdge[] = [];
	const values: Record<string, ValueInfo> = {};
	for (const a of args)
		values[a.name] = { defBlock: "argument", def: a.type, uses: [] };
	for (const b of raw) {
		const term = b.lines[b.lines.length - 1] ?? "";
		const cond = /^br i1 [^,]+, label %([\w.]+), label %([\w.]+)/.exec(term);
		const uncond = /^br label %([\w.]+)/.exec(term);
		if (cond) {
			edges.push({ from: b.name, to: cond[1], label: "true" });
			edges.push({ from: b.name, to: cond[2], label: "false" });
		} else if (uncond) edges.push({ from: b.name, to: uncond[1] });
		for (const l of b.lines) {
			const def = /^(%[\w.]+) = (.*)$/.exec(l);
			if (def)
				values[def[1]] = {
					...(values[def[1]] ?? { uses: [] }),
					defBlock: b.name,
					def: def[2],
				};
		}
	}
	for (const b of blocks) {
		for (const l of b.lines) {
			for (const s of l) {
				if (s.value && !s.def) {
					values[s.value] ??= { defBlock: "?", def: "", uses: [] };
					values[s.value].uses.push(b.name);
				}
			}
		}
	}
	return { signature, blocks, edges, values };
}

export interface PlacedBlock extends Block {
	x: number;
	y: number;
	w: number;
	h: number;
}

export interface CfgLayout {
	blocks: PlacedBlock[];
	edges: (CfgEdge & {
		d: string;
		labelAt?: { x: number; y: number };
		self: boolean;
	})[];
	width: number;
	height: number;
}

// Text widths are estimated from character counts at the figure's 11px monospace font.
export const CFG_CHAR = 6.7;
export const CFG_LINE = 15;
export const CFG_HEAD = 20;
const PAD = 10;
const LOOP = 34; // room on the right of a block for its self-loop arc

export function layoutCfg(cfg: Cfg): CfgLayout {
	const g = new graphlib.Graph({ multigraph: true });
	g.setGraph({
		rankdir: "TB",
		nodesep: 40,
		ranksep: 46,
		marginx: 8,
		marginy: 8,
	});
	g.setDefaultEdgeLabel(() => ({}));
	const selfLoops = new Set(
		cfg.edges.filter((e) => e.from === e.to).map((e) => e.from),
	);
	for (const b of cfg.blocks) {
		const chars = Math.max(
			`${b.name}:`.length,
			...b.lines.map((l) => l.reduce((n, s) => n + s.text.length, 0)),
		);
		const w = Math.ceil(chars * CFG_CHAR + 2 * PAD);
		g.setNode(b.name, {
			width: w + (selfLoops.has(b.name) ? LOOP : 0),
			height: CFG_HEAD + b.lines.length * CFG_LINE + PAD,
			boxWidth: w,
		});
	}
	cfg.edges.forEach((e, i) => {
		if (e.from !== e.to) g.setEdge(e.from, e.to, {}, `e${i}`);
	});
	layout(g);

	const blocks = cfg.blocks.map((b) => {
		const n = g.node(b.name);
		const w = n.boxWidth as number;
		// dagre centres the node including the loop margin; the box itself sits on the left.
		return {
			...b,
			x: n.x - n.width / 2,
			y: n.y - n.height / 2,
			w,
			h: n.height,
		};
	});
	const box = new Map(blocks.map((b) => [b.name, b]));
	const edges = cfg.edges.map((e, i) => {
		if (e.from === e.to) {
			const b = box.get(e.from) as PlacedBlock;
			const x = b.x + b.w;
			const y0 = b.y + b.h * 0.7;
			const y1 = b.y + b.h * 0.3;
			const d = `M${x},${y0} C${x + LOOP * 1.3},${y0} ${x + LOOP * 1.3},${y1} ${x + 2},${y1}`;
			return {
				...e,
				d,
				self: true,
				labelAt: e.label
					? { x: x + LOOP * 0.55, y: (y0 + y1) / 2 + 4 }
					: undefined,
			};
		}
		const pts = g.edge({ v: e.from, w: e.to, name: `e${i}` }).points;
		const d = pts
			.map((p, k) => `${k ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
			.join(" ");
		const a = pts[0];
		const b = pts[Math.min(1, pts.length - 1)];
		const labelAt = e.label
			? { x: a.x + (b.x - a.x) * 0.5 + 6, y: a.y + (b.y - a.y) * 0.5 }
			: undefined;
		return { ...e, d, self: false, labelAt };
	});
	const gr = g.graph();
	const fin = (v: number | undefined) =>
		v !== undefined && Number.isFinite(v) ? Math.ceil(v) : 0;
	return { blocks, edges, width: fin(gr.width), height: fin(gr.height) };
}
