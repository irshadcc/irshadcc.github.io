// Data model for GraphSteps.astro: a small computation graph (network units or the variables of an
// evaluation trace) drawn at fixed grid positions, and a list of steps. Each step says what every
// node shows (value lines), which nodes and edges are lit and in which direction information is
// flowing (forward = accent, backward = red, drawn pointing from the child to the parent), an
// optional label per edge, an optional table and a note. Notes may contain $...$ TeX, which tex()
// renders with KaTeX at build time.
import katex from "katex";

export type Flow = "fwd" | "bwd";

export interface GraphNode {
	id: string;
	/** Name drawn in the node's header, e.g. "z₁" or "v₃" (plain text, drawn in SVG). */
	label: string;
	/** Second header line, e.g. the operation "v₁·v₂" or "tanh". */
	sub?: string;
	/** Grid column (left to right) and row (top to bottom); fractions are allowed. */
	col: number;
	row: number;
	/** Small round node for the constant 1 that carries a bias. */
	bias?: boolean;
}

export interface GraphEdge {
	from: string;
	to: string;
}

export interface StepNode {
	/** Value lines shown under the header, e.g. ["a = 0.57", "z = 0.515"]. */
	lines?: string[];
	hi?: Flow;
}

export interface StepEdge {
	hi?: Flow;
	/** Text drawn at the middle of the edge. */
	label?: string;
}

export interface GraphStep {
	label: string;
	/** HTML; $...$ and $$...$$ are rendered by tex(). */
	note: string;
	nodes: Record<string, StepNode>;
	/** Keyed by "from>to". */
	edges?: Record<string, StepEdge>;
	table?: { columns: string[]; rows: string[][]; caption?: string };
}

export interface GraphStepsData {
	nodes: GraphNode[];
	edges: GraphEdge[];
	steps: GraphStep[];
}

export const edgeKey = (e: GraphEdge) => `${e.from}>${e.to}`;

export function validateGraphSteps(d: GraphStepsData): string | null {
	const ids = new Set(d.nodes.map((n) => n.id));
	if (ids.size !== d.nodes.length) return "duplicate node id";
	for (const e of d.edges) {
		if (!ids.has(e.from) || !ids.has(e.to))
			return `edge ${edgeKey(e)} names an unknown node`;
	}
	const keys = new Set(d.edges.map(edgeKey));
	if (d.steps.length === 0) return "no steps";
	for (const [k, s] of d.steps.entries()) {
		for (const id of Object.keys(s.nodes))
			if (!ids.has(id)) return `step ${k}: unknown node ${id}`;
		for (const key of Object.keys(s.edges ?? {}))
			if (!keys.has(key)) return `step ${k}: unknown edge ${key}`;
		const t = s.table;
		if (t?.rows.some((r) => r.length !== t.columns.length))
			return `step ${k}: table row length`;
	}
	return null;
}

/** Render $$...$$ (display) and $...$ (inline) TeX in an HTML string with KaTeX. */
export function tex(html: string): string {
	return html
		.replace(/\$\$([^$]+)\$\$/g, (_, s: string) =>
			katex.renderToString(s, { displayMode: true, throwOnError: true }),
		)
		.replace(/\$([^$]+)\$/g, (_, s: string) =>
			katex.renderToString(s, { throwOnError: true }),
		);
}

/** Format a number with `digits` decimals, using a real minus sign. */
export function fmt(x: number, digits = 3): string {
	const s = x.toFixed(digits);
	return (Number(s) === 0 ? (0).toFixed(digits) : s).replace("-", "−");
}
