// Data model and helpers for SymbolicShapes.astro: an FX graph traced with symbolic sizes, the
// concrete shapes the same graph produces for several input sizes, and the guards that decide
// whether each input size can reuse the graph.

/** A size symbol such as s77, and where its value is read from. */
export interface ShapeSymbol {
	name: string;
	/** e.g. "x.size(0)", or "x.size(1) = w.size(0)" when duck sizing shared the symbol. */
	source: string;
}

/** One set of input sizes. */
export interface ShapeConfig {
	label: string;
	/** Value of every symbol for these inputs. */
	values: Record<string, number>;
	/** Whether Dynamo compiled a new graph for these inputs (observed, not derived). */
	recompiles: boolean;
}

/** One FX node. */
export interface ShapeRow {
	name: string;
	op: string;
	code: string;
	/** Symbolic type as print_readable shows it, e.g. "f32[2*s77, (s21//2)]" or "Sym(2*s77)". */
	symbolic: string;
	/** Concrete type for each config, in config order. */
	concrete: string[];
}

/** A guard on the symbols, and whether it holds for each config. */
export interface ShapeGuard {
	expr: string;
	why: string;
	holds: boolean[];
}

export interface SymbolicShapesData {
	symbols: ShapeSymbol[];
	configs: ShapeConfig[];
	rows: ShapeRow[];
	guards: ShapeGuard[];
}

/** A run of text; `sym` is the index of the symbol it names, if any. */
export interface Segment {
	text: string;
	sym?: number;
}

/** Split text into runs, marking whole-word occurrences of the given symbol names. */
export function splitSymbols(text: string, symbols: string[]): Segment[] {
	if (symbols.length === 0) return [{ text }];
	const alt = [...symbols]
		.sort((a, b) => b.length - a.length)
		.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
		.join("|");
	const re = new RegExp(`(?<![\\w])(${alt})(?![\\w])`, "g");
	const out: Segment[] = [];
	let last = 0;
	for (const m of text.matchAll(re)) {
		const at = m.index ?? 0;
		if (at > last) out.push({ text: text.slice(last, at) });
		out.push({ text: m[0], sym: symbols.indexOf(m[0]) });
		last = at + m[0].length;
	}
	if (last < text.length) out.push({ text: text.slice(last) });
	return out;
}

/** Return an error message if the data is inconsistent, else null. */
export function validateShapes(d: SymbolicShapesData): string | null {
	const n = d.configs.length;
	if (n === 0) return "no configs";
	const names = new Set(d.symbols.map((s) => s.name));
	for (const c of d.configs) {
		for (const s of names)
			if (!(s in c.values)) return `config "${c.label}" has no value for ${s}`;
	}
	for (const r of d.rows)
		if (r.concrete.length !== n)
			return `row "${r.name}" has ${r.concrete.length} values for ${n} configs`;
	for (const g of d.guards)
		if (g.holds.length !== n)
			return `guard "${g.expr}" has ${g.holds.length} results for ${n} configs`;
	for (const [i, c] of d.configs.entries()) {
		const fails = d.guards.some((g) => !g.holds[i]);
		if (fails !== c.recompiles)
			return `config "${c.label}": recompiles=${c.recompiles} but ${fails ? "a guard fails" : "every guard holds"}`;
	}
	return null;
}

/** One-line verdict for a config. */
export function verdict(d: SymbolicShapesData, i: number): string {
	const failed = d.guards.filter((g) => !g.holds[i]).length;
	if (failed === 0)
		return "Every guard holds, so this call reuses the traced graph.";
	return `${failed} guard${failed > 1 ? "s fail" : " fails"}, so Dynamo traces and compiles a new graph.`;
}
