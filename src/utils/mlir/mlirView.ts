// Highlights MLIR text by dialect and counts its operations, for LoweringExplorer.astro.
//
// An operation is a `dialect.name` word that isn't a type (`!llvm.ptr`), an attribute
// (`#arith.overflow`) or a value or block name. The func dialect's `return` and `call` are printed
// without their prefix, so they are counted as func too.

export const DIALECTS: Record<string, string> = {
	func: "#9fc8ef",
	arith: "#f5d38c",
	tensor: "#a9dba6",
	linalg: "#f4a7a3",
	bufferization: "#9fe0d6",
	memref: "#c8b4ee",
	scf: "#f3b6d6",
	cf: "#d9c7a4",
	vector: "#b8d98a",
	llvm: "#f0b88a",
	ub: "#d9d6cc",
};

const OP = /(^|[\s(=])([a-z_]+)\.([a-z_][\w.]*)/g;
const BARE = /(^|=\s|^\s+)(return|call)\b/;

const esc = (s: string) =>
	s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

export function dialectOf(word: string): string | undefined {
	return word in DIALECTS ? word : undefined;
}

/** Counts the operations of each dialect, one per line at most (MLIR prints one op per line). */
export function countOps(ir: string): Record<string, number> {
	const counts: Record<string, number> = {};
	for (const line of ir.split("\n")) {
		const code = line.split("//")[0];
		let d: string | undefined;
		for (const m of code.matchAll(OP)) {
			d = dialectOf(m[2]);
			if (d) break;
		}
		if (!d && BARE.test(code)) d = "func";
		if (d) counts[d] = (counts[d] ?? 0) + 1;
	}
	return counts;
}

/** HTML for the IR, with every operation name wrapped in a span coloured by its dialect. */
export function highlight(ir: string): string {
	return ir
		.split("\n")
		.map((line) => {
			const [code, ...rest] = line.split("//");
			let html = esc(code).replace(OP, (all, pre, d, name) =>
				dialectOf(d)
					? `${pre}<span class="op" data-d="${d}" style="--c:${DIALECTS[d]}">${d}.${name}</span>`
					: all,
			);
			html = html.replace(
				BARE,
				(all, pre, name) =>
					`${pre}<span class="op" data-d="func" style="--c:${DIALECTS.func}">${name}</span>`,
			);
			const comment = rest.length
				? `<span class="cm">//${esc(rest.join("//"))}</span>`
				: "";
			return html + comment;
		})
		.join("\n");
}
