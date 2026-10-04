// Data model and checks for JaxprSteps.astro: two program listings and a list of steps. Each step
// highlights lines of the top listing (the program being read), reveals and highlights lines of
// the bottom listing (the program being written), and shows a small table of values.

export interface JaxprListing {
	/** Shown above the listing, e.g. "model.py" or "jaxpr being built". */
	title: string;
	lines: string[];
}

export interface JaxprRow {
	cells: string[];
	/** Highlight this row (new in this step). */
	hi?: boolean;
}

export interface JaxprStep {
	/** Short label for the step's timeline segment. */
	label: string;
	/** Lines of the top listing being read in this step. */
	top: number[];
	/** Lines of the bottom listing that exist after this step. */
	shown: number[];
	/** Lines of the bottom listing written in this step. */
	bottom: number[];
	rows: JaxprRow[];
	/** What happens in this step, shown under the figure. HTML allowed. */
	note: string;
}

export interface JaxprStepsData {
	top: JaxprListing;
	bottom: JaxprListing;
	/** Column headings of the value table. */
	columns: string[];
	steps: JaxprStep[];
}

/** Lines 0..n-1 plus the given extra lines: the usual "everything emitted so far". */
export function upTo(n: number, ...extra: number[]): number[] {
	return [...Array.from({ length: n }, (_, i) => i), ...extra];
}

export function validateJaxprSteps(d: JaxprStepsData): string | undefined {
	if (d.steps.length === 0) return "no steps";
	const inRange = (xs: number[], n: number) =>
		xs.every((x) => Number.isInteger(x) && x >= 0 && x < n);
	for (const [i, s] of d.steps.entries()) {
		if (!inRange(s.top, d.top.lines.length))
			return `step ${i}: top line out of range`;
		if (!inRange(s.shown, d.bottom.lines.length))
			return `step ${i}: shown line out of range`;
		if (!s.bottom.every((b) => s.shown.includes(b)))
			return `step ${i}: highlighted bottom line is not shown`;
		for (const r of s.rows) {
			if (r.cells.length !== d.columns.length)
				return `step ${i}: row has ${r.cells.length} cells, expected ${d.columns.length}`;
		}
	}
	return undefined;
}
