// The polyhedral model on one small loop nest, for PolyhedralSchedule.astro:
//
//   for (i = 1; i < n; i++)
//     for (j = 1; j < n - 1; j++)
//       A[i][j] = A[i-1][j+1] + A[i][j-1];      // statement S(i, j)
//
// Everything is computed by brute force over the integer points of the domain, so it can be
// checked against a polyhedral library (the post checks it against isl): the dependences
// (which earlier iteration last wrote each value an iteration reads), the order a schedule runs
// the points in, the dependences that order violates, and the array the order computes.
//
// A schedule maps a point (i, j) to a vector; points run in lexicographic order of their
// vectors. Ties cannot happen for the schedules here because each one is injective.

export type Point = [number, number];

export interface Dependence {
	src: Point;
	dst: Point;
	kind: "flow" | "anti" | "output";
}

export interface Schedule {
	key: string;
	label: string;
	/** Formula shown in the figure, e.g. "(i, i + j)". */
	formula: string;
	map: (i: number, j: number) => number[];
	/** Number of leading coordinates that identify a tile, 0 when not tiled. */
	tileDims: number;
	/** Colour points by their first coordinate (wavefront time) instead of by tile. */
	byTime?: boolean;
}

export const TILE = 3;

export const SCHEDULES: Schedule[] = [
	{
		key: "original",
		label: "Original",
		formula: "(i, j)",
		map: (i, j) => [i, j],
		tileDims: 0,
	},
	{
		key: "interchange",
		label: "Interchange",
		formula: "(j, i)",
		map: (i, j) => [j, i],
		tileDims: 0,
	},
	{
		key: "rect",
		label: `Tile ${TILE}×${TILE}`,
		formula: `(⌊i/${TILE}⌋, ⌊j/${TILE}⌋, i, j)`,
		map: (i, j) => [Math.floor(i / TILE), Math.floor(j / TILE), i, j],
		tileDims: 2,
	},
	{
		key: "skew",
		label: "Skew",
		formula: "(i, i + j)",
		map: (i, j) => [i, i + j],
		tileDims: 0,
	},
	{
		key: "skewtile",
		label: `Skew, then tile ${TILE}×${TILE}`,
		formula: `(⌊i/${TILE}⌋, ⌊(i+j)/${TILE}⌋, i, i + j)`,
		map: (i, j) => [Math.floor(i / TILE), Math.floor((i + j) / TILE), i, i + j],
		tileDims: 2,
	},
	{
		key: "wavefront",
		label: "Wavefront",
		formula: "(2i + j, i)",
		map: (i, j) => [2 * i + j, i],
		tileDims: 0,
		byTime: true,
	},
];

export function domain(n: number): Point[] {
	const pts: Point[] = [];
	for (let i = 1; i < n; i++) for (let j = 1; j < n - 1; j++) pts.push([i, j]);
	return pts;
}

const writeOf = (i: number, j: number): Point => [i, j];
const readsOf = (i: number, j: number): Point[] => [
	[i - 1, j + 1],
	[i, j - 1],
];
const key = (p: Point) => `${p[0]},${p[1]}`;

/** All dependences between iterations, found by walking the original order. */
export function dependences(n: number): Dependence[] {
	const pts = domain(n); // already in original (lexicographic) order
	const lastWrite = new Map<string, Point>();
	const readsSinceWrite = new Map<string, Point[]>();
	const deps: Dependence[] = [];
	for (const p of pts) {
		for (const loc of readsOf(...p)) {
			const w = lastWrite.get(key(loc));
			if (w) deps.push({ src: w, dst: p, kind: "flow" });
			const rs = readsSinceWrite.get(key(loc)) ?? [];
			rs.push(p);
			readsSinceWrite.set(key(loc), rs);
		}
		const loc = writeOf(...p);
		for (const r of readsSinceWrite.get(key(loc)) ?? [])
			if (key(r) !== key(p)) deps.push({ src: r, dst: p, kind: "anti" });
		const w = lastWrite.get(key(loc));
		if (w) deps.push({ src: w, dst: p, kind: "output" });
		lastWrite.set(key(loc), p);
		readsSinceWrite.set(key(loc), []);
	}
	return deps;
}

const lexLess = (a: number[], b: number[]) => {
	for (let k = 0; k < Math.min(a.length, b.length); k++)
		if (a[k] !== b[k]) return a[k] < b[k];
	return a.length < b.length;
};

/** Points in the order the schedule runs them. */
export function order(n: number, s: Schedule): Point[] {
	return domain(n).sort((p, q) => {
		const a = s.map(...p);
		const b = s.map(...q);
		return lexLess(a, b) ? -1 : lexLess(b, a) ? 1 : 0;
	});
}

/** Dependences whose destination the schedule runs before their source. */
export function violations(
	n: number,
	s: Schedule,
	deps = dependences(n),
): Dependence[] {
	return deps.filter((d) => lexLess(s.map(...d.dst), s.map(...d.src)));
}

/** Distance vectors dst - src, deduplicated, as "(di, dj)". */
export function distances(deps: Dependence[]): string[] {
	return [
		...new Set(
			deps.map((d) => `(${d.dst[0] - d.src[0]}, ${d.dst[1] - d.src[1]})`),
		),
	].sort();
}

/** Runs the loop body in the schedule's order on a fixed initial array. */
export function simulate(n: number, s: Schedule): number[][] {
	const A = Array.from({ length: n }, (_, r) =>
		Array.from({ length: n }, (_, c) => ((3 * r + 7 * c) % 11) + 1),
	);
	for (const [i, j] of order(n, s)) A[i][j] = A[i - 1][j + 1] + A[i][j - 1];
	return A;
}

export function cellsDiffering(n: number, s: Schedule): number {
	const ref = simulate(n, SCHEDULES[0]);
	const got = simulate(n, s);
	let diff = 0;
	for (let r = 0; r < n; r++)
		for (let c = 0; c < n; c++) if (ref[r][c] !== got[r][c]) diff++;
	return diff;
}

/** Group id of each point: its tile, or its wavefront time step. */
export function groupOf(s: Schedule, p: Point): string {
	const v = s.map(...p);
	if (s.byTime) return `t${v[0]}`;
	if (s.tileDims) return v.slice(0, s.tileDims).join(",");
	return "";
}

const hueOf = (k: number) => (25 + k * 137.508) % 360;

/** The figure: the domain in (i, j) space, each point numbered by when it runs. */
export function drawSchedule(n: number, s: Schedule): string {
	const deps = dependences(n);
	const bad = new Set(
		violations(n, s, deps).map((d) => `${key(d.src)}>${key(d.dst)}`),
	);
	const runs = order(n, s);
	const rank = new Map(runs.map((p, k) => [key(p), k + 1]));
	const groups = [...new Set(runs.map((p) => groupOf(s, p)))];
	const G = 44;
	const left = 40;
	const top = 30;
	const x = (j: number) => left + (j - 1) * G + G / 2;
	const y = (i: number) => top + (i - 1) * G + G / 2;
	const w = left + (n - 2) * G + 10;
	const h = top + (n - 1) * G + 8;
	const parts: string[] = [
		`<defs><marker id="pd-a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 8 4 0 8z" class="ah"/></marker><marker id="pd-b" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 8 4 0 8z" class="ah bad"/></marker></defs>`,
		`<text class="axis" x="${left}" y="14">j →</text>`,
		`<text class="axis" x="6" y="${top + 12}">i ↓</text>`,
	];
	for (let j = 1; j < n - 1; j++)
		parts.push(`<text class="tick" x="${x(j)}" y="${top - 6}">${j}</text>`);
	for (let i = 1; i < n; i++)
		parts.push(
			`<text class="tick" x="${left - 14}" y="${y(i) + 4}">${i}</text>`,
		);
	const r = 13;
	for (const d of deps) {
		const isBad = bad.has(`${key(d.src)}>${key(d.dst)}`);
		const [x1, y1, x2, y2] = [
			x(d.src[1]),
			y(d.src[0]),
			x(d.dst[1]),
			y(d.dst[0]),
		];
		const len = Math.hypot(x2 - x1, y2 - y1);
		const ux = (x2 - x1) / len;
		const uy = (y2 - y1) / len;
		parts.push(
			`<line class="dep${isBad ? " bad" : ""}" x1="${(x1 + ux * r).toFixed(1)}" y1="${(y1 + uy * r).toFixed(1)}" x2="${(x2 - ux * (r + 2)).toFixed(1)}" y2="${(y2 - uy * (r + 2)).toFixed(1)}" marker-end="url(#${isBad ? "pd-b" : "pd-a"})"/>`,
		);
	}
	for (const p of runs) {
		const g = groupOf(s, p);
		const hue = g ? hueOf(groups.indexOf(g)).toFixed(1) : "";
		parts.push(
			`<g class="pt${g ? " grouped" : ""}"${g ? ` style="--hue:${hue}"` : ""}><circle cx="${x(p[1])}" cy="${y(p[0])}" r="${r}"/><text x="${x(p[1])}" y="${y(p[0]) + 4}">${rank.get(key(p))}</text></g>`,
		);
	}
	return `<svg viewBox="0 0 ${w} ${h}" width="${Math.round(w * 1.15)}" height="${Math.round(h * 1.15)}" role="img" aria-label="Iteration domain run in ${s.label} order">${parts.join("")}</svg>`;
}

export function summary(n: number, s: Schedule): string[] {
	const deps = dependences(n);
	const bad = violations(n, s, deps);
	const diff = cellsDiffering(n, s);
	const lines = [
		`${s.label}: S(i, j) runs at ${s.formula}`,
		bad.length
			? `${bad.length} of ${deps.length} dependences violated (red); the result differs in ${diff} cells`
			: `all ${deps.length} dependences respected; the result matches the original`,
	];
	if (s.byTime) {
		const steps = new Set(domain(n).map((p) => s.map(...p)[0])).size;
		lines.push(
			`${steps} time steps; points of one colour have no dependences between them and can run in parallel`,
		);
	} else if (s.tileDims) {
		lines.push(
			`${new Set(domain(n).map((p) => groupOf(s, p))).size} tiles, one colour each`,
		);
	}
	return lines;
}
