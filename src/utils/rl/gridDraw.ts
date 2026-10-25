// Browser-side drawing of a grid world into an SVG: cells (forbidden / target colours), optional
// heat tint, values, policy arrows (length grows with π(a|s)), a dashed trajectory and the agent.
// Shared by GridWorld, DpSteps and TdLearning. Styles are in rl.css.

import { el } from "../sampling/plot";
import type { GridSpec } from "./gridWorld";

export const CELL = 60;
export const PAD = 18;

export interface GridDrawing {
	/** Policy as π(a|s) rows; arrows are drawn for every action with positive probability. */
	policy?: number[][];
	values?: (number | null)[];
	digits?: number;
	/** Values whose cell should be emphasised (e.g. changed in this step). */
	changed?: boolean[];
	/** Heat tint per cell in [0, 1]. */
	heat?: number[];
	/** Cell indices to connect with a dashed line. */
	trail?: number[];
	/** Cell to mark with the agent dot. */
	agent?: number;
	hot?: number;
	/** Draw the value in the middle of the cell, large (when no policy is shown). */
	bigValues?: boolean;
}

export function gridSize(g: GridSpec): { w: number; h: number } {
	return { w: PAD + g.cols * CELL + 4, h: PAD + g.rows * CELL + 4 };
}

const center = (g: GridSpec, s: number): [number, number] => [
	PAD + (s % g.cols) * CELL + CELL / 2,
	PAD + Math.floor(s / g.cols) * CELL + CELL / 2,
];

const DIRS: [number, number][] = [
	[0, -1],
	[1, 0],
	[0, 1],
	[-1, 0],
];

export function drawGrid(
	svg: SVGSVGElement,
	g: GridSpec,
	d: GridDrawing,
): SVGElement[] {
	const { w, h } = gridSize(g);
	svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
	// Keep cells the same on-screen size whatever the grid dimensions.
	svg.style.maxWidth = `${(w / 324) * 22}rem`;
	svg.replaceChildren();
	const forb = new Set(g.forbidden.map(([r, c]) => r * g.cols + c));
	const tgt = g.target[0] * g.cols + g.target[1];
	for (let c = 0; c < g.cols; c++)
		el(
			"text",
			{
				x: PAD + c * CELL + CELL / 2,
				y: PAD - 6,
				class: "rl-axis",
				"text-anchor": "middle",
			},
			svg,
			String(c + 1),
		);
	for (let r = 0; r < g.rows; r++)
		el(
			"text",
			{
				x: PAD - 6,
				y: PAD + r * CELL + CELL / 2 + 3.5,
				class: "rl-axis",
				"text-anchor": "end",
			},
			svg,
			String(r + 1),
		);
	const n = g.rows * g.cols;
	for (let s = 0; s < n; s++) {
		const x = PAD + (s % g.cols) * CELL;
		const y = PAD + Math.floor(s / g.cols) * CELL;
		const kind = s === tgt ? " target" : forb.has(s) ? " forbidden" : "";
		el(
			"rect",
			{
				x,
				y,
				width: CELL,
				height: CELL,
				class: `rl-cell${kind}${d.hot === s ? " hot" : ""}`,
			},
			svg,
		);
		if (d.heat)
			el(
				"rect",
				{
					x: x + 5,
					y: y + 5,
					width: CELL - 10,
					height: CELL - 10,
					class: "rl-heat",
					style: `fill: rgba(var(--rl-heat), ${(0.55 * d.heat[s]).toFixed(3)})`,
				},
				svg,
			);
	}
	if (d.policy)
		for (let s = 0; s < n; s++) {
			const [cx, cy0] = center(g, s);
			const cy = d.values && !d.bigValues ? cy0 - 7 : cy0;
			d.policy[s].forEach((p, a) => {
				if (p <= 1e-9) return;
				const scale = 0.55 + 0.45 * p;
				if (a === 4) {
					el("circle", { cx, cy, r: 2 + 5 * scale, class: "rl-stay" }, svg);
					return;
				}
				const [dx, dy] = DIRS[a];
				const len = CELL * (d.values && !d.bigValues ? 0.28 : 0.36) * scale;
				const x2 = cx + dx * len;
				const y2 = cy + dy * len;
				el(
					"line",
					{
						x1: cx + dx * 3,
						y1: cy + dy * 3,
						x2: x2 - dx * 4,
						y2: y2 - dy * 4,
						class: "rl-arrow",
					},
					svg,
				);
				const px = -dy;
				const py = dx;
				el(
					"path",
					{
						d: `M${x2},${y2}L${x2 - dx * 7 + px * 4},${y2 - dy * 7 + py * 4}L${x2 - dx * 7 - px * 4},${y2 - dy * 7 - py * 4}Z`,
						class: "rl-arrowhead",
					},
					svg,
				);
			});
		}
	if (d.values)
		d.values.forEach((v, s) => {
			if (v === null) return;
			const [cx, cy] = center(g, s);
			const big = d.bigValues || !d.policy;
			el(
				"text",
				{
					x: cx,
					y: big ? cy + 4.5 : cy + CELL / 2 - 7,
					class: `rl-val${big ? " big" : ""}${d.changed?.[s] ? " changed" : ""}`,
					"text-anchor": "middle",
				},
				svg,
				(Math.abs(v) < 0.5 * 10 ** -(d.digits ?? 1) ? 0 : v).toFixed(
					d.digits ?? 1,
				),
			);
		});
	if (d.trail && d.trail.length > 1) {
		const pts = d.trail.map((s) => center(g, s).join(",")).join(" ");
		el("polyline", { points: pts, class: "rl-trail" }, svg);
	}
	if (d.agent !== undefined) {
		const [cx, cy] = center(g, d.agent);
		el("circle", { cx, cy, r: 7, class: "rl-agent" }, svg);
	}
	// Transparent hit boxes on top so hover works over arrows and text.
	const hits: SVGElement[] = [];
	for (let s = 0; s < n; s++)
		hits.push(
			el(
				"rect",
				{
					x: PAD + (s % g.cols) * CELL,
					y: PAD + Math.floor(s / g.cols) * CELL,
					width: CELL,
					height: CELL,
					class: "rl-hit",
					"data-s": s,
				},
				svg,
			),
		);
	return hits;
}

/** Cell name as the book writes it: (row, column), 1-based. */
export const cellName = (g: GridSpec, s: number) =>
	`(${Math.floor(s / g.cols) + 1}, ${(s % g.cols) + 1})`;
