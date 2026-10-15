// Small SVG helpers shared by the sampling figures (browser only): elements, axes, paths,
// histograms and the play / step controls. Styles live in sampling.css.

export const NS = "http://www.w3.org/2000/svg";

export function el(
	tag: string,
	attrs: Record<string, string | number>,
	parent: Element,
	text?: string,
): SVGElement {
	const e = document.createElementNS(NS, tag) as SVGElement;
	for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
	if (text !== undefined) e.textContent = text;
	parent.append(e);
	return e;
}

export const fmt = (v: number, d = 2) =>
	(Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d);

/** Round tick step: 1, 2 or 5 times a power of ten. */
export function tickStep(range: number, target = 6): number {
	const raw = range / target;
	const p = 10 ** Math.floor(Math.log10(raw));
	const m = raw / p;
	return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

export interface Box {
	x: number;
	y: number;
	w: number;
	h: number;
}

export interface Axes {
	box: Box;
	x: [number, number];
	y: [number, number];
	sx: (v: number) => number;
	sy: (v: number) => number;
}

export interface AxesOptions {
	xLabel?: string;
	yLabel?: string;
	xTicks?: number;
	yTicks?: number;
	/** Skip the y tick labels (for densities whose scale does not matter). */
	noYLabels?: boolean;
}

/** Draw a framed plot area with grid lines, ticks and labels; return the scales. */
export function axes(
	parent: Element,
	box: Box,
	x: [number, number],
	y: [number, number],
	opt: AxesOptions = {},
): Axes {
	const sx = (v: number) => box.x + ((v - x[0]) / (x[1] - x[0])) * box.w;
	const sy = (v: number) =>
		box.y + box.h - ((v - y[0]) / (y[1] - y[0])) * box.h;
	const g = el("g", { class: "smp-axes" }, parent);
	const xs = tickStep(x[1] - x[0], opt.xTicks ?? 6);
	const xd = Math.max(0, -Math.floor(Math.log10(xs) + 1e-9));
	for (let t = Math.ceil(x[0] / xs - 1e-9) * xs; t <= x[1] + 1e-9; t += xs) {
		el(
			"line",
			{ x1: sx(t), x2: sx(t), y1: box.y, y2: box.y + box.h, class: "smp-grid" },
			g,
		);
		el(
			"text",
			{
				x: sx(t),
				y: box.y + box.h + 13,
				class: "smp-tick",
				"text-anchor": "middle",
			},
			g,
			fmt(t, xd),
		);
	}
	const ys = tickStep(y[1] - y[0], opt.yTicks ?? 4);
	const yd = Math.max(0, -Math.floor(Math.log10(ys) + 1e-9));
	for (let t = Math.ceil(y[0] / ys - 1e-9) * ys; t <= y[1] + 1e-9; t += ys) {
		el(
			"line",
			{ x1: box.x, x2: box.x + box.w, y1: sy(t), y2: sy(t), class: "smp-grid" },
			g,
		);
		if (!opt.noYLabels)
			el(
				"text",
				{
					x: box.x - 5,
					y: sy(t) + 3.5,
					class: "smp-tick",
					"text-anchor": "end",
				},
				g,
				fmt(t, yd),
			);
	}
	el(
		"rect",
		{ x: box.x, y: box.y, width: box.w, height: box.h, class: "smp-plotbox" },
		g,
	);
	if (opt.xLabel)
		el(
			"text",
			{
				x: box.x + box.w / 2,
				y: box.y + box.h + 28,
				class: "smp-label",
				"text-anchor": "middle",
			},
			g,
			opt.xLabel,
		);
	if (opt.yLabel) {
		const cy = box.y + box.h / 2;
		const lx = box.x - (opt.noYLabels ? 10 : 34);
		el(
			"text",
			{
				x: lx,
				y: cy,
				class: "smp-label",
				"text-anchor": "middle",
				transform: `rotate(-90 ${lx} ${cy})`,
			},
			g,
			opt.yLabel,
		);
	}
	return { box, x, y, sx, sy };
}

/** SVG path through (x, f(x)) for `n` points across the axes' x range; y is clamped to the box. */
export function curve(
	a: Axes,
	f: (x: number) => number,
	n = 300,
	lo = a.x[0],
	hi = a.x[1],
): string {
	let d = "";
	let pen = false;
	for (let i = 0; i <= n; i++) {
		const x = lo + ((hi - lo) * i) / n;
		const y = f(x);
		if (!Number.isFinite(y)) {
			pen = false;
			continue;
		}
		const py = Math.max(a.box.y - 2, Math.min(a.box.y + a.box.h, a.sy(y)));
		d += `${pen ? "L" : "M"}${a.sx(x).toFixed(1)},${py.toFixed(1)}`;
		pen = true;
	}
	return d;
}

/** Closed area under f, for shading. */
export function area(
	a: Axes,
	f: (x: number) => number,
	n = 300,
	lo = a.x[0],
	hi = a.x[1],
): string {
	return `${curve(a, f, n, lo, hi)}L${a.sx(hi).toFixed(1)},${a.sy(a.y[0]).toFixed(1)}L${a.sx(lo).toFixed(1)},${a.sy(a.y[0]).toFixed(1)}Z`;
}

/** Density histogram of `values` (weighted if `weights` given) over [lo, hi] with `bins` bins. */
export function histogram(
	values: number[],
	lo: number,
	hi: number,
	bins: number,
	weights?: number[],
): number[] {
	const h = new Array<number>(bins).fill(0);
	const width = (hi - lo) / bins;
	let total = 0;
	values.forEach((v, i) => {
		const w = weights ? weights[i] : 1;
		total += w;
		const b = Math.floor((v - lo) / width);
		if (b >= 0 && b < bins) h[b] += w;
	});
	return total ? h.map((c) => c / (total * width)) : h;
}

/** Draw a histogram as bars into `parent`. */
export function bars(
	parent: Element,
	a: Axes,
	h: number[],
	lo: number,
	hi: number,
	cls: string,
) {
	const width = (hi - lo) / h.length;
	h.forEach((v, i) => {
		if (v <= 0) return;
		const x0 = a.sx(lo + i * width);
		const x1 = a.sx(lo + (i + 1) * width);
		const top = Math.max(a.box.y, a.sy(v));
		el(
			"rect",
			{
				x: x0 + 0.5,
				y: top,
				width: Math.max(0, x1 - x0 - 1),
				height: a.sy(a.y[0]) - top,
				class: cls,
			},
			parent,
		);
	});
}

export interface Player {
	stop: () => void;
}

/**
 * Wire the play / step buttons inside `root` (data-act = first | prev | play | next). `step(dir)`
 * moves one step and returns false at the end, which stops playback.
 */
export function bindPlayer(
	root: Element,
	step: (dir: 1 | -1) => boolean,
	reset: () => void,
	interval = 120,
): Player {
	const play = root.querySelector<HTMLButtonElement>("[data-act=play]");
	let timer: number | undefined;
	const stop = () => {
		if (timer !== undefined) window.clearInterval(timer);
		timer = undefined;
		play?.classList.remove("playing");
		play?.setAttribute("aria-label", "Play");
	};
	const start = () => {
		play?.classList.add("playing");
		play?.setAttribute("aria-label", "Pause");
		timer = window.setInterval(() => {
			if (!step(1)) stop();
		}, interval);
	};
	for (const b of root.querySelectorAll<HTMLButtonElement>("[data-act]")) {
		b.addEventListener("click", () => {
			const act = b.dataset.act;
			if (act === "play") return timer === undefined ? start() : stop();
			stop();
			if (act === "first") reset();
			if (act === "prev") step(-1);
			if (act === "next") step(1);
		});
	}
	return { stop };
}
