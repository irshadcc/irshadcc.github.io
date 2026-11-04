// SVG drawing for the order book figures. Every function returns an SVG string whose elements
// carry `obk-` classes; colours come from orderBook.css (light and dark).
import type { BookFrame, SearchStep, WalkStep } from "./orderBook";

const esc = (s: string) =>
	s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const txt = (x: number, y: number, s: string, cls: string, anchor = "middle") =>
	`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`;

// ------------------------------------------------------------------ price ladder

export interface LadderOptions {
	lo: number;
	hi: number;
}

/** Bids to the left of the price column, asks to the right, queue head next to the price. */
export function drawLadder(f: BookFrame, { lo, hi }: LadderOptions): string {
	const unit = 13;
	const rowH = 24;
	const side = 230;
	const priceW = 56;
	const W = side * 2 + priceW;
	const cx = side + priceW / 2;
	const top = 30;
	const rows = hi - lo + 1;
	const H = top + rows * rowH + 8;
	const out: string[] = [];
	out.push(txt(side - 6, 18, "BIDS (buy)  ← queue", "obk-head", "end"));
	out.push(txt(cx, 18, "price", "obk-head"));
	out.push(
		txt(side + priceW + 6, 18, "queue →  ASKS (sell)", "obk-head", "start"),
	);
	const bb = f.bestBid;
	const ba = f.bestAsk;
	for (let p = hi; p >= lo; p -= 1) {
		const y = top + (hi - p) * rowH;
		const inSpread = bb !== null && ba !== null && p > bb && p < ba;
		const isBest = p === bb || p === ba;
		out.push(
			`<rect x="0" y="${y}" width="${W}" height="${rowH}" class="obk-row${inSpread ? " spread" : ""}${p % 2 ? " odd" : ""}"/>`,
		);
		out.push(
			txt(
				cx,
				y + rowH / 2 + 4,
				`${p}`,
				`obk-price${isBest ? (p === bb ? " best-bid" : " best-ask") : ""}`,
			),
		);
		const queue = f.levels.get(p) ?? [];
		const ghosts = f.ghosts.filter((g) => g.price === p).map((g) => g.order);
		const cancelled =
			f.cancelled && f.cancelled.price === p ? [f.cancelled] : [];
		// consumed orders were at the front of the queue; a cancelled one is shown where it was
		type Chip = {
			id: number;
			qty: number;
			side: string;
			kind: "live" | "ghost" | "cancel";
		};
		const chips: Chip[] = [
			...ghosts.map((o) => ({ ...o, kind: "ghost" as const })),
			...queue.map((o) => ({ ...o, kind: "live" as const })),
			...cancelled.map((o) => ({
				id: o.id,
				qty: o.qty,
				side: o.side,
				kind: "cancel" as const,
			})),
		];
		let off = 4;
		for (const c of chips) {
			const w = Math.max(30, c.qty * unit + 16);
			const isBid = c.side === "buy";
			const x = isBid ? side - off - w : side + priceW + off;
			const filled = f.filled.get(c.id);
			const cls = [
				"obk-chip",
				isBid ? "bid" : "ask",
				c.kind,
				f.added === c.id ? "added" : "",
				filled !== undefined && c.kind === "live" ? "partial" : "",
			]
				.filter(Boolean)
				.join(" ");
			const label =
				c.kind === "live" && filled !== undefined
					? `#${c.id} ${c.qty} (−${filled})`
					: `#${c.id} ${c.qty}`;
			out.push(
				`<g class="${cls}"><title>${esc(`order #${c.id}: ${c.side} ${c.qty} at ${p}${c.kind === "ghost" ? " (filled)" : c.kind === "cancel" ? " (cancelled)" : ""}`)}</title><rect x="${x}" y="${y + 3}" width="${w}" height="${rowH - 6}" rx="4"/>${txt(x + w / 2, y + rowH / 2 + 4, label, "obk-chip-t")}</g>`,
			);
			off += w + 3;
		}
		if (queue.length > 0) {
			const total = queue.reduce((s, o) => s + o.qty, 0);
			const isBid = queue[0].side === "buy";
			out.push(
				txt(
					isBid ? 4 : W - 4,
					y + rowH / 2 + 4,
					`Σ ${total}`,
					"obk-total",
					isBid ? "start" : "end",
				),
			);
		}
	}
	return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="order book ladder">${out.join("")}</svg>`;
}

// ------------------------------------------------------------------ sorted vector

export interface VectorOptions {
	/** text shown in each cell; defaults to the key */
	show?: (key: number, i: number) => string;
	target: number;
	caption?: string;
}

export function drawVector(
	keys: number[],
	steps: SearchStep[],
	at: number,
	opt: VectorOptions,
): string {
	const cw = 40;
	const ch = 34;
	const left = 10;
	const n = keys.length;
	const W = Math.max(left * 2 + n * cw, 300);
	const top = 34;
	const H = top + ch + 50;
	const s = steps[at];
	const seen = steps
		.slice(0, at + 1)
		.map((x, k) => [x.probe, k] as const)
		.filter(([p]) => p !== null) as [number, number][];
	const out: string[] = [];
	if (opt.caption) out.push(txt(left, 14, opt.caption, "obk-head", "start"));
	// possible range for binary searches
	const [r0, r1] = s.range;
	if (s.probe !== null && (r0 > 0 || r1 < n))
		out.push(
			`<rect x="${left + r0 * cw}" y="${top - 6}" width="${(r1 - r0) * cw}" height="${ch + 12}" rx="6" class="obk-range"/>`,
		);
	for (let i = 0; i < n; i += 1) {
		const x = left + i * cw;
		const order = seen.filter(([p]) => p === i).map(([, k]) => k + 1);
		const cls = [
			"obk-cell",
			s.probe === i ? "probe" : order.length ? "seen" : "",
			i === n - 1 ? "back" : "",
		]
			.filter(Boolean)
			.join(" ");
		out.push(
			`<rect x="${x + 1}" y="${top}" width="${cw - 2}" height="${ch}" rx="3" class="${cls}"/>`,
		);
		out.push(
			txt(
				x + cw / 2,
				top + ch / 2 + 4,
				opt.show ? opt.show(keys[i], i) : `${keys[i]}`,
				"obk-cell-t",
			),
		);
		out.push(txt(x + cw / 2, top + ch + 14, `${i}`, "obk-idx"));
		if (order.length)
			out.push(txt(x + cw / 2, top - 10, order.join(","), "obk-order"));
	}
	out.push(txt(left + (n - 0.5) * cw, top + ch + 30, "best ↑", "obk-idx"));
	if (s.result !== null) {
		const r = s.result;
		const exists = r < n && keys[r] === opt.target;
		const x = left + r * cw;
		out.push(
			exists
				? `<rect x="${x - 1}" y="${top - 3}" width="${cw + 2}" height="${ch + 6}" rx="5" class="obk-hit"/>`
				: `<path d="M${x} ${top - 8}V${top + ch + 4}" class="obk-insert"/>${txt(x, top + ch + 30, `insert ${opt.show ? opt.show(opt.target, -1) : opt.target}`, "obk-insert-t")}`,
		);
	}
	return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="sorted vector search">${out.join("")}</svg>`;
}

// ------------------------------------------------------------------ tick array and bitmap

export interface TickOptions {
	lo: number;
	wordBits: number;
	bitmap: boolean;
	from: number;
}

export function drawTicks(
	occupied: boolean[],
	steps: WalkStep[],
	at: number,
	opt: TickOptions,
): string {
	const n = occupied.length;
	const cw = 10;
	const gap = 4;
	const left = 10;
	const words = Math.ceil(n / opt.wordBits);
	const xOf = (i: number) => left + i * cw + Math.floor(i / opt.wordBits) * gap;
	const W = xOf(n - 1) + cw + left;
	const sumTop = 26;
	const top = opt.bitmap ? 112 : 26;
	const H = top + 60;
	const s = steps[at];
	const visited = new Set(steps.slice(0, at + 1).flatMap((x) => x.cells));
	const out: string[] = [];
	if (opt.bitmap) {
		out.push(
			txt(
				left,
				sumTop - 8,
				"summary: one bit per word (is any tick in the word occupied?)",
				"obk-head",
				"start",
			),
		);
		const sw = 22;
		for (let k = 0; k < words; k += 1) {
			const x = xOf(k * opt.wordBits) + (opt.wordBits * cw - sw) / 2;
			const on = occupied
				.slice(k * opt.wordBits, (k + 1) * opt.wordBits)
				.some(Boolean);
			const below = k < Math.floor((opt.from - 1) / opt.wordBits);
			const cls = ["obk-bit", on ? "on" : "", s.summary && below ? "probe" : ""]
				.filter(Boolean)
				.join(" ");
			out.push(
				`<rect x="${x}" y="${sumTop}" width="${sw}" height="20" rx="3" class="${cls}"/>`,
			);
			out.push(txt(x + sw / 2, sumTop + 14, on ? "1" : "0", "obk-bit-t"));
			out.push(txt(x + sw / 2, sumTop + 32, `w${k}`, "obk-idx"));
			// line from the summary bit to its word
			const wx = xOf(k * opt.wordBits) + (opt.wordBits * cw) / 2;
			out.push(
				`<path d="M${x + sw / 2} ${sumTop + 36}L${wx} ${top - 22}" class="obk-link${s.summary && below && on ? " hot" : ""}"/>`,
			);
		}
		out.push(
			txt(
				left,
				top - 8,
				"level 0: one bit per tick (is the level non-empty?)",
				"obk-head",
				"start",
			),
		);
	} else {
		out.push(
			txt(
				left,
				top - 8,
				"one price level per tick (filled = has orders)",
				"obk-head",
				"start",
			),
		);
	}
	for (let k = 0; k < words; k += 1)
		if (s.words.includes(k))
			out.push(
				`<rect x="${xOf(k * opt.wordBits) - 3}" y="${top - 3}" width="${opt.wordBits * cw + 6}" height="26" rx="5" class="obk-word"/>`,
			);
	for (let i = 0; i < n; i += 1) {
		const x = xOf(i);
		const cls = [
			"obk-tick",
			occupied[i] ? "on" : "",
			i === opt.from ? "emptied" : "",
			s.found === i
				? "found"
				: s.cells.includes(i)
					? "probe"
					: visited.has(i)
						? "seen"
						: "",
		]
			.filter(Boolean)
			.join(" ");
		out.push(
			`<rect x="${x}" y="${top}" width="${cw - 2}" height="20" rx="2" class="${cls}"/>`,
		);
		if (opt.bitmap)
			out.push(
				txt(x + (cw - 2) / 2, top + 13, occupied[i] ? "1" : "0", "obk-bit-t"),
			);
		if (i % opt.wordBits === 0)
			out.push(txt(x, top + 34, `${opt.lo + i}`, "obk-idx", "start"));
	}
	out.push(
		txt(
			xOf(opt.from) + cw,
			top + 50,
			"best level just emptied ↑",
			"obk-insert-t",
			"end",
		),
	);
	return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="tick array">${out.join("")}</svg>`;
}

// ------------------------------------------------------------------ three indexes (tree + FIFOs + id map)

export interface IndexOrder {
	id: number;
	qty: number;
	slot: number;
}
export interface IndexLevel {
	price: number;
	side: "buy" | "sell";
	orders: IndexOrder[];
}
export interface IndexState {
	levels: IndexLevel[]; // any order; drawn by price, high to low
	/** highlighted things */
	hotId?: number;
	hotLevel?: number;
	hotOrders?: number[];
	ghost?: { price: number; order: IndexOrder; at: number };
	/** tree nodes visited from the root, by price */
	treePath?: number[];
	/** price whose node is being inserted / erased */
	treeChange?: { price: number; kind: "insert" | "erase" };
	freeSlots: number[];
	/** explicit tree shape (price -> parent price), else a balanced tree per side */
	parents?: Record<number, number | null>;
	note: string;
	label: string;
}

/** depth of each key in a balanced BST built from sorted keys (median as root) */
function bstShape(
	keys: number[],
): Map<number, { depth: number; parent: number | null }> {
	const m = new Map<number, { depth: number; parent: number | null }>();
	const build = (lo: number, hi: number, d: number, parent: number | null) => {
		if (lo > hi) return;
		const mid = Math.floor((lo + hi + 1) / 2);
		m.set(keys[mid], { depth: d, parent });
		build(lo, mid - 1, d + 1, keys[mid]);
		build(mid + 1, hi, d + 1, keys[mid]);
	};
	build(0, keys.length - 1, 0, null);
	return m;
}

function shapeFrom(keys: number[], parents: Record<number, number | null>) {
	const m = new Map<number, { depth: number; parent: number | null }>();
	const depth = (k: number): number => {
		const p = parents[k];
		return p === null || p === undefined ? 0 : 1 + depth(p);
	};
	for (const k of keys)
		m.set(k, { depth: depth(k), parent: parents[k] ?? null });
	return m;
}

export function drawIndexes(st: IndexState): string {
	const rowH = 34;
	const top = 30;
	const treeX = 18;
	const depthW = 26;
	const lvX = 120;
	const lvW = 112;
	const chainX = lvX + lvW + 30;
	const ow = 66;
	const levels = [...st.levels].sort((a, b) => b.price - a.price);
	const asks = levels.filter((l) => l.side === "sell");
	const bids = levels.filter((l) => l.side === "buy");
	const rowOf = new Map<number, number>();
	let row = 0;
	for (const l of asks) rowOf.set(l.price, row++);
	const gapRow = row;
	row += 0.6;
	for (const l of bids) rowOf.set(l.price, row++);
	const yOf = (p: number) => top + (rowOf.get(p) as number) * rowH;
	const idsTop = top + row * rowH + 26;
	const allOrders = levels.flatMap((l) =>
		l.orders.map((o) => ({ ...o, price: l.price })),
	);
	const ids = [
		...allOrders.map((o) => o.id),
		...(st.ghost ? [st.ghost.order.id] : []),
	].sort((a, b) => a - b);
	const maxId = Math.max(8, ...ids);
	const cellW = 40;
	const W = Math.max(
		chainX + 3 * (ow + 22) + 10,
		10 + (maxId + 1) * cellW + 110,
	);
	const H = idsTop + 42;
	const out: string[] = [];
	out.push(txt(treeX, 16, "std::map (tree)", "obk-head", "start"));
	out.push(txt(lvX, 16, "Limit (level)", "obk-head", "start"));
	out.push(
		txt(chainX, 16, "orders: intrusive FIFO, head → tail", "obk-head", "start"),
	);
	out.push(
		`<path d="M4 ${top + gapRow * rowH + 0.3 * rowH}H${W - 4}" class="obk-sep"/>`,
		txt(
			W - 6,
			top + gapRow * rowH + 0.3 * rowH - 4,
			"asks ↑   bids ↓",
			"obk-idx",
			"end",
		),
	);
	// trees: one per side, sideways (depth to the right), rows sorted by price
	for (const sideLevels of [asks, bids]) {
		const keys = sideLevels.map((l) => l.price).sort((a, b) => a - b);
		const shape = st.parents ? shapeFrom(keys, st.parents) : bstShape(keys);
		const pos = (p: number) => ({
			x: treeX + 8 + (shape.get(p)?.depth ?? 0) * depthW,
			y: yOf(p) + rowH / 2,
		});
		for (const [p, sh] of shape)
			if (sh.parent !== null) {
				const a = pos(sh.parent);
				const b = pos(p);
				const hot =
					st.treePath?.includes(p) && st.treePath?.includes(sh.parent);
				out.push(
					`<path d="M${a.x} ${a.y}L${b.x} ${b.y}" class="obk-tree-e${hot ? " hot" : ""}"/>`,
				);
			}
		for (const [p] of shape) {
			const { x, y } = pos(p);
			const hot = st.treePath?.includes(p);
			const ch = st.treeChange?.price === p ? ` ${st.treeChange.kind}` : "";
			out.push(`<path d="M${x} ${y}H${lvX}" class="obk-dot-link"/>`);
			out.push(
				`<circle cx="${x}" cy="${y}" r="7" class="obk-tree-n${hot ? " hot" : ""}${ch}"/>`,
			);
		}
	}
	for (const l of levels) {
		const y = yOf(l.price);
		const total = l.orders.reduce((s, o) => s + o.qty, 0);
		const hot = st.hotLevel === l.price;
		const ch = st.treeChange?.price === l.price ? ` ${st.treeChange.kind}` : "";
		const sideCls = l.side === "buy" ? "bid" : "ask";
		out.push(
			`<g class="obk-level ${sideCls}${hot ? " hot" : ""}${ch}"><rect x="${lvX}" y="${y + 3}" width="${lvW}" height="${rowH - 6}" rx="5"/>${txt(lvX + 8, y + rowH / 2 + 4, `${l.price}`, "obk-lv-p", "start")}${txt(lvX + lvW - 8, y + rowH / 2 + 4, l.orders.length ? `total ${total}` : "empty", "obk-lv-t", "end")}</g>`,
		);
		const chain = [...l.orders];
		const ghostHere = st.ghost && st.ghost.price === l.price ? st.ghost : null;
		const items: { o: IndexOrder; ghost: boolean }[] = chain.map((o) => ({
			o,
			ghost: false,
		}));
		if (ghostHere)
			items.splice(ghostHere.at, 0, { o: ghostHere.order, ghost: true });
		items.forEach(({ o, ghost }, k) => {
			const x = chainX + k * (ow + 22);
			const hotO = st.hotOrders?.includes(o.id) || st.hotId === o.id;
			out.push(
				`<g class="obk-node ${sideCls}${hotO ? " hot" : ""}${ghost ? " ghost" : ""}"><rect x="${x}" y="${y + 4}" width="${ow}" height="${rowH - 8}" rx="4"/>${txt(x + ow / 2, y + rowH / 2 + 4, `#${o.id} · ${o.qty}`, "obk-chip-t")}${txt(x + ow - 3, y + 11, `s${o.slot}`, "obk-slot", "end")}</g>`,
			);
			if (k > 0)
				out.push(
					`<path d="M${x - 20} ${y + rowH / 2 - 3}H${x - 3}M${x - 3} ${y + rowH / 2 + 3}H${x - 20}" class="obk-ll${ghost || items[k - 1].ghost ? " cut" : ""}"/>`,
				);
		});
		if (items.length)
			out.push(
				`<path d="M${lvX + lvW} ${y + rowH / 2}H${chainX - 2}" class="obk-ll head"/>`,
			);
	}
	// id -> slot index
	out.push(
		txt(
			10,
			idsTop - 8,
			"order id → pool slot (flat array or hash map)",
			"obk-head",
			"start",
		),
	);
	const live = new Map(allOrders.map((o) => [o.id, o]));
	for (let id = 1; id <= maxId; id += 1) {
		const x = 10 + (id - 1) * cellW;
		const o = live.get(id);
		const hot = st.hotId === id;
		out.push(
			`<rect x="${x}" y="${idsTop}" width="${cellW - 3}" height="32" rx="3" class="obk-idcell${o ? " on" : ""}${hot ? " hot" : ""}"/>`,
		);
		out.push(txt(x + (cellW - 3) / 2, idsTop + 13, `${id}`, "obk-idx"));
		out.push(
			txt(
				x + (cellW - 3) / 2,
				idsTop + 27,
				o ? `s${o.slot}` : "–",
				"obk-cell-t",
			),
		);
	}
	out.push(
		txt(
			10 + maxId * cellW + 8,
			idsTop + 21,
			`free slots: ${st.freeSlots.length ? st.freeSlots.join(", ") : "none"}`,
			"obk-idx",
			"start",
		),
	);
	// arrow from the hot id to its order node
	if (st.hotId !== undefined) {
		const o =
			live.get(st.hotId) ??
			(st.ghost?.order.id === st.hotId
				? { ...st.ghost.order, price: st.ghost.price }
				: undefined);
		if (o) {
			const lvl = levels.find((l) => l.price === o.price);
			const k = lvl ? lvl.orders.findIndex((x) => x.id === o.id) : -1;
			const at = k >= 0 ? k : (st.ghost?.at ?? 0);
			const x1 = 10 + (st.hotId - 1) * cellW + (cellW - 3) / 2;
			const x2 = chainX + at * (ow + 22) + ow / 2;
			const y2 = yOf(o.price) + rowH - 4;
			out.push(
				`<path d="M${x1} ${idsTop}C${x1} ${idsTop - 30} ${x2} ${y2 + 30} ${x2} ${y2 + 2}" class="obk-arrow"/>`,
			);
		}
	}
	return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="tree of limits with order lists">${out.join("")}</svg>`;
}

// ------------------------------------------------------------------ horizontal bar chart

export interface BarGroup {
	name: string;
	bars: { label: string; value: number; hot?: boolean }[];
}

export function drawBars(
	groups: BarGroup[],
	unit: string,
	maxValue?: number,
): string {
	const labelW = 150;
	const barMax = 330;
	const bh = 15;
	const W = labelW + barMax + 70;
	const max =
		maxValue ?? Math.max(...groups.flatMap((g) => g.bars.map((b) => b.value)));
	const out: string[] = [];
	let y = 4;
	for (const g of groups) {
		out.push(txt(0, y + 12, g.name, "obk-head", "start"));
		y += 20;
		for (const b of g.bars) {
			const w = Math.max(1, (b.value / max) * barMax);
			const clipped = b.value > max;
			out.push(txt(labelW - 8, y + bh - 3, b.label, "obk-bar-l", "end"));
			out.push(
				`<rect x="${labelW}" y="${y}" width="${Math.min(w, barMax)}" height="${bh}" rx="2" class="obk-bar${b.hot ? " hot" : ""}"/>`,
			);
			out.push(
				txt(
					labelW + Math.min(w, barMax) + 6,
					y + bh - 3,
					`${b.value.toFixed(1)} ${unit}${clipped ? " →" : ""}`,
					"obk-bar-v",
					"start",
				),
			);
			y += bh + 4;
		}
		y += 8;
	}
	return `<svg viewBox="0 0 ${W} ${y}" width="${W}" height="${y}" role="img" aria-label="bar chart">${out.join("")}</svg>`;
}

/** Vertical bars on a log-ish category axis: `xs` labels, several series side by side. */
export function drawColumns(
	xs: string[],
	series: { name: string; values: number[]; cls: string }[],
	yLabel: string,
	log = false,
): string {
	const left = 52;
	const top = 26;
	const h = 190;
	const gw = 52;
	const W = left + xs.length * gw + 10;
	const H = top + h + 44;
	const all = series.flatMap((s) => s.values);
	const max = Math.max(...all);
	const min = Math.min(...all.filter((v) => v > 0));
	const lo = log ? 10 ** Math.floor(Math.log10(min)) : 0;
	const step =
		[1, 2, 2.5, 5, 10, 20, 25, 50, 100].find((s) => s * 4 >= max) ?? max / 4;
	const hi = log
		? 10 ** Math.ceil(Math.log10(max))
		: step * Math.ceil(max / step);
	const yOf = (v: number) =>
		top +
		h -
		(log
			? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))
			: (v - lo) / (hi - lo)) *
			h;
	const out: string[] = [];
	const ticks: number[] = [];
	if (log) for (let t = lo; t <= hi * 1.0001; t *= 10) ticks.push(t);
	else for (let t = 0; t <= hi + 1e-9; t += step) ticks.push(t);
	for (const t of ticks) {
		const y = yOf(t);
		out.push(`<path d="M${left} ${y}H${W - 6}" class="obk-grid"/>`);
		out.push(
			txt(
				left - 6,
				y + 4,
				t >= 1000 ? `${t / 1000}k` : `${+t.toPrecision(3)}`,
				"obk-idx",
				"end",
			),
		);
	}
	out.push(txt(4, 14, yLabel, "obk-head", "start"));
	const bw = (gw - 12) / series.length;
	xs.forEach((x, i) => {
		const gx = left + i * gw + 6;
		series.forEach((s, k) => {
			const v = s.values[i];
			const y = yOf(Math.max(v, log ? lo : 0));
			out.push(
				`<rect x="${gx + k * bw}" y="${y}" width="${bw - 1}" height="${top + h - y}" class="obk-col ${s.cls}"><title>${esc(`${s.name}, ${x}: ${v}`)}</title></rect>`,
			);
		});
		out.push(txt(gx + (gw - 12) / 2, top + h + 15, x, "obk-idx"));
	});
	let lx = left;
	for (const s of series) {
		out.push(
			`<rect x="${lx}" y="${H - 14}" width="10" height="10" class="obk-col ${s.cls}"/>`,
		);
		out.push(txt(lx + 14, H - 5, s.name, "obk-bar-l", "start"));
		lx += 14 + s.name.length * 6.5 + 18;
	}
	return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="column chart">${out.join("")}</svg>`;
}
