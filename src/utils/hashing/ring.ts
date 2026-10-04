// Consistent hashing on a 32-bit ring, used by HashRing.astro and the consistent-hashing post.
//
// hash32(s) is 32-bit FNV-1a over the UTF-8 bytes of s, followed by MurmurHash3's 32-bit
// finalizer (fmix32). Plain FNV-1a puts strings that differ only in their last byte close
// together ("node-A#0" ... "node-A#7" all land within 3% of the ring), so virtual nodes would
// clump; the finalizer spreads them out. The C++ code in the post computes the same hash.
//
// Server i is named `${prefix}${letter}` (node-A, node-B, ...); its virtual node j is placed at
// hash32(`${name}#${j}`). A key belongs to the first point at or clockwise after hash32(key).

export const RING = 2 ** 32;

/** 32-bit FNV-1a of the UTF-8 bytes of s (same as fnv1a32 in payments/sharding.ts). */
export function fnv1a32(s: string): number {
	let h = 0x811c9dc5;
	for (const b of new TextEncoder().encode(s)) {
		h ^= b;
		h = Math.imul(h, 0x01000193) >>> 0;
	}
	return h >>> 0;
}

/** MurmurHash3's 32-bit finalizer. */
export function fmix32(x: number): number {
	let h = x >>> 0;
	h ^= h >>> 16;
	h = Math.imul(h, 0x85ebca6b) >>> 0;
	h ^= h >>> 13;
	h = Math.imul(h, 0xc2b2ae35) >>> 0;
	h ^= h >>> 16;
	return h >>> 0;
}

export function hash32(s: string): number {
	return fmix32(fnv1a32(s));
}

export function nodeName(i: number, prefix = "node-"): string {
	return `${prefix}${String.fromCharCode(65 + i)}`;
}

export interface Point {
	/** Position on the ring, 0 .. 2^32 - 1. */
	pos: number;
	/** Server index. */
	node: number;
	/** Virtual node index within the server. */
	v: number;
}

/** Ring points of servers 0..n-1 with `vnodes` points each, sorted by position. */
export function ringPoints(
	n: number,
	vnodes: number,
	prefix = "node-",
): Point[] {
	const pts: Point[] = [];
	for (let node = 0; node < n; node++)
		for (let v = 0; v < vnodes; v++)
			pts.push({ pos: hash32(`${nodeName(node, prefix)}#${v}`), node, v });
	return pts.sort((a, b) => a.pos - b.pos || a.node - b.node);
}

/** Index into `pts` of the first point at or after h, wrapping to 0. Binary search. */
export function successor(pts: readonly Point[], h: number): number {
	let lo = 0;
	let hi = pts.length;
	while (lo < hi) {
		const mid = (lo + hi) >> 1;
		if (pts[mid].pos < h) lo = mid + 1;
		else hi = mid;
	}
	return lo === pts.length ? 0 : lo;
}

/** Fraction of the ring each server owns: the arc from each of its points back to the previous point. */
export function ringShare(pts: readonly Point[], n: number): number[] {
	const share = new Array<number>(n).fill(0);
	pts.forEach((p, i) => {
		const prev = pts[(i - 1 + pts.length) % pts.length].pos;
		const arc = pts.length === 1 ? RING : (p.pos - prev + RING) % RING;
		share[p.node] += arc / RING;
	});
	return share;
}

export interface RingPlacement {
	points: Point[];
	/** Hash of each key. */
	hash: number[];
	/** Server of each key. */
	owner: number[];
	/** Index into points of the point each key maps to. */
	point: number[];
	/** Keys per server. */
	load: number[];
	/** Fraction of the ring per server. */
	share: number[];
}

export function placeKeys(
	keys: readonly string[],
	n: number,
	vnodes: number,
	prefix = "node-",
): RingPlacement {
	const points = ringPoints(n, vnodes, prefix);
	const hash = keys.map(hash32);
	const point = hash.map((h) => successor(points, h));
	const owner = point.map((i) => points[i].node);
	const load = new Array<number>(n).fill(0);
	for (const o of owner) load[o]++;
	return { points, hash, owner, point, load, share: ringShare(points, n) };
}

// --- Drawing (HashRing.astro) --------------------------------------------------------------------

export interface DrawRingOptions {
	nodes: number;
	vnodes: number;
	/** Outline keys whose server changed since nodes - 1 servers. */
	showMoves: boolean;
	prefix?: string;
	size?: number;
}

const f1 = (x: number) => x.toFixed(1);
const hex = (h: number) => `0x${h.toString(16).padStart(8, "0")}`;

/** Angle of ring position h, clockwise from 12 o'clock, in radians. */
function angle(h: number): number {
	return (h / RING) * 2 * Math.PI;
}

function xy(cx: number, cy: number, r: number, a: number): [number, number] {
	return [cx + r * Math.sin(a), cy - r * Math.cos(a)];
}

/** Clockwise arc of radius r from angle a0 to a1. */
function arcPath(
	cx: number,
	cy: number,
	r: number,
	a0: number,
	a1: number,
): string {
	let sweep = a1 - a0;
	if (sweep < 0) sweep += 2 * Math.PI;
	if (sweep > 2 * Math.PI - 1e-6) {
		// Full circle: two half arcs.
		const [x0, y0] = xy(cx, cy, r, a0);
		const [xm, ym] = xy(cx, cy, r, a0 + Math.PI);
		return `M${f1(x0)},${f1(y0)}A${r},${r} 0 1 1 ${f1(xm)},${f1(ym)}A${r},${r} 0 1 1 ${f1(x0)},${f1(y0)}`;
	}
	const [x0, y0] = xy(cx, cy, r, a0);
	const [x1, y1] = xy(cx, cy, r, a0 + sweep);
	return `M${f1(x0)},${f1(y0)}A${r},${r} 0 ${sweep > Math.PI ? 1 : 0} 1 ${f1(x1)},${f1(y1)}`;
}

export function drawRing(
	keys: readonly string[],
	o: DrawRingOptions,
): { svg: string; moved: number; load: number[]; share: number[] } {
	const prefix = o.prefix ?? "node-";
	const size = o.size ?? 340;
	const cx = size / 2;
	const cy = size / 2;
	const R = size / 2 - 44;
	const now = placeKeys(keys, o.nodes, o.vnodes, prefix);
	const before =
		o.nodes > 1 ? placeKeys(keys, o.nodes - 1, o.vnodes, prefix) : now;
	const moved = new Set(
		now.owner.flatMap((s, i) =>
			o.showMoves && s !== before.owner[i] ? [i] : [],
		),
	);
	const parts: string[] = [];
	parts.push(`<circle class="hr-base" cx="${cx}" cy="${cy}" r="${R}"/>`);
	// Arcs: each point owns the arc from the previous point up to itself.
	const pts = now.points;
	pts.forEach((p, i) => {
		const prev = pts[(i - 1 + pts.length) % pts.length];
		parts.push(
			`<path class="hr-arc s${p.node}" d="${arcPath(cx, cy, R, angle(prev.pos), angle(p.pos))}"/>`,
		);
	});
	// 12 o'clock tick: position 0.
	parts.push(
		`<line class="hr-zero" x1="${cx}" y1="${cy - R - 9}" x2="${cx}" y2="${cy - R + 9}"/><text class="hr-zt" x="${cx}" y="${cy - R + 40}">0</text>`,
	);
	// Keys, inside the ring, with the clockwise walk to their point shown on hover.
	const rk = R - 22;
	keys.forEach((k, i) => {
		const a = angle(now.hash[i]);
		const p = pts[now.point[i]];
		const s = now.owner[i];
		const mv = moved.has(i);
		const [x, y] = xy(cx, cy, rk, a);
		const walk = `M${f1(x)},${f1(y)}L${f1(xy(cx, cy, R - 8, a)[0])},${f1(xy(cx, cy, R - 8, a)[1])}${arcPath(cx, cy, R - 8, a, angle(p.pos)).replace(/^M[^A]+/, "")}`;
		const vn = o.vnodes > 1 ? `#${p.v}` : "";
		parts.push(
			`<g class="hr-key s${s}${mv ? " mv" : ""}"><path class="hr-walk" d="${walk}"/><circle class="hr-kd" cx="${f1(x)}" cy="${f1(y)}" r="5.5"><title>${k}: hash ${hex(now.hash[i])} → next point ${nodeName(s, prefix)}${vn} at ${hex(p.pos)}${mv ? ` (was ${nodeName(before.owner[i], prefix)})` : ""}</title></circle></g>`,
		);
	});
	// Server points, on the ring, labelled outside when there are few of them.
	const label = pts.length <= 16;
	for (const p of pts) {
		const a = angle(p.pos);
		const [x, y] = xy(cx, cy, R, a);
		const name = nodeName(p.node, prefix);
		const vn = o.vnodes > 1 ? `#${p.v}` : "";
		parts.push(
			`<g class="hr-node s${p.node}"><circle class="hr-nd" cx="${f1(x)}" cy="${f1(y)}" r="${label ? 7 : 4.5}"><title>${name}${vn} at ${hex(p.pos)}</title></circle>${
				label
					? `<text class="hr-nt" x="${f1(xy(cx, cy, R + 22, a)[0])}" y="${f1(xy(cx, cy, R + 22, a)[1] + 3.5)}">${o.vnodes > 1 ? `${name.slice(prefix.length)}${p.v}` : name}</text>`
					: ""
			}</g>`,
		);
	}
	return {
		svg: `<svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}" role="img">${parts.join("")}</svg>`,
		moved: moved.size,
		load: now.load,
		share: now.share,
	};
}
