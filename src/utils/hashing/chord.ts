// Chord (Stoica et al., SIGCOMM 2001) on an m-bit identifier circle, used by ChordSteps.astro
// and the consistent-hashing post. The functions follow the paper's pseudocode: Figure 4
// (find_successor, find_predecessor, closest_preceding_finger) and Figure 7 (join, stabilize,
// notify, fix_fingers). Remote calls are plain function calls on an in-memory network.
//
// Fingers are 1-indexed as in the paper: finger[i].start = (n + 2^(i-1)) mod 2^m, 1 <= i <= m,
// and finger[i].node = successor(finger[i].start). finger[1] is the successor.

export interface ChordNode {
	id: number;
	successor: number;
	/** -1 = nil. */
	predecessor: number;
	/** finger[i] for i = 1..m, stored at index i - 1. */
	finger: number[];
}

export type Net = Map<number, ChordNode>;

/** x in the open circular interval (a, b). With a == b, the whole circle except a. */
export function inOpen(x: number, a: number, b: number): boolean {
	if (a < b) return a < x && x < b;
	return x !== a && (x > a || x < b);
}

/** x in the half-open circular interval (a, b]. */
export function inHalf(x: number, a: number, b: number): boolean {
	return x === b || inOpen(x, a, b);
}

export const fingerStart = (n: number, i: number, m: number) =>
	(n + 2 ** (i - 1)) % 2 ** m;

/** Owner of id among sorted node ids: the first node >= id, wrapping. */
export function successorOf(ids: readonly number[], id: number): number {
	return ids.find((n) => n >= id) ?? ids[0];
}

/** A stable network: every successor, predecessor and finger correct. */
export function stableNet(m: number, ids: readonly number[]): Net {
	const s = [...ids].sort((a, b) => a - b);
	const net: Net = new Map();
	s.forEach((id, k) => {
		net.set(id, {
			id,
			successor: s[(k + 1) % s.length],
			predecessor: s[(k - 1 + s.length) % s.length],
			finger: Array.from({ length: m }, (_, i) =>
				successorOf(s, fingerStart(id, i + 1, m)),
			),
		});
	});
	return net;
}

export function closestPrecedingFinger(
	net: Net,
	n: number,
	id: number,
	m: number,
): { node: number; i: number } {
	const node = net.get(n) as ChordNode;
	for (let i = m; i >= 1; i--)
		if (inOpen(node.finger[i - 1], n, id))
			return { node: node.finger[i - 1], i };
	return { node: n, i: 0 };
}

export interface Hop {
	/** Node asked at this step. */
	at: number;
	/** Finger index used to leave this node (0 = none: id is in (at, at.successor]). */
	finger: number;
	/** Node forwarded to (or at.successor, the answer, on the last hop). */
	to: number;
}

/** find_successor(id) started at node `from`: the hops of find_predecessor, then the answer. */
export function lookup(
	net: Net,
	from: number,
	id: number,
	m: number,
): { owner: number; hops: Hop[] } {
	const hops: Hop[] = [];
	let n = from;
	for (let guard = 0; guard < 4 * m + net.size; guard++) {
		const node = net.get(n) as ChordNode;
		if (inHalf(id, n, node.successor)) {
			hops.push({ at: n, finger: 0, to: node.successor });
			return { owner: node.successor, hops };
		}
		const f = closestPrecedingFinger(net, n, id, m);
		hops.push({ at: n, finger: f.i, to: f.node });
		n = f.node;
	}
	throw new Error("lookup did not terminate");
}

// --- Join and stabilization (Figure 7) -----------------------------------------------------------

export function join(net: Net, n: number, via: number, m: number): void {
	const succ = via === n ? n : lookup(net, via, n, m).owner;
	net.set(n, {
		id: n,
		successor: succ,
		predecessor: -1,
		finger: Array.from({ length: m }, (_, i) => (i === 0 ? succ : n)),
	});
}

export function notify(net: Net, n: number, candidate: number): boolean {
	const node = net.get(n) as ChordNode;
	if (node.predecessor === -1 || inOpen(candidate, node.predecessor, n)) {
		const changed = node.predecessor !== candidate;
		node.predecessor = candidate;
		return changed;
	}
	return false;
}

/** n.stabilize(); returns what changed, for the step text. */
export function stabilize(
	net: Net,
	n: number,
): { x: number; newSucc: boolean; predOf: number; newPred: boolean } {
	const node = net.get(n) as ChordNode;
	const x = (net.get(node.successor) as ChordNode).predecessor;
	let newSucc = false;
	if (x !== -1 && inOpen(x, n, node.successor)) {
		node.successor = x;
		node.finger[0] = x;
		newSucc = true;
	}
	const newPred = notify(net, node.successor, n);
	return { x, newSucc, predOf: node.successor, newPred };
}

/** Refresh every finger of n (the paper's simulator does all entries per call). */
export function fixFingers(net: Net, n: number, m: number): number {
	const node = net.get(n) as ChordNode;
	let changed = 0;
	for (let i = 2; i <= m; i++) {
		const f = lookup(net, n, fingerStart(n, i, m), m).owner;
		if (f !== node.finger[i - 1]) changed++;
		node.finger[i - 1] = f;
	}
	return changed;
}

export function cloneNet(net: Net): Net {
	return new Map(
		[...net].map(([k, v]) => [k, { ...v, finger: [...v.finger] }]),
	);
}

// --- Steps for ChordSteps.astro ------------------------------------------------------------------

export interface ChordStep {
	label: string;
	/** Plain text; `N8`-style names. */
	text: string;
	net: Net;
	/** Node whose finger table is shown. */
	focus: number;
	/** Finger row highlighted in the table (1-based, 0 = none). */
	row: number;
	/** Hops drawn so far, as [from, to] pairs; the last one is highlighted. */
	path: [number, number][];
	/** Key marker. */
	key?: number;
	/** Nodes drawn with a dashed outline (joining, not yet linked). */
	pending?: number[];
	/** Nodes whose predecessor and successor are listed in the table (instead of a finger table). */
	pointers?: number[];
	/** Pointer arrows to draw instead of fingers: successor outside, predecessor inside (dashed). */
	arrows?: [number, "succ" | "pred"][];
	/** Arrows that changed in this step, as "26.succ"; drawn highlighted. */
	changed?: string[];
}

const N = (n: number) => (n === -1 ? "nil" : `N${n}`);

export function lookupSteps(
	m: number,
	ids: readonly number[],
	from: number,
	key: number,
): ChordStep[] {
	const net = stableNet(m, ids);
	const { owner, hops } = lookup(net, from, key, m);
	const steps: ChordStep[] = [
		{
			label: "start",
			text: `${N(from)} wants successor(${key}), the node that stores key K${key}. It only knows its ${m} fingers, shown in its finger table: finger[i] is the first node at or after ${from} + 2^(i-1).`,
			net,
			focus: from,
			row: 0,
			path: [],
			key,
		},
	];
	const path: [number, number][] = [];
	hops.forEach((h, k) => {
		const node = net.get(h.at) as ChordNode;
		path.push([h.at, h.to]);
		if (h.finger > 0) {
			steps.push({
				label: `${N(h.at)}`,
				text: `${key} is not in (${h.at}, ${node.successor}], so ${N(h.at)} is not the predecessor. closest_preceding_finger scans from finger[${m}] down; the first finger in the open interval (${h.at}, ${key}) is finger[${h.finger}] = ${N(h.to)}, so the query moves to ${N(h.to)}.`,
				net,
				focus: h.at,
				row: h.finger,
				path: [...path],
				key,
			});
		} else {
			steps.push({
				label: `${N(h.at)} → ${N(owner)}`,
				text: `${key} is in (${h.at}, ${node.successor}]: ${N(h.at)} is the predecessor of ${key}, and its successor ${N(owner)} stores K${key}. The query took ${k} hop${k === 1 ? "" : "s"} (${hops.map((x) => N(x.at)).join(" → ")}), then ${N(h.at)} answered with its successor.`,
				net,
				focus: h.at,
				row: 1,
				path: [...path],
				key,
			});
		}
	});
	return steps;
}

/**
 * Node `n` joins via `via`, then the stabilization calls that link it in, in the order the
 * paper's example uses: n.stabilize, then its predecessor's stabilize, then fix_fingers everywhere.
 */
export function joinSteps(
	m: number,
	ids: readonly number[],
	n: number,
	via: number,
): ChordStep[] {
	const net = stableNet(m, ids);
	const steps: ChordStep[] = [];
	const s0 = successorOf(
		[...ids].sort((a, b) => a - b),
		n,
	);
	const p0 = (net.get(s0) as ChordNode).predecessor;
	const show = [p0, n, s0];
	const arrows: [number, "succ" | "pred"][] = [
		[p0, "succ"],
		[n, "succ"],
		[n, "pred"],
		[s0, "pred"],
	];
	steps.push({
		label: "before",
		text: `A stable ring. ${N(n)} wants to join between ${N(p0)} and ${N(s0)}: ${N(p0)}.successor = ${N(s0)} and ${N(s0)}.predecessor = ${N(p0)}.`,
		net: cloneNet(net),
		focus: p0,
		row: 0,
		path: [],
		pending: [n],
		pointers: show,
		arrows,
		changed: [],
	});
	join(net, n, via, m);
	steps.push({
		label: "join",
		text: `${N(n)}.join(${N(via)}): predecessor = nil, successor = ${N(via)}.find_successor(${n}) = ${N(s0)}. No other node knows about ${N(n)} yet, so lookups still go to ${N(s0)}.`,
		net: cloneNet(net),
		focus: n,
		row: 1,
		path: [],
		pointers: show,
		arrows,
		changed: [`${n}.succ`],
	});
	let r = stabilize(net, n);
	steps.push({
		label: `${N(n)}.stabilize`,
		text: `${N(n)}.stabilize(): x = ${N(s0)}.predecessor = ${N(r.x)}, which is not in (${n}, ${s0}), so ${N(n)} keeps its successor. Then ${N(s0)}.notify(${N(n)}): ${n} is in (${p0}, ${s0}), so ${N(s0)}.predecessor becomes ${N(n)}.`,
		net: cloneNet(net),
		focus: n,
		row: 1,
		path: [],
		pointers: show,
		arrows,
		changed: [`${s0}.pred`],
	});
	r = stabilize(net, p0);
	steps.push({
		label: `${N(p0)}.stabilize`,
		text: `${N(p0)}.stabilize(): x = ${N(s0)}.predecessor = ${N(r.x)}, which is in (${p0}, ${s0}), so ${N(p0)}.successor becomes ${N(n)}. Then ${N(n)}.notify(${N(p0)}): its predecessor was nil, so it becomes ${N(p0)}. All successor and predecessor pointers are now correct, and the keys in (${p0}, ${n}] move from ${N(s0)} to ${N(n)}.`,
		net: cloneNet(net),
		focus: p0,
		row: 1,
		path: [],
		pointers: show,
		arrows,
		changed: [`${p0}.succ`, `${n}.pred`],
	});
	let changed = 0;
	const touched: number[] = [];
	for (const id of [...net.keys()].sort((a, b) => a - b)) {
		const c = fixFingers(net, id, m);
		if (c > 0 && id !== n) touched.push(id);
		changed += c;
	}
	steps.push({
		label: "fix_fingers",
		text: `fix_fingers() on every node: ${N(n)} fills its own fingers, and ${touched.length} other node${touched.length === 1 ? "" : "s"} (${touched.map(N).join(", ")}) now point a finger at ${N(n)}; ${changed} finger entries changed in all. Lookups were already correct before this step, just possibly slower.`,
		net: cloneNet(net),
		focus: n,
		row: 0,
		path: [],
	});
	return steps;
}

// --- Drawing ---------------------------------------------------------------------------------------

const r1 = (x: number) => x.toFixed(1);

/** SVG of one step: the identifier circle, nodes, key, finger chords or pointers, and the path. */
export function drawChordStep(
	step: ChordStep,
	m: number,
	uid: string,
	size = 340,
): string {
	const c = size / 2;
	const R = size / 2 - 40;
	const M = 2 ** m;
	const ang = (id: number) => (id / M) * 2 * Math.PI;
	const at = (id: number, r: number): [number, number] => [
		c + r * Math.sin(ang(id)),
		c - r * Math.cos(ang(id)),
	];
	/** Curve from node a to node b, bowed toward the centre (bow < 1) or outward (bow > 1). */
	const curve = (a: number, b: number, bow: number, ra = R, rb = R) => {
		const [x0, y0] = at(a, ra);
		const [x1, y1] = at(b, rb);
		const mx = (x0 + x1) / 2;
		const my = (y0 + y1) / 2;
		const qx = c + (mx - c) * bow;
		const qy = c + (my - c) * bow;
		return `M${r1(x0)},${r1(y0)}Q${r1(qx)},${r1(qy)} ${r1(x1)},${r1(y1)}`;
	};
	const parts: string[] = [
		`<defs><marker id="ca-${uid}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="cs-head-on" d="M0,0 L8,4 L0,8 z"/></marker><marker id="cm-${uid}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="cs-head" d="M0,0 L8,4 L0,8 z"/></marker></defs>`,
		`<circle class="cs-ring" cx="${c}" cy="${c}" r="${R}"/>`,
	];
	for (let id = 0; id < M; id++) {
		const [x0, y0] = at(id, R - 3);
		const [x1, y1] = at(id, R + 3);
		parts.push(
			`<line class="cs-tick${id % 8 === 0 ? " major" : ""}" x1="${r1(x0)}" y1="${r1(y0)}" x2="${r1(x1)}" y2="${r1(y1)}"/>`,
		);
	}
	const [zx, zy] = at(0, R - 13);
	parts.push(`<text class="cs-zero" x="${r1(zx)}" y="${r1(zy + 3)}">0</text>`);
	const focus = step.net.get(step.focus);
	const arrow = (d: string, cls: string, on: boolean) =>
		`<path class="${cls}" d="${d}" marker-end="url(#${on ? "ca" : "cm"}-${uid})"/>`;
	if (step.arrows) {
		for (const [id, kind] of step.arrows) {
			const n = step.net.get(id);
			const to = n ? (kind === "succ" ? n.successor : n.predecessor) : -1;
			if (to === -1 || to === id) continue;
			const on = step.changed?.includes(`${id}.${kind}`) ?? false;
			const d =
				kind === "succ"
					? curve(id, to, 1.3, R + 9, R + 9)
					: curve(id, to, 0.6, R - 9, R - 9);
			parts.push(arrow(d, `cs-${kind}${on ? " on" : ""}`, on));
		}
	} else if (focus) {
		const seen = new Set<number>();
		focus.finger.forEach((f, i) => {
			if (f === step.focus || seen.has(f)) return;
			seen.add(f);
			const on =
				step.row > 0 &&
				focus.finger[step.row - 1] === f &&
				step.path.length > 0;
			if (!on)
				parts.push(
					`<path class="cs-finger" d="${curve(step.focus, f, 0.55)}"><title>finger[${i + 1}] of N${step.focus}</title></path>`,
				);
		});
	}
	step.path.forEach(([a, b], k) => {
		const last = k === step.path.length - 1;
		parts.push(arrow(curve(a, b, 0.3), last ? "cs-hop on" : "cs-hop", last));
	});
	if (step.key !== undefined) {
		const [kx, ky] = at(step.key, R - 22);
		parts.push(
			`<g class="cs-key"><rect x="${r1(kx - 4.5)}" y="${r1(ky - 4.5)}" width="9" height="9" transform="rotate(45 ${r1(kx)} ${r1(ky)})"/><text x="${r1(kx)}" y="${r1(ky - 9)}">K${step.key}</text></g>`,
		);
	}
	const ids = new Set([...step.net.keys(), ...(step.pending ?? [])]);
	for (const id of ids) {
		const [x, y] = at(id, R);
		const [lx, ly] = at(id, R + 20);
		const pending = step.pending?.includes(id);
		const cls =
			id === step.focus
				? "cs-node focus"
				: pending
					? "cs-node pending"
					: "cs-node";
		parts.push(
			`<g class="${cls}"><circle cx="${r1(x)}" cy="${r1(y)}" r="6"><title>N${id}</title></circle><text x="${r1(lx)}" y="${r1(ly + 3.5)}">N${id}</text></g>`,
		);
	}
	return `<svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}" role="img">${parts.join("")}</svg>`;
}

/** Table beside the ring: the focus node's finger table, or the pointers of the shown nodes. */
export function chordTable(step: ChordStep, m: number): string {
	const N = (n: number) => (n === -1 ? "nil" : `N${n}`);
	if (step.pointers) {
		const rows = step.pointers
			.filter((id) => step.net.has(id))
			.map((id) => {
				const n = step.net.get(id) as ChordNode;
				return `<tr${id === step.focus ? ' class="on"' : ""}><td>${N(id)}</td><td>${N(n.predecessor)}</td><td>${N(n.successor)}</td></tr>`;
			});
		return `<table><caption>pointers</caption><thead><tr><th>node</th><th>predecessor</th><th>successor</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
	}
	const n = step.net.get(step.focus) as ChordNode;
	const rows = n.finger.map((f, k) => {
		const i = k + 1;
		const s = fingerStart(step.focus, i, m);
		const e = i < m ? fingerStart(step.focus, i + 1, m) : step.focus;
		return `<tr${i === step.row ? ' class="on"' : ""}><td>${i}</td><td>${s}</td><td>[${s}, ${e})</td><td>${N(f)}</td></tr>`;
	});
	return `<table><caption>finger table of ${N(step.focus)}</caption><thead><tr><th>i</th><th>start</th><th>interval</th><th>node</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
}
