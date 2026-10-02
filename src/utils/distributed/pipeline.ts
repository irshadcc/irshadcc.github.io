// Pipeline-parallel schedules and a small simulator that times them, used by
// PipelineSchedule.astro.
//
// A schedule is, for every device, the ordered list of operations it runs. An operation is the
// forward (F), the backward (B) or the weight-gradient half of the backward (W) of one
// microbatch through one stage. Schedules that do not split the backward use a "full" B that
// computes both gradients. DualPipe also runs an F of one microbatch and a full B of another as a
// single overlapped operation (FB).
//
// The simulator starts each operation as soon as its device is free and its inputs have
// arrived: F needs the previous stage's F of the same microbatch, B the next stage's B (or, on the
// last stage, its own F), W its own B. Communication is free, so every gap in the result is a
// pipeline bubble.
//
//   gpipe        all forwards, then all backwards
//   1f1b         p - d - 1 warm-up forwards on device d, then alternate F and B
//   interleaved  Megatron's interleaved 1F1B: v model chunks per device, m a multiple of p
//   zb-h1        1F1B with B and W split; W's fill the gaps, at most d of them deferred
//   dualpipe     DeepSeek's bidirectional schedule, following the eight phases of dualpipe.py

export type ScheduleKind = "gpipe" | "1f1b" | "interleaved" | "zb-h1" | "dualpipe";

export const SCHEDULE_NAMES: Record<ScheduleKind, string> = {
	gpipe: "GPipe",
	"1f1b": "1F1B",
	interleaved: "Interleaved 1F1B",
	"zb-h1": "ZB-H1",
	dualpipe: "DualPipe",
};

export type OpKind = "F" | "B" | "W";

/** One microbatch through one stage. */
export interface Part {
	kind: OpKind;
	/** DualPipe direction (0 enters at device 0, 1 at device p - 1); 0 elsewhere. */
	dir: number;
	mb: number;
	/** Stage along the microbatch's path: 0 .. p - 1, or 0 .. pv - 1 when interleaved. */
	stage: number;
	/** A B that also computes the weight gradient. */
	full?: boolean;
}

export interface Op {
	device: number;
	/** One part, or an overlapped F and B. */
	parts: Part[];
	start: number;
	end: number;
}

export interface Durations {
	/** Forward of one microbatch through one stage. */
	F: number;
	/** Input-gradient half of the backward. */
	B: number;
	/** Weight-gradient half of the backward. */
	W: number;
}

export interface Config {
	kind: ScheduleKind;
	/** Devices (pipeline ranks). */
	p: number;
	/** Microbatches per step. */
	m: number;
	/** Model chunks per device for interleaved 1F1B. */
	v?: number;
	t?: Durations;
}

export interface Result {
	config: Required<Config>;
	ops: Op[];
	makespan: number;
	/** Idle fraction of p × makespan. */
	bubble: number;
	/**
	 * Peak number of microbatches whose activations a device holds at once, in units of one
	 * microbatch through one full stage (an interleaved chunk counts 1/v).
	 */
	peakActivations: number;
	/** Copies of the parameters each device holds, relative to one stage. */
	params: number;
	/** Number of stages along a microbatch's path. */
	stages: number;
}

/** Display label for a microbatch: DualPipe numbers the second direction after the first. */
export const mbLabel = (part: Part, m: number) =>
	part.dir === 0 ? part.mb : m / 2 + part.mb;

/** Which variant of a colour a part gets: its model chunk, or its DualPipe direction. */
export const variantOf = (part: Part, p: number) =>
	part.dir || Math.floor(part.stage / p);

export function validate(c: Config): string | undefined {
	const { kind, p, m, v = 2 } = c;
	if (p < 2) return "A pipeline needs at least two devices.";
	if (m < 1) return "Use at least one microbatch.";
	if (kind === "interleaved" && m % p !== 0)
		return `Interleaved 1F1B needs the microbatches to be a multiple of p (m = ${m}, p = ${p}).`;
	if (kind === "interleaved" && v < 2) return "Interleaving needs v ≥ 2.";
	if (kind === "dualpipe" && p % 2 !== 0)
		return `DualPipe needs an even number of devices (p = ${p}).`;
	if (kind === "dualpipe" && (m % 2 !== 0 || m < 2 * p))
		return `DualPipe needs an even m of at least 2p, half for each direction (m = ${m}, p = ${p}).`;
	return undefined;
}

// ---- Per-device orders.

type Order = Part[][]; // one entry per device; a two-part entry is an overlapped F and B

const F = (mb: number, stage: number, dir = 0): Part => ({
	kind: "F",
	dir,
	mb,
	stage,
});
const fullB = (mb: number, stage: number, dir = 0): Part => ({
	kind: "B",
	dir,
	mb,
	stage,
	full: true,
});

function gpipe(p: number, m: number): Order[] {
	return Array.from({ length: p }, (_, d) => [
		...Array.from({ length: m }, (_, i) => [F(i, d)]),
		...Array.from({ length: m }, (_, i) => [fullB(i, d)]),
	]);
}

function oneFOneB(p: number, m: number): Order[] {
	return Array.from({ length: p }, (_, d) => {
		const warmup = Math.min(p - d - 1, m);
		const order: Order = [];
		for (let i = 0; i < warmup; i++) order.push([F(i, d)]);
		for (let i = 0; i < m - warmup; i++) {
			order.push([F(warmup + i, d)]);
			order.push([fullB(i, d)]);
		}
		for (let i = m - warmup; i < m; i++) order.push([fullB(i, d)]);
		return order;
	});
}

// Megatron-LM's forward_backward_pipelining_with_interleaving: microbatches go through the
// chunks in groups of p, and device d warms up with (p - d - 1) * 2 + (v - 1) * p forwards.
function interleaved(p: number, m: number, v: number): Order[] {
	const total = m * v;
	const chunk = (k: number, forward: boolean) => {
		const c = Math.floor((k % (p * v)) / p);
		return forward ? c : v - 1 - c;
	};
	const mbOf = (k: number) => Math.floor(k / (p * v)) * p + (k % p);
	return Array.from({ length: p }, (_, d) => {
		const fwd = (k: number) => [F(mbOf(k), chunk(k, true) * p + d)];
		const bwd = (k: number) => [fullB(mbOf(k), chunk(k, false) * p + d)];
		const warmup = Math.min((p - d - 1) * 2 + (v - 1) * p, total);
		const order: Order = [];
		for (let k = 0; k < warmup; k++) order.push(fwd(k));
		for (let k = 0; k < total - warmup; k++) {
			order.push(fwd(warmup + k));
			order.push(bwd(k));
		}
		for (let k = total - warmup; k < total; k++) order.push(bwd(k));
		return order;
	});
}

// Zero Bubble's H1 schedule: 1F1B's order of F's and B's, with each backward split into B and W.
// The W's are not placed in the order: the simulator runs a pending W whenever the device's next
// F or B is still waiting for its input, so they fill the gaps where 1F1B sits idle, and as soon
// as device d has more than d of them pending, which bounds their memory. Later stages defer
// more, so their B's, which the stages before them are waiting on, go out sooner.
function zbH1(p: number, m: number): Order[] {
	return oneFOneB(p, m).map((order) =>
		order.map((parts) => parts.map((q) => ({ ...q, full: false }))),
	);
}

// DeepSeek's DualPipe (deepseek-ai/DualPipe, dualpipe.py). Half the microbatches enter at
// device 0 and flow down, half enter at device p - 1 and flow up, so device d holds stage d of
// one direction and stage p - 1 - d of the other: two copies of the parameters. "Chunk 0" is the
// direction for which d is in the first half of the pipeline.
function dualpipe(p: number, m: number): Order[] {
	const half = m / 2;
	return Array.from({ length: p }, (_, d) => {
		const halfRank = Math.min(d, p - 1 - d);
		const isMiddle = halfRank === p / 2 - 1;
		const dirOf = (chunk: number) => (d < p / 2 ? chunk : 1 - chunk);
		const stageOf = (dir: number) => (dir === 0 ? d : p - 1 - d);
		const nextF = [0, 0];
		const nextB = [0, 0];
		const pendingW: Part[] = [];
		const order: Order = [];

		const fPart = (chunk: number): Part => {
			const dir = dirOf(chunk);
			return F(nextF[dir]++, stageOf(dir), dir);
		};
		const bPart = (chunk: number, zb = false): Part => {
			const dir = dirOf(chunk);
			const mb = nextB[dir]++;
			const stage = stageOf(dir);
			if (!zb) return fullB(mb, stage, dir);
			pendingW.push({ kind: "W", dir, mb, stage });
			return { kind: "B", dir, mb, stage };
		};
		const fwd = (chunk: number) => order.push([fPart(chunk)]);
		const bwd = (chunk: number, zb = false) => order.push([bPart(chunk, zb)]);
		const weight = () => {
			const w = pendingW.shift();
			if (w) order.push([w]);
		};
		const fwdBwd = (f: number, b: number) => order.push([fPart(f), bPart(b)]);

		const quarter = p / 2 - halfRank - 1;
		// 1: nF0
		for (let i = 0; i < quarter * 2; i++) fwd(0);
		// 2: nF0F1
		for (let i = 0; i < halfRank + 1; i++) {
			fwd(0);
			fwd(1);
		}
		// 3: nB1W1F1, zero bubble
		for (let i = 0; i < quarter; i++) {
			bwd(1, true);
			weight();
			fwd(1);
		}
		// 4: main step, nF0B1F1B0 with each F and B overlapped
		for (let i = 0; i < half - p + halfRank + 1; i++) {
			if (i === 0 && isMiddle) {
				// The middle ranks run the first pair back to back to shorten the bubble.
				fwd(0);
				bwd(1);
			} else fwdBwd(0, 1);
			fwdBwd(1, 0);
		}
		// 5: nB1F1B0
		for (let i = 0; i < quarter; i++) {
			bwd(1);
			fwdBwd(1, 0);
		}
		// 6: nB1B0, the second half with zero bubble
		let zb = false;
		const step6 = halfRank + 1;
		for (let i = 0; i < step6; i++) {
			if (i === Math.floor(step6 / 2) && halfRank % 2 === 1) zb = true;
			bwd(1, zb);
			if (i === Math.floor(step6 / 2) && halfRank % 2 === 0) zb = true;
			bwd(0, zb);
		}
		// 7: nWB0, zero bubble
		for (let i = 0; i < quarter; i++) {
			weight();
			bwd(0, true);
		}
		// 8: nW
		while (pendingW.length) weight();
		return order;
	});
}

// ---- Simulation.

export function simulate(config: Config): Result {
	const err = validate(config);
	if (err) throw new Error(err);
	const c: Required<Config> = {
		v: 2,
		t: { F: 1, B: 1, W: 1 },
		...config,
	};
	const { kind, p, m, t } = c;
	const v = kind === "interleaved" ? c.v : 1;
	const stages = p * v;
	const orders =
		kind === "gpipe"
			? gpipe(p, m)
			: kind === "1f1b"
				? oneFOneB(p, m)
				: kind === "interleaved"
					? interleaved(p, m, v)
					: kind === "zb-h1"
						? zbH1(p, m)
						: dualpipe(p, m);

	const key = (k: OpKind, part: Part, stage = part.stage) =>
		`${k}:${part.dir}:${part.mb}:${stage}`;
	const dur = (part: Part) =>
		(part.kind === "F"
			? t.F
			: part.kind === "W"
				? t.W
				: part.full
					? t.B + t.W
					: t.B) / v;
	const deps = (part: Part) =>
		part.kind === "F"
			? part.stage > 0
				? [key("F", part, part.stage - 1)]
				: []
			: part.kind === "B"
				? [key(part.stage < stages - 1 ? "B" : "F", part, part.stage < stages - 1 ? part.stage + 1 : part.stage)]
				: [key("B", part)];

	// Event-driven: at each moment some operation ends, every idle device starts its next
	// operation if its inputs are in, or else (ZB-H1 only) a pending W.
	const flexW = kind === "zb-h1";
	const done = new Map<string, number>();
	const free = new Array(p).fill(0);
	const cursor = new Array(p).fill(0);
	const pendingW: Part[][] = Array.from({ length: p }, () => []);
	const ops: Op[] = [];
	const run = (d: number, parts: Part[], start: number) => {
		const end = start + parts.reduce((s, q) => s + dur(q), 0);
		for (const q of parts) done.set(key(q.kind, q), end);
		ops.push({ device: d, parts, start, end });
		free[d] = end;
	};
	const left = (d: number) => cursor[d] < orders[d].length || pendingW[d].length > 0;
	let now = 0;
	while ([...Array(p).keys()].some(left)) {
		for (let d = 0; d < p; d++) {
			if (free[d] > now || !left(d)) continue;
			const parts = orders[d][cursor[d]];
			const ready =
				parts !== undefined &&
				parts.flatMap(deps).every((k) => (done.get(k) ?? Number.POSITIVE_INFINITY) <= now);
			// ZB-H1 lets device d defer at most d W's. With the p - d microbatches 1F1B keeps in
			// flight there, no device holds more than p microbatches' activations.
			if (flexW && pendingW[d].length > d) {
				run(d, [pendingW[d].shift() as Part], now);
			} else if (ready) {
				run(d, parts, now);
				cursor[d]++;
				if (flexW)
					for (const q of parts)
						if (q.kind === "B") pendingW[d].push({ ...q, kind: "W" });
			} else if (pendingW[d].length) {
				run(d, [pendingW[d].shift() as Part], now);
			}
		}
		const next = Math.min(
			...[...done.values(), ...free].filter((x) => x > now),
		);
		if (!Number.isFinite(next))
			throw new Error(`Schedule ${kind} deadlocks for p = ${p}, m = ${m}.`);
		now = next;
	}

	const makespan = Math.max(...free);
	const busy = ops.reduce((s, o) => s + o.end - o.start, 0);

	// Activations live from the start of a microbatch's F on a stage to the end of its last
	// backward operation there: the full B, or the W.
	let peak = 0;
	for (let d = 0; d < p; d++) {
		const events: [number, number][] = [];
		for (const o of ops.filter((o) => o.device === d))
			for (const q of o.parts) {
				if (q.kind === "F") events.push([o.start, 1 / v]);
				if (q.kind === "W" || q.full) events.push([o.end, -1 / v]);
			}
		events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
		let live = 0;
		for (const [, delta] of events) {
			live += delta;
			peak = Math.max(peak, live);
		}
	}

	return {
		config: { ...c, v },
		ops,
		makespan,
		bubble: 1 - busy / (p * makespan),
		peakActivations: Math.round(peak * 100) / 100,
		params: kind === "dualpipe" ? 2 : 1,
		stages,
	};
}

// ---- Drawing.

export interface DrawOptions {
	/** Pixels per unit of time. */
	unit?: number;
	/** Prefix for SVG ids, unique per figure. */
	id?: string;
}

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

/** The schedule as an SVG timeline: one row per device, one block per operation. */
export function drawSchedule(r: Result, opts: DrawOptions = {}): string {
	const unit = opts.unit ?? 18;
	const id = opts.id ?? "pp";
	const { p, m } = r.config;
	const ROW = 24;
	const GAP = 4;
	const LEFT = 58;
	const TOP = 4;
	const AXIS = 22;
	const W = LEFT + r.makespan * unit + 8;
	const H = TOP + p * (ROW + GAP) - GAP + AXIS;
	const rowY = (d: number) => TOP + d * (ROW + GAP);
	const out: string[] = [];

	out.push(
		`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" style="min-width:${Math.round(Math.min(W, 640) * 0.75)}px" role="img" aria-label="${esc(SCHEDULE_NAMES[r.config.kind])} on ${p} devices with ${m} microbatches">`,
		`<defs><pattern id="${id}-idle" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" class="idle-bg"/><line x1="0" y1="0" x2="0" y2="6" class="idle-line"/></pattern></defs>`,
	);
	for (let d = 0; d < p; d++) {
		out.push(
			`<text class="dev" x="${LEFT - 8}" y="${rowY(d) + ROW / 2 + 4}">GPU ${d}</text>`,
			`<rect class="idle" x="${LEFT}" y="${rowY(d)}" width="${r.makespan * unit}" height="${ROW}" fill="url(#${id}-idle)"/>`,
		);
	}

	const describe = (q: Part) => {
		const what =
			q.kind === "F"
				? "forward"
				: q.kind === "W"
					? "weight gradient"
					: q.full
						? "backward"
						: "input gradient";
		const where =
			r.config.kind === "interleaved"
				? `, chunk ${Math.floor(q.stage / p)} (stage ${q.stage})`
				: r.config.kind === "dualpipe"
					? `, direction ${q.dir} (stage ${q.stage})`
					: "";
		return `${q.kind} · microbatch ${mbLabel(q, m)}: ${what}${where}`;
	};

	for (const o of r.ops) {
		const x = LEFT + o.start * unit;
		const w = Math.max(1, (o.end - o.start) * unit - 1);
		const y = rowY(o.device);
		const h = ROW / o.parts.length;
		const keys = o.parts.map((q) => `${q.dir}:${q.mb}`).join(" ");
		const tip = `${o.parts.map(describe).join("\n")}${o.parts.length > 1 ? "\n(overlapped)" : ""}\nt = ${+o.start.toFixed(2)} – ${+o.end.toFixed(2)}`;
		out.push(`<g class="op${o.parts.length > 1 ? " pair" : ""}" data-mb="${keys}"><title>${esc(tip)}</title>`);
		o.parts.forEach((q, i) => {
			const cls = `${q.kind}${q.full ? " full" : ""} v${variantOf(q, p)}`;
			out.push(`<rect class="${cls}" x="${x}" y="${y + i * h}" width="${w}" height="${h}"/>`);
			if (w >= 9)
				out.push(
					`<text x="${x + w / 2}" y="${y + i * h + h / 2 + (o.parts.length > 1 ? 3 : 3.5)}"${o.parts.length > 1 ? ' class="small"' : ""}>${mbLabel(q, m)}</text>`,
				);
		});
		if (o.parts.length > 1)
			out.push(`<rect class="pair-outline" x="${x}" y="${y}" width="${w}" height="${ROW}"/>`);
		out.push("</g>");
	}

	// Time axis: a tick every F's worth of time.
	const axisY = rowY(p - 1) + ROW + 4;
	const step = r.makespan > 60 ? 10 : 5;
	out.push(`<line class="axis" x1="${LEFT}" y1="${axisY}" x2="${LEFT + r.makespan * unit}" y2="${axisY}"/>`);
	for (let s = 0; s <= r.makespan + 1e-9; s += step)
		out.push(
			`<line class="axis" x1="${LEFT + s * unit}" y1="${axisY}" x2="${LEFT + s * unit}" y2="${axisY + 4}"/>`,
			`<text class="tick" x="${LEFT + s * unit}" y="${axisY + 14}">${s}</text>`,
		);
	out.push(`<text class="tick end" x="${LEFT - 8}" y="${axisY + 14}">time</text>`);
	out.push("</svg>");
	return out.join("");
}

/** One-line summary of a simulated schedule. */
export function summarize(r: Result): { label: string; value: string }[] {
	const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
	return [
		{ label: "Step time", value: `${+r.makespan.toFixed(2)} F` },
		{ label: "Bubble", value: pct(r.bubble) },
		{ label: "Peak activations", value: `${r.peakActivations} µb` },
		{ label: "Parameters", value: `${r.params}×` },
	];
}
