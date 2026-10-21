// A tiny SIMT interpreter for a subset of PTX, used by PtxTrace.astro to step through a kernel
// lane by lane. One warp of `lanes` threads (8 by default, a scaled-down 32) runs the program.
// Every lane has its own program counter; at each step the warp issues the instruction at the
// smallest PC among the lanes that have not exited, for exactly the lanes waiting there. That is
// enough to show divergence (the two sides of a branch issue one after the other) and
// reconvergence (the lanes meet again at the join label) for the structured code compilers emit.
//
// Supported: mov and cvt (registers, immediates, %tid.x, %ntid.x, %ctaid.x, %laneid), ld.param,
// ld.global, st.global, cvta.to.global, add/sub/mul(.lo/.wide)/mad.lo/shl/shr/and/or/xor on
// integers, add/sub/mul/fma/neg/sqrt and sin/cos/ex2 .approx on f32, setp, selp, bra, ret,
// exit, shfl.sync (up, down, bfly, idx with the c-operand clamp and segment mask, as in the PTX
// ISA's pseudocode) and vote.sync.ballot. Registers hold plain numbers; the instruction's type
// decides how they are read, as in PTX. Anything else throws, so a figure cannot silently show a
// wrong value.

export interface PtxBuffer {
	/** Byte address of element 0. */
	base: number;
	values: number[];
	/** Bytes per element (default 4). */
	elem?: number;
}

export interface PtxSimConfig {
	/** PTX instructions, labels and comments; `.reg` lines and braces are ignored. */
	code: string;
	lanes?: number;
	ntid?: number;
	ctaid?: number;
	/** Values returned by `ld.param` for each parameter name. */
	params?: Record<string, number>;
	buffers?: Record<string, PtxBuffer>;
	/** Registers shown as rows of the figure's table, in order. */
	show: string[];
	/** Registers among `show` that hold f32 values. */
	floats?: string[];
	/** Notes keyed by 1-based source line, shown when that line issues. */
	notes?: Record<number, string>;
}

export interface PtxStep {
	/** 0-based source line that issued. */
	line: number;
	text: string;
	/** Lanes the instruction issued for (waiting at this PC). */
	active: boolean[];
	/** Active lanes whose guard predicate was true. */
	executed: boolean[];
	/** Register values after the step, `values[reg][lane]`, formatted. */
	values: string[][];
	/** Registers changed in this step, per lane. */
	changed: boolean[][];
	/** Where each lane waits after the step: a 1-based source line, or "exit". */
	at: string[];
	/** Issue slots so far, and lane-instructions that did work so far. */
	issued: number;
	useful: number;
	note: string;
	/** Memory written in this step, as "y[3] = 4.5" strings. */
	stores: string[];
}

export interface PtxTraceResult {
	lines: string[];
	steps: PtxStep[];
	lanes: number;
	show: string[];
}

interface Instr {
	line: number;
	text: string;
	guard?: { reg: string; neg: boolean };
	op: string[];
	args: string[];
}

const SPECIAL = new Set(["%tid.x", "%ntid.x", "%ctaid.x", "%laneid"]);

function parse(code: string): {
	lines: string[];
	instrs: Instr[];
	labels: Map<string, number>;
} {
	const lines = code.replace(/^\n+|\s+$/g, "").split("\n");
	const instrs: Instr[] = [];
	const labels = new Map<string, number>();
	lines.forEach((raw, line) => {
		let s = raw.replace(/\/\/.*$/, "").trim();
		const lab = s.match(/^([$\w]+):\s*(.*)$/);
		if (lab) {
			labels.set(lab[1], instrs.length);
			s = lab[2].trim();
		}
		if (!s || s.startsWith(".") || s === "{" || s === "}") return;
		s = s.replace(/;$/, "").trim();
		let guard: Instr["guard"];
		const g = s.match(/^@(!?)(%\w+)\s+(.*)$/);
		if (g) {
			guard = { reg: g[2], neg: g[1] === "!" };
			s = g[3];
		}
		const sp = s.search(/\s/);
		const opcode = sp < 0 ? s : s.slice(0, sp);
		const rest = sp < 0 ? "" : s.slice(sp).trim();
		const args = rest ? splitArgs(rest) : [];
		instrs.push({ line, text: raw.trim(), guard, op: opcode.split("."), args });
	});
	return { lines, instrs, labels };
}

function splitArgs(s: string): string[] {
	const out: string[] = [];
	let depth = 0;
	let cur = "";
	for (const ch of s) {
		if (ch === "[" || ch === "{") depth++;
		if (ch === "]" || ch === "}") depth--;
		if (ch === "," && depth === 0) {
			out.push(cur.trim());
			cur = "";
		} else cur += ch;
	}
	if (cur.trim()) out.push(cur.trim());
	return out;
}

function floatFromHex(h: string): number {
	const b = new DataView(new ArrayBuffer(4));
	b.setUint32(0, Number.parseInt(h, 16));
	return b.getFloat32(0);
}

export function formatValue(v: number | undefined, float: boolean): string {
	if (v === undefined) return "";
	if (float) {
		if (Number.isNaN(v)) return "NaN";
		if (Number.isInteger(v) && Math.abs(v) < 1e6) return String(v);
		return Number(v.toPrecision(4)).toString();
	}
	if (v >= 0x10000 && Number.isInteger(v)) return `0x${v.toString(16)}`;
	return String(v);
}

export function simulatePtx(cfg: PtxSimConfig): PtxTraceResult {
	const lanes = cfg.lanes ?? 8;
	const { lines, instrs, labels } = parse(cfg.code);
	const floats = new Set(cfg.floats ?? []);
	const regs: Map<string, number | boolean>[] = Array.from(
		{ length: lanes },
		() => new Map(),
	);
	const pc = Array.from({ length: lanes }, () => 0);
	const done = Array.from({ length: lanes }, () => false);
	const mem = new Map<number, number>();
	for (const b of Object.values(cfg.buffers ?? {}))
		b.values.forEach((v, i) => mem.set(b.base + i * (b.elem ?? 4), v));
	const bufferAt = (addr: number) => {
		for (const [name, b] of Object.entries(cfg.buffers ?? {})) {
			const i = (addr - b.base) / (b.elem ?? 4);
			if (Number.isInteger(i) && i >= 0 && i < b.values.length)
				return `${name}[${i}]`;
		}
		return `[0x${addr.toString(16)}]`;
	};

	const steps: PtxStep[] = [];
	let issued = 0;
	let useful = 0;
	const at = () =>
		pc.map((p, l) =>
			done[l] ? "exit" : p < instrs.length ? String(instrs[p].line + 1) : "end",
		);

	for (let guardSteps = 0; guardSteps < 10000; guardSteps++) {
		const live = pc
			.map((_, l) => l)
			.filter((l) => !done[l] && pc[l] < instrs.length);
		if (live.length === 0) break;
		const p = Math.min(...live.map((l) => pc[l]));
		const ins = instrs[p];
		const active = pc.map((q, l) => !done[l] && q === p);
		const read = (lane: number, a: string): number => {
			const r = regs[lane];
			if (SPECIAL.has(a)) {
				if (a === "%tid.x" || a === "%laneid") return lane;
				if (a === "%ntid.x") return cfg.ntid ?? lanes;
				return cfg.ctaid ?? 0;
			}
			if (a.startsWith("%")) {
				const v = r.get(a);
				if (v === undefined)
					throw new Error(`line ${ins.line + 1}: ${a} read before written`);
				return Number(v);
			}
			if (/^0[fF][0-9a-fA-F]{8}$/.test(a)) return floatFromHex(a.slice(2));
			const n = Number(a);
			if (Number.isNaN(n))
				throw new Error(`line ${ins.line + 1}: bad operand ${a}`);
			return n;
		};
		const pred = (lane: number, a: string): boolean => {
			const v = regs[lane].get(a);
			if (typeof v !== "boolean")
				throw new Error(`line ${ins.line + 1}: ${a} is not a predicate`);
			return v;
		};
		const executed = active.map(
			(on, l) => on && (!ins.guard || pred(l, ins.guard.reg) !== ins.guard.neg),
		);
		const changed: Set<string>[] = Array.from(
			{ length: lanes },
			() => new Set(),
		);
		const write = (lane: number, reg: string, v: number | boolean) => {
			regs[lane].set(reg, v);
			changed[lane].add(reg);
		};
		const stores: string[] = [];
		const [base, ...mods] = ins.op;
		const type = mods[mods.length - 1];
		const isF = type === "f32";
		const signed = type?.startsWith("s");
		const int = (x: number) => {
			if (type === "u64" || type === "s64" || type === "b64") return x;
			return signed ? x | 0 : x >>> 0;
		};
		const next = pc.slice();

		// Instructions that need every executing lane at once.
		if (base === "shfl") {
			const mode = mods[1];
			const [dst, a, b, c, mask] = ins.args;
			const [d, pd] = dst.split("|");
			const src = executed.map((on, l) => (on ? read(l, a) : 0));
			const m = read(0, mask) >>> 0;
			for (let l = 0; l < lanes; l++) {
				if (!executed[l]) continue;
				for (let k = 0; k < lanes; k++)
					if ((m >>> k) & 1 && !executed[k])
						throw new Error(
							`line ${ins.line + 1}: lane ${k} is in the mask but not here`,
						);
				const bval = read(l, b) & 31;
				const cv = read(l, c);
				const cval = cv & 31;
				const seg = (cv >>> 8) & 31;
				const maxLane = (l & seg) | (cval & ~seg);
				const minLane = l & seg;
				let j: number;
				let ok: boolean;
				if (mode === "up") {
					j = l - bval;
					ok = j >= maxLane;
				} else if (mode === "down") {
					j = l + bval;
					ok = j <= maxLane;
				} else if (mode === "bfly") {
					j = l ^ bval;
					ok = j <= maxLane;
				} else {
					j = minLane | (bval & ~seg);
					ok = j <= maxLane;
				}
				if (!ok) j = l;
				if (j >= lanes)
					throw new Error(`line ${ins.line + 1}: lane ${l} reads lane ${j}`);
				write(l, d, src[j]);
				if (pd) write(l, pd, ok);
			}
		} else if (base === "vote") {
			const [d, a, mask] = ins.args;
			const m = read(0, mask) >>> 0;
			let ballot = 0;
			for (let l = 0; l < lanes; l++)
				if (
					executed[l] &&
					(m >>> l) & 1 &&
					pred(l, a.replace(/^!/, "")) !== a.startsWith("!")
				)
					ballot |= 1 << l;
			for (let l = 0; l < lanes; l++)
				if (executed[l]) write(l, d, ballot >>> 0);
		} else {
			for (let l = 0; l < lanes; l++) {
				if (!active[l]) continue;
				next[l] = pc[l] + 1;
				if (!executed[l]) continue;
				const A = ins.args;
				const r = (i: number) => read(l, A[i]);
				switch (base) {
					case "mov":
					case "cvt":
					case "cvta":
						write(l, A[0], r(A.length - 1));
						break;
					case "ld": {
						const addr = A[1].slice(1, -1);
						if (mods[0] === "param") {
							const v = cfg.params?.[addr];
							if (v === undefined) throw new Error(`no param ${addr}`);
							write(l, A[0], v);
						} else {
							const [reg, off] = addr.split("+");
							const ea = read(l, reg) + Number(off ?? 0);
							const v = mem.get(ea);
							if (v === undefined)
								throw new Error(
									`line ${ins.line + 1}: load from 0x${ea.toString(16)}`,
								);
							write(l, A[0], v);
						}
						break;
					}
					case "st": {
						const [reg, off] = A[0].slice(1, -1).split("+");
						const ea = read(l, reg) + Number(off ?? 0);
						const v = r(1);
						mem.set(ea, v);
						stores.push(
							`${bufferAt(ea)} = ${formatValue(v, isF || floats.has(A[1]))}`,
						);
						break;
					}
					case "add":
						write(l, A[0], isF ? Math.fround(r(1) + r(2)) : int(r(1) + r(2)));
						break;
					case "sub":
						write(l, A[0], isF ? Math.fround(r(1) - r(2)) : int(r(1) - r(2)));
						break;
					case "mul":
						if (isF) write(l, A[0], Math.fround(r(1) * r(2)));
						else if (mods[0] === "wide") write(l, A[0], r(1) * r(2));
						else write(l, A[0], int(Math.imul(r(1), r(2))));
						break;
					case "mad":
						write(l, A[0], int(Math.imul(r(1), r(2)) + r(3)));
						break;
					case "fma":
						// Exact for the small values in the figures: one rounding of a*b + c.
						write(l, A[0], Math.fround(r(1) * r(2) + r(3)));
						break;
					case "neg":
						write(l, A[0], isF ? -r(1) : int(-r(1)));
						break;
					case "sqrt":
						write(l, A[0], Math.fround(Math.sqrt(r(1))));
						break;
					case "sin":
						write(l, A[0], Math.fround(Math.sin(r(1))));
						break;
					case "cos":
						write(l, A[0], Math.fround(Math.cos(r(1))));
						break;
					case "ex2":
						write(l, A[0], Math.fround(2 ** r(1)));
						break;
					case "shl":
						write(
							l,
							A[0],
							type === "b64" ? r(1) * 2 ** r(2) : int(r(1) << r(2)),
						);
						break;
					case "shr":
						write(l, A[0], signed ? r(1) >> r(2) : r(1) >>> r(2));
						break;
					case "and":
						write(l, A[0], int(r(1) & r(2)));
						break;
					case "or":
						write(l, A[0], int(r(1) | r(2)));
						break;
					case "xor":
						write(l, A[0], int(r(1) ^ r(2)));
						break;
					case "setp": {
						const cmp = mods[0];
						const x = isF ? r(1) : int(r(1));
						const y = isF ? r(2) : int(r(2));
						const nan = Number.isNaN(x) || Number.isNaN(y);
						const res: Record<string, boolean> = {
							eq: x === y,
							ne: x !== y,
							lt: x < y,
							le: x <= y,
							gt: x > y,
							ge: x >= y,
							leu: nan || x <= y,
							ltu: nan || x < y,
							geu: nan || x >= y,
							gtu: nan || x > y,
						};
						if (!(cmp in res)) throw new Error(`setp.${cmp} not supported`);
						write(l, A[0], res[cmp]);
						break;
					}
					case "selp":
						write(l, A[0], pred(l, A[3]) ? r(1) : r(2));
						break;
					case "bra": {
						const t = labels.get(A[0]);
						if (t === undefined) throw new Error(`no label ${A[0]}`);
						next[l] = t;
						break;
					}
					case "ret":
					case "exit":
						done[l] = true;
						break;
					default:
						throw new Error(
							`line ${ins.line + 1}: ${ins.op.join(".")} not supported`,
						);
				}
			}
		}
		if (base === "shfl" || base === "vote")
			for (let l = 0; l < lanes; l++) if (active[l]) next[l] = pc[l] + 1;
		for (let l = 0; l < lanes; l++) pc[l] = next[l];

		issued++;
		const nExec = executed.filter(Boolean).length;
		useful += nExec;
		const nAct = active.filter(Boolean).length;
		const span = (mask: boolean[]) => laneSpan(mask);
		const parts: string[] = [];
		if (nAct < lanes) {
			const waiting = active.map((a, l) => !a && !done[l] && !executed[l]);
			const idle = waiting.some(Boolean)
				? `; ${span(waiting)} wait elsewhere`
				: "";
			parts.push(`Issued for ${span(active)}${idle}.`);
		} else parts.push("Issued for every lane.");
		if (ins.guard && nExec < nAct) {
			const off = active.map((a, l) => a && !executed[l]);
			parts.push(
				`Guard ${ins.guard.neg ? "!" : ""}${ins.guard.reg} is false in ${span(off)}, so they skip it.`,
			);
		}
		const custom = cfg.notes?.[ins.line + 1];
		steps.push({
			line: ins.line,
			text: ins.text,
			active,
			executed,
			values: cfg.show.map((reg) =>
				regs.map((r) => {
					const v = r.get(reg);
					return typeof v === "boolean"
						? v
							? "T"
							: "F"
						: formatValue(v, floats.has(reg));
				}),
			),
			changed: cfg.show.map((reg) => changed.map((c) => c.has(reg))),
			at: at(),
			issued,
			useful,
			note: custom ? `${custom} ${parts.join(" ")}` : parts.join(" "),
			stores,
		});
	}
	return { lines, steps, lanes, show: cfg.show };
}

/** "lanes 0–3", "lane 5", "lanes 0, 2, 4". */
export function laneSpan(mask: boolean[]): string {
	const on = mask.map((m, i) => (m ? i : -1)).filter((i) => i >= 0);
	if (on.length === 0) return "no lanes";
	if (on.length === 1) return `lane ${on[0]}`;
	const contiguous = on.every((v, i) => i === 0 || v === on[i - 1] + 1);
	return contiguous
		? `lanes ${on[0]}–${on[on.length - 1]}`
		: `lanes ${on.join(", ")}`;
}
