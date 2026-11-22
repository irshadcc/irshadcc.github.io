// A tiny cycle-level model of an out-of-order core, for OooSteps.astro. It is a teaching model,
// not a simulator of a real chip: `width` instructions are renamed into the reorder buffer (ROB)
// per cycle in program order, any instruction whose sources are ready issues to a free port of
// its kind (oldest first, one instruction per port per cycle, ports fully pipelined), its result
// is ready `lat` cycles after issue, and up to `width` finished instructions leave the ROB per
// cycle, again in program order. Register renaming is implicit: only true (read-after-write)
// dependences delay an instruction, which is what renaming achieves on a real core.

export type PortKind = "alu" | "fp" | "load" | "store" | "branch";

export interface Instr {
	/** Assembly text shown in the figure. */
	text: string;
	kind: PortKind;
	/** Destination register (renamed), if any. */
	dst?: string;
	/** Source registers: the instruction waits for the latest earlier writer of each. */
	srcs?: string[];
	/** Cycles from issue until the result is ready. */
	lat: number;
	/** Short note, e.g. "L1 miss". */
	tag?: string;
}

export interface Machine {
	/** Rename and retire width (instructions per cycle). */
	width: number;
	rob: number;
	ports: { name: string; kind: PortKind }[];
}

export interface Timing {
	/** Cycle the instruction entered the ROB. */
	dispatch: number;
	issue: number;
	/** First cycle its result can be used (issue + lat). */
	done: number;
	retire: number;
	port: string;
}

export interface CycleState {
	cycle: number;
	/** Indices of instructions in the ROB at the end of this cycle. */
	rob: number[];
	/** Instruction issued on each port this cycle (or -1). */
	portUse: number[];
	dispatched: number[];
	issued: number[];
	retired: number[];
	/** Why rename stopped this cycle, if it did. */
	stall?: string;
}

export interface Simulation {
	timing: Timing[];
	cycles: CycleState[];
	/** Cycles until the last instruction retired. */
	total: number;
}

export function simulate(
	prog: readonly Instr[],
	m: Machine,
	maxCycles = 500,
): Simulation {
	const n = prog.length;
	const t: Timing[] = prog.map(() => ({
		dispatch: -1,
		issue: -1,
		done: -1,
		retire: -1,
		port: "",
	}));
	// producer[i][k] = index of the instruction that writes srcs[k] of instruction i, or -1.
	const producer = prog.map((ins, i) =>
		(ins.srcs ?? []).map((r) => {
			for (let j = i - 1; j >= 0; j--) if (prog[j].dst === r) return j;
			return -1;
		}),
	);
	const cycles: CycleState[] = [];
	let nextDispatch = 0;
	let nextRetire = 0;
	const inRob = new Set<number>();
	for (let c = 0; c < maxCycles && nextRetire < n; c++) {
		const st: CycleState = {
			cycle: c,
			rob: [],
			portUse: m.ports.map(() => -1),
			dispatched: [],
			issued: [],
			retired: [],
		};
		// Retire: in order, finished by the start of this cycle.
		while (
			st.retired.length < m.width &&
			nextRetire < n &&
			t[nextRetire].done >= 0 &&
			t[nextRetire].done <= c &&
			t[nextRetire].issue < c
		) {
			t[nextRetire].retire = c;
			inRob.delete(nextRetire);
			st.retired.push(nextRetire);
			nextRetire++;
		}
		// Issue: oldest first, sources ready, a free port of the right kind.
		for (const i of [...inRob].sort((a, b) => a - b)) {
			if (t[i].issue >= 0 || t[i].dispatch >= c) continue;
			const ready = producer[i].every(
				(p) => p < 0 || (t[p].done >= 0 && t[p].done <= c),
			);
			if (!ready) continue;
			const port = m.ports.findIndex(
				(p, k) => p.kind === prog[i].kind && st.portUse[k] < 0,
			);
			if (port < 0) continue;
			st.portUse[port] = i;
			t[i].issue = c;
			t[i].done = c + prog[i].lat;
			t[i].port = m.ports[port].name;
			st.issued.push(i);
		}
		// Rename / dispatch: in order, while the ROB has room.
		while (st.dispatched.length < m.width && nextDispatch < n) {
			if (inRob.size >= m.rob) {
				st.stall = `ROB full (${m.rob} entries): rename waits for the oldest instruction to retire`;
				break;
			}
			t[nextDispatch].dispatch = c;
			inRob.add(nextDispatch);
			st.dispatched.push(nextDispatch);
			nextDispatch++;
		}
		st.rob = [...inRob].sort((a, b) => a - b);
		cycles.push(st);
	}
	return { timing: t, cycles, total: cycles.length };
}

/** State letter of instruction i in cycle c, for the pipeline diagram. */
export function cellState(
	tm: Timing,
	c: number,
): "" | "D" | "w" | "E" | "e" | "r" | "R" {
	if (tm.dispatch < 0 || c < tm.dispatch) return "";
	if (tm.retire >= 0 && c > tm.retire) return "";
	if (c === tm.retire) return "R";
	if (c === tm.dispatch) return "D";
	if (tm.issue < 0 || c < tm.issue) return "w";
	if (c === tm.issue) return "E";
	if (c < tm.done) return "e";
	return "r";
}
