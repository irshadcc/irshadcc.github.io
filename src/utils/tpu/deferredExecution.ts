// Logic for EagerModeSteps.astro: how one small PyTorch program is cut into TPU executables by
// TorchTPU's execution modes (strict eager, defer-and-fuse, torch.compile) and by PyTorch/XLA's
// lazy tensors. Each step is one event of the program (an op dispatched, a sync); the result says
// which ops are still deferred, which executables exist, and whether each was compiled or found
// in the compilation cache.
//
// The repetition detector is a port of StreamState::CheckLastRepeatedSequence and
// StreamState::FindRepeatedSequence in TorchTPU's csrc/eager/events_queue.cc (commit 02cc8fe):
// sequences of 2..128 op hashes repeated at least twice end a fused region. An op's hash is
// DeferredOp::Hash(): op name, input shapes, donated indices, output shapes and op parameters.
//
// Simplifications (stated in the post): the program keeps one live tensor `x`, so every flush
// produces a single-output graph (GraphSplitMode::kSplitGraph then makes no extra split), and a
// cache key is the sequence of op signatures in the flushed region.

export type Mode = "strict" | "fuse" | "compile" | "lazy";
export type Opt = "O1" | "O2" | "default";

export interface ProgEvent {
	kind: "op" | "sync" | "markStep";
	/** Op name shown on the chip, e.g. "mm". */
	name: string;
	/** Op signature: name + shapes + parameters; equal signatures hash equally. */
	sig: string;
	/** Python source line that issued the event. */
	line: string;
	/** Loop iteration (1-based) or 0 outside the loop. */
	iter: number;
	/** For mode "compile": true for ops inside the compiled function. */
	inCompiled?: boolean;
}

export interface Executable {
	/** Indices into the program of the ops this executable computes. */
	ops: number[];
	/** Cache key: the joined op signatures. */
	key: string;
	/** True on a cache miss (compiled now), false on a hit. */
	compiled: boolean;
	/** XLA optimisation level used for the compile ("default" for PyTorch/XLA). */
	opt: Opt;
	/** Why the region was flushed. */
	reason: string;
	/** torch.compile: a later call reuses the callable's executable without a cache lookup. */
	reused?: boolean;
}

export interface Step {
	/** Index of the program event this step processes. */
	at: number;
	/** For every op in the program: -2 not yet issued, -1 deferred, k = in executable k. */
	opState: number[];
	executables: Executable[];
	/** Executable created in this step, if any (indices into executables). */
	fresh: number[];
	text: string;
	compiles: number;
	hits: number;
	reuses: number;
}

const MIN_LEN = 2;
const MAX_LEN = 128;
const MIN_COUNT = 2;

export type Detect = "none" | "wait" | "complete" | "found";

/** Port of TorchTPU's repeated-op heuristic for one stream. */
export class RepetitionDetector {
	/** Hashes of ops still deferred on the stream (deferred_ops_). */
	queue: string[] = [];
	/** The last repeated pattern (repetition_hashes_). */
	pattern: string[] = [];
	/** Position inside `pattern` expected next (repetition_i_). */
	i = 0;

	/** RecordDeferredOpAndDetectRepetition: true means "materialize the stream now". */
	record(hash: string): { flush: boolean; how: Detect; length: number } {
		this.queue.push(hash);
		// Fast path: CheckLastRepeatedSequence.
		if (this.i < this.pattern.length) {
			if (hash === this.pattern[this.i]) {
				this.i++;
				if (this.i === this.pattern.length) {
					this.i = 0;
					return { flush: true, how: "complete", length: this.pattern.length };
				}
				return { flush: false, how: "wait", length: this.pattern.length };
			}
			this.i = 0;
			this.pattern = [];
		}
		const len = this.find(hash);
		if (len === 0) return { flush: false, how: "none", length: 0 };
		this.pattern = this.queue.slice(this.queue.length - len);
		return { flush: true, how: "found", length: len };
	}

	/** FindRepeatedSequence: length L of a sequence repeated MIN_COUNT times at the end, or 0. */
	find(hash: string): number {
		const q = this.queue;
		const last = q.length - 1;
		if (last < 0) return 0;
		let len = Math.min(MIN_LEN, q.length);
		let prior = last - len; // prior_repetition_end_it, as an index from the front
		while (prior >= 0 && len <= MAX_LEN) {
			if (q[prior] !== hash) {
				len++;
				prior--;
				continue;
			}
			const need = (MIN_COUNT - 1) * len;
			let matches = 0;
			let p = prior;
			let c = last;
			while (matches < need && p >= 0) {
				if (q[p] !== q[c]) break;
				if (++matches === need) break;
				p--;
				c--;
			}
			if (matches === need) return len;
			len++;
			prior--;
		}
		return 0;
	}

	/** Materialized ops leave the queue (their DeferredOps are gone). */
	drain(): void {
		this.queue = [];
	}
}

/**
 * The running example: `x = x * 0.5`, then `iters` iterations of
 * `x = torch.relu(x @ w + b)`, then `loss = x.sum()` and `loss.item()`.
 * For mode "lazy" an `xm.mark_step()` ends every iteration.
 */
export function loopProgram(iters: number, lazy = false): ProgEvent[] {
	const ev: ProgEvent[] = [
		{
			kind: "op",
			name: "mul",
			sig: "mul[8x8,0.5]",
			line: "x = x * 0.5",
			iter: 0,
		},
	];
	for (let k = 1; k <= iters; k++) {
		ev.push(
			{
				kind: "op",
				name: "mm",
				sig: "mm[8x8,8x8]",
				line: "x @ w",
				iter: k,
				inCompiled: true,
			},
			{
				kind: "op",
				name: "add",
				sig: "add[8x8,8]",
				line: "... + b",
				iter: k,
				inCompiled: true,
			},
			{
				kind: "op",
				name: "relu",
				sig: "relu[8x8]",
				line: "x = torch.relu(...)",
				iter: k,
				inCompiled: true,
			},
		);
		if (lazy)
			ev.push({
				kind: "markStep",
				name: "mark_step",
				sig: "",
				line: "xm.mark_step()",
				iter: k,
			});
	}
	ev.push(
		{
			kind: "op",
			name: "sum",
			sig: "sum[8x8]",
			line: "loss = x.sum()",
			iter: 0,
		},
		{ kind: "sync", name: "item", sig: "", line: "loss.item()", iter: 0 },
	);
	return ev;
}

/** Simulates one mode over a program and returns one Step per event. */
export function simulate(mode: Mode, prog: ProgEvent[]): Step[] {
	const opState = prog.map(() => -2);
	const exes: Executable[] = [];
	const cache = new Set<string>();
	const det = new RepetitionDetector();
	let pending: number[] = [];
	let compiles = 0;
	let hits = 0;
	const steps: Step[] = [];

	const flush = (ops: number[], reason: string, opt: Opt): number | null => {
		if (ops.length === 0) return null;
		const key = ops.map((j) => prog[j].sig).join(" → ");
		const compiled = !cache.has(key);
		cache.add(key);
		if (compiled) compiles++;
		else hits++;
		exes.push({ ops: [...ops], key, compiled, opt, reason });
		const k = exes.length - 1;
		for (const j of ops) opState[j] = k;
		return k;
	};
	const fmtExe = (k: number | null) => {
		if (k === null) return "";
		const e = exes[k];
		const names = e.ops.map((j) => prog[j].name).join(", ");
		return e.compiled
			? ` Executable ${k + 1} (${names}) is new: cache miss, compiled${e.opt === "default" ? "" : ` at -${e.opt}`}.`
			: ` Executable ${k + 1} (${names}) has the same key as an earlier one: cache hit, no compile.`;
	};

	// torch.compile: ops inside the compiled function run as one call per iteration.
	let callOps: number[] = [];

	prog.forEach((ev, idx) => {
		const fresh: number[] = [];
		let text = "";
		const push = (k: number | null) => {
			if (k !== null) fresh.push(k);
		};
		if (ev.kind === "op") opState[idx] = -1;

		if (mode === "strict") {
			if (ev.kind === "op") {
				const k = flush([idx], "op dispatched", "O1");
				push(k);
				text = `<code>${ev.line}</code> creates a DeferredOp and immediately flags the stream; the background worker turns it into its own executable.${fmtExe(k)}`;
			} else {
				text = `<code>${ev.line}</code> copies a value to the host. Every op already ran, so it only waits for the TPU.`;
			}
		} else if (mode === "fuse") {
			if (ev.kind === "op") {
				pending.push(idx);
				const r = det.record(ev.sig);
				if (r.flush) {
					const k = flush(
						pending,
						r.how === "found" ? "repetition found" : "repetition complete",
						"O2",
					);
					push(k);
					pending = [];
					det.drain();
					text =
						r.how === "found"
							? `<code>${ev.line}</code> ends a sequence of ${r.length} op hashes seen twice in a row, so the detector asks for a graph break. All deferred ops become one graph, since only <code>x</code> is still alive.${fmtExe(k)}`
							: `<code>${ev.line}</code> completes another copy of the known ${r.length}-op loop, so the loop body is flushed.${fmtExe(k)}`;
				} else {
					text =
						r.how === "wait"
							? `<code>${ev.line}</code> matches the next op of the known loop; it stays deferred until the loop body is complete.`
							: `<code>${ev.line}</code> is deferred. No repetition yet, so nothing runs.`;
				}
			} else {
				const k = flush(pending, "host transfer", "O2");
				push(k);
				pending = [];
				det.drain();
				text = `<code>${ev.line}</code> needs the value on the host, which forces materialization of everything deferred.${fmtExe(k)}`;
			}
		} else if (mode === "compile") {
			if (ev.kind === "op" && ev.inCompiled) {
				callOps.push(idx);
				const last =
					idx + 1 >= prog.length ||
					!prog[idx + 1].inCompiled ||
					prog[idx + 1].iter !== ev.iter;
				if (last) {
					const first = ev.iter === 1;
					const k = flush(callOps, "compiled call", "O2");
					// A compiled callable skips the cache lookup: Dynamo's guards already matched.
					if (!first && k !== null) {
						exes[k].compiled = false;
						exes[k].reused = true;
					}
					push(k);
					text = first
						? `The first call of <code>step(x)</code>: Dynamo captures an FX graph, TorchTPU traces it with placeholder tensors into one StableHLO program and compiles it at -O2.${fmtExe(k)}`
						: `Call ${ev.iter} of <code>step(x)</code>: Dynamo's guards pass, so the callable pushes the ready executable to PJRT. No DeferredOps, no graph analysis, no cache lookup.`;
					callOps = [];
				} else {
					text = `<code>${ev.line}</code> is inside the compiled function. It is not dispatched one op at a time.`;
				}
			} else if (ev.kind === "op") {
				const k = flush([idx], "op dispatched", "O1");
				push(k);
				text = `<code>${ev.line}</code> is outside <code>step</code>, so it runs in strict eager mode as its own executable.${fmtExe(k)}`;
			} else {
				text = `<code>${ev.line}</code> waits for the TPU and copies the value to the host.`;
			}
		} else {
			// PyTorch/XLA lazy tensors.
			if (ev.kind === "op") {
				pending.push(idx);
				text = `<code>${ev.line}</code> adds a node to the lazy IR graph. Nothing is compiled or run.`;
			} else {
				const k = flush(
					pending,
					ev.kind === "markStep" ? "mark_step" : "host transfer",
					"default",
				);
				push(k);
				pending = [];
				text =
					ev.kind === "markStep"
						? `<code>xm.mark_step()</code> cuts the graph: the IR is hashed, looked up in the computation cache and run.${fmtExe(k)}`
						: `<code>${ev.line}</code> needs the value, so the pending graph is cut here too.${fmtExe(k)}`;
			}
		}

		// Recompute counters from the executables so "compile" mode's patched flags count right.
		compiles = exes.filter((e) => e.compiled).length;
		const reuses = exes.filter((e) => e.reused).length;
		hits = exes.length - compiles - reuses;
		steps.push({
			at: idx,
			opState: [...opState],
			executables: exes.map((e) => ({ ...e, ops: [...e.ops] })),
			fresh,
			text,
			compiles,
			hits,
			reuses,
		});
	});
	return steps;
}

/** One-line summary of a finished run, e.g. "14 executables, 5 compiles". */
export function summary(steps: Step[]): {
	launches: number;
	compiles: number;
	hits: number;
	reuses: number;
} {
	const last = steps[steps.length - 1];
	return {
		launches: last.executables.length,
		compiles: last.compiles,
		hits: last.hits,
		reuses: last.reuses,
	};
}

/** The Python program a mode runs, and for each program event the line that issued it. */
export function codeFor(
	mode: Mode,
	prog: ProgEvent[],
): { lines: string[]; lineOf: number[] } {
	const iters = Math.max(...prog.map((e) => e.iter));
	const loop = `for _ in range(${iters}):`;
	let lines: string[];
	if (mode === "compile") {
		lines = [
			'@torch.compile(backend="tpu")',
			"def step(x):",
			"    return torch.relu(x @ w + b)",
			"",
			"x = x * 0.5",
			loop,
			"    x = step(x)",
			"loss = x.sum()",
			"loss.item()",
		];
	} else if (mode === "lazy") {
		lines = [
			"x = x * 0.5",
			loop,
			"    x = torch.relu(x @ w + b)",
			"    xm.mark_step()",
			"loss = x.sum()",
			"loss.item()",
		];
	} else {
		lines = [
			"x = x * 0.5",
			loop,
			"    x = torch.relu(x @ w + b)",
			"loss = x.sum()",
			"loss.item()",
		];
	}
	const find = (s: string) => lines.findIndex((l) => l.trim().startsWith(s));
	const lineOf = prog.map((e) => {
		if (e.kind === "markStep") return find("xm.mark_step");
		if (e.kind === "sync") return find("loss.item");
		if (e.name === "mul") return find("x = x * 0.5");
		if (e.name === "sum") return find("loss = x.sum");
		return mode === "compile"
			? find("return torch.relu")
			: find("x = torch.relu");
	});
	return { lines, lineOf };
}
