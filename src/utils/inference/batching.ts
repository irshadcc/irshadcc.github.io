// Iteration-level batching as vLLM and SGLang do it, simulated step by step for BatchTimeline.astro.
// Each step is one forward pass. A request has a prompt of `prompt` tokens and stops after
// `output` generated tokens; the step that computes its last prompt token also samples its
// first output token, and every later step that runs it decodes one more.
//
// "vllm" follows Scheduler.schedule() in vllm/v1/core/sched/scheduler.py: one token budget per
// step, running requests first (a decode costs 1 token, an unfinished prefill takes the next
// chunk), then waiting requests in arrival order, with chunked prefill. Prefill chunks and
// decodes share the same forward pass.
// "sglang" follows Scheduler.get_next_batch_to_run() in sglang/srt/managers/scheduler.py with
// its defaults (no --enable-mixed-chunk): if any prefill work can be scheduled, the step is a
// prefill-only batch of at most `budget` tokens (the chunked request first, then waiting
// requests in arrival order, at most one of them cut into a chunk); otherwise it is a decode
// batch of every running request.
// Memory limits, preemption and retraction are left out.

export type BatchPolicy = "vllm" | "sglang";

export interface BatchRequest {
	id: string;
	prompt: number;
	output: number;
	/** Step at which the request reaches the server. */
	arrival: number;
}

export interface CellWork {
	/** Tokens this request contributed to the step's forward pass. */
	tokens: number;
	kind: "prefill" | "decode";
	/** This step sampled an output token for the request. */
	emits: boolean;
}

export type CellState = "absent" | "waiting" | "idle" | "done";

export interface BatchStep {
	/** Work done per request id in this step (missing = no work). */
	work: Record<string, CellWork>;
	/** What each request was doing when it did no work. */
	state: Record<string, CellState>;
	total: number;
	kind: "mixed" | "prefill" | "decode" | "empty";
}

export interface BatchResult {
	steps: BatchStep[];
	/** Step at which each request emitted its first token (time to first token, in steps). */
	firstToken: Record<string, number>;
	/** Step at which each request emitted its last token. */
	finish: Record<string, number>;
}

interface Live {
	req: BatchRequest;
	computed: number;
	emitted: number;
}

export function simulateBatching(
	policy: BatchPolicy,
	reqs: readonly BatchRequest[],
	budget: number,
	maxSteps = 64,
): BatchResult {
	if (budget < 1) throw new Error("budget must be at least 1");
	for (const r of reqs)
		if (r.prompt < 1 || r.output < 1)
			throw new Error(`request ${r.id}: prompt and output must be >= 1`);
	const pending = [...reqs].sort((a, b) => a.arrival - b.arrival);
	const waiting: Live[] = [];
	const running: Live[] = [];
	const done = new Set<string>();
	const steps: BatchStep[] = [];
	const firstToken: Record<string, number> = {};
	const finish: Record<string, number> = {};

	const run = (l: Live, n: number, step: number, work: BatchStep["work"]) => {
		const prefill = l.computed < l.req.prompt;
		l.computed += n;
		const emits = l.computed >= l.req.prompt;
		work[l.req.id] = { tokens: n, kind: prefill ? "prefill" : "decode", emits };
		if (emits) {
			l.emitted++;
			if (firstToken[l.req.id] === undefined) firstToken[l.req.id] = step;
		}
	};

	for (let t = 0; t < maxSteps; t++) {
		while (pending.length && pending[0].arrival <= t) {
			const req = pending.shift() as BatchRequest;
			waiting.push({ req, computed: 0, emitted: 0 });
		}
		if (!waiting.length && !running.length) {
			if (!pending.length) break;
			steps.push(emptyStep(reqs, t, done));
			continue;
		}
		const work: BatchStep["work"] = {};
		let left = budget;
		let kind: BatchStep["kind"];

		if (policy === "vllm") {
			for (const l of running) {
				if (left === 0) break;
				const need = l.computed < l.req.prompt ? l.req.prompt - l.computed : 1;
				const n = Math.min(need, left);
				run(l, n, t, work);
				left -= n;
			}
			while (left > 0 && waiting.length) {
				const l = waiting.shift() as Live;
				const n = Math.min(l.req.prompt, left);
				running.push(l);
				run(l, n, t, work);
				left -= n;
			}
			const kinds = new Set(Object.values(work).map((w) => w.kind));
			kind =
				kinds.size === 2
					? "mixed"
					: kinds.has("prefill")
						? "prefill"
						: "decode";
		} else {
			// Prefill batch if there is any prefill work: the chunked request, then new requests.
			const chunked = running.find((l) => l.computed < l.req.prompt);
			if (chunked || waiting.length) {
				if (chunked) {
					const n = Math.min(chunked.req.prompt - chunked.computed, left);
					run(chunked, n, t, work);
					left -= n;
				}
				let cut =
					chunked !== undefined && chunked.computed < chunked.req.prompt;
				while (left > 0 && waiting.length && !cut) {
					const l = waiting.shift() as Live;
					const n = Math.min(l.req.prompt, left);
					running.push(l);
					run(l, n, t, work);
					left -= n;
					cut = l.computed < l.req.prompt;
				}
				kind = "prefill";
			} else {
				for (const l of running) run(l, 1, t, work);
				kind = "decode";
			}
		}

		const state: BatchStep["state"] = {};
		for (const r of reqs) {
			if (work[r.id]) continue;
			if (done.has(r.id)) state[r.id] = "done";
			else if (r.arrival > t) state[r.id] = "absent";
			else if (waiting.some((l) => l.req.id === r.id)) state[r.id] = "waiting";
			else state[r.id] = "idle";
		}
		for (const l of [...running])
			if (l.emitted >= l.req.output) {
				running.splice(running.indexOf(l), 1);
				done.add(l.req.id);
				finish[l.req.id] = t;
			}
		const total = Object.values(work).reduce((s, w) => s + w.tokens, 0);
		steps.push({ work, state, total, kind });
		if (!waiting.length && !running.length && !pending.length) break;
	}
	return { steps, firstToken, finish };
}

function emptyStep(
	reqs: readonly BatchRequest[],
	t: number,
	done: Set<string>,
): BatchStep {
	const state: BatchStep["state"] = {};
	for (const r of reqs)
		state[r.id] = done.has(r.id) ? "done" : r.arrival > t ? "absent" : "idle";
	return { work: {}, state, total: 0, kind: "empty" };
}

/** The scaled-down workload used in the vLLM and SGLang posts. */
export const BATCH_WORKLOAD: BatchRequest[] = [
	{ id: "A", prompt: 5, output: 6, arrival: 0 },
	{ id: "B", prompt: 3, output: 6, arrival: 0 },
	{ id: "C", prompt: 14, output: 3, arrival: 2 },
	{ id: "D", prompt: 4, output: 3, arrival: 3 },
];
export const BATCH_BUDGET = 8;

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

/** The schedule as a grid: one row per request, one column per step. */
export function drawBatching(
	res: BatchResult,
	reqs: readonly BatchRequest[],
	budget: number,
	colors: readonly string[],
): string {
	const cw = 30;
	const ch = 26;
	const gap = 3;
	const left = 34;
	const top = 22;
	const n = res.steps.length;
	const W = left + n * (cw + gap);
	const H = top + (reqs.length + 1) * (ch + gap) + 6;
	const out: string[] = [];
	for (let t = 0; t < n; t++) {
		const x = left + t * (cw + gap);
		out.push(`<text class="bt-step" x="${x + cw / 2}" y="14">${t}</text>`);
	}
	reqs.forEach((r, i) => {
		const y = top + i * (ch + gap);
		const col = colors[i % colors.length];
		out.push(
			`<text class="bt-req" x="${left - 10}" y="${y + ch / 2 + 4}">${esc(r.id)}</text>`,
		);
		res.steps.forEach((s, t) => {
			const x = left + t * (cw + gap);
			const w = s.work[r.id];
			if (w) {
				const tip = `Step ${t}, request ${r.id}: ${w.tokens} ${w.kind} token${w.tokens > 1 ? "s" : ""}${w.emits ? ", samples a token" : ", no token yet"}`;
				out.push(
					`<g class="bt-cell ${w.kind}"><title>${esc(tip)}</title>`,
					`<rect x="${x}" y="${y}" width="${cw}" height="${ch}" rx="3" style="fill:${col}"/>`,
					`<text x="${x + cw / 2}" y="${y + ch / 2 + 4}">${w.tokens}</text>`,
					w.emits ? `<circle cx="${x + cw - 5}" cy="${y + 5}" r="2.6"/>` : "",
					"</g>",
				);
				return;
			}
			const st = s.state[r.id];
			if (st === "waiting" || st === "idle") {
				const tip =
					st === "waiting"
						? `Step ${t}, request ${r.id}: waiting to be admitted`
						: `Step ${t}, request ${r.id}: running but not in this batch`;
				out.push(
					`<rect class="bt-${st}" x="${x + 0.5}" y="${y + 0.5}" width="${cw - 1}" height="${ch - 1}" rx="3"><title>${esc(tip)}</title></rect>`,
				);
			}
		});
	});
	const y = top + reqs.length * (ch + gap) + 4;
	out.push(
		`<text class="bt-req" x="${left - 10}" y="${y + ch / 2 + 2}">Σ</text>`,
	);
	res.steps.forEach((s, t) => {
		const x = left + t * (cw + gap);
		const full = s.total === budget;
		out.push(
			`<text class="bt-total${full ? " full" : ""}" x="${x + cw / 2}" y="${y + ch / 2 + 2}">${s.total}<title>Step ${t}: ${s.total} of ${budget} tokens, ${s.kind} batch</title></text>`,
		);
	});
	return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Tokens scheduled per request and step">${out.join("")}</svg>`;
}
