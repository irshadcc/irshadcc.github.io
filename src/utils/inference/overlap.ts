// CPU and GPU timelines of a serving loop, for OverlapTimeline.astro. Each batch k needs CPU
// time to schedule it (S), GPU time to run it (F) and CPU time to process its output (P).
// "normal" follows SGLang's Scheduler.event_loop_normal: S_k, F_k, P_k strictly in turn.
// "overlap" follows event_loop_overlap: after scheduling and launching batch k the CPU
// processes batch k-1's result while the GPU runs batch k, so the GPU only waits when
// S + P is longer than F.

export type LoopKind = "normal" | "overlap";

export interface Span {
	lane: "cpu" | "gpu";
	kind: "S" | "F" | "P";
	batch: number;
	start: number;
	end: number;
}

export interface LoopTimes {
	schedule: number;
	forward: number;
	process: number;
}

export function simulateLoop(kind: LoopKind, n: number, t: LoopTimes): Span[] {
	const spans: Span[] = [];
	if (kind === "normal") {
		let now = 0;
		for (let k = 1; k <= n; k++) {
			spans.push({
				lane: "cpu",
				kind: "S",
				batch: k,
				start: now,
				end: now + t.schedule,
			});
			now += t.schedule;
			spans.push({
				lane: "gpu",
				kind: "F",
				batch: k,
				start: now,
				end: now + t.forward,
			});
			now += t.forward;
			spans.push({
				lane: "cpu",
				kind: "P",
				batch: k,
				start: now,
				end: now + t.process,
			});
			now += t.process;
		}
		return spans;
	}
	let cpu = 0;
	let gpuFree = 0;
	const fEnd: number[] = [];
	for (let k = 1; k <= n + 1; k++) {
		if (k <= n) {
			spans.push({
				lane: "cpu",
				kind: "S",
				batch: k,
				start: cpu,
				end: cpu + t.schedule,
			});
			cpu += t.schedule;
			const start = Math.max(cpu, gpuFree);
			spans.push({
				lane: "gpu",
				kind: "F",
				batch: k,
				start,
				end: start + t.forward,
			});
			gpuFree = start + t.forward;
			fEnd[k] = gpuFree;
		}
		if (k > 1) {
			const start = Math.max(cpu, fEnd[k - 1]);
			spans.push({
				lane: "cpu",
				kind: "P",
				batch: k - 1,
				start,
				end: start + t.process,
			});
			cpu = start + t.process;
		}
	}
	return spans;
}

/** Fraction of the time until the last span ends during which the GPU is busy. */
export function gpuBusy(spans: Span[]): number {
	const end = Math.max(...spans.map((s) => s.end));
	const busy = spans
		.filter((s) => s.lane === "gpu")
		.reduce((a, s) => a + s.end - s.start, 0);
	return busy / end;
}

export function drawLoop(spans: Span[], scale: number, total: number): string {
	const laneH = 24;
	const left = 40;
	const out: string[] = [];
	(["cpu", "gpu"] as const).forEach((lane, i) => {
		const y = i * (laneH + 8);
		out.push(
			`<text class="ol-lane" x="0" y="${y + 16}">${lane.toUpperCase()}</text>`,
		);
		out.push(
			`<line class="ol-axis" x1="${left}" x2="${left + total * scale}" y1="${y + laneH}" y2="${y + laneH}"/>`,
		);
	});
	for (const s of spans) {
		const y = (s.lane === "cpu" ? 0 : 1) * (laneH + 8);
		const x = left + s.start * scale;
		const w = (s.end - s.start) * scale - 2;
		const name =
			s.kind === "S" ? "schedule" : s.kind === "F" ? "forward" : "process";
		out.push(
			`<g class="ol-span ${s.kind}"><title>${name} batch ${s.batch}: t = ${s.start} to ${s.end}</title><rect x="${x}" y="${y}" width="${w}" height="${laneH}" rx="3"/><text x="${x + w / 2}" y="${y + 16}">${s.kind}${s.batch}</text></g>`,
		);
	}
	const W = left + total * scale + 4;
	const H = 2 * (laneH + 8);
	return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="CPU and GPU timeline">${out.join("")}</svg>`;
}
