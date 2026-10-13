// A toy timeline of a persistent tcgen05 GEMM CTA, for UmmaPipeline.astro: a TMA producer loads
// k-blocks of A and B into `abStages` SMEM buffers, a single thread issues the MMAs for each
// k-block into one of `accStages` TMEM accumulators, and the epilogue warps drain a finished
// accumulator (TMEM → registers → global memory). Durations are illustrative units, not
// measurements: one unit is one k-block of MMAs.
//
// Dependencies, as the mbarrier pipelines of the CUTLASS examples enforce them
// (PipelineTmaUmma for A/B, PipelineUmmaAsync for the accumulator):
//   load g      waits for the MMA of load g − abStages (its buffer is released by tcgen05.commit)
//   MMA  g      waits for load g, and on a tile's first k-block for the epilogue of tile t − accStages
//   epilogue t  waits for the last MMA of tile t
//
//   const r = simulatePipeline({ tiles: 4, kBlocks: 4, abStages: 4, accStages: 2, tma: 1, mma: 1, epi: 3 });
//   r.end   // 19

export interface PipelineConfig {
	/** Output tiles this CTA computes. */
	tiles: number;
	/** k-blocks per output tile (K / BK). */
	kBlocks: number;
	abStages: number;
	accStages: number;
	/** Duration of one k-block's TMA load, its MMAs, and one tile's epilogue. */
	tma: number;
	mma: number;
	epi: number;
}

export type Lane = "TMA" | "MMA" | "Epilogue";

export interface Span {
	lane: Lane;
	start: number;
	end: number;
	tile: number;
	/** k-block, for TMA and MMA spans. */
	k?: number;
	/** SMEM buffer (TMA, MMA) or TMEM accumulator stage (MMA, Epilogue). */
	buffer: number;
	acc?: number;
}

export interface PipelineResult {
	spans: Span[];
	end: number;
	/** Fraction of [0, end] the MMA lane is busy. */
	mmaBusy: number;
}

export function simulatePipeline(c: PipelineConfig): PipelineResult {
	if (c.abStages < 1 || c.accStages < 1)
		throw new Error("need at least one stage");
	const n = c.tiles * c.kBlocks;
	const tmaEnd: number[] = [];
	const mmaEnd: number[] = [];
	const epiEnd: number[] = [];
	const spans: Span[] = [];
	let tmaFree = 0;
	let mmaFree = 0;
	let epiFree = 0;
	// Loads run ahead of the MMAs as far as the buffers allow, so walk both in index order,
	// interleaving: load g only depends on MMA g − abStages, which comes earlier.
	for (let g = 0; g < n; g++) {
		const t = Math.floor(g / c.kBlocks);
		const k = g % c.kBlocks;
		const freeBuf = g - c.abStages >= 0 ? mmaEnd[g - c.abStages] : 0;
		const ls = Math.max(tmaFree, freeBuf);
		tmaEnd[g] = ls + c.tma;
		tmaFree = tmaEnd[g];
		spans.push({
			lane: "TMA",
			start: ls,
			end: tmaEnd[g],
			tile: t,
			k,
			buffer: g % c.abStages,
		});

		const accFree =
			k === 0 && t - c.accStages >= 0 ? epiEnd[t - c.accStages] : 0;
		const ms = Math.max(mmaFree, tmaEnd[g], accFree);
		mmaEnd[g] = ms + c.mma;
		mmaFree = mmaEnd[g];
		spans.push({
			lane: "MMA",
			start: ms,
			end: mmaEnd[g],
			tile: t,
			k,
			buffer: g % c.abStages,
			acc: t % c.accStages,
		});

		if (k === c.kBlocks - 1) {
			const es = Math.max(epiFree, mmaEnd[g]);
			epiEnd[t] = es + c.epi;
			epiFree = epiEnd[t];
			spans.push({
				lane: "Epilogue",
				start: es,
				end: epiEnd[t],
				tile: t,
				buffer: t % c.accStages,
				acc: t % c.accStages,
			});
		}
	}
	const end = Math.max(...spans.map((s) => s.end));
	const busy = spans
		.filter((s) => s.lane === "MMA")
		.reduce((a, s) => a + s.end - s.start, 0);
	return { spans, end, mmaBusy: busy / end };
}

const LANES: Lane[] = ["TMA", "MMA", "Epilogue"];
const LANE_LABEL: Record<Lane, string> = {
	TMA: "TMA loads",
	MMA: "MMA (1 thread)",
	Epilogue: "epilogue warps",
};

/** The timeline as SVG; `unitPx` pixels per time unit, `endAt` fixes the axis length. */
export function pipelineSvg(
	r: PipelineResult,
	title: string,
	unitPx: number,
	endAt = r.end,
): string {
	const left = 104;
	const laneH = 22;
	const top = 18;
	const w = left + endAt * unitPx + 8;
	const h = top + LANES.length * (laneH + 4) + 16;
	const out: string[] = [];
	out.push(`<text class="up-title" x="0" y="11">${title}</text>`);
	LANES.forEach((lane, i) => {
		const y = top + i * (laneH + 4);
		out.push(
			`<text class="up-lane" x="${left - 8}" y="${y + laneH / 2}">${LANE_LABEL[lane]}</text>`,
		);
		out.push(
			`<rect class="up-track" x="${left}" y="${y}" width="${(endAt * unitPx).toFixed(1)}" height="${laneH}"/>`,
		);
	});
	for (const s of r.spans) {
		const i = LANES.indexOf(s.lane);
		const y = top + i * (laneH + 4);
		const x = left + s.start * unitPx;
		const sw = (s.end - s.start) * unitPx;
		const label = s.k === undefined ? `t${s.tile}` : `${s.tile}.${s.k}`;
		const what =
			s.lane === "TMA"
				? `TMA: tile ${s.tile}, k-block ${s.k} into SMEM stage ${s.buffer}`
				: s.lane === "MMA"
					? `MMA: tile ${s.tile}, k-block ${s.k} from SMEM stage ${s.buffer} into TMEM accumulator ${s.acc}`
					: `Epilogue: tile ${s.tile} from TMEM accumulator ${s.acc}`;
		out.push(
			`<rect class="up-span up-t${s.tile % 6}" x="${(x + 0.5).toFixed(1)}" y="${y + 1}" width="${Math.max(1, sw - 1).toFixed(1)}" height="${laneH - 2}" rx="3" data-d="${what}, time ${s.start}–${s.end}"/>`,
		);
		if (sw >= 22)
			out.push(
				`<text class="up-label" x="${(x + sw / 2).toFixed(1)}" y="${y + laneH / 2}">${label}</text>`,
			);
	}
	const axisY = top + LANES.length * (laneH + 4) + 10;
	for (let t = 0; t <= endAt; t += endAt > 40 ? 10 : 5)
		out.push(
			`<text class="up-tick" x="${left + t * unitPx}" y="${axisY}">${t}</text>`,
		);
	return `<svg viewBox="0 0 ${Math.ceil(w)} ${h}" role="img" aria-label="${title}">${out.join("")}</svg>`;
}
