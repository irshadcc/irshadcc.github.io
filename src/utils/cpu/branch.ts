// Two toy branch predictors for BranchSteps.astro, run over one branch's outcome string
// ("T" taken, "N" not taken). Both use 2-bit saturating counters: 0 and 1 predict not taken,
// 2 and 3 predict taken, and each outcome moves the counter one step towards itself.
//   bimodal: one counter for the branch (what a predictor indexed by address alone does).
//   history: a table of 2^h counters indexed by the branch's last h outcomes (a local-history
//   predictor; real cores use TAGE, which combines several history lengths).

export interface BranchStep {
	outcome: "T" | "N";
	bimodal: { counter: number; predict: "T" | "N"; miss: boolean };
	history: {
		index: number;
		pattern: string;
		counter: number;
		predict: "T" | "N";
		miss: boolean;
	};
	/** History-table counters after the update. */
	table: number[];
	missesBimodal: number;
	missesHistory: number;
}

const pred = (c: number): "T" | "N" => (c >= 2 ? "T" : "N");
const update = (c: number, o: "T" | "N") =>
	o === "T" ? Math.min(3, c + 1) : Math.max(0, c - 1);

export function runPredictors(outcomes: string, h = 3, init = 2): BranchStep[] {
	let bim = init;
	const table = new Array<number>(1 << h).fill(init);
	let hist = 0;
	let mb = 0;
	let mh = 0;
	const out: BranchStep[] = [];
	for (const ch of outcomes) {
		const o = ch === "T" ? "T" : "N";
		const bp = pred(bim);
		const idx = hist;
		const hp = pred(table[idx]);
		if (bp !== o) mb++;
		if (hp !== o) mh++;
		const step: BranchStep = {
			outcome: o,
			bimodal: { counter: bim, predict: bp, miss: bp !== o },
			history: {
				index: idx,
				pattern: idx
					.toString(2)
					.padStart(h, "0")
					.replace(/1/g, "T")
					.replace(/0/g, "N"),
				counter: table[idx],
				predict: hp,
				miss: hp !== o,
			},
			table: [],
			missesBimodal: mb,
			missesHistory: mh,
		};
		bim = update(bim, o);
		table[idx] = update(table[idx], o);
		hist = ((hist << 1) | (o === "T" ? 1 : 0)) & ((1 << h) - 1);
		step.table = [...table];
		out.push(step);
	}
	return out;
}
