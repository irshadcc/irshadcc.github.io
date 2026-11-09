// Lays out a parsed circuit (quantum.ts) as columns for CircuitSteps.astro. Each step becomes
// one or more sub-columns: a gate goes into the step's last sub-column unless its qubit span
// overlaps a gate already there. No DOM; the component turns the result into SVG.
import type { Op, Step } from "./quantum";

export const BLOCK = new Set(["ORACLE", "DIFF", "QFT", "IQFT"]);

export interface PlacedOp {
	op: Op;
	x: number;
	w: number;
}

export interface Column {
	step: number;
	x: number;
	w: number;
	ops: PlacedOp[];
}

export interface CircuitLayout {
	columns: Column[];
	/** Caption per step, centred over its columns. */
	captions: {
		step: number;
		x: number;
		text: string;
		error: boolean;
		row: number;
	}[];
	width: number;
	height: number;
	rowY: (q: number) => number;
	labelW: number;
}

export function opWidth(op: Op): number {
	if (BLOCK.has(op.name)) return 54;
	if (["CX", "CZ", "CCX", "CCZ", "SWAP"].includes(op.name)) return 20;
	if (op.paramText !== undefined) return 38;
	return 26;
}

const span = (op: Op) => [Math.min(...op.qubits), Math.max(...op.qubits)];

export function layoutCircuit(
	n: number,
	steps: Step[],
	labelW = 62,
): CircuitLayout {
	const top = 46;
	const row = 38;
	const rowY = (q: number) => top + q * row;
	const columns: Column[] = [];
	const captions: CircuitLayout["captions"] = [];
	let x = labelW + 8;
	steps.forEach((st, k) => {
		const cols: { ops: Op[] }[] = [];
		for (const op of st.ops) {
			const [lo, hi] = span(op);
			const last = cols.at(-1);
			const clash = last?.ops.some((o) => {
				const [a, b] = span(o);
				return !(hi < a || lo > b);
			});
			if (!last || clash) cols.push({ ops: [op] });
			else last.ops.push(op);
		}
		const start = x;
		for (const c of cols) {
			const w = Math.max(...c.ops.map(opWidth));
			columns.push({
				step: k + 1,
				x: x + w / 2,
				w,
				ops: c.ops.map((op) => ({ op, x: x + w / 2, w: opWidth(op) })),
			});
			x += w + 10;
		}
		const labels = [
			...new Set(st.ops.map((o) => o.label).filter((l): l is string => !!l)),
		];
		if (labels.length) {
			const text = labels.join(", ");
			const cx = (start + x - 10) / 2;
			// Two caption rows: drop to the second when this one would overlap the previous.
			const prev = captions.at(-1);
			const halfW = (t: string) => t.length * 2.9;
			const clash =
				prev &&
				prev.row === 0 &&
				prev.x + halfW(prev.text) + 4 > cx - halfW(text);
			captions.push({
				step: k + 1,
				x: cx,
				text,
				error: labels.some((l) => /error/i.test(l)),
				row: clash ? 1 : 0,
			});
		}
		x += 8;
	});
	return {
		columns,
		captions,
		width: x + 6,
		height: rowY(n - 1) + 24,
		rowY,
		labelW,
	};
}

export const GATE_TEXT: Record<string, string> = {
	SDG: "S†",
	TDG: "T†",
	ORACLE: "Oracle",
	DIFF: "Diffuse",
	QFT: "QFT",
	IQFT: "QFT†",
};

const esc = (s: string) =>
	s
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;");

/** Inner SVG markup of the circuit: wires, labels and one <g class="qc-col"> per sub-column. */
export function circuitSvg(
	n: number,
	L: CircuitLayout,
	labels: string[],
	init: string,
): string {
	const out: string[] = [];
	const y = L.rowY;
	for (let q = 0; q < n; q++) {
		out.push(
			`<line class="qc-wire" x1="${L.labelW}" x2="${L.width - 4}" y1="${y(q)}" y2="${y(q)}"/>`,
		);
		out.push(
			`<text class="qc-qlabel" x="${L.labelW - 4}" y="${y(q) + 4}" text-anchor="end">${esc(labels[q])} |${init[q] ?? "0"}⟩</text>`,
		);
	}
	for (const c of L.captions)
		out.push(
			`<text class="qc-caption${c.error ? " err" : ""}" x="${c.x}" y="${12 + 12 * c.row}" text-anchor="middle">${esc(c.text)}</text>`,
		);
	const box = (
		cx: number,
		cy: number,
		w: number,
		text: string,
		sub?: string,
	) => {
		const h = sub ? 30 : 24;
		let s = `<rect class="qc-box" x="${cx - w / 2}" y="${cy - h / 2}" width="${w}" height="${h}" rx="3"/>`;
		s += `<text class="qc-gtext" x="${cx}" y="${cy + (sub ? -1 : 4)}" text-anchor="middle">${esc(text)}</text>`;
		if (sub)
			s += `<text class="qc-ptext" x="${cx}" y="${cy + 10}" text-anchor="middle">${esc(sub)}</text>`;
		return s;
	};
	const dot = (cx: number, q: number) =>
		`<circle class="qc-dot" cx="${cx}" cy="${y(q)}" r="4"/>`;
	const link = (cx: number, qs: number[]) =>
		`<line class="qc-link" x1="${cx}" x2="${cx}" y1="${y(Math.min(...qs))}" y2="${y(Math.max(...qs))}"/>`;
	const target = (cx: number, q: number) =>
		`<circle class="qc-target" cx="${cx}" cy="${y(q)}" r="8"/><line class="qc-link" x1="${cx - 8}" x2="${cx + 8}" y1="${y(q)}" y2="${y(q)}"/><line class="qc-link" x1="${cx}" x2="${cx}" y1="${y(q) - 8}" y2="${y(q) + 8}"/>`;
	const cross = (cx: number, q: number) =>
		`<path class="qc-link" d="M${cx - 5},${y(q) - 5}L${cx + 5},${y(q) + 5}M${cx - 5},${y(q) + 5}L${cx + 5},${y(q) - 5}"/>`;
	for (const col of L.columns) {
		const g: string[] = [];
		const err = col.ops.some((p) => /error/i.test(p.op.label ?? ""));
		g.push(
			`<rect class="qc-colbg" x="${col.x - col.w / 2 - 4}" y="28" width="${col.w + 8}" height="${L.height - 32}" rx="4"/>`,
		);
		for (const { op, x } of col.ops) {
			const q = op.qubits;
			switch (op.name) {
				case "CX":
					g.push(link(x, q), dot(x, q[0]), target(x, q[1]));
					break;
				case "CCX":
					g.push(link(x, q), dot(x, q[0]), dot(x, q[1]), target(x, q[2]));
					break;
				case "CZ":
				case "CCZ":
					g.push(link(x, q), ...q.map((v) => dot(x, v)));
					break;
				case "SWAP":
					g.push(link(x, q), cross(x, q[0]), cross(x, q[1]));
					break;
				case "CY":
				case "CH":
				case "CP":
					g.push(
						link(x, q),
						dot(x, q[0]),
						box(x, y(q[1]), opWidth(op), op.name.slice(1), op.paramText),
					);
					break;
				case "M":
					for (const v of q) {
						const cy = y(v);
						g.push(
							`<rect class="qc-box" x="${x - 13}" y="${cy - 12}" width="26" height="24" rx="3"/>`,
						);
						g.push(
							`<path class="qc-link" fill="none" d="M${x - 8},${cy + 5}A9,9 0 0 1 ${x + 8},${cy + 5}M${x},${cy + 6}L${x + 6},${cy - 6}"/>`,
						);
					}
					break;
				default:
					if (BLOCK.has(op.name)) {
						const lo = y(Math.min(...q)) - 14;
						const hi = y(Math.max(...q)) + 14;
						const w = opWidth(op);
						g.push(
							`<rect class="qc-block" x="${x - w / 2}" y="${lo}" width="${w}" height="${hi - lo}" rx="4"/>`,
						);
						const mid = (lo + hi) / 2;
						const sub = op.marked
							? op.marked
									.map((m) => `|${m.toString(2).padStart(q.length, "0")}⟩`)
									.join(" ")
							: "";
						g.push(
							`<text class="qc-gtext" x="${x}" y="${mid + (sub ? -2 : 4)}" text-anchor="middle">${GATE_TEXT[op.name]}</text>`,
						);
						if (sub)
							g.push(
								`<text class="qc-ptext" x="${x}" y="${mid + 11}" text-anchor="middle">${esc(sub)}</text>`,
							);
						// Wires that pass through the block untouched get a small gap marker.
						for (let v = Math.min(...q); v <= Math.max(...q); v++)
							if (!q.includes(v))
								g.push(
									`<text class="qc-ptext" x="${x}" y="${y(v) + 3}" text-anchor="middle">(skip)</text>`,
								);
					} else {
						g.push(
							box(
								x,
								y(q[0]),
								opWidth(op),
								GATE_TEXT[op.name] ?? op.name,
								op.paramText,
							),
						);
					}
			}
		}
		out.push(
			`<g class="qc-col${err ? " err-gate" : ""}" data-step="${col.step}" role="button" tabindex="0" aria-label="Go to step ${col.step}">${g.join("")}</g>`,
		);
	}
	return out.join("");
}
