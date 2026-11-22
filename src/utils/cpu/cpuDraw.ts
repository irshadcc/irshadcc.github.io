// SVG drawing for the CPU figures. Classes are styled by cpu.css.
import type { BranchStep } from "./branch";
import { type CacheGeom, type Placement, numSets, split } from "./cacheSets";
import { type Instr, type Machine, type Simulation, cellState } from "./ooo";

const esc = (s: string) =>
	s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const CELL = 15;
const ROW = 18;
const LABEL_W = 178;
const HEAD = 30;

/** Instructions × cycles diagram. Every cell carries data-c so the stepper can hide the future. */
export function drawPipeline(prog: readonly Instr[], sim: Simulation): string {
	const w = LABEL_W + sim.total * CELL + 8;
	const h = HEAD + prog.length * ROW + 6;
	const parts: string[] = [];
	parts.push(
		`<text class="t-head" x="4" y="12">instruction</text><text class="t-head" x="${LABEL_W}" y="12">cycle →</text>`,
	);
	for (let c = 0; c < sim.total; c++) {
		if (c % 5 === 0)
			parts.push(
				`<text class="t-small" x="${LABEL_W + c * CELL + CELL / 2}" y="25" text-anchor="middle">${c}</text>`,
			);
		if (sim.cycles[c].stall)
			parts.push(
				`<text class="stall-mark" data-c="${c}" x="${LABEL_W + c * CELL + CELL / 2}" y="${h - 0}">!</text>`,
			);
	}
	prog.forEach((ins, i) => {
		const y = HEAD + i * ROW;
		parts.push(
			`<g class="prow${ins.tag === "miss" ? " miss" : ""}"><text class="t-small" x="4" y="${y + 12}">${i}</text><text class="t-mono" x="20" y="${y + 12}">${esc(ins.text)}</text></g>`,
		);
		for (let c = 0; c < sim.total; c++) {
			const s = cellState(sim.timing[i], c);
			if (!s) continue;
			const x = LABEL_W + c * CELL;
			const letter = s === "e" ? "" : s === "w" ? "" : s === "r" ? "" : s;
			parts.push(
				`<g class="pc ${s}" data-c="${c}"><rect x="${x}" y="${y + 2}" width="${CELL}" height="${ROW - 3}" rx="2"/>${letter ? `<text x="${x + CELL / 2}" y="${y + 12.5}">${letter}</text>` : ""}</g>`,
			);
		}
	});
	parts.push(
		`<rect class="pcol" x="${LABEL_W}" y="${HEAD - 1}" width="${CELL}" height="${prog.length * ROW + 2}" rx="2"/>`,
	);
	return `<svg viewBox="0 0 ${w} ${h + 4}" width="${w}" height="${h + 4}" role="img">${parts.join("")}</svg>`;
}

/** ROB slots and ports in one cycle. */
export function drawStrip(sim: Simulation, m: Machine, c: number): string {
	const st = sim.cycles[c];
	const sw = 22;
	const parts: string[] = [];
	parts.push(
		`<text class="t-head" x="0" y="11">ROB (${st.rob.length}/${m.rob})</text>`,
	);
	for (let k = 0; k < m.rob; k++) {
		const i = st.rob[k];
		const x = 78 + k * sw;
		parts.push(
			`<rect class="slot${i !== undefined ? " used" : ""}${k === 0 && i !== undefined ? " head" : ""}" x="${x}" y="0" width="${sw - 2}" height="16" rx="3"/>${i !== undefined ? `<text class="t-mono" x="${x + (sw - 2) / 2}" y="12" text-anchor="middle" style="font-size:9px">${i}</text>` : ""}`,
		);
	}
	const py = 26;
	parts.push(`<text class="t-head" x="0" y="${py + 11}">ports</text>`);
	m.ports.forEach((p, k) => {
		const i = st.portUse[k];
		const x = 78 + k * 58;
		parts.push(
			`<rect class="port${i >= 0 ? " busy" : ""}" x="${x}" y="${py}" width="54" height="16" rx="3"/><text class="t-mono" x="${x + 27}" y="${py + 12}" text-anchor="middle" style="font-size:9px">${p.name}${i >= 0 ? ` ← ${i}` : ""}</text>`,
		);
	});
	const w = Math.max(78 + m.rob * sw, 78 + m.ports.length * 58) + 4;
	return `<svg viewBox="0 0 ${w} 46" width="${w}" height="46" role="img">${parts.join("")}</svg>`;
}

/** Outcomes so far, the two predictors' guesses, and the history counter table. */
export function drawBranch(
	steps: readonly BranchStep[],
	at: number,
	h: number,
): string {
	const bw = 22;
	const x0 = 92;
	const parts: string[] = [];
	const rows: [string, (s: BranchStep) => { t: string; miss: boolean }][] = [
		["outcome", (s) => ({ t: s.outcome, miss: false })],
		["one counter", (s) => ({ t: s.bimodal.predict, miss: s.bimodal.miss })],
		[
			`last ${h} outcomes`,
			(s) => ({ t: s.history.predict, miss: s.history.miss }),
		],
	];
	rows.forEach(([label, f], r) => {
		const y = r * 26;
		parts.push(`<text class="t-head" x="0" y="${y + 15}">${label}</text>`);
		steps.forEach((s, i) => {
			if (i > at) return;
			const v = f(s);
			const x = x0 + i * bw;
			parts.push(
				`<rect class="bo ${v.t}${v.miss ? " miss" : ""}${i === at && r === 0 ? " cur" : ""}" x="${x}" y="${y}" width="${bw - 3}" height="20" rx="3"/><text class="t-mono" x="${x + (bw - 3) / 2}" y="${y + 14}" text-anchor="middle">${v.t}</text>`,
			);
		});
	});
	const ty = 3 * 26 + 18;
	parts.push(`<text class="t-head" x="0" y="${ty + 22}">history table</text>`);
	const s = steps[at];
	const before = at > 0 ? steps[at - 1].table : new Array(1 << h).fill(2);
	for (let k = 0; k < 1 << h; k++) {
		const x = x0 + k * 46;
		const lit = k === s.history.index;
		const pat = k
			.toString(2)
			.padStart(h, "0")
			.replace(/1/g, "T")
			.replace(/0/g, "N");
		const c = before[k];
		parts.push(
			`<text class="t-mono" x="${x + 21}" y="${ty + 6}" text-anchor="middle" style="font-size:9px">${pat}</text><rect class="ctr${lit ? " lit" : ""}" x="${x}" y="${ty + 10}" width="42" height="20" rx="3"/><rect class="ctr-fill" x="${x + 1}" y="${ty + 11}" width="${(40 * c) / 3}" height="18" rx="2"/><text class="t-mono" x="${x + 21}" y="${ty + 24}" text-anchor="middle">${c}</text>`,
		);
	}
	const w = Math.max(x0 + steps.length * bw, x0 + (1 << h) * 46) + 4;
	return `<svg viewBox="0 0 ${w} ${ty + 34}" width="${w}" height="${ty + 34}" role="img">${parts.join("")}</svg>`;
}

/** Sets × ways grid of a cache with the strided lines placed; overflowing sets in red. */
export function drawCache(
	g: CacheGeom,
	p: Placement,
	stride: number,
	pick: number,
): string {
	const sets = numSets(g);
	const cols = Math.min(sets, 64);
	const groups = Math.ceil(sets / cols);
	const cw = 9;
	const parts: string[] = [];
	const bandH = g.ways * cw + 20;
	for (let b = 0; b < groups; b++) {
		const yb = 30 + b * bandH;
		parts.push(
			`<text class="t-small" x="0" y="${yb - 4}">sets ${b * cols}–${b * cols + cols - 1}</text>`,
		);
		for (let k = 0; k < cols; k++) {
			const set = b * cols + k;
			const n = p.perSet[set] ?? 0;
			const x = 44 + k * cw;
			for (let w = 0; w < g.ways; w++) {
				const on = w < n;
				parts.push(
					`<rect class="way${on ? (n > g.ways ? " over" : " on") : ""}" x="${x}" y="${yb + w * cw}" width="${cw - 1}" height="${cw - 1}"/>`,
				);
			}
			if (n > g.ways)
				parts.push(
					`<text class="setcount" x="${x + cw / 2}" y="${yb - 4 + 0}">${n}</text>`,
				);
		}
		if (b === 0)
			parts.push(
				`<text class="t-small" x="0" y="${yb + 9}">way 0</text><text class="t-small" x="0" y="${yb + g.ways * cw}">way ${g.ways - 1}</text>`,
			);
	}
	// address bits of one line
	const addr = pick * stride;
	const sp = split(addr, g);
	const nb = 20;
	const ay = 30 + groups * bandH + 8;
	parts.push(`<text class="t-head" x="0" y="${ay + 12}">line ${pick}</text>`);
	for (let k = 0; k < nb; k++) {
		const bit = nb - 1 - k;
		const kind =
			bit < sp.offBits ? "off" : bit < sp.offBits + sp.setBits ? "set" : "tag";
		const x = 44 + k * 14;
		parts.push(
			`<rect class="bitbox ${kind}" x="${x}" y="${ay}" width="13" height="17" rx="2"/><text class="t-mono" x="${x + 6.5}" y="${ay + 12.5}" text-anchor="middle">${Math.floor(addr / 2 ** bit) % 2}</text>`,
		);
	}
	parts.push(
		`<text class="t-mono" x="${44 + nb * 14 + 8}" y="${ay + 12.5}">0x${addr.toString(16)} → tag ${sp.tag}, set ${sp.set}, offset ${sp.offset}</text>`,
	);
	const w = Math.max(44 + cols * cw, 44 + nb * 14 + 300) + 4;
	return `<svg viewBox="0 0 ${w} ${ay + 24}" width="${w}" height="${ay + 24}" role="img">${parts.join("")}</svg>`;
}
