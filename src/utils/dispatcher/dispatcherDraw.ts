// SVG drawing for KeySetSteps.astro (a dispatch key set as bits) and DispatchTable.astro (an
// operator's 138-slot dispatch table). Both return SVG markup so the server render and the
// client-side step updates share one code path.
import {
	BACKENDS,
	type Entry,
	FUNCTIONALITIES,
	NUM_BACKENDS,
	SLOT_KEYS,
	backendBit,
	functionalityBit,
} from "./dispatchKeys";
import type { TableRow } from "./dispatcherFigures";

const esc = (s: string) =>
	s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// ---------- Key set bit strip ----------

const CELL = 13;
const NF = FUNCTIONALITIES.length - 1; // 47 functionality bits (Undefined has none)

/**
 * Two rows: functionality bits (bit 62 on the left, bit 16 on the right) and backend bits (bit 15
 * to bit 0). Bits in `labels` get their key name. `prev` marks bits added or removed by this stage;
 * `top` marks the highest functionality and backend bits used by the table lookup.
 */
export const drawBits = (
	ks: bigint,
	prev: bigint | undefined,
	labels: bigint,
	top?: { f: number; b: number },
) => {
	const left = 6;
	const labelH = 118;
	const rowA = labelH;
	const rowB = rowA + CELL + 34;
	const width = left * 2 + NF * CELL;
	const height = rowB + CELL + 52;
	const parts: string[] = [];
	const cell = (
		x: number,
		y: number,
		bit: bigint,
		name: string,
		bitIndex: number,
	) => {
		const on = (ks & bit) !== 0n;
		const was = prev === undefined ? on : (prev & bit) !== 0n;
		const cls = on ? (was ? "on" : "on added") : was ? "off removed" : "off";
		parts.push(
			`<rect class="bit ${cls}" x="${x + 0.5}" y="${y}" width="${CELL - 1}" height="${CELL}" rx="2"><title>bit ${bitIndex}: ${esc(name)}</title></rect>`,
		);
	};
	parts.push(
		`<text class="row-label" x="${left}" y="${rowA - 6 - 92}">functionality bits 62 … 16 (higher = higher priority)</text>`,
	);
	for (let i = 0; i < NF; i++) {
		const f = NF - i; // leftmost cell is the highest functionality
		const x = left + i * CELL;
		const bit = functionalityBit(f);
		cell(x, rowA, bit, FUNCTIONALITIES[f], NUM_BACKENDS + f - 1);
		if (labels & bit) {
			const cx = x + CELL / 2;
			parts.push(
				`<text class="bit-name${ks & bit ? " set" : ""}" transform="translate(${cx + 3},${rowA - 4}) rotate(-60)">${esc(FUNCTIONALITIES[f])}</text>`,
			);
		}
		if (top && top.f === f)
			parts.push(
				`<path class="marker" d="M${x + CELL / 2} ${rowA + CELL + 2} l-4 7 h8 z"><title>highest functionality bit</title></path>`,
			);
	}
	parts.push(
		`<text class="row-label" x="${left}" y="${rowB - 6}">backend bits 15 … 0</text>`,
	);
	for (let i = 0; i < NUM_BACKENDS; i++) {
		const b = NUM_BACKENDS - 1 - i;
		const x = left + i * CELL;
		const bit = backendBit(b);
		cell(x, rowB, bit, `${BACKENDS[b]}Bit`, b);
		if (labels & bit) {
			const cx = x + CELL / 2;
			parts.push(
				`<text class="bit-name${ks & bit ? " set" : ""}" transform="translate(${cx - 3},${rowB + CELL + 5}) rotate(60)">${esc(BACKENDS[b])}</text>`,
			);
		}
		if (top && top.b === b)
			parts.push(
				`<path class="marker" d="M${x + CELL / 2} ${rowB - 2} l-4 -7 h8 z"><title>highest backend bit</title></path>`,
			);
	}
	return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Dispatch key set bits">${parts.join("")}</svg>`;
};

// ---------- Dispatch table grid ----------

const SHORT: Record<string, string> = {
	Undefined: "Undef",
	FPGA: "FPGA",
	Vulkan: "Vulk",
	Metal: "Metal",
	CustomRNGKeyId: "RNG",
	MkldnnCPU: "Mkldnn",
	BackendSelect: "BkSel",
	Python: "Py",
	Fake: "Fake",
	FuncTorchDynamicLayerBackMode: "DLBack",
	Functionalize: "Func",
	Named: "Named",
	Conjugate: "Conj",
	Negative: "Neg",
	ZeroTensor: "Zero",
	ADInplaceOrView: "ADInV",
	AutogradOther: "AGOth",
	AutogradNestedTensor: "AGNest",
	Tracer: "Tracer",
	FuncTorchBatched: "FTBat",
	BatchedNestedTensor: "BatNT",
	FuncTorchVmapMode: "FTVmap",
	Batched: "Batch",
	VmapMode: "Vmap",
	FuncTorchGradWrapper: "FTGrad",
	DeferredInit: "Defer",
	PythonTLSSnapshot: "PyTLS",
	FuncTorchDynamicLayerFrontMode: "DLFront",
	TESTING_ONLY_GenericWrapper: "TestW",
	TESTING_ONLY_GenericMode: "TestM",
	PreDispatch: "PreDsp",
	PythonDispatcher: "PyDsp",
	PrivateUse1: "PU1",
	PrivateUse2: "PU2",
	PrivateUse3: "PU3",
};

/** Cell label: the backend for per-backend rows (cell i is backend i), else a short key name. */
const shortName = (key: string, backend?: number) => {
	if (backend !== undefined)
		return SHORT[BACKENDS[backend]] ?? BACKENDS[backend];
	if (key.startsWith("Autocast"))
		return `AC${SHORT[key.slice(8)] ?? key.slice(8)}`;
	return SHORT[key] ?? key;
};

/** CSS class for an entry: what kind of kernel fills the slot. */
export const entryClass = (e: Entry) => {
	if (e.kernel?.fallthrough) return "fallthrough";
	switch (e.source) {
		case "kernel":
			return "kernel";
		case "default backend kernel":
		case "math kernel":
		case "nested kernel":
		case "batched kernel":
			return "alias";
		case "autograd kernel":
			return "autograd";
		case "backend fallback":
			return "fallback";
		default:
			return "missing";
	}
};

const TCELL_W = 36;
const TCELL_H = 22;
const LABEL_W = 116;

export const drawTable = (
	table: Entry[],
	rows: TableRow[],
	changed: number[],
	highlight: string[] = [],
) => {
	const width = LABEL_W + NUM_BACKENDS * TCELL_W + 4;
	const rowH = TCELL_H + 6;
	const height = rows.length * rowH + 4;
	const changedSet = new Set(changed);
	const parts: string[] = [];
	rows.forEach((row, r) => {
		const y = 2 + r * rowH;
		const label = row.perBackend
			? `${row.label} ${row.slots[0]}–${row.slots[row.slots.length - 1]}`
			: `slot ${row.label}`;
		parts.push(
			`<text class="row-label" x="${LABEL_W - 6}" y="${y + TCELL_H / 2 + 3.5}">${esc(label)}</text>`,
		);
		row.slots.forEach((slot, i) => {
			const e = table[slot];
			const x = LABEL_W + i * TCELL_W;
			const cls = `${entryClass(e)}${changedSet.has(slot) ? " changed" : ""}${highlight.includes(e.key) ? " hl" : ""}`;
			const site = e.kernel ? e.kernel.site : "";
			parts.push(
				`<g class="slot ${cls}" data-slot="${slot}" data-key="${esc(e.key)}" data-src="${esc(e.source)}${e.kernel?.fallthrough ? " (fallthrough)" : ""}" data-site="${esc(site)}" tabindex="0">` +
					`<rect x="${x + 1}" y="${y}" width="${TCELL_W - 2}" height="${TCELL_H}" rx="2.5"></rect>` +
					`<text x="${x + TCELL_W / 2}" y="${y + TCELL_H / 2 + 3}">${esc(shortName(SLOT_KEYS[slot], row.perBackend ? i : undefined))}</text>` +
					`<title>${slot}: ${esc(e.key)} · ${esc(e.source)}${e.kernel?.fallthrough ? " (fallthrough)" : ""}${site ? ` · ${esc(site)}` : ""}</title></g>`,
			);
		});
	});
	return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Operator dispatch table">${parts.join("")}</svg>`;
};

// ---------- The 64-bit word of a key set ----------

const WCELL = 20;
const WROW = 32;

/**
 * The key set's uint64_t as two rows of 32 bits (63 … 32 on top, 31 … 0 below), each cell tinted
 * by its region: bit 63 unused, bits 62 … 16 functionality, bits 15 … 0 backend. Set bits are
 * filled. Names sit above the top row and below the bottom row; braces between the rows mark the
 * regions.
 */
export const drawWord = (ks: bigint) => {
	const left = 8;
	const labelH = 128;
	const rowA = labelH;
	const braceA = rowA + WCELL + 4;
	const rowB = braceA + 62;
	const braceB = rowB - 4;
	const width = left + WROW * WCELL + 44;
	const height = rowB + WCELL + labelH + 4;
	const parts: string[] = [];
	const nameOf = (bit: number) =>
		bit === 63
			? "(unused)"
			: bit >= NUM_BACKENDS
				? FUNCTIONALITIES[bit - NUM_BACKENDS + 1]
				: BACKENDS[bit];
	const regionOf = (bit: number) =>
		bit === 63 ? "unused" : bit >= NUM_BACKENDS ? "func" : "backend";
	for (let bit = 63; bit >= 0; bit--) {
		const top = bit >= WROW;
		const i = top ? 63 - bit : WROW - 1 - bit;
		const x = left + i * WCELL;
		const y = top ? rowA : rowB;
		const on = (ks & (1n << BigInt(bit))) !== 0n;
		const name = nameOf(bit);
		const detail =
			bit === 63
				? "bit 63: unused"
				: bit >= NUM_BACKENDS
					? `bit ${bit}: functionality ${name} (f = ${bit - NUM_BACKENDS + 1})`
					: `bit ${bit}: backend ${name}`;
		parts.push(
			`<g class="wbit ${regionOf(bit)}${on ? " on" : ""}"><rect x="${x + 0.5}" y="${y}" width="${WCELL - 1}" height="${WCELL}" rx="2"></rect>` +
				`<text class="idx" x="${x + WCELL / 2}" y="${y + WCELL / 2 + 3}">${bit}</text><title>${esc(detail)}</title></g>`,
		);
		const cx = x + WCELL / 2;
		parts.push(
			top
				? `<text class="wname${on ? " set" : ""}" transform="translate(${cx + 2},${y - 4}) rotate(-60)">${esc(name)}</text>`
				: `<text class="wname${on ? " set" : ""}" transform="translate(${cx - 2},${y + WCELL + 4}) rotate(60)">${esc(name)}</text>`,
		);
	}
	// A brace from cell i0 to cell i1 (inclusive), opening towards the row at y (down = row below).
	const brace = (
		i0: number,
		i1: number,
		y: number,
		down: boolean,
		cls: string,
		label: string,
	) => {
		const x0 = left + i0 * WCELL + 2;
		const x1 = left + (i1 + 1) * WCELL - 2;
		const d = down ? 6 : -6;
		const mid = (x0 + x1) / 2;
		parts.push(
			`<path class="brace ${cls}" d="M${x0} ${y} v${-d} H${mid - 4} l4 ${-d} l4 ${d} H${x1} v${d}"></path>`,
			`<text class="region ${cls}" x="${mid}" y="${y - 2 * d + (down ? -3 : 9)}">${esc(label)}</text>`,
		);
	};
	brace(0, 0, braceA, false, "unused", "63");
	brace(1, WROW - 1, braceA, false, "func", "functionality bits 62 … 32");
	brace(0, 15, braceB, true, "func", "functionality bits 31 … 16");
	brace(16, WROW - 1, braceB, true, "backend", "backend bits 15 … 0");
	return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="The 64 bits of a DispatchKeySet">${parts.join("")}</svg>`;
};
