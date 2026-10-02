// A sequence diagram drawn as SVG, for SequenceSteps.astro: one vertical lifeline per
// participant, one row per message, top to bottom in time. A message is an arrow between two
// lifelines, a self-step on one lifeline (from === to), or a lost message (lost: true, drawn as
// an arrow that stops halfway with a cross). Every row carries data-i so the component can show
// rows 0..i and highlight row i.

export interface Participant {
	id: string;
	label: string;
	/** Second line under the label, e.g. "Visa / Mastercard". */
	sub?: string;
}

export interface Message {
	from: string;
	to: string;
	/** Text on the arrow. */
	label: string;
	/** Short label for the step's timeline segment. */
	step: string;
	/** What happens in this step, shown under the figure. HTML allowed. */
	note: string;
	/** Dashed arrow: a response rather than a request. */
	reply?: boolean;
	/** The message never arrives. */
	lost?: boolean;
	/** Colour group: 0 = neutral, 1 = good, 2 = bad, 3 = money. */
	tone?: 0 | 1 | 2 | 3;
	/** Start a new phase above this row with this heading. */
	phase?: string;
}

export interface SequenceLayout {
	width: number;
	height: number;
	svg: string;
}

const esc = (s: string) =>
	s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

export function validateSequence(
	parts: readonly Participant[],
	msgs: readonly Message[],
): string | undefined {
	const ids = new Set(parts.map((p) => p.id));
	for (const [i, m] of msgs.entries()) {
		if (!ids.has(m.from))
			return `message ${i}: unknown participant "${m.from}"`;
		if (!ids.has(m.to)) return `message ${i}: unknown participant "${m.to}"`;
	}
	return undefined;
}

export function drawSequence(
	parts: readonly Participant[],
	msgs: readonly Message[],
	opts: { colWidth?: number; rowHeight?: number; id?: string } = {},
): SequenceLayout {
	const col = opts.colWidth ?? 112;
	const row = opts.rowHeight ?? 34;
	const phaseH = 20;
	const head = 46;
	const pad = 8;
	const width = parts.length * col;
	const x = new Map(parts.map((p, i) => [p.id, i * col + col / 2]));
	const marker = `${opts.id ?? "seq"}-arrow`;

	let y = head + 14;
	const rows: string[] = [];
	for (const [i, m] of msgs.entries()) {
		if (m.phase) {
			rows.push(
				`<g class="phase" data-i="${i}"><line x1="${pad}" x2="${width - pad}" y1="${y + 4}" y2="${y + 4}"/><text x="${pad + 2}" y="${y + 16}">${esc(m.phase)}</text></g>`,
			);
			y += phaseH;
		}
		const x1 = x.get(m.from) as number;
		const x2 = x.get(m.to) as number;
		const ly = y + row - 10;
		const cls = `msg t${m.tone ?? 0}${m.reply ? " reply" : ""}${m.lost ? " lost" : ""}`;
		let shape: string;
		let tx: number;
		let anchor = "middle";
		if (x1 === x2) {
			// Self-step: a small loop to the right of the lifeline.
			const w = 18;
			shape = `<path class="line" d="M${x1} ${ly - 8} h${w} v10 h${-w + 3}" marker-end="url(#${marker}${m.tone ?? 0})"/>`;
			tx = x1 + w + 4;
			anchor = "start";
		} else if (m.lost) {
			const mid = (x1 + x2) / 2;
			const c = 5;
			shape = `<line class="line" x1="${x1}" x2="${mid}" y1="${ly}" y2="${ly}"/><path class="cross" d="M${mid - c} ${ly - c} L${mid + c} ${ly + c} M${mid - c} ${ly + c} L${mid + c} ${ly - c}"/>`;
			tx = (x1 + mid) / 2;
		} else {
			const dir = Math.sign(x2 - x1);
			shape = `<line class="line" x1="${x1}" x2="${x2 - dir * 3}" y1="${ly}" y2="${ly}" marker-end="url(#${marker}${m.tone ?? 0})"/>`;
			tx = (x1 + x2) / 2;
		}
		rows.push(
			`<g class="${cls}" data-i="${i}"><rect class="hit" x="0" y="${y}" width="${width}" height="${row}"/>${shape}<text class="lbl" x="${tx}" y="${ly - 6}" text-anchor="${anchor}">${esc(m.label)}</text></g>`,
		);
		y += row;
	}
	const height = y + 10;
	const heads = parts
		.map((p) => {
			const cx = x.get(p.id) as number;
			return `<g class="part"><rect x="${cx - col / 2 + 5}" y="4" width="${col - 10}" height="${head - 8}" rx="6"/><text class="pl" x="${cx}" y="${p.sub ? 21 : 28}">${esc(p.label)}</text>${p.sub ? `<text class="ps" x="${cx}" y="35">${esc(p.sub)}</text>` : ""}</g><line class="life" x1="${cx}" x2="${cx}" y1="${head - 4}" y2="${height - 4}"/>`;
		})
		.join("");
	// One arrowhead per tone: markers don't inherit the colour of the line that uses them.
	const defs = `<defs>${[0, 1, 2, 3]
		.map(
			(t) =>
				`<marker id="${marker}${t}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head t${t}" d="M0 0 L8 4 L0 8 z"/></marker>`,
		)
		.join("")}</defs>`;
	return {
		width,
		height,
		svg: `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img">${defs}${heads}${rows.join("")}</svg>`,
	};
}
