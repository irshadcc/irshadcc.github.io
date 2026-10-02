// SVG drawing for PaxosRun.astro (a message sequence diagram with acceptor state) and
// RaftRun.astro (each server's role, term and log). Both return SVG markup so that the server
// render and client-side step updates share one code path.
import type { Frame as PaxosFrame, Scenario as PaxosScenario } from "./paxos";
import type { Frame as RaftFrame } from "./raft";

const esc = (s: string) =>
	s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// ---------- Paxos ----------

const COL = 82;
const BAND = 58;
const HEAD = 34;

/**
 * Lifelines for P1, A1..An, P2 (left to right). Step t shows the messages of frames 1..t, one
 * horizontal band per step: requests go out on the upper half, replies come back on the lower
 * half. The current step is drawn at full strength, earlier ones faded. Below the lifelines, each
 * acceptor's promised number and accepted proposal after step t.
 */
export const drawPaxos = (
	sc: PaxosScenario,
	frames: PaxosFrame[],
	t: number,
) => {
	const n = sc.acceptors;
	const cols = n + sc.values.length;
	const left = 46;
	const width = left * 2 + (cols - 1) * COL;
	const steps = frames.length - 1;
	const bottom = HEAD + steps * BAND + 10;
	const height = bottom + 78;
	const xA = (i: number) => left + (i + 1) * COL;
	const xP = (p: number) => (p === 1 ? left : left + (n + 1) * COL);
	const parts: string[] = [];
	const names = [
		"P1",
		...Array.from({ length: n }, (_, i) => `A${i + 1}`),
		"P2",
	];
	names.forEach((name, i) => {
		const x = left + i * COL;
		const cls = name.startsWith("P") ? "proposer" : "acceptor";
		parts.push(`<text class="lane-name ${cls}" x="${x}" y="16">${name}</text>`);
		parts.push(
			`<line class="lane" x1="${x}" y1="${HEAD - 10}" x2="${x}" y2="${bottom}"></line>`,
		);
	});
	parts.push(
		`<defs><marker id="pxa" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L8 4L0 8z" class="arrowhead"></path></marker></defs>`,
	);
	for (let k = 1; k <= t; k++) {
		const f = frames[k];
		const y0 = HEAD + (k - 1) * BAND;
		const cls = k === t ? "band now" : "band past";
		parts.push(`<g class="${cls}">`);
		if (k === t)
			parts.push(
				`<rect class="band-bg" x="4" y="${y0 - 4}" width="${width - 8}" height="${BAND - 4}" rx="4"></rect>`,
			);
		const p = f.messages[0]?.p;
		if (p !== undefined) {
			const px = xP(p);
			const label = f.messages[0].req;
			const lx = p === 1 ? px + 6 : px - 6;
			parts.push(
				`<text class="req-label" x="${lx}" y="${y0 + 8}" text-anchor="${p === 1 ? "start" : "end"}">${esc(label)}</text>`,
			);
			for (const m of f.messages) {
				const ax = xA(m.a);
				const yReq = y0 + 22;
				const yRep = y0 + 42;
				parts.push(
					`<line class="msg req" x1="${px}" y1="${y0 + 12}" x2="${ax}" y2="${yReq}" marker-end="url(#pxa)"><title>${esc(`${m.from} → ${m.to}: ${m.req}`)}</title></line>`,
				);
				parts.push(
					`<line class="msg rep ${m.ok ? "ok" : "no"}" x1="${ax}" y1="${yReq + 2}" x2="${px}" y2="${yRep}" marker-end="url(#pxa)"><title>${esc(`${m.to} → ${m.from}: ${m.reply}`)}</title></line>`,
				);
				const short = m.ok
					? m.reply.startsWith("promise")
						? m.reply.includes("–")
							? "promise"
							: `promise ${m.reply.slice(m.reply.indexOf(", ") + 2, -1)}`
						: "accepted"
					: `nack ${m.reply.slice(5, -1)}`;
				parts.push(
					`<text class="rep-label ${m.ok ? "ok" : "no"}" x="${ax}" y="${yReq - 3}">${esc(short)}</text>`,
				);
			}
		}
		parts.push("</g>");
	}
	const f = frames[t];
	for (let i = 0; i < n; i++) {
		const a = f.acceptors[i];
		const x = xA(i);
		const acc = a.accepted ? `(${a.accepted.n}, ${a.accepted.v})` : "–";
		const chosenHere = f.chosen && a.accepted && a.accepted.v === f.chosen.v;
		parts.push(
			`<rect class="state${chosenHere ? " chosen" : ""}" x="${x - COL / 2 + 4}" y="${bottom + 8}" width="${COL - 8}" height="40" rx="4"></rect>`,
		);
		parts.push(
			`<text class="state-k" x="${x}" y="${bottom + 22}">promised ${a.promised || "–"}</text>`,
		);
		parts.push(
			`<text class="state-v" x="${x}" y="${bottom + 39}">${esc(acc)}</text>`,
		);
	}
	f.proposers.forEach((pr, j) => {
		const x = xP(j + 1);
		const txt = pr.n ? `n=${pr.n}${pr.value ? `, v=${pr.value}` : ""}` : "idle";
		parts.push(
			`<text class="state-k" x="${x}" y="${bottom + 22}">${esc(txt)}</text>`,
		);
		parts.push(
			`<text class="state-k faint" x="${x}" y="${bottom + 39}">wants ${esc(sc.values[j])}</text>`,
		);
	});
	const verdict = f.chosen
		? `chosen: ${f.chosen.v} (by proposal ${f.chosen.n})`
		: "nothing chosen yet";
	parts.push(
		`<text class="verdict${f.chosen ? " yes" : ""}" x="${width / 2}" y="${height - 8}">${esc(verdict)}</text>`,
	);
	return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Paxos message sequence">${parts.join("")}</svg>`;
};

// ---------- Raft ----------

const CELL = 34;
const ROW = 40;

/** One row per server: name, role, term and vote, then its log (one box per entry, coloured by
 * term). Committed entries (index <= that server's commitIndex) are solid; others are pale with a
 * dashed border. Entries that changed since the previous frame are outlined. */
export const drawRaft = (
	f: RaftFrame,
	prev: RaftFrame | undefined,
	maxLog: number,
) => {
	const labelW = 150;
	const width = labelW + Math.max(maxLog, 4) * CELL + 12;
	const top = 22;
	const height = top + f.servers.length * ROW + 6;
	const parts: string[] = [];
	for (let i = 1; i <= Math.max(maxLog, 4); i++)
		parts.push(
			`<text class="idx" x="${labelW + (i - 0.5) * CELL}" y="14">${i}</text>`,
		);
	f.servers.forEach((s, k) => {
		const y = top + k * ROW;
		const role = !s.up ? "down" : s.role;
		parts.push(`<g class="srv ${role}">`);
		parts.push(`<text class="srv-name" x="6" y="${y + 21}">S${k + 1}</text>`);
		parts.push(
			`<rect class="badge" x="30" y="${y + 8}" width="62" height="18" rx="9"></rect>`,
		);
		parts.push(`<text class="badge-text" x="61" y="${y + 21}">${role}</text>`);
		const vote = s.votedFor === null ? "" : ` · voted S${s.votedFor + 1}`;
		parts.push(`<text class="term" x="98" y="${y + 15}">term ${s.term}</text>`);
		parts.push(
			`<text class="vote" x="98" y="${y + 28}">${esc(vote.slice(3))}</text>`,
		);
		const old = prev?.servers[k];
		s.log.forEach((e, j) => {
			const x = labelW + j * CELL;
			const committed = j + 1 <= s.commit;
			const was = old?.log[j];
			const changed = !was || was.term !== e.term || was.cmd !== e.cmd;
			const cls = `entry t${((e.term - 1) % 6) + 1}${committed ? " committed" : ""}${changed && prev ? " changed" : ""}`;
			parts.push(
				`<g class="${cls}"><rect x="${x + 2}" y="${y + 4}" width="${CELL - 4}" height="${ROW - 10}" rx="3"></rect>` +
					`<text class="cmd" x="${x + CELL / 2}" y="${y + 19}">${esc(e.cmd)}</text>` +
					`<text class="eterm" x="${x + CELL / 2}" y="${y + 30}">t${e.term}</text>` +
					`<title>S${k + 1} index ${j + 1}: ${esc(e.cmd)}, term ${e.term}${committed ? ", committed" : ""}</title></g>`,
			);
		});
		if (s.commit > 0) {
			const cx = labelW + s.commit * CELL;
			parts.push(
				`<line class="commit-mark" x1="${cx}" y1="${y + 2}" x2="${cx}" y2="${y + ROW - 4}"><title>commitIndex = ${s.commit}</title></line>`,
			);
		}
		parts.push("</g>");
	});
	return `<svg viewBox="0 0 ${width} ${height}" width="${Math.round(width * 1.3)}" role="img" aria-label="Raft servers and logs">${parts.join("")}</svg>`;
};
