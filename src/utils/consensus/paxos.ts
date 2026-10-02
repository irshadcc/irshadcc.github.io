// Single-decree Paxos, following Lamport, "Paxos Made Simple" (2001), section 2.2. A scenario is
// a list of proposer actions; each action sends one kind of request to some acceptors, which
// answer immediately (an acceptor that is not in `to` never hears of the request, as if the
// message were lost). run() applies the acceptor and proposer rules and records one frame per
// action, which PaxosRun.astro draws. Proposal numbers are 10 * round + proposer id, so different
// proposers never use the same number.

export interface Proposal {
	n: number;
	v: string;
}

export interface AcceptorState {
	/** Highest prepare request number this acceptor has responded to (0 = none). */
	promised: number;
	/** Highest-numbered proposal it has accepted, if any. */
	accepted?: Proposal;
}

export interface ProposerState {
	/** The value this proposer wants, used if phase 1 reports no accepted proposal. */
	own: string;
	/** Proposal number of the current attempt (0 = none yet). */
	n: number;
	/** Acceptors that promised for n, with what they reported. */
	promises: Map<number, Proposal | undefined>;
	/** The value fixed by phase 1 for proposal n, once a majority promised. */
	value?: string;
}

export type Action =
	| { kind: "prepare"; p: number; n: number; to: number[]; note?: string }
	| { kind: "accept"; p: number; n: number; to: number[]; note?: string };

export interface Message {
	from: string;
	to: string;
	/** Request label, e.g. "prepare(11)". */
	req: string;
	/** Reply label, e.g. "promise(11, -)" or "nack(12)". */
	reply: string;
	ok: boolean;
	p: number;
	a: number;
}

export interface Frame {
	title: string;
	note: string;
	messages: Message[];
	acceptors: AcceptorState[];
	proposers: { n: number; value?: string; promises: number }[];
	/** The value chosen so far (accepted by a majority under one proposal number), if any. */
	chosen?: Proposal;
}

export interface Scenario {
	id: string;
	label: string;
	summary: string;
	acceptors: number;
	/** Each proposer's own value; proposer ids are 1-based. */
	values: string[];
	actions: Action[];
}

const fmt = (p?: Proposal) => (p ? `(${p.n}, ${p.v})` : "–");

export const run = (s: Scenario): Frame[] => {
	const majority = Math.floor(s.acceptors / 2) + 1;
	const acc: AcceptorState[] = Array.from({ length: s.acceptors }, () => ({
		promised: 0,
	}));
	const props: ProposerState[] = s.values.map((own) => ({
		own,
		n: 0,
		promises: new Map(),
	}));
	// Every (acceptor, proposal) acceptance that ever happened, for the learner's view.
	const acceptances = new Map<number, Set<number>>();
	const values = new Map<number, string>();
	let chosen: Proposal | undefined;

	const snapshot = (
		title: string,
		note: string,
		messages: Message[],
	): Frame => ({
		title,
		note,
		messages,
		acceptors: acc.map((a) => ({
			promised: a.promised,
			accepted: a.accepted && { ...a.accepted },
		})),
		proposers: props.map((p) => ({
			n: p.n,
			value: p.value,
			promises: p.promises.size,
		})),
		chosen: chosen && { ...chosen },
	});

	const frames: Frame[] = [
		snapshot("Start", "No acceptor has promised or accepted anything.", []),
	];
	for (const act of s.actions) {
		const prop = props[act.p - 1];
		const pname = `P${act.p}`;
		const messages: Message[] = [];
		if (act.kind === "prepare") {
			if (act.n % 10 !== act.p)
				throw new Error(`P${act.p} may not use number ${act.n}`);
			if (act.n <= prop.n)
				throw new Error(
					`P${act.p} must use a new, higher number than ${prop.n}`,
				);
			prop.n = act.n;
			prop.promises = new Map();
			prop.value = undefined;
			for (const i of act.to) {
				const a = acc[i];
				// Phase 1b: respond only to a prepare numbered higher than any already answered.
				const ok = act.n > a.promised;
				if (ok) {
					a.promised = act.n;
					prop.promises.set(i, a.accepted && { ...a.accepted });
				}
				messages.push({
					from: pname,
					to: `A${i + 1}`,
					req: `prepare(${act.n})`,
					reply: ok
						? `promise(${act.n}, ${fmt(a.accepted)})`
						: `nack(${a.promised})`,
					ok,
					p: act.p,
					a: i,
				});
			}
			if (prop.promises.size >= majority) {
				// Phase 2a's value rule: the highest-numbered accepted proposal among the promises.
				let best: Proposal | undefined;
				for (const r of prop.promises.values())
					if (r && (!best || r.n > best.n)) best = r;
				prop.value = best ? best.v : prop.own;
			}
		} else {
			if (act.n !== prop.n || prop.value === undefined)
				throw new Error(`P${act.p} has no majority of promises for ${act.n}`);
			const v = prop.value;
			values.set(act.n, v);
			for (const i of act.to) {
				const a = acc[i];
				// Phase 2b: accept unless a prepare numbered higher than n was answered.
				const ok = act.n >= a.promised;
				if (ok) {
					a.promised = act.n;
					a.accepted = { n: act.n, v };
					const set = acceptances.get(act.n) ?? new Set<number>();
					set.add(i);
					acceptances.set(act.n, set);
				}
				messages.push({
					from: pname,
					to: `A${i + 1}`,
					req: `accept(${act.n}, ${v})`,
					reply: ok ? `accepted(${act.n})` : `nack(${a.promised})`,
					ok,
					p: act.p,
					a: i,
				});
			}
			for (const [n, set] of acceptances) {
				if (set.size < majority) continue;
				const p = { n, v: values.get(n) as string };
				if (chosen && chosen.v !== p.v)
					throw new Error(
						`safety violated: ${chosen.v} and ${p.v} both chosen`,
					);
				if (!chosen || n < chosen.n) chosen = p;
			}
		}
		const title =
			act.kind === "prepare"
				? `${pname} sends prepare(${act.n}) to ${act.to.map((i) => `A${i + 1}`).join(", ")}`
				: `${pname} sends accept(${act.n}, ${prop.value}) to ${act.to.map((i) => `A${i + 1}`).join(", ")}`;
		frames.push(snapshot(title, act.note ?? "", messages));
	}
	return frames;
};

const ALL = [0, 1, 2, 3, 4];

export const SCENARIOS: Scenario[] = [
	{
		id: "basic",
		label: "One proposer",
		summary: "P1 proposes X to five acceptors and nothing goes wrong.",
		acceptors: 5,
		values: ["X", "Y"],
		actions: [
			{
				kind: "prepare",
				p: 1,
				n: 11,
				to: ALL,
				note: "Phase 1. No acceptor has answered a prepare yet, so all five promise never to accept a proposal numbered below 11, and report that they have accepted nothing.",
			},
			{
				kind: "accept",
				p: 1,
				n: 11,
				to: ALL,
				note: "Phase 2. No promise reported an accepted proposal, so P1 is free to propose its own value X. Once three of five acceptors accept (11, X), X is chosen.",
			},
		],
	},
	{
		id: "adopt",
		label: "A second proposer",
		summary:
			"X is chosen by A1–A3; then P2 runs phase 1 with A3–A5 and must propose X too.",
		acceptors: 5,
		values: ["X", "Y"],
		actions: [
			{
				kind: "prepare",
				p: 1,
				n: 11,
				to: [0, 1, 2],
				note: "P1 only needs a majority: A1, A2 and A3 promise.",
			},
			{
				kind: "accept",
				p: 1,
				n: 11,
				to: [0, 1, 2],
				note: "A1, A2 and A3 accept (11, X): X is chosen. A4 and A5 have heard nothing.",
			},
			{
				kind: "prepare",
				p: 2,
				n: 12,
				to: [2, 3, 4],
				note: "P2 wants Y and asks A3, A4, A5. Any majority overlaps A1–A3, here in A3, which reports that it accepted (11, X).",
			},
			{
				kind: "accept",
				p: 2,
				n: 12,
				to: [2, 3, 4],
				note: "So P2 must propose X, not Y: the highest-numbered accepted proposal among its promises was (11, X). The chosen value stays X.",
			},
		],
	},
	{
		id: "preempt",
		label: "Promises block a slow proposer",
		summary:
			"P1's accept reaches only A1 before P2 runs phase 1; P2 is free to choose Y, and P1's late accepts are refused.",
		acceptors: 5,
		values: ["X", "Y"],
		actions: [
			{
				kind: "prepare",
				p: 1,
				n: 11,
				to: [0, 1, 2],
				note: "A1, A2 and A3 promise for 11.",
			},
			{
				kind: "accept",
				p: 1,
				n: 11,
				to: [0],
				note: "P1's accept(11, X) reaches A1 only; the copies to A2 and A3 are delayed in the network. One acceptance is not a majority, so nothing is chosen.",
			},
			{
				kind: "prepare",
				p: 2,
				n: 12,
				to: [2, 3, 4],
				note: "P2 asks A3, A4, A5. None has accepted anything, so P2 may propose its own value. A3's promise for 12 replaces its promise for 11.",
			},
			{
				kind: "accept",
				p: 2,
				n: 12,
				to: [2, 3, 4],
				note: "A3, A4, A5 accept (12, Y): Y is chosen.",
			},
			{
				kind: "accept",
				p: 1,
				n: 11,
				to: [1, 2],
				note: "P1's delayed accept(11, X) finally arrives. A2 has only promised 11, so it accepts (11, X); but that is two acceptances, never a majority. A3 promised 12 and refuses. Y stays chosen.",
			},
		],
	},
	{
		id: "duel",
		label: "Duelling proposers",
		summary:
			"Two proposers keep preempting each other's phase 1, so nothing is ever chosen (section 2.4).",
		acceptors: 5,
		values: ["X", "Y"],
		actions: [
			{
				kind: "prepare",
				p: 1,
				n: 11,
				to: [0, 1, 2],
				note: "P1 completes phase 1 for 11.",
			},
			{
				kind: "prepare",
				p: 2,
				n: 12,
				to: [0, 1, 2],
				note: "Before P1's accepts arrive, P2 completes phase 1 for 12.",
			},
			{
				kind: "accept",
				p: 1,
				n: 11,
				to: [0, 1, 2],
				note: "P1's accept(11, X) is refused: every acceptor has promised 12.",
			},
			{
				kind: "prepare",
				p: 1,
				n: 21,
				to: [0, 1, 2],
				note: "P1 retries with 21, which preempts P2's 12.",
			},
			{
				kind: "accept",
				p: 2,
				n: 12,
				to: [0, 1, 2],
				note: "P2's accept(12, Y) is refused.",
			},
			{
				kind: "prepare",
				p: 2,
				n: 22,
				to: [0, 1, 2],
				note: "P2 retries with 22, preempting 21. And so on: safe, but no progress.",
			},
			{
				kind: "accept",
				p: 1,
				n: 21,
				to: [0, 1, 2],
				note: "Refused again. The fix is to let one distinguished proposer (a leader) issue proposals.",
			},
		],
	},
];
