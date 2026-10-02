// A small Raft simulator following Figure 2 of Ongaro & Ousterhout, "In Search of an
// Understandable Consensus Algorithm" (USENIX ATC 2014): RequestVote and AppendEntries with their
// receiver rules, the election restriction (§5.4.1) and the commit rule for the current term
// (§5.4.2). A scenario is a script of events; each RPC is answered immediately, and `to` limits
// which servers a message reaches (the rest are lost). run() records one frame per event for
// RaftRun.astro.

export interface Entry {
	term: number;
	cmd: string;
}

export type Role = "follower" | "candidate" | "leader";

export interface Server {
	term: number;
	votedFor: number | null;
	log: Entry[];
	commit: number;
	role: Role;
	up: boolean;
	/** Leader only: next log index to send to each server, and highest index known replicated. */
	nextIndex: number[];
	matchIndex: number[];
}

export type Event =
	| { kind: "timeout"; s: number; to?: number[]; note?: string }
	| { kind: "client"; s: number; cmd: string; note?: string }
	| { kind: "replicate"; s: number; to?: number[]; note?: string }
	| { kind: "crash"; s: number; note?: string }
	| { kind: "restart"; s: number; note?: string };

export interface Message {
	from: number;
	to: number;
	text: string;
	ok: boolean;
}

export interface Frame {
	title: string;
	note: string;
	servers: Server[];
	messages: Message[];
	/** Highest index any server knows to be committed, for marking the committed prefix. */
	committed: number;
}

export interface Init {
	terms: number[];
	logs: Entry[][];
	commit: number[];
	votedFor?: (number | null)[];
	leader?: number;
}

export interface Scenario {
	id: string;
	label: string;
	summary: string;
	servers: number;
	init?: Init;
	events: Event[];
}

const lastTerm = (log: Entry[]) => (log.length ? log[log.length - 1].term : 0);

/** §5.4.1: is log a at least as up-to-date as log b? */
export const atLeastAsUpToDate = (a: Entry[], b: Entry[]) =>
	lastTerm(a) !== lastTerm(b)
		? lastTerm(a) > lastTerm(b)
		: a.length >= b.length;

const clone = (s: Server): Server => ({
	...s,
	log: s.log.map((e) => ({ ...e })),
	nextIndex: [...s.nextIndex],
	matchIndex: [...s.matchIndex],
});

/** unsafeCommit drops the "log[N].term == currentTerm" check, to show why it is needed. */
export const run = (
	sc: Scenario,
	opts: { unsafeCommit?: boolean } = {},
): Frame[] => {
	const N = sc.servers;
	const majority = Math.floor(N / 2) + 1;
	const sv: Server[] = Array.from({ length: N }, (_, i) => ({
		term: sc.init?.terms[i] ?? 0,
		votedFor: sc.init?.votedFor?.[i] ?? null,
		log: (sc.init?.logs[i] ?? []).map((e) => ({ ...e })),
		commit: sc.init?.commit[i] ?? 0,
		role: "follower",
		up: true,
		nextIndex: new Array(N).fill(0),
		matchIndex: new Array(N).fill(0),
	}));
	const becomeLeader = (i: number) => {
		const s = sv[i];
		s.role = "leader";
		s.nextIndex = new Array(N).fill(s.log.length + 1);
		s.matchIndex = new Array(N).fill(0);
	};
	if (sc.init?.leader !== undefined) becomeLeader(sc.init.leader);
	const name = (i: number) => `S${i + 1}`;
	// "If RPC request or response contains term T > currentTerm: set currentTerm = T, convert to follower"
	const observe = (i: number, t: number) => {
		if (t > sv[i].term) {
			sv[i].term = t;
			sv[i].votedFor = null;
			sv[i].role = "follower";
		}
	};
	const others = (i: number, to?: number[]) =>
		(to ?? Array.from({ length: N }, (_, k) => k)).filter(
			(k) => k !== i && sv[k].up,
		);
	const frame = (title: string, note: string, messages: Message[]): Frame => ({
		title,
		note,
		servers: sv.map(clone),
		messages,
		committed: Math.max(...sv.map((s) => s.commit)),
	});
	const frames: Frame[] = [
		frame(
			"Start",
			sc.init
				? "The cluster's state when the scenario begins."
				: "Five servers, all followers in term 0 with empty logs.",
			[],
		),
	];

	for (const ev of sc.events) {
		const s = sv[ev.s];
		const messages: Message[] = [];
		let title = "";
		if (!s.up && ev.kind !== "restart")
			throw new Error(`${name(ev.s)} is down`);
		switch (ev.kind) {
			case "timeout": {
				// Candidates (§5.2): increment currentTerm, vote for self, send RequestVote to all.
				s.term += 1;
				s.role = "candidate";
				s.votedFor = ev.s;
				let votes = 1;
				const lastIndex = s.log.length;
				for (const k of others(ev.s, ev.to)) {
					if (s.role !== "candidate") break;
					const r = sv[k];
					observe(k, s.term);
					// RequestVote receiver: reply false if term < currentTerm; grant if not voted
					// for someone else and the candidate's log is at least as up-to-date.
					const grant =
						s.term >= r.term &&
						(r.votedFor === null || r.votedFor === ev.s) &&
						atLeastAsUpToDate(s.log, r.log);
					if (grant) r.votedFor = ev.s;
					const why = grant
						? "vote granted"
						: s.term < r.term
							? `refused: term ${r.term} > ${s.term}`
							: r.votedFor !== null && r.votedFor !== ev.s
								? `refused: already voted for ${name(r.votedFor)}`
								: "refused: candidate's log is less up-to-date";
					messages.push({
						from: ev.s,
						to: k,
						text: `RequestVote(term ${s.term}, last ${lastIndex}/${lastTerm(s.log)}) → ${why}`,
						ok: grant,
					});
					observe(ev.s, r.term);
					if (grant) votes += 1;
				}
				if (s.role === "candidate" && votes >= majority) becomeLeader(ev.s);
				title =
					s.role === "leader"
						? `${name(ev.s)} times out, wins term ${s.term} with ${votes} votes`
						: `${name(ev.s)} times out, starts term ${s.term}, gets ${votes} vote${votes === 1 ? "" : "s"} (needs ${majority})`;
				break;
			}
			case "client": {
				if (s.role !== "leader") throw new Error(`${name(ev.s)} is not leader`);
				s.log.push({ term: s.term, cmd: ev.cmd });
				title = `Client sends ${ev.cmd} to ${name(ev.s)}, which appends it at index ${s.log.length}`;
				break;
			}
			case "replicate": {
				if (s.role !== "leader") throw new Error(`${name(ev.s)} is not leader`);
				for (const k of others(ev.s, ev.to)) {
					const r = sv[k];
					for (;;) {
						if (s.role !== "leader") break;
						const prev = s.nextIndex[k] - 1;
						const prevTerm = prev > 0 ? s.log[prev - 1].term : 0;
						const entries = s.log.slice(prev);
						observe(k, s.term);
						let ok = false;
						let why = "";
						if (s.term < r.term) why = `refused: term ${r.term} > ${s.term}`;
						else {
							r.role = "follower";
							if (
								prev > 0 &&
								(r.log.length < prev || r.log[prev - 1].term !== prevTerm)
							)
								why = `refused: no entry ${prev} with term ${prevTerm}`;
							else {
								ok = true;
								let deleted = 0;
								entries.forEach((e, j) => {
									const idx = prev + j + 1;
									if (r.log.length >= idx && r.log[idx - 1].term !== e.term) {
										deleted += r.log.length - idx + 1;
										r.log.length = idx - 1;
									}
									if (r.log.length < idx) r.log.push({ ...e });
								});
								const lastNew = prev + entries.length;
								if (s.commit > r.commit) r.commit = Math.min(s.commit, lastNew);
								why = `${entries.length ? `appended ${entries.length}` : "heartbeat"}${deleted ? `, deleted ${deleted} conflicting` : ""}`;
							}
						}
						const what = entries.length
							? `${entries.length} entr${entries.length === 1 ? "y" : "ies"}`
							: "no entries";
						messages.push({
							from: ev.s,
							to: k,
							text: `AppendEntries(term ${s.term}, prev ${prev}/${prevTerm}, ${what}, commit ${s.commit}) → ${why}`,
							ok,
						});
						observe(ev.s, r.term);
						if (ok) {
							s.matchIndex[k] = prev + entries.length;
							s.nextIndex[k] = s.matchIndex[k] + 1;
							break;
						}
						if (s.role !== "leader") break;
						s.nextIndex[k] -= 1; // log inconsistency: decrement nextIndex and retry
					}
				}
				if (s.role === "leader") {
					// Commit rule: N > commitIndex, a majority of matchIndex >= N, log[N].term == currentTerm.
					for (let n = s.log.length; n > s.commit; n--) {
						const count =
							1 + s.matchIndex.filter((m, k) => k !== ev.s && m >= n).length;
						if (
							count >= majority &&
							(opts.unsafeCommit || s.log[n - 1].term === s.term)
						) {
							s.commit = n;
							break;
						}
					}
				}
				title = `${name(ev.s)} sends AppendEntries to ${others(ev.s, ev.to).map(name).join(", ") || "nobody"}`;
				break;
			}
			case "crash": {
				s.up = false;
				title = `${name(ev.s)} crashes`;
				break;
			}
			case "restart": {
				// currentTerm, votedFor and log are persistent; commitIndex and the role are not.
				s.up = true;
				s.role = "follower";
				s.commit = 0;
				title = `${name(ev.s)} restarts as a follower`;
				break;
			}
		}
		frames.push(frame(title, ev.note ?? "", messages));
	}
	return frames;
};

/** Raft's safety properties (Figure 3), checked on a run's frames. Returns the violations. */
export const checkSafety = (frames: Frame[]): string[] => {
	const bad: string[] = [];
	const leaders = new Map<number, number>();
	const committed = new Map<number, Entry>();
	for (const [t, f] of frames.entries()) {
		f.servers.forEach((s, i) => {
			if (s.role === "leader" && s.up) {
				const other = leaders.get(s.term);
				if (other !== undefined && other !== i)
					bad.push(`frame ${t}: two leaders in term ${s.term}`);
				leaders.set(s.term, i);
			}
			for (let n = 1; n <= s.commit; n++) {
				const e = s.log[n - 1];
				const c = committed.get(n);
				if (!e) bad.push(`frame ${t}: S${i + 1} commit ${n} beyond its log`);
				else if (c && (c.term !== e.term || c.cmd !== e.cmd))
					bad.push(`frame ${t}: index ${n} committed as ${c.cmd} and ${e.cmd}`);
				else committed.set(n, e);
			}
		});
		// Leader Completeness: every up leader holds every committed entry.
		f.servers.forEach((s, i) => {
			if (s.role !== "leader" || !s.up) return;
			for (const [n, e] of committed)
				if (s.log[n - 1]?.term !== e.term || s.log[n - 1]?.cmd !== e.cmd)
					bad.push(`frame ${t}: leader S${i + 1} lacks committed index ${n}`);
		});
	}
	return bad;
};

const e = (term: number, cmd: string): Entry => ({ term, cmd });

// Figure 8's prefix: (a) S1 leads term 2; index 1 (term 1) is everywhere.
const FIG8_INIT: Init = {
	terms: [2, 2, 2, 2, 2],
	logs: [[e(1, "a")], [e(1, "a")], [e(1, "a")], [e(1, "a")], [e(1, "a")]],
	commit: [1, 1, 1, 1, 1],
	votedFor: [0, 0, 0, 0, 0],
	leader: 0,
};

const FIG8_PREFIX: Event[] = [
	{
		kind: "client",
		s: 0,
		cmd: "b",
		note: "(a) S1 leads term 2 and appends b at index 2.",
	},
	{
		kind: "replicate",
		s: 0,
		to: [1],
		note: "S1 manages to replicate b only to S2 before it crashes.",
	},
	{
		kind: "crash",
		s: 0,
		note: "(b) S1 crashes. b is on 2 of 5 servers: not committed.",
	},
	{
		kind: "timeout",
		s: 4,
		to: [2, 3],
		note: "S5 (log [a]) is elected for term 3 by S3, S4 and itself. S2, which would refuse (its log ends in term 2), is not reached.",
	},
	{
		kind: "client",
		s: 4,
		cmd: "c",
		note: "S5 appends c at index 2, in term 3, and crashes before replicating it.",
	},
	{ kind: "crash", s: 4, note: "" },
	{ kind: "restart", s: 0, note: "(c) S1 restarts. Its term is still 2." },
	{
		kind: "timeout",
		s: 0,
		note: "S1 tries term 3. S2 grants, but S3 and S4 already voted for S5 in term 3, so S1 has 2 votes of the 3 it needs and stays a candidate.",
	},
	{
		kind: "timeout",
		s: 0,
		note: "S1 times out again and wins term 4: its log [a, b] is more up-to-date than S3's and S4's [a].",
	},
	{
		kind: "replicate",
		s: 0,
		to: [1, 2],
		note: "S1 replicates b to S3. b (term 2) is now on S1, S2 and S3, a majority. But b is from an older term, so S1 may not count replicas to commit it: its commitIndex does not move.",
	},
	{ kind: "client", s: 0, cmd: "d", note: "S1 appends d (term 4) at index 3." },
];

export const SCENARIOS: Scenario[] = [
	{
		id: "elect",
		label: "Election and replication",
		summary: "S3 times out first, wins term 1, and replicates two commands.",
		servers: 5,
		events: [
			{
				kind: "timeout",
				s: 2,
				note: "S3's randomised election timeout expires first. It becomes a candidate for term 1, votes for itself and asks the others; all four grant.",
			},
			{
				kind: "replicate",
				s: 2,
				note: "The new leader sends empty AppendEntries (heartbeats) so that no one else starts an election.",
			},
			{
				kind: "client",
				s: 2,
				cmd: "x←1",
				note: "The entry is only in S3's log, so it is not committed yet.",
			},
			{ kind: "client", s: 2, cmd: "y←2", note: "" },
			{
				kind: "replicate",
				s: 2,
				note: "One round of AppendEntries copies both entries. A majority now stores them in S3's own term, so S3 advances commitIndex to 2 and can apply them and answer the client.",
			},
			{
				kind: "replicate",
				s: 2,
				note: "Followers learn the commit index from the next AppendEntries (here a heartbeat) and apply the entries too.",
			},
		],
	},
	{
		id: "repair",
		label: "Leader crash and log repair",
		summary:
			"S1 crashes with an uncommitted entry; the new leader must be up-to-date, and it repairs every follower's log.",
		servers: 5,
		events: [
			{ kind: "timeout", s: 0, note: "S1 wins term 1." },
			{ kind: "client", s: 0, cmd: "x", note: "" },
			{ kind: "replicate", s: 0, note: "x is on all five servers: committed." },
			{
				kind: "crash",
				s: 4,
				note: "S5 crashes; four servers are still a majority.",
			},
			{ kind: "client", s: 0, cmd: "y", note: "" },
			{
				kind: "replicate",
				s: 0,
				note: "y reaches S1–S4 and is committed (4 of 5). S5 misses it.",
			},
			{ kind: "client", s: 0, cmd: "z", note: "" },
			{
				kind: "replicate",
				s: 0,
				to: [1],
				note: "z reaches only S2 before S1 crashes: not committed.",
			},
			{ kind: "crash", s: 0, note: "" },
			{ kind: "restart", s: 4, note: "S5 comes back with only [x]." },
			{
				kind: "timeout",
				s: 2,
				note: "S3 runs for term 2. S2 refuses: its log [x, y, z] is longer with the same last term, so it is more up-to-date. S4 and S5 grant, which is enough. S3 has y, as every possible winner must.",
			},
			{
				kind: "client",
				s: 2,
				cmd: "w",
				note: "S3 appends w at index 3, in term 2.",
			},
			{
				kind: "replicate",
				s: 2,
				note: "S2's z conflicts with w at index 3 and is deleted. S5 lacks index 2, so its first AppendEntries fails; S3 decrements nextIndex and retries from index 2. w is committed (term 2, on 4 servers), and with it everything before it.",
			},
			{ kind: "restart", s: 0, note: "S1 restarts with [x, y, z] in term 1." },
			{
				kind: "replicate",
				s: 2,
				note: "S1 accepts the leader's higher term, its z is overwritten by w, and the heartbeat's commit index tells everyone that index 3 is committed.",
			},
		],
	},
	{
		id: "fig8d",
		label: "Figure 8: why old entries can't be counted",
		summary:
			"The paper's Figure 8 (a)–(d): an entry stored on a majority is overwritten, which is safe only because Raft never counted it as committed.",
		servers: 5,
		init: FIG8_INIT,
		events: [
			...FIG8_PREFIX,
			{ kind: "crash", s: 0, note: "(d) S1 crashes before d is replicated." },
			{
				kind: "restart",
				s: 4,
				note: "S5 restarts with [a, c]: c is from term 3.",
			},
			{
				kind: "timeout",
				s: 4,
				note: "S5's term 4 is already taken (S2–S4 voted for S1), so it is refused.",
			},
			{
				kind: "timeout",
				s: 4,
				note: "In term 5, S5 wins with S2, S3 and S4: its last entry (term 3) is newer than theirs (term 2), so its log counts as more up-to-date.",
			},
			{
				kind: "replicate",
				s: 4,
				note: "S5 overwrites b with c on S2 and S3. Had S1 declared b committed when it reached a majority, a committed entry would now be lost.",
			},
		],
	},
	{
		id: "fig8e",
		label: "Figure 8: committing in the current term",
		summary:
			"Figure 8 (e): S1 replicates d from its own term to a majority, which commits b too, and S5 can no longer win.",
		servers: 5,
		init: FIG8_INIT,
		events: [
			...FIG8_PREFIX,
			{
				kind: "replicate",
				s: 0,
				to: [1, 2],
				note: "(e) d, from S1's current term, reaches S2 and S3: d is committed, and with it b at index 2.",
			},
			{ kind: "crash", s: 0, note: "S1 crashes after committing." },
			{ kind: "restart", s: 4, note: "" },
			{ kind: "timeout", s: 4, note: "S5's term 4 is taken." },
			{
				kind: "timeout",
				s: 4,
				note: "In term 5, S2 and S3 refuse: their logs end in term 4, newer than S5's term 3. Only S4 grants, so S5 cannot win, and b and d are safe.",
			},
		],
	},
];
