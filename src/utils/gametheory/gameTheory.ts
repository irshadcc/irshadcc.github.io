// Logic for the game theory figures (src/utils/gametheory): bimatrix games, best responses,
// pure and mixed Nash equilibria (support enumeration), iterated elimination of dominated
// strategies, correlated-equilibrium checks, backward induction on game trees, repeated
// Prisoner's Dilemma strategies, replicator / fictitious-play / Hedge dynamics, Gale–Shapley,
// Shapley values, Rosenthal potentials and auctions. No imports, so it runs directly with
// `node --experimental-strip-types`.

export type Matrix = number[][];

/** A two-player game in strategic form: A holds the row player's payoffs, B the column player's. */
export interface Bimatrix {
	name: string;
	players: [string, string];
	rows: string[];
	cols: string[];
	A: Matrix;
	B: Matrix;
}

const bimatrix = (
	name: string,
	players: [string, string],
	rows: string[],
	cols: string[],
	cells: [number, number][][],
): Bimatrix => ({
	name,
	players,
	rows,
	cols,
	A: cells.map((r) => r.map((c) => c[0])),
	B: cells.map((r) => r.map((c) => c[1])),
});

/** The games used in the post. Payoffs are utilities: larger is better. */
export const games: Record<string, Bimatrix> = {
	prisoners: bimatrix(
		"Prisoner's Dilemma",
		["Row", "Column"],
		["Cooperate", "Defect"],
		["Cooperate", "Defect"],
		[
			[
				[3, 3],
				[0, 5],
			],
			[
				[5, 0],
				[1, 1],
			],
		],
	),
	bos: bimatrix(
		"Battle of the Sexes",
		["Row", "Column"],
		["Opera", "Football"],
		["Opera", "Football"],
		[
			[
				[2, 1],
				[0, 0],
			],
			[
				[0, 0],
				[1, 2],
			],
		],
	),
	stagHunt: bimatrix(
		"Stag Hunt",
		["Row", "Column"],
		["Stag", "Hare"],
		["Stag", "Hare"],
		[
			[
				[4, 4],
				[0, 3],
			],
			[
				[3, 0],
				[3, 3],
			],
		],
	),
	chicken: bimatrix(
		"Chicken",
		["Row", "Column"],
		["Dare", "Chicken"],
		["Dare", "Chicken"],
		[
			[
				[0, 0],
				[7, 2],
			],
			[
				[2, 7],
				[6, 6],
			],
		],
	),
	pennies: bimatrix(
		"Matching Pennies",
		["Row", "Column"],
		["Heads", "Tails"],
		["Heads", "Tails"],
		[
			[
				[1, -1],
				[-1, 1],
			],
			[
				[-1, 1],
				[1, -1],
			],
		],
	),
	rps: bimatrix(
		"Rock–Paper–Scissors",
		["Row", "Column"],
		["Rock", "Paper", "Scissors"],
		["Rock", "Paper", "Scissors"],
		[
			[
				[0, 0],
				[-1, 1],
				[1, -1],
			],
			[
				[1, -1],
				[0, 0],
				[-1, 1],
			],
			[
				[-1, 1],
				[1, -1],
				[0, 0],
			],
		],
	),
	// Iterated elimination needs three rounds, and the second one removes a row that only a
	// mixture of the other two rows dominates.
	iesds: bimatrix(
		"A 3 × 3 game solved by elimination",
		["Row", "Column"],
		["Top", "Middle", "Bottom"],
		["Left", "Center", "Right"],
		[
			[
				[4, 1],
				[0, 2],
				[1, 0],
			],
			[
				[1, 4],
				[1, 1],
				[5, 0],
			],
			[
				[0, 1],
				[4, 3],
				[0, 0],
			],
		],
	),
	// A 3 × 3 game with five Nash equilibria (two pure, three mixed), used for support enumeration.
	support: bimatrix(
		"A 3 × 3 game for support enumeration",
		["Row", "Column"],
		["a", "b", "c"],
		["x", "y", "z"],
		[
			[
				[3, 3],
				[3, 2],
				[0, 0],
			],
			[
				[2, 2],
				[5, 6],
				[0, 1],
			],
			[
				[0, 3],
				[6, 1],
				[4, 4],
			],
		],
	),
	// A zero-sum game with no saddle point: row's payoffs only (column gets the negative).
	zeroSum: bimatrix(
		"A zero-sum game without a saddle point",
		["Row", "Column"],
		["r1", "r2", "r3"],
		["c1", "c2", "c3"],
		[
			[
				[3, -3],
				[-1, 1],
				[2, -2],
			],
			[
				[-2, 2],
				[4, -4],
				[1, -1],
			],
			[
				[1, -1],
				[0, 0],
				[-1, 1],
			],
		],
	),
	// The entry game of the extensive-form section, written in strategic form.
	entry: bimatrix(
		"Entry deterrence in strategic form",
		["Entrant", "Incumbent"],
		["Out", "In"],
		["Fight", "Accommodate"],
		[
			[
				[0, 4],
				[0, 4],
			],
			[
				[-1, -1],
				[1, 2],
			],
		],
	),
	hawkDove: bimatrix(
		"Hawk–Dove (V = 2, C = 4)",
		["Row", "Column"],
		["Hawk", "Dove"],
		["Hawk", "Dove"],
		[
			[
				[-1, -1],
				[2, 0],
			],
			[
				[0, 2],
				[1, 1],
			],
		],
	),
};

const EPS = 1e-9;

/** Expected payoff x^T M y. */
export function expected(M: Matrix, x: number[], y: number[]): number {
	let s = 0;
	for (let i = 0; i < x.length; i++)
		for (let j = 0; j < y.length; j++) s += x[i] * M[i][j] * y[j];
	return s;
}

/** Payoff of each pure row against the column mixture y: (M y)_i. */
export const rowPayoffs = (M: Matrix, y: number[]) =>
	M.map((r) => r.reduce((s, v, j) => s + v * y[j], 0));

/** Payoff of each pure column against the row mixture x: (x^T M)_j. */
export const colPayoffs = (M: Matrix, x: number[]) =>
	M[0].map((_, j) => M.reduce((s, r, i) => s + x[i] * r[j], 0));

/** br[i][j] = [is row i a best response to column j, is column j a best response to row i]. */
export function bestResponses(g: Bimatrix): [boolean, boolean][][] {
	const m = g.rows.length;
	const n = g.cols.length;
	const colMax = g.cols.map((_, j) => Math.max(...g.A.map((r) => r[j])));
	const rowMax = g.rows.map((_, i) => Math.max(...g.B[i]));
	return Array.from({ length: m }, (_, i) =>
		Array.from({ length: n }, (_, j) => [
			g.A[i][j] >= colMax[j] - EPS,
			g.B[i][j] >= rowMax[i] - EPS,
		]),
	);
}

/** Pure Nash equilibria as [row, column] pairs. */
export function pureNash(g: Bimatrix): [number, number][] {
	const out: [number, number][] = [];
	bestResponses(g).forEach((r, i) => {
		r.forEach(([a, b], j) => {
			if (a && b) out.push([i, j]);
		});
	});
	return out;
}

/** Solve the square system M z = b by Gaussian elimination with partial pivoting; null if singular. */
export function solve(M: Matrix, b: number[]): number[] | null {
	const n = b.length;
	const a = M.map((r, i) => [...r, b[i]]);
	for (let c = 0; c < n; c++) {
		let p = c;
		for (let r = c + 1; r < n; r++)
			if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
		if (Math.abs(a[p][c]) < 1e-12) return null;
		[a[c], a[p]] = [a[p], a[c]];
		for (let r = 0; r < n; r++) {
			if (r === c) continue;
			const f = a[r][c] / a[c][c];
			for (let k = c; k <= n; k++) a[r][k] -= f * a[c][k];
		}
	}
	return a.map((r, i) => r[n] / r[i]);
}

function subsets(n: number, k: number): number[][] {
	const out: number[][] = [];
	const rec = (start: number, cur: number[]) => {
		if (cur.length === k) {
			out.push([...cur]);
			return;
		}
		for (let i = start; i < n; i++) rec(i + 1, [...cur, i]);
	};
	rec(0, []);
	return out;
}

/**
 * Mixed strategy on `support` (of `size` strategies) that makes the opponent indifferent:
 * solves sum_s z_s M'[t][s] = v for every t in `otherSupport`, sum z = 1. M' is the payoff
 * matrix seen from the opponent, indexed [their strategy][our strategy].
 */
function indifferent(
	Mt: Matrix,
	support: number[],
	otherSupport: number[],
	size: number,
): { z: number[]; v: number } | null {
	const k = support.length;
	const rows: Matrix = otherSupport.map((t) => [
		...support.map((s) => Mt[t][s]),
		-1,
	]);
	rows.push([...support.map(() => 1), 0]);
	const sol = solve(rows, [...otherSupport.map(() => 0), 1]);
	if (!sol) return null;
	const z = new Array<number>(size).fill(0);
	for (let i = 0; i < k; i++) {
		if (sol[i] < -EPS) return null;
		z[support[i]] = Math.max(0, sol[i]);
	}
	return { z, v: sol[k] };
}

export interface MixedEquilibrium {
	x: number[];
	y: number[];
	u1: number;
	u2: number;
	rowSupport: number[];
	colSupport: number[];
}

/**
 * All Nash equilibria of a nondegenerate bimatrix game by support enumeration: for each pair of
 * equal-size supports, solve the indifference conditions and keep solutions with nonnegative
 * probabilities and no profitable pure strategy outside the support.
 */
export function supportEnumeration(g: Bimatrix): MixedEquilibrium[] {
	const m = g.rows.length;
	const n = g.cols.length;
	const At = g.A; // [row][col]: row player's payoffs, indexed by the opponent's (row) strategy for y
	const Bt = g.B[0].map((_, j) => g.B.map((r) => r[j])); // [col][row]
	const out: MixedEquilibrium[] = [];
	for (let k = 1; k <= Math.min(m, n); k++) {
		for (const I of subsets(m, k)) {
			for (const J of subsets(n, k)) {
				// y makes the row player indifferent over I; x makes the column player indifferent over J.
				const ry = indifferent(At, J, I, n);
				const rx = indifferent(Bt, I, J, m);
				if (!ry || !rx) continue;
				const { z: y } = ry;
				const { z: x } = rx;
				const pr = rowPayoffs(g.A, y);
				const pc = colPayoffs(g.B, x);
				const u1 = expected(g.A, x, y);
				const u2 = expected(g.B, x, y);
				if (Math.max(...pr) > u1 + 1e-7 || Math.max(...pc) > u2 + 1e-7)
					continue;
				if (
					out.some(
						(e) =>
							e.x.every((v, i) => Math.abs(v - x[i]) < 1e-7) &&
							e.y.every((v, j) => Math.abs(v - y[j]) < 1e-7),
					)
				)
					continue;
				out.push({ x, y, u1, u2, rowSupport: I, colSupport: J });
			}
		}
	}
	return out;
}

/** The fully mixed equilibrium of a 2 × 2 game: p = P(row plays row 0), q = P(column plays col 0). */
export function mixed2x2(g: Bimatrix): { p: number; q: number } | null {
	const { A, B } = g;
	const dq = A[0][0] - A[0][1] - A[1][0] + A[1][1];
	const dp = B[0][0] - B[0][1] - B[1][0] + B[1][1];
	if (Math.abs(dq) < EPS || Math.abs(dp) < EPS) return null;
	const q = (A[1][1] - A[0][1]) / dq;
	const p = (B[1][1] - B[1][0]) / dp;
	if (p < -EPS || p > 1 + EPS || q < -EPS || q > 1 + EPS) return null;
	return { p, q };
}

/** Security levels of the zero-sum game A: max over rows of min over columns, and min-max. */
export function pureSecurity(A: Matrix): { maxmin: number; minmax: number } {
	const maxmin = Math.max(...A.map((r) => Math.min(...r)));
	const minmax = Math.min(
		...A[0].map((_, j) => Math.max(...A.map((r) => r[j]))),
	);
	return { maxmin, minmax };
}

// ---------------------------------------------------------------------------------------------
// Iterated elimination of strictly dominated strategies

export interface EliminationStep {
	player: 0 | 1;
	removed: number;
	/** The dominating strategy: one pure strategy, or a mixture of two with weight `w` on the first. */
	by: number[];
	w?: number;
	rows: number[];
	cols: number[];
	text: string;
}

/**
 * Strict dominance of `t` by a pure strategy or a mixture of two, restricted to the surviving
 * opponent strategies. `pay(s, o)` is the payoff of own strategy s against opponent o. Returns
 * the dominator (and the weight on its first member for a mixture), or null.
 */
function dominator(
	t: number,
	own: number[],
	opp: number[],
	pay: (s: number, o: number) => number,
): { by: number[]; w?: number } | null {
	for (const s of own)
		if (s !== t && opp.every((o) => pay(s, o) > pay(t, o) + EPS))
			return { by: [s] };
	for (const s of own)
		for (const r of own) {
			if (s >= r || s === t || r === t) continue;
			// Need w pay(s,o) + (1-w) pay(r,o) > pay(t,o) for all o: one linear inequality in w each.
			let lo = 0;
			let hi = 1;
			for (const o of opp) {
				const a = pay(s, o) - pay(r, o);
				const b = pay(t, o) - pay(r, o);
				if (Math.abs(a) < EPS) {
					if (b >= -EPS) {
						lo = 2;
						break;
					}
				} else if (a > 0) lo = Math.max(lo, b / a);
				else hi = Math.min(hi, b / a);
			}
			if (lo < hi - EPS) return { by: [s, r], w: (lo + hi) / 2 };
		}
	return null;
}

/** Remove strictly dominated strategies until none is left, row player first in each round. */
export function iesds(g: Bimatrix): EliminationStep[] {
	let rows = g.rows.map((_, i) => i);
	let cols = g.cols.map((_, j) => j);
	const steps: EliminationStep[] = [];
	const name = (p: 0 | 1, i: number) => (p === 0 ? g.rows[i] : g.cols[i]);
	let changed = true;
	while (changed) {
		changed = false;
		for (const p of [0, 1] as const) {
			const own = p === 0 ? rows : cols;
			const opp = p === 0 ? cols : rows;
			const pay =
				p === 0
					? (s: number, o: number) => g.A[s][o]
					: (s: number, o: number) => g.B[o][s];
			for (const t of own) {
				const d = dominator(t, own, opp, pay);
				if (!d) continue;
				if (p === 0) rows = rows.filter((r) => r !== t);
				else cols = cols.filter((c) => c !== t);
				const who =
					d.by.length === 1
						? name(p, d.by[0])
						: `${fmtNum(d.w ?? 0)}·${name(p, d.by[0])} + ${fmtNum(1 - (d.w ?? 0))}·${name(p, d.by[1])}`;
				steps.push({
					player: p,
					removed: t,
					by: d.by,
					w: d.w,
					rows: [...rows],
					cols: [...cols],
					text: `${g.players[p]} never plays ${name(p, t)}: ${who} pays strictly more against every remaining ${p === 0 ? "column" : "row"}.`,
				});
				changed = true;
				break;
			}
			if (changed) break;
		}
	}
	return steps;
}

export const fmtNum = (v: number, d = 2) => {
	const r = Number(v.toFixed(d));
	return Object.is(r, -0) ? "0" : String(r);
};

// ---------------------------------------------------------------------------------------------
// Correlated equilibrium

/**
 * Largest gain any player gets by disobeying a recommendation of the joint distribution `p`
 * (p[i][j] = probability of recommending row i and column j). p is a correlated equilibrium
 * when the result is at most 0.
 */
export function ceMaxGain(g: Bimatrix, p: Matrix): number {
	let worst = Number.NEGATIVE_INFINITY;
	const m = g.rows.length;
	const n = g.cols.length;
	for (let i = 0; i < m; i++)
		for (let k = 0; k < m; k++) {
			if (k === i) continue;
			let gain = 0;
			for (let j = 0; j < n; j++) gain += p[i][j] * (g.A[k][j] - g.A[i][j]);
			worst = Math.max(worst, gain);
		}
	for (let j = 0; j < n; j++)
		for (let k = 0; k < n; k++) {
			if (k === j) continue;
			let gain = 0;
			for (let i = 0; i < m; i++) gain += p[i][j] * (g.B[i][k] - g.B[i][j]);
			worst = Math.max(worst, gain);
		}
	return worst;
}

// ---------------------------------------------------------------------------------------------
// Extensive-form games

export interface TreeNode {
	id: string;
	/** Player index (0, 1, ...) at a decision node, "chance" for a chance node; absent at a leaf. */
	player?: number | "chance";
	/** Payoffs at a leaf. */
	payoff?: number[];
	/** Nodes with the same infoSet belong to one information set (drawn joined by a dashed line). */
	infoSet?: string;
	/** Optional fixed position [x, y] in the figure's 640-wide viewBox (else laid out by depth). */
	at?: [number, number];
	children?: { action: string; prob?: number; node: TreeNode }[];
}

export interface InductionStep {
	node: string;
	action: string;
	value: number[];
	text: string;
}

/**
 * Backward induction: solve the deepest decision nodes first; at each one the mover picks the
 * action with the highest own payoff (the first on ties), chance nodes average. Returns the steps
 * in solving order, plus the value of every node.
 */
export function backwardInduction(
	root: TreeNode,
	playerNames: string[],
): {
	steps: InductionStep[];
	value: Map<string, number[]>;
	choice: Map<string, string>;
} {
	const steps: InductionStep[] = [];
	const value = new Map<string, number[]>();
	const choice = new Map<string, string>();
	const show = (v: number[]) => `(${v.map((x) => fmtNum(x)).join(", ")})`;
	const rec = (nd: TreeNode): number[] => {
		if (!nd.children) {
			value.set(nd.id, nd.payoff ?? []);
			return nd.payoff ?? [];
		}
		const vals = nd.children.map((c) => rec(c.node));
		let v: number[];
		if (nd.player === "chance") {
			v = vals[0].map((_, k) =>
				vals.reduce((s, vv, c) => s + (nd.children?.[c].prob ?? 0) * vv[k], 0),
			);
			steps.push({
				node: nd.id,
				action: "",
				value: v,
				text: `Chance node: expected payoffs ${show(v)}.`,
			});
		} else {
			const p = nd.player ?? 0;
			let best = 0;
			vals.forEach((vv, c) => {
				if (vv[p] > vals[best][p] + EPS) best = c;
			});
			v = vals[best];
			const act = nd.children[best].action;
			choice.set(nd.id, act);
			const alts = nd.children
				.map((c, k) => `${c.action} → ${fmtNum(vals[k][p])}`)
				.join(", ");
			steps.push({
				node: nd.id,
				action: act,
				value: v,
				text: `${playerNames[p]} compares ${alts} and picks ${act}; this node is worth ${show(v)}.`,
			});
		}
		value.set(nd.id, v);
		return v;
	};
	rec(root);
	return { steps, value, choice };
}

const leaf = (id: string, ...payoff: number[]): TreeNode => ({ id, payoff });

/** Game trees used in the post. */
export const trees: Record<
	string,
	{ players: string[]; root: TreeNode; height?: number }
> = {
	entry: {
		players: ["Entrant", "Incumbent"],
		root: {
			id: "E",
			player: 0,
			children: [
				{ action: "Out", node: leaf("o", 0, 4) },
				{
					action: "In",
					node: {
						id: "I",
						player: 1,
						children: [
							{ action: "Fight", node: leaf("f", -1, -1) },
							{ action: "Accommodate", node: leaf("a", 1, 2) },
						],
					},
				},
			],
		},
	},
	centipede: {
		players: ["Ann", "Bob"],
		root: {
			id: "n1",
			player: 0,
			children: [
				{ action: "Take", node: leaf("t1", 2, 0) },
				{
					action: "Pass",
					node: {
						id: "n2",
						player: 1,
						children: [
							{ action: "Take", node: leaf("t2", 1, 3) },
							{
								action: "Pass",
								node: {
									id: "n3",
									player: 0,
									children: [
										{ action: "Take", node: leaf("t3", 4, 2) },
										{
											action: "Pass",
											node: {
												id: "n4",
												player: 1,
												children: [
													{ action: "Take", node: leaf("t4", 3, 5) },
													{ action: "Pass", node: leaf("t5", 6, 4) },
												],
											},
										},
									],
								},
							},
						],
					},
				},
			],
		},
	},
	// Ultimatum game over 4 coins, offers of 1, 2 or 3 coins to the responder.
	ultimatum: {
		players: ["Proposer", "Responder"],
		root: {
			id: "P",
			player: 0,
			children: [1, 2, 3].map((k) => ({
				action: `offer ${k}`,
				node: {
					id: `R${k}`,
					player: 1,
					children: [
						{ action: "Accept", node: leaf(`a${k}`, 4 - k, k) },
						{ action: "Reject", node: leaf(`r${k}`, 0, 0) },
					],
				},
			})),
		},
	},
	// Simultaneous moves drawn as a tree: Column does not see Row's move (one information set).
	pdTree: {
		players: ["Row", "Column"],
		root: {
			id: "r",
			player: 0,
			children: ["C", "D"].map((a, i) => ({
				action: a,
				node: {
					id: `c${a}`,
					player: 1,
					infoSet: "col",
					children: ["C", "D"].map((b, j) => ({
						action: b,
						node: leaf(
							`${a}${b}`,
							games.prisoners.A[i][j],
							games.prisoners.B[i][j],
						),
					})),
				},
			})),
		},
	},
	// Spence-style signalling game: chance picks the worker's type (productivity 2 or 1), the
	// worker picks education e = 0 or 1.5 at cost e / productivity, the firm sees only e and pays
	// w = 2 or 1. Payoffs (worker w - cost, firm -(w - productivity)^2): the firm wants to pay
	// what the worker is worth. Drawn in the usual signalling-game layout.
	signalling: {
		players: ["Worker", "Firm"],
		height: 330,
		root: {
			id: "N",
			player: "chance",
			at: [320, 165],
			children: [
				{
					action: "high, ½",
					prob: 0.5,
					node: {
						id: "H",
						player: 0,
						at: [320, 70],
						children: [
							{
								action: "e = 1.5",
								node: {
									id: "H1",
									player: 1,
									infoSet: "e1",
									at: [190, 70],
									children: [
										{
											action: "w = 2",
											node: { ...leaf("H1h", 1.25, 0), at: [95, 30] },
										},
										{
											action: "w = 1",
											node: { ...leaf("H1l", 0.25, -1), at: [95, 110] },
										},
									],
								},
							},
							{
								action: "e = 0",
								node: {
									id: "H0",
									player: 1,
									infoSet: "e0",
									at: [450, 70],
									children: [
										{
											action: "w = 2",
											node: { ...leaf("H0h", 2, 0), at: [545, 30] },
										},
										{
											action: "w = 1",
											node: { ...leaf("H0l", 1, -1), at: [545, 110] },
										},
									],
								},
							},
						],
					},
				},
				{
					action: "low, ½",
					prob: 0.5,
					node: {
						id: "L",
						player: 0,
						at: [320, 260],
						children: [
							{
								action: "e = 1.5",
								node: {
									id: "L1",
									player: 1,
									infoSet: "e1",
									at: [190, 260],
									children: [
										{
											action: "w = 2",
											node: { ...leaf("L1h", 0.5, -1), at: [95, 220] },
										},
										{
											action: "w = 1",
											node: { ...leaf("L1l", -0.5, 0), at: [95, 300] },
										},
									],
								},
							},
							{
								action: "e = 0",
								node: {
									id: "L0",
									player: 1,
									infoSet: "e0",
									at: [450, 260],
									children: [
										{
											action: "w = 2",
											node: { ...leaf("L0h", 2, -1), at: [545, 220] },
										},
										{
											action: "w = 1",
											node: { ...leaf("L0l", 1, 0), at: [545, 300] },
										},
									],
								},
							},
						],
					},
				},
			],
		},
	},
};

// ---------------------------------------------------------------------------------------------
// Repeated Prisoner's Dilemma

export type Move = "C" | "D";

/** A strategy picks a move from its own history and the opponent's; `rng` is for random ones. */
export interface RepeatedStrategy {
	name: string;
	note: string;
	play: (mine: Move[], theirs: Move[], rng: () => number) => Move;
}

export const repeatedStrategies: Record<string, RepeatedStrategy> = {
	allc: { name: "Always cooperate", note: "C every round.", play: () => "C" },
	alld: { name: "Always defect", note: "D every round.", play: () => "D" },
	tft: {
		name: "Tit for tat",
		note: "C first, then copy the opponent's last move.",
		play: (_m, t) => (t.length ? t[t.length - 1] : "C"),
	},
	grim: {
		name: "Grim trigger",
		note: "C until the opponent defects once, then D forever.",
		play: (_m, t) => (t.includes("D") ? "D" : "C"),
	},
	wsls: {
		name: "Win-stay, lose-shift",
		note: "C first; repeat the last move after a payoff of 3 or 5, switch after 0 or 1.",
		play: (m, t) => {
			if (!m.length) return "C";
			const last = m[m.length - 1];
			const win = t[t.length - 1] === "C";
			return win ? last : last === "C" ? "D" : "C";
		},
	},
	tf2t: {
		name: "Tit for two tats",
		note: "D only after two defections in a row.",
		play: (_m, t) =>
			t.length >= 2 && t[t.length - 1] === "D" && t[t.length - 2] === "D"
				? "D"
				: "C",
	},
	random: {
		name: "Random",
		note: "C or D with probability ½ each.",
		play: (_m, _t, rng) => (rng() < 0.5 ? "C" : "D"),
	},
};

/** Payoffs of the stage game in the post: T = 5, R = 3, P = 1, S = 0. */
export const pdPayoff = (a: Move, b: Move): [number, number] =>
	a === "C" ? (b === "C" ? [3, 3] : [0, 5]) : b === "C" ? [5, 0] : [1, 1];

/** Seeded uniforms on [0, 1) (mulberry32). */
export function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

export interface Match {
	a: Move[];
	b: Move[];
	pa: number[];
	pb: number[];
	/** Rounds where noise flipped the intended move. */
	fa: boolean[];
	fb: boolean[];
}

/** Play `rounds` rounds; with probability `noise` each intended move is flipped. */
export function playMatch(
	sa: RepeatedStrategy,
	sb: RepeatedStrategy,
	rounds: number,
	noise = 0,
	seed = 1,
): Match {
	const rng = mulberry32(seed);
	const a: Move[] = [];
	const b: Move[] = [];
	const pa: number[] = [];
	const pb: number[] = [];
	const fa: boolean[] = [];
	const fb: boolean[] = [];
	const flip = (m: Move, f: boolean[]): Move => {
		const hit = rng() < noise;
		f.push(hit);
		return hit ? (m === "C" ? "D" : "C") : m;
	};
	for (let t = 0; t < rounds; t++) {
		const x = flip(sa.play(a, b, rng), fa);
		const y = flip(sb.play(b, a, rng), fb);
		a.push(x);
		b.push(y);
		const [u, v] = pdPayoff(x, y);
		pa.push(u);
		pb.push(v);
	}
	return { a, b, pa, pb, fa, fb };
}

/** Normalised discounted payoff (1 - δ) Σ δ^t u_t of a payoff stream. */
export const discounted = (u: number[], delta: number) =>
	(1 - delta) * u.reduce((s, v, t) => s + delta ** t * v, 0);

/** Round-robin tournament (each pair once, plus a copy of itself): average payoff per round. */
export function tournament(
	keys: string[],
	rounds: number,
	noise = 0,
	seed = 1,
): { key: string; score: number; vs: number[] }[] {
	const res = keys.map((k) => ({ key: k, score: 0, vs: keys.map(() => 0) }));
	keys.forEach((ka, i) => {
		keys.forEach((kb, j) => {
			if (j < i) return;
			const m = playMatch(
				repeatedStrategies[ka],
				repeatedStrategies[kb],
				rounds,
				noise,
				seed + 97 * i + j,
			);
			const sa = m.pa.reduce((s, v) => s + v, 0) / rounds;
			const sb = m.pb.reduce((s, v) => s + v, 0) / rounds;
			res[i].vs[j] = sa;
			res[j].vs[i] = sb;
		});
	});
	for (const r of res) r.score = r.vs.reduce((s, v) => s + v, 0) / keys.length;
	return res;
}

/**
 * Best normalised discounted payoff against tit for tat, by value iteration on TFT's two states
 * (it will play C or D this round). Returns the value from the start (TFT plays C) and the best
 * first move from each state.
 */
export function bestAgainstTft(delta: number): {
	value: number;
	policy: { C: Move; D: Move };
} {
	let vC = 0;
	let vD = 0;
	for (let it = 0; it < 5000; it++) {
		// In state s (TFT plays s), our move m gives pdPayoff(m, s) and moves TFT to state m.
		const nC = Math.max(3 + delta * vC, 5 + delta * vD);
		const nD = Math.max(0 + delta * vC, 1 + delta * vD);
		vC = nC;
		vD = nD;
	}
	return {
		value: (1 - delta) * vC,
		policy: {
			C: 3 + delta * vC >= 5 + delta * vD - EPS ? "C" : "D",
			D: 0 + delta * vC >= 1 + delta * vD - EPS ? "C" : "D",
		},
	};
}

// ---------------------------------------------------------------------------------------------
// Dynamics on the simplex

/** Generalised Rock–Paper–Scissors: a win pays `win`, a loss costs `lose`, a tie pays 0. */
export const rpsMatrix = (win: number, lose: number): Matrix => [
	[0, -lose, win],
	[win, 0, -lose],
	[-lose, win, 0],
];

/** Replicator dynamics dx_i/dt = x_i ((A x)_i - x^T A x). */
export function replicatorField(A: Matrix, x: number[]): number[] {
	const f = rowPayoffs(A, x);
	const avg = x.reduce((s, v, i) => s + v * f[i], 0);
	return x.map((v, i) => v * (f[i] - avg));
}

/** Integrate the replicator dynamics with RK4 for `steps` steps of size `h`. */
export function replicatorPath(
	A: Matrix,
	x0: number[],
	steps: number,
	h: number,
): number[][] {
	const out = [x0];
	let x = x0;
	const add = (a: number[], b: number[], s: number) =>
		a.map((v, i) => v + s * b[i]);
	for (let t = 0; t < steps; t++) {
		const k1 = replicatorField(A, x);
		const k2 = replicatorField(A, add(x, k1, h / 2));
		const k3 = replicatorField(A, add(x, k2, h / 2));
		const k4 = replicatorField(A, add(x, k3, h));
		x = x.map((v, i) => v + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
		const s = x.reduce((a, b) => a + b, 0);
		x = x.map((v) => Math.max(0, v / s));
		out.push(x);
	}
	return out;
}

/**
 * Symmetric fictitious play: one population best-responds to the empirical frequency of past
 * play (starting from `counts`), ties broken towards the lowest index. Returns the empirical
 * frequencies after each round.
 */
export function fictitiousPlay(
	A: Matrix,
	counts: number[],
	rounds: number,
): number[][] {
	const c = [...counts];
	const out: number[][] = [];
	for (let t = 0; t < rounds; t++) {
		const tot = c.reduce((a, b) => a + b, 0);
		const freq = c.map((v) => v / tot);
		out.push(freq);
		const f = rowPayoffs(A, freq);
		let best = 0;
		f.forEach((v, i) => {
			if (v > f[best] + EPS) best = i;
		});
		c[best] += 1;
	}
	const tot = c.reduce((a, b) => a + b, 0);
	out.push(c.map((v) => v / tot));
	return out;
}

/**
 * Two players run Hedge (multiplicative weights) against each other in the zero-sum game A
 * (row gets A, column gets -A). Returns the row player's last iterate and running average of
 * its mixed strategies, plus both players' regrets after each round.
 */
export function hedgeSelfPlay(
	A: Matrix,
	x0: number[],
	y0: number[],
	eta: number,
	rounds: number,
): {
	last: number[][];
	avg: number[][];
	regretRow: number[];
	regretCol: number[];
} {
	let lx = x0.map(Math.log);
	let ly = y0.map(Math.log);
	const norm = (l: number[]) => {
		const m = Math.max(...l);
		const e = l.map((v) => Math.exp(v - m));
		const s = e.reduce((a, b) => a + b, 0);
		return e.map((v) => v / s);
	};
	const last: number[][] = [];
	const avg: number[][] = [];
	const regretRow: number[] = [];
	const regretCol: number[] = [];
	const sum = x0.map(() => 0);
	const cumRow = A.map(() => 0); // cumulative payoff of each pure row
	const cumCol = A[0].map(() => 0);
	let gotRow = 0;
	let gotCol = 0;
	for (let t = 1; t <= rounds; t++) {
		const x = norm(lx);
		const y = norm(ly);
		last.push(x);
		x.forEach((v, i) => {
			sum[i] += v;
		});
		avg.push(sum.map((v) => v / t));
		const fr = rowPayoffs(A, y);
		const fc = colPayoffs(A, x).map((v) => -v);
		gotRow += x.reduce((s, v, i) => s + v * fr[i], 0);
		gotCol += y.reduce((s, v, j) => s + v * fc[j], 0);
		fr.forEach((v, i) => {
			cumRow[i] += v;
		});
		fc.forEach((v, j) => {
			cumCol[j] += v;
		});
		regretRow.push(Math.max(...cumRow) - gotRow);
		regretCol.push(Math.max(...cumCol) - gotCol);
		lx = lx.map((v, i) => v + eta * fr[i]);
		ly = ly.map((v, j) => v + eta * fc[j]);
	}
	return { last, avg, regretRow, regretCol };
}

/**
 * Regret matching (Hart & Mas-Colell) for both players of a bimatrix game: each round play each
 * action with probability proportional to its positive cumulative regret (uniform if none).
 * Returns the empirical joint distribution of play and its largest CE / CCE violations.
 */
export function regretMatching(
	g: Bimatrix,
	rounds: number,
	seed = 1,
): { joint: Matrix; ceGain: number; cceGain: number } {
	const rng = mulberry32(seed);
	const m = g.rows.length;
	const n = g.cols.length;
	const rr = new Array<number>(m).fill(0);
	const rc = new Array<number>(n).fill(0);
	const joint: Matrix = g.rows.map(() => new Array<number>(n).fill(0));
	const draw = (r: number[]) => {
		const pos = r.map((v) => Math.max(0, v));
		const s = pos.reduce((a, b) => a + b, 0);
		const p = s > 0 ? pos.map((v) => v / s) : r.map(() => 1 / r.length);
		let u = rng();
		for (let k = 0; k < p.length; k++) {
			u -= p[k];
			if (u < 0) return k;
		}
		return p.length - 1;
	};
	for (let t = 0; t < rounds; t++) {
		const i = draw(rr);
		const j = draw(rc);
		joint[i][j] += 1 / rounds;
		for (let k = 0; k < m; k++) rr[k] += g.A[k][j] - g.A[i][j];
		for (let k = 0; k < n; k++) rc[k] += g.B[i][k] - g.B[i][j];
	}
	// Coarse CE: no fixed deviation k beats following the distribution.
	let cce = Number.NEGATIVE_INFINITY;
	for (let k = 0; k < m; k++) {
		let s = 0;
		for (let i = 0; i < m; i++)
			for (let j = 0; j < n; j++) s += joint[i][j] * (g.A[k][j] - g.A[i][j]);
		cce = Math.max(cce, s);
	}
	for (let k = 0; k < n; k++) {
		let s = 0;
		for (let i = 0; i < m; i++)
			for (let j = 0; j < n; j++) s += joint[i][j] * (g.B[i][k] - g.B[i][j]);
		cce = Math.max(cce, s);
	}
	return { joint, ceGain: ceMaxGain(g, joint), cceGain: cce };
}

// ---------------------------------------------------------------------------------------------
// Matching

export interface GsStep {
	proposer: number;
	target: number;
	accepted: boolean;
	/** The proposer the target dropped, if any. */
	dropped: number | null;
	/** engaged[target] = proposer index or -1, after this step. */
	engaged: number[];
	text: string;
}

/**
 * Deferred acceptance with the proposers in `prop` (prop[i] = proposer i's ranking of targets,
 * best first) and the targets' rankings in `recv`. One proposal per step: the lowest-numbered
 * free proposer proposes to the best target not yet tried.
 */
export function galeShapley(
	prop: number[][],
	recv: number[][],
	pNames: string[],
	rNames: string[],
): { steps: GsStep[]; match: number[] } {
	const n = prop.length;
	const next = new Array<number>(n).fill(0);
	const engaged = new Array<number>(recv.length).fill(-1);
	const rank = recv.map((r) => {
		const out = new Array<number>(n);
		r.forEach((p, k) => {
			out[p] = k;
		});
		return out;
	});
	const steps: GsStep[] = [];
	for (;;) {
		const free = [...Array(n).keys()].find(
			(p) => !engaged.includes(p) && next[p] < prop[p].length,
		);
		if (free === undefined) break;
		const t = prop[free][next[free]++];
		const cur = engaged[t];
		let accepted = false;
		let dropped: number | null = null;
		let text: string;
		if (cur === -1) {
			engaged[t] = free;
			accepted = true;
			text = `${pNames[free]} proposes to ${rNames[t]}, who has a free place and holds the offer.`;
		} else if (rank[t][free] < rank[t][cur]) {
			engaged[t] = free;
			accepted = true;
			dropped = cur;
			text = `${pNames[free]} proposes to ${rNames[t]}, who prefers ${pNames[free]} to ${pNames[cur]}: ${pNames[cur]} is free again.`;
		} else {
			text = `${pNames[free]} proposes to ${rNames[t]}, who rejects it: ${rNames[t]} ranks ${pNames[cur]} higher.`;
		}
		steps.push({
			proposer: free,
			target: t,
			accepted,
			dropped,
			engaged: [...engaged],
			text,
		});
	}
	const match = new Array<number>(n).fill(-1);
	engaged.forEach((p, t) => {
		if (p >= 0) match[p] = t;
	});
	return { steps, match };
}

/** Blocking pairs of a matching (match[p] = target of proposer p). */
export function blockingPairs(
	prop: number[][],
	recv: number[][],
	match: number[],
): [number, number][] {
	const out: [number, number][] = [];
	const partner = new Array<number>(recv.length).fill(-1);
	match.forEach((t, p) => {
		if (t >= 0) partner[t] = p;
	});
	prop.forEach((pref, p) => {
		for (const t of pref) {
			if (t === match[p]) break;
			const q = partner[t];
			if (q === -1 || recv[t].indexOf(p) < recv[t].indexOf(q)) out.push([p, t]);
		}
	});
	return out;
}

/**
 * The matching example in the post: 4 doctors (A–D) propose to 4 hospitals (W–Z) with one
 * place each. It has three stable matchings, so the doctor-optimal and hospital-optimal ones differ.
 */
export const matchingExample = {
	pNames: ["A", "B", "C", "D"],
	rNames: ["W", "X", "Y", "Z"],
	prop: [
		[0, 1, 3, 2],
		[3, 0, 2, 1],
		[3, 1, 2, 0],
		[0, 2, 3, 1],
	],
	recv: [
		[2, 3, 1, 0],
		[3, 1, 2, 0],
		[0, 2, 3, 1],
		[3, 0, 1, 2],
	],
};

// ---------------------------------------------------------------------------------------------
// Cooperative games

/**
 * Shapley value of an n-player TU game with characteristic function v (a bitmask of the
 * coalition), with each player's marginal contribution in every ordering.
 */
export function shapley(
	n: number,
	v: (mask: number) => number,
): { phi: number[]; orders: { order: number[]; marginal: number[] }[] } {
	const orders: { order: number[]; marginal: number[] }[] = [];
	const perm = (cur: number[], rest: number[]) => {
		if (!rest.length) {
			const marginal = new Array<number>(n).fill(0);
			let mask = 0;
			for (const p of cur) {
				marginal[p] = v(mask | (1 << p)) - v(mask);
				mask |= 1 << p;
			}
			orders.push({ order: cur, marginal });
			return;
		}
		rest.forEach((p, k) =>
			perm(
				[...cur, p],
				rest.filter((_, j) => j !== k),
			),
		);
	};
	perm([], [...Array(n).keys()]);
	const phi = new Array<number>(n).fill(0);
	for (const o of orders)
		o.marginal.forEach((m, p) => {
			phi[p] += m / orders.length;
		});
	return { phi, orders };
}

/** Largest amount by which some coalition is short-changed by allocation x (≤ 0 means x is in the core). */
export function coreExcess(
	n: number,
	v: (mask: number) => number,
	x: number[],
): number {
	let worst = Number.NEGATIVE_INFINITY;
	for (let mask = 1; mask < 1 << n; mask++) {
		let s = 0;
		for (let p = 0; p < n; p++) if (mask & (1 << p)) s += x[p];
		worst = Math.max(worst, v(mask) - s);
	}
	return worst;
}

/** Glove game: player 0 owns a left glove, players 1 and 2 a right glove each; a pair is worth 1. */
export const gloveGame = (mask: number) => (mask & 1 && mask & 6 ? 1 : 0);

/** Airport game: runway costs 1, 2 and 3 for planes 0, 1, 2; a coalition pays for its largest plane. */
export const airportCost = (mask: number) =>
	mask & 4 ? 3 : mask & 2 ? 2 : mask & 1 ? 1 : 0;

// ---------------------------------------------------------------------------------------------
// Congestion games

/**
 * Best-response dynamics in a two-road congestion game with `players` drivers: road A costs
 * a(k) and road B costs b(k) per driver when k drivers use it. Starts with everyone on B and moves
 * one driver at a time while some driver strictly gains. Returns the states with Rosenthal's
 * potential Φ = Σ_{k ≤ nA} a(k) + Σ_{k ≤ nB} b(k).
 */
export function congestionDynamics(
	players: number,
	a: (k: number) => number,
	b: (k: number) => number,
): {
	nA: number;
	nB: number;
	costA: number;
	costB: number;
	phi: number;
	text: string;
}[] {
	const phi = (nA: number) => {
		let s = 0;
		for (let k = 1; k <= nA; k++) s += a(k);
		for (let k = 1; k <= players - nA; k++) s += b(k);
		return s;
	};
	const out = [];
	let nA = 0;
	for (;;) {
		const nB = players - nA;
		const row = {
			nA,
			nB,
			costA: nA ? a(nA) : 0,
			costB: nB ? b(nB) : 0,
			phi: phi(nA),
			text: "",
		};
		out.push(row);
		if (nB > 0 && a(nA + 1) < b(nB)) {
			row.text = `a driver on B pays ${b(nB)}, would pay ${a(nA + 1)} on A: switch`;
			nA++;
		} else if (nA > 0 && b(nB + 1) < a(nA)) {
			row.text = `a driver on A pays ${a(nA)}, would pay ${b(nB + 1)} on B: switch`;
			nA--;
		} else {
			row.text = "no driver gains by switching: Nash equilibrium";
			break;
		}
	}
	return out;
}

// ---------------------------------------------------------------------------------------------
// Auctions

/** Monte Carlo revenue of first-price, second-price and all-pay auctions with n uniform bidders. */
export function auctionRevenue(
	n: number,
	draws: number,
	seed = 1,
	reserve = 0,
): { first: number; second: number; allPay: number; secondReserve: number } {
	const rng = mulberry32(seed);
	let first = 0;
	let second = 0;
	let allPay = 0;
	let secondReserve = 0;
	for (let d = 0; d < draws; d++) {
		const v = Array.from({ length: n }, () => rng()).sort((x, y) => y - x);
		first += ((n - 1) / n) * v[0];
		second += v[1];
		allPay += v.reduce((s, x) => s + ((n - 1) / n) * x ** n, 0);
		if (v[0] >= reserve) secondReserve += Math.max(v[1], reserve);
	}
	return {
		first: first / draws,
		second: second / draws,
		allPay: allPay / draws,
		secondReserve: secondReserve / draws,
	};
}
