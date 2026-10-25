// Grid-world MDP from Shiyu Zhao's "Mathematical Foundations of Reinforcement Learning": a
// rows × cols board with forbidden cells and one target cell, five actions (up, right, down,
// left, stay), deterministic moves, and rewards that depend on the cell the agent tries to
// enter: r_boundary for bumping into the wall (the agent stays put), r_forbidden for entering a
// forbidden cell, r_target for entering the target, r_other otherwise. The target is not
// terminal: staying in it earns r_target forever, so returns are discounted by γ < 1.
//
// The functions are plain TypeScript (no DOM) so they run at build time, in the browser and in
// node --experimental-strip-types. Checked against the book's printed tables (Figures 3.4 and
// 8.6) and an independent NumPy implementation.

export type Cell = [number, number]; // [row, col], 0-based

export interface GridSpec {
	rows: number;
	cols: number;
	forbidden: Cell[];
	target: Cell;
	rBoundary: number;
	rForbidden: number;
	rTarget: number;
	rOther: number;
	gamma: number;
}

/** Action order a1..a5 as in the book. */
export const ACTIONS = ["up", "right", "down", "left", "stay"] as const;
export const ARROWS = ["↑", "→", "↓", "←", "○"];
const MOVES: Cell[] = [
	[-1, 0],
	[0, 1],
	[1, 0],
	[0, -1],
	[0, 0],
];
export const NA = 5;

/** The 5 × 5 grid used throughout the book (forbidden cells and target of Figure 3.4). */
export function bookGrid(over: Partial<GridSpec> = {}): GridSpec {
	return {
		rows: 5,
		cols: 5,
		forbidden: [
			[1, 1],
			[1, 2],
			[2, 2],
			[3, 1],
			[3, 3],
			[4, 1],
		],
		target: [3, 2],
		rBoundary: -1,
		rForbidden: -1,
		rTarget: 1,
		rOther: 0,
		gamma: 0.9,
		...over,
	};
}

/** The 2 × 2 grid of the book's chapters 2–4: s2 forbidden, s4 the target. */
export function smallGrid(over: Partial<GridSpec> = {}): GridSpec {
	return {
		rows: 2,
		cols: 2,
		forbidden: [[0, 1]],
		target: [1, 1],
		rBoundary: -1,
		rForbidden: -1,
		rTarget: 1,
		rOther: 0,
		gamma: 0.9,
		...over,
	};
}

export interface Model {
	n: number;
	/** next[s * 5 + a] = s' (moves are deterministic). */
	next: Int32Array;
	/** reward[s * 5 + a] = r(s, a). */
	reward: Float64Array;
	kind: ("free" | "forbidden" | "target")[];
}

export function buildModel(g: GridSpec): Model {
	const n = g.rows * g.cols;
	const kind: Model["kind"] = new Array(n).fill("free");
	for (const [r, c] of g.forbidden) kind[r * g.cols + c] = "forbidden";
	kind[g.target[0] * g.cols + g.target[1]] = "target";
	const next = new Int32Array(n * NA);
	const reward = new Float64Array(n * NA);
	for (let s = 0; s < n; s++) {
		const r = Math.floor(s / g.cols);
		const c = s % g.cols;
		for (let a = 0; a < NA; a++) {
			let nr = r + MOVES[a][0];
			let nc = c + MOVES[a][1];
			let rew: number;
			if (nr < 0 || nr >= g.rows || nc < 0 || nc >= g.cols) {
				nr = r;
				nc = c;
				rew = g.rBoundary;
			} else {
				const k = kind[nr * g.cols + nc];
				rew =
					k === "target"
						? g.rTarget
						: k === "forbidden"
							? g.rForbidden
							: g.rOther;
			}
			next[s * NA + a] = nr * g.cols + nc;
			reward[s * NA + a] = rew;
		}
	}
	return { n, next, reward, kind };
}

/** π(a | s) as an n × 5 table. */
export type Policy = number[][];

/**
 * Policy from one string per row, one character per cell: ^ > v < o for a deterministic action,
 * * for uniform over the five actions, b for right or down with probability 0.5 each.
 */
export function parsePolicy(rows: string[]): Policy {
	const map: Record<string, number> = { "^": 0, ">": 1, v: 2, "<": 3, o: 4 };
	const pi: Policy = [];
	for (const row of rows)
		for (const ch of row) {
			const p = new Array(NA).fill(0);
			if (ch === "*") p.fill(1 / NA);
			else if (ch === "b") {
				p[1] = 0.5;
				p[2] = 0.5;
			} else if (ch in map) p[map[ch]] = 1;
			else throw new Error(`parsePolicy: unknown cell '${ch}'`);
			pi.push(p);
		}
	return pi;
}

export function deterministic(actions: number[]): Policy {
	return actions.map((a) => {
		const p = new Array(NA).fill(0);
		p[a] = 1;
		return p;
	});
}

export function uniformPolicy(n: number): Policy {
	return Array.from({ length: n }, () => new Array(NA).fill(1 / NA));
}

/** One sweep of iterative policy evaluation: v' = r_π + γ P_π v. */
export function evaluateStep(
	m: Model,
	g: GridSpec,
	pi: Policy,
	v: number[],
): number[] {
	const out = new Array(m.n).fill(0);
	for (let s = 0; s < m.n; s++)
		for (let a = 0; a < NA; a++) {
			const p = pi[s][a];
			if (p)
				out[s] += p * (m.reward[s * NA + a] + g.gamma * v[m.next[s * NA + a]]);
		}
	return out;
}

/** v_π by iterating the Bellman equation to a tolerance (the closed form needs an inverse). */
export function evaluatePolicy(
	m: Model,
	g: GridSpec,
	pi: Policy,
	tol = 1e-12,
): number[] {
	let v = new Array(m.n).fill(0);
	for (let k = 0; k < 100000; k++) {
		const w = evaluateStep(m, g, pi, v);
		const d = maxAbsDiff(w, v);
		v = w;
		if (d < tol) break;
	}
	return v;
}

/** q(s, a) = r(s, a) + γ v(s'). */
export function qValues(m: Model, g: GridSpec, v: number[]): number[][] {
	return Array.from({ length: m.n }, (_, s) =>
		Array.from(
			{ length: NA },
			(_, a) => m.reward[s * NA + a] + g.gamma * v[m.next[s * NA + a]],
		),
	);
}

/** All actions within tol of the best one. */
export function argmaxAll(q: number[], tol = 1e-9): number[] {
	const best = Math.max(...q);
	return q.flatMap((x, a) => (x >= best - tol ? [a] : []));
}

/** Greedy deterministic policy; ties go to the lowest action index (as Python's argmax). */
export function greedy(q: number[][]): number[] {
	return q.map((row) => argmaxAll(row)[0]);
}

export function maxAbsDiff(a: number[], b: number[]): number {
	let d = 0;
	for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
	return d;
}

export function rmse(a: number[], b: number[]): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
	return Math.sqrt(s / a.length);
}

/** Optimal values and action values by value iteration to a tight tolerance. */
export function solveOptimal(
	m: Model,
	g: GridSpec,
	tol = 1e-12,
): { v: number[]; q: number[][] } {
	let v = new Array(m.n).fill(0);
	for (let k = 0; k < 100000; k++) {
		const q = qValues(m, g, v);
		const w = q.map((row) => Math.max(...row));
		const d = maxAbsDiff(w, v);
		v = w;
		if (d < tol) break;
	}
	return { v, q: qValues(m, g, v) };
}

export interface DpStep {
	/** Value estimate after this iteration. */
	v: number[];
	/** Greedy policy chosen in this iteration (one action per state). */
	pi: number[];
	/** ‖v − v*‖∞ after this iteration. */
	err: number;
	/** Bellman sweeps spent in this iteration's evaluation (1 for value iteration). */
	sweeps: number;
}

/** Value iteration from v0: π_{k+1} = greedy(q_k), v_{k+1} = max_a q_k. Entry 0 is the start. */
export function valueIteration(
	m: Model,
	g: GridSpec,
	iters: number,
	v0?: number[],
): DpStep[] {
	const vStar = solveOptimal(m, g).v;
	let v = v0 ? [...v0] : new Array(m.n).fill(0);
	const out: DpStep[] = [
		{ v, pi: greedy(qValues(m, g, v)), err: maxAbsDiff(v, vStar), sweeps: 0 },
	];
	for (let k = 0; k < iters; k++) {
		const q = qValues(m, g, v);
		const pi = greedy(q);
		v = q.map((row) => Math.max(...row));
		out.push({ v, pi, err: maxAbsDiff(v, vStar), sweeps: 1 });
	}
	return out;
}

/**
 * Truncated policy iteration: evaluate π_k with `j` sweeps of the Bellman equation started from
 * the previous estimate, then improve greedily. j = Infinity is policy iteration (evaluated to
 * 1e-12), j = 1 is value iteration. Entry 0 holds π_0 and v_0 = 0.
 */
export function policyIteration(
	m: Model,
	g: GridSpec,
	pi0: number[],
	iters: number,
	j = Number.POSITIVE_INFINITY,
): DpStep[] {
	const vStar = solveOptimal(m, g).v;
	let pi = [...pi0];
	let v = new Array(m.n).fill(0);
	const out: DpStep[] = [{ v, pi, err: maxAbsDiff(v, vStar), sweeps: 0 }];
	for (let k = 0; k < iters; k++) {
		const P = deterministic(pi);
		let sweeps = 0;
		while (sweeps < j) {
			const w = evaluateStep(m, g, P, v);
			const d = maxAbsDiff(w, v);
			v = w;
			sweeps++;
			if (!Number.isFinite(j) && d < 1e-12) break;
		}
		pi = greedy(qValues(m, g, v));
		out.push({ v, pi, err: maxAbsDiff(v, vStar), sweeps });
	}
	return out;
}

/** Small seeded generator (mulberry32) so figures are reproducible. */
export function rng(seed: number): () => number {
	let t = seed >>> 0;
	return () => {
		t = (t + 0x6d2b79f5) >>> 0;
		let x = Math.imul(t ^ (t >>> 15), 1 | t);
		x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
		return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
	};
}

/** ε-greedy action w.r.t. q(s, ·): random action with probability ε, else a greedy one. */
function epsGreedy(q: number[], eps: number, rand: () => number): number {
	if (rand() < eps) return Math.floor(rand() * NA);
	const best = argmaxAll(q);
	return best[Math.floor(rand() * best.length)];
}

export interface TdOptions {
	algorithm: "q-learning" | "sarsa";
	/** Constant learning rate α. */
	alpha: number;
	/** ε of the ε-greedy behaviour policy (taken w.r.t. the current estimate). ε = 1 is uniform. */
	eps: number;
	seed: number;
	/** Start cell index. */
	start: number;
	/** Initial value of every q(s, a). */
	q0?: number;
}

export interface TdUpdate {
	t: number;
	s: number;
	a: number;
	r: number;
	s2: number;
	a2: number;
	target: number;
	before: number;
	after: number;
}

export interface TdLearner {
	q: number[][];
	visits: number[];
	/** Current state (the next update starts here). */
	state: () => number;
	/** One environment step and one update of q(s, a). */
	step: () => TdUpdate;
}

/**
 * Tabular Q-learning or Sarsa on one long episode (as in the book's Figure 7.4) with an ε-greedy
 * behaviour policy. Q-learning's target uses max_a q(s′, a) (Bellman optimality equation);
 * Sarsa's uses q(s′, a′) for the action a′ it will actually take (Bellman equation of the
 * ε-greedy policy it follows).
 */
export function makeTdLearner(m: Model, g: GridSpec, o: TdOptions): TdLearner {
	const rand = rng(o.seed);
	const q = Array.from({ length: m.n }, () => new Array(NA).fill(o.q0 ?? 0));
	const visits = new Array(m.n).fill(0);
	let s = o.start;
	let a = epsGreedy(q[s], o.eps, rand);
	let t = 0;
	return {
		q,
		visits,
		state: () => s,
		step() {
			t++;
			visits[s]++;
			const r = m.reward[s * NA + a];
			const s2 = m.next[s * NA + a];
			const a2 = epsGreedy(q[s2], o.eps, rand);
			const target =
				o.algorithm === "q-learning"
					? r + g.gamma * Math.max(...q[s2])
					: r + g.gamma * q[s2][a2];
			const before = q[s][a];
			q[s][a] -= o.alpha * (q[s][a] - target);
			const u = { t, s, a, r, s2, a2, target, before, after: q[s][a] };
			s = s2;
			a = a2;
			return u;
		},
	};
}

export interface TdRun {
	q: number[][];
	/** [step, RMSE of max_a q(s, a) against v*, states whose greedy action is optimal]. */
	curve: [number, number, number][];
	visits: number[];
}

/** RMSE of max_a q against v* and the number of states whose greedy action is optimal. */
export function tdScore(
	q: number[][],
	vStar: number[],
	qStar: number[][],
): [number, number] {
	const v = q.map((row) => Math.max(...row));
	const good = q.reduce(
		(c, row, s) =>
			c + (argmaxAll(qStar[s], 1e-6).includes(greedy([row])[0]) ? 1 : 0),
		0,
	);
	return [rmse(v, vStar), good];
}

/**
 * Run a learner for `steps` steps, recording the score every `every` steps. For Sarsa the error
 * levels off above zero because Sarsa learns the values of the ε-greedy policy it follows.
 */
export function tdControl(
	m: Model,
	g: GridSpec,
	o: TdOptions & { steps: number; every: number },
): TdRun {
	const { v: vStar, q: qStar } = solveOptimal(m, g);
	const L = makeTdLearner(m, g, o);
	const curve: TdRun["curve"] = [[0, ...tdScore(L.q, vStar, qStar)]];
	for (let t = 1; t <= o.steps; t++) {
		L.step();
		if (t % o.every === 0) curve.push([t, ...tdScore(L.q, vStar, qStar)]);
	}
	return { q: L.q, curve, visits: L.visits };
}

/** Stationary distribution d_π (d_πᵀ P_π = d_πᵀ) by power iteration from the uniform vector. */
export function stationary(m: Model, pi: Policy, iters = 20000): number[] {
	let d = new Array(m.n).fill(1 / m.n);
	for (let k = 0; k < iters; k++) {
		const e = new Array(m.n).fill(0);
		for (let s = 0; s < m.n; s++)
			for (let a = 0; a < NA; a++)
				if (pi[s][a]) e[m.next[s * NA + a]] += d[s] * pi[s][a];
		d = e;
	}
	return d;
}

/** ε-greedy version of a deterministic policy: 1 − ε(|A| − 1)/|A| on its action, ε/|A| elsewhere. */
export function epsGreedyPolicy(actions: number[], eps: number): Policy {
	return actions.map((a) => {
		const p = new Array(NA).fill(eps / NA);
		p[a] = 1 - (eps * (NA - 1)) / NA;
		return p;
	});
}

/**
 * Best policy among the ε-greedy ones (policy iteration with an ε-greedy improvement step, as in
 * the book's MC ε-Greedy). Returns its values and greedy actions.
 */
export function bestEpsGreedy(
	m: Model,
	g: GridSpec,
	eps: number,
): { v: number[]; pi: number[] } {
	let pi = greedy(solveOptimal(m, g).q);
	for (let k = 0; k < 200; k++) {
		const v = evaluatePolicy(m, g, epsGreedyPolicy(pi, eps));
		const next = greedy(qValues(m, g, v));
		if (next.every((a, s) => a === pi[s])) return { v, pi };
		pi = next;
	}
	return { v: evaluatePolicy(m, g, epsGreedyPolicy(pi, eps)), pi };
}

/** Polynomial features of the book's eq. (8.15)–(8.17): x = column, y = row, both scaled to [−1, 1]. */
export function polyFeatures(g: GridSpec, degree: 1 | 2 | 3): number[][] {
	return Array.from({ length: g.rows * g.cols }, (_, s) => {
		const x = (2 * (s % g.cols)) / (g.cols - 1) - 1;
		const y = (2 * Math.floor(s / g.cols)) / (g.rows - 1) - 1;
		const f = [1, x, y];
		if (degree >= 2) f.push(x * x, y * y, x * y);
		if (degree >= 3) f.push(x ** 3, y ** 3, x * x * y, x * y * y);
		return f;
	});
}

/** Solve A x = b by Gaussian elimination with partial pivoting. */
export function solveLinear(A: number[][], b: number[]): number[] {
	const n = b.length;
	const M = A.map((row, i) => [...row, b[i]]);
	for (let c = 0; c < n; c++) {
		let p = c;
		for (let r = c + 1; r < n; r++)
			if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
		[M[c], M[p]] = [M[p], M[c]];
		for (let r = c + 1; r < n; r++) {
			const f = M[r][c] / M[c][c];
			for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
		}
	}
	const x = new Array(n).fill(0);
	for (let r = n - 1; r >= 0; r--) {
		let s = M[r][n];
		for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
		x[r] = s / M[r][r];
	}
	return x;
}

export interface LinearFit {
	/** Stationary distribution used as the weighting d_π. */
	d: number[];
	/** True values v_π. */
	v: number[];
	/** Where TD-Linear converges: Φᵀ D (I − γ P_π) Φ w = Φᵀ D r_π. */
	vTd: number[];
	/** Best fit in the d_π-weighted norm: Φᵀ D Φ w = Φᵀ D v_π. */
	vBest: number[];
}

/** TD-Linear fixed point and best weighted least-squares fit of v_π with features Φ. */
export function linearValueFit(
	m: Model,
	g: GridSpec,
	pi: Policy,
	phi: number[][],
): LinearFit {
	const v = evaluatePolicy(m, g, pi);
	const d = stationary(m, pi);
	const k = phi[0].length;
	const rPi = pi.map((p, s) =>
		p.reduce((acc, x, a) => acc + x * m.reward[s * NA + a], 0),
	);
	// (P_π Φ)(s) = Σ_a π(a|s) Φ(s'(s, a))
	const pPhi = pi.map((p, s) =>
		Array.from({ length: k }, (_, j) =>
			p.reduce((acc, x, a) => acc + x * phi[m.next[s * NA + a]][j], 0),
		),
	);
	const A = Array.from({ length: k }, () => new Array(k).fill(0));
	const bTd = new Array(k).fill(0);
	const B = Array.from({ length: k }, () => new Array(k).fill(0));
	const bBest = new Array(k).fill(0);
	for (let s = 0; s < m.n; s++)
		for (let i = 0; i < k; i++) {
			bTd[i] += d[s] * phi[s][i] * rPi[s];
			bBest[i] += d[s] * phi[s][i] * v[s];
			for (let j = 0; j < k; j++) {
				A[i][j] += d[s] * phi[s][i] * (phi[s][j] - g.gamma * pPhi[s][j]);
				B[i][j] += d[s] * phi[s][i] * phi[s][j];
			}
		}
	const wTd = solveLinear(A, bTd);
	const wBest = solveLinear(B, bBest);
	const dot = (f: number[], w: number[]) =>
		f.reduce((acc, x, i) => acc + x * w[i], 0);
	return {
		d,
		v,
		vTd: phi.map((f) => dot(f, wTd)),
		vBest: phi.map((f) => dot(f, wBest)),
	};
}
