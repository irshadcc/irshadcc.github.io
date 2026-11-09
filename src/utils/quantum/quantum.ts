// A small state-vector simulator for the quantum computing figures (no DOM, so it runs at build
// time and in node tests). Convention: big-endian, as in Nielsen & Chuang. Qubit 0 is the
// leftmost bit of a basis label, so |q0 q1 q2> = |110> is index 6. (Qiskit labels the other way
// round: its qubit 0 is the rightmost bit.)
//
// A circuit is a list of steps. Each step is a string of gates separated by ";", optionally
// followed by "| narration". Gates:
//   H X Y Z S SDG T TDG q          single-qubit gates
//   RX(a) RY(a) RZ(a) P(a) q       rotations and phase, angle like pi/4, -pi/2, 2*pi/3, 0.3
//   CX c t, CY c t, CZ c t, CP(a) c t, CH c t
//   CCX c1 c2 t, CCZ a b c, SWAP a b
//   ORACLE m1,m2 q...              flip the sign of basis states of qubits q... whose value
//                                  (big-endian over q...) is one of m1, m2, ...
//   DIFF q...                      Grover diffusion 2|s><s| - I on qubits q...
//   QFT q..., IQFT q...            quantum Fourier transform (q... most significant first)
//   M q...                         drawn as a measurement; leaves the state unchanged
//   I q                            identity (a labelled no-op, e.g. "no error here")
// Any gate may end with "@label" to show a different label in the diagram, e.g. "X 1 @error".

export type Complex = [number, number];

export interface Op {
	name: string;
	qubits: number[];
	param?: number;
	/** The angle as written, for labels. */
	paramText?: string;
	marked?: number[];
	label?: string;
}

export interface Step {
	ops: Op[];
	note: string;
}

export interface State {
	re: Float64Array;
	im: Float64Array;
}

// ---------------------------------------------------------------------------------------------
// Parsing

/** Evaluate an angle expression built from numbers, pi, + - * / and parentheses. */
export function parseAngle(src: string): number {
	const s = src.replace(/\s+/g, "").toLowerCase().replace(/π/g, "pi");
	let i = 0;
	const peek = () => s[i];
	function atom(): number {
		if (peek() === "-") {
			i++;
			return -atom();
		}
		if (peek() === "+") {
			i++;
			return atom();
		}
		if (peek() === "(") {
			i++;
			const v = expr();
			if (s[i++] !== ")") throw new Error(`bad angle "${src}"`);
			return v;
		}
		if (s.startsWith("pi", i)) {
			i += 2;
			return Math.PI;
		}
		const m = /^\d*\.?\d+(e[-+]?\d+)?/.exec(s.slice(i));
		if (!m) throw new Error(`bad angle "${src}"`);
		i += m[0].length;
		return Number(m[0]);
	}
	function term(): number {
		let v = atom();
		while (peek() === "*" || peek() === "/") {
			const op = s[i++];
			const r = atom();
			v = op === "*" ? v * r : v / r;
		}
		return v;
	}
	function expr(): number {
		let v = term();
		while (peek() === "+" || peek() === "-") {
			const op = s[i++];
			const r = term();
			v = op === "+" ? v + r : v - r;
		}
		return v;
	}
	const v = expr();
	if (i !== s.length) throw new Error(`bad angle "${src}"`);
	return v;
}

const ARITY: Record<string, number> = {
	H: 1,
	X: 1,
	Y: 1,
	Z: 1,
	S: 1,
	SDG: 1,
	T: 1,
	TDG: 1,
	RX: 1,
	RY: 1,
	RZ: 1,
	P: 1,
	I: 1,
	CX: 2,
	CY: 2,
	CZ: 2,
	CP: 2,
	CH: 2,
	SWAP: 2,
	CCX: 3,
	CCZ: 3,
};
const PARAM = new Set(["RX", "RY", "RZ", "P", "CP"]);
const VARIADIC = new Set(["ORACLE", "DIFF", "QFT", "IQFT", "M"]);

export function parseGate(src: string, n: number): Op {
	let text = src.trim();
	let label: string | undefined;
	const at = text.indexOf("@");
	if (at >= 0) {
		label = text.slice(at + 1).trim();
		text = text.slice(0, at).trim();
	}
	const m = /^([A-Za-z]+)(?:\(([^)]*)\))?\s*(.*)$/.exec(text);
	if (!m) throw new Error(`bad gate "${src}"`);
	const name = m[1].toUpperCase();
	const rest = m[3].trim().split(/\s+/).filter(Boolean);
	const op: Op = { name, qubits: [], label };
	if (PARAM.has(name)) {
		if (m[2] === undefined) throw new Error(`${name} needs an angle: "${src}"`);
		op.param = parseAngle(m[2]);
		op.paramText = m[2].trim().replace(/pi/gi, "π").replace(/\*/g, "");
	}
	if (name === "ORACLE") {
		op.marked = (rest.shift() ?? "").split(",").map(Number);
		if (op.marked.some((v) => !Number.isInteger(v)))
			throw new Error(`bad marked list: "${src}"`);
	}
	op.qubits = rest.map(Number);
	if (VARIADIC.has(name) && !op.qubits.length)
		op.qubits = Array.from({ length: n }, (_, q) => q);
	if (!(name in ARITY) && !VARIADIC.has(name))
		throw new Error(`unknown gate "${name}"`);
	if (name in ARITY && op.qubits.length !== ARITY[name])
		throw new Error(`${name} takes ${ARITY[name]} qubit(s): "${src}"`);
	for (const q of op.qubits)
		if (!Number.isInteger(q) || q < 0 || q >= n)
			throw new Error(`qubit out of range in "${src}"`);
	if (new Set(op.qubits).size !== op.qubits.length)
		throw new Error(`repeated qubit in "${src}"`);
	if (op.marked?.some((v) => v < 0 || v >= 2 ** op.qubits.length))
		throw new Error(`marked value out of range in "${src}"`);
	return op;
}

export function parseStep(src: string, n: number): Step {
	const bar = src.indexOf("|");
	const gates = bar >= 0 ? src.slice(0, bar) : src;
	const note = bar >= 0 ? src.slice(bar + 1).trim() : "";
	const ops = gates
		.split(";")
		.map((g) => g.trim())
		.filter(Boolean)
		.map((g) => parseGate(g, n));
	return { ops, note };
}

// ---------------------------------------------------------------------------------------------
// Simulation

export function basisState(n: number, bits = ""): State {
	const re = new Float64Array(2 ** n);
	const im = new Float64Array(2 ** n);
	const b = bits.padEnd(n, "0");
	if (!/^[01]+$/.test(b) || b.length !== n)
		throw new Error(`bad initial bits "${bits}"`);
	re[Number.parseInt(b, 2)] = 1;
	return { re, im };
}

const R = Math.SQRT1_2;
/** 2×2 matrices as [a, b, c, d] (row-major), each entry [re, im]. */
export function matrix(name: string, t = 0): Complex[] {
	switch (name) {
		case "H":
			return [
				[R, 0],
				[R, 0],
				[R, 0],
				[-R, 0],
			];
		case "X":
			return [
				[0, 0],
				[1, 0],
				[1, 0],
				[0, 0],
			];
		case "Y":
			return [
				[0, 0],
				[0, -1],
				[0, 1],
				[0, 0],
			];
		case "Z":
			return [
				[1, 0],
				[0, 0],
				[0, 0],
				[-1, 0],
			];
		case "S":
			return [
				[1, 0],
				[0, 0],
				[0, 0],
				[0, 1],
			];
		case "SDG":
			return [
				[1, 0],
				[0, 0],
				[0, 0],
				[0, -1],
			];
		case "T":
			return [
				[1, 0],
				[0, 0],
				[0, 0],
				[R, R],
			];
		case "TDG":
			return [
				[1, 0],
				[0, 0],
				[0, 0],
				[R, -R],
			];
		case "I":
			return [
				[1, 0],
				[0, 0],
				[0, 0],
				[1, 0],
			];
		case "RX":
			return [
				[Math.cos(t / 2), 0],
				[0, -Math.sin(t / 2)],
				[0, -Math.sin(t / 2)],
				[Math.cos(t / 2), 0],
			];
		case "RY":
			return [
				[Math.cos(t / 2), 0],
				[-Math.sin(t / 2), 0],
				[Math.sin(t / 2), 0],
				[Math.cos(t / 2), 0],
			];
		case "RZ":
			return [
				[Math.cos(t / 2), -Math.sin(t / 2)],
				[0, 0],
				[0, 0],
				[Math.cos(t / 2), Math.sin(t / 2)],
			];
		case "P":
			return [
				[1, 0],
				[0, 0],
				[0, 0],
				[Math.cos(t), Math.sin(t)],
			];
	}
	throw new Error(`no matrix for ${name}`);
}

const bit = (n: number, q: number) => 1 << (n - 1 - q);

function apply1(
	s: State,
	n: number,
	target: number,
	controls: number[],
	m: Complex[],
) {
	const tb = bit(n, target);
	let cmask = 0;
	for (const c of controls) cmask |= bit(n, c);
	const [a, b, c, d] = m;
	for (let i = 0; i < s.re.length; i++) {
		if (i & tb || (i & cmask) !== cmask) continue;
		const j = i | tb;
		const r0 = s.re[i];
		const i0 = s.im[i];
		const r1 = s.re[j];
		const i1 = s.im[j];
		s.re[i] = a[0] * r0 - a[1] * i0 + b[0] * r1 - b[1] * i1;
		s.im[i] = a[0] * i0 + a[1] * r0 + b[0] * i1 + b[1] * r1;
		s.re[j] = c[0] * r0 - c[1] * i0 + d[0] * r1 - d[1] * i1;
		s.im[j] = c[0] * i0 + c[1] * r0 + d[0] * i1 + d[1] * r1;
	}
}

/** Value of qubits qs (first = most significant) in basis index i. */
function sub(i: number, n: number, qs: number[]): number {
	let v = 0;
	for (const q of qs) v = (v << 1) | (i & bit(n, q) ? 1 : 0);
	return v;
}

/** Group basis indices by the value of the qubits not in qs; each group is ordered by sub(qs). */
function groups(n: number, qs: number[]): number[][] {
	const mask = qs.reduce((m, q) => m | bit(n, q), 0);
	const out = new Map<number, number[]>();
	for (let i = 0; i < 2 ** n; i++) {
		const key = i & ~mask;
		let g = out.get(key);
		if (!g) {
			g = new Array(2 ** qs.length);
			out.set(key, g);
		}
		g[sub(i, n, qs)] = i;
	}
	return [...out.values()];
}

export function applyOp(s: State, n: number, op: Op) {
	const q = op.qubits;
	switch (op.name) {
		case "M":
			return;
		case "CX":
		case "CY":
		case "CZ":
		case "CH":
			return apply1(s, n, q[1], [q[0]], matrix(op.name.slice(1)));
		case "CP":
			return apply1(s, n, q[1], [q[0]], matrix("P", op.param));
		case "CCX":
			return apply1(s, n, q[2], [q[0], q[1]], matrix("X"));
		case "CCZ":
			return apply1(s, n, q[2], [q[0], q[1]], matrix("Z"));
		case "SWAP":
			apply1(s, n, q[1], [q[0]], matrix("X"));
			apply1(s, n, q[0], [q[1]], matrix("X"));
			apply1(s, n, q[1], [q[0]], matrix("X"));
			return;
		case "ORACLE":
			for (let i = 0; i < s.re.length; i++)
				if (op.marked?.includes(sub(i, n, q))) {
					s.re[i] = -s.re[i];
					s.im[i] = -s.im[i];
				}
			return;
		case "DIFF":
			for (const g of groups(n, q)) {
				let mr = 0;
				let mi = 0;
				for (const i of g) {
					mr += s.re[i];
					mi += s.im[i];
				}
				mr /= g.length;
				mi /= g.length;
				for (const i of g) {
					s.re[i] = 2 * mr - s.re[i];
					s.im[i] = 2 * mi - s.im[i];
				}
			}
			return;
		case "QFT":
		case "IQFT": {
			const N = 2 ** q.length;
			const sign = op.name === "QFT" ? 1 : -1;
			for (const g of groups(n, q)) {
				const r = g.map((i) => s.re[i]);
				const m = g.map((i) => s.im[i]);
				for (let y = 0; y < N; y++) {
					let sr = 0;
					let si = 0;
					for (let x = 0; x < N; x++) {
						const ang = (sign * 2 * Math.PI * x * y) / N;
						sr += r[x] * Math.cos(ang) - m[x] * Math.sin(ang);
						si += r[x] * Math.sin(ang) + m[x] * Math.cos(ang);
					}
					s.re[g[y]] = sr / Math.sqrt(N);
					s.im[g[y]] = si / Math.sqrt(N);
				}
			}
			return;
		}
	}
	apply1(s, n, q[0], [], matrix(op.name, op.param));
}

export interface Run {
	n: number;
	steps: Step[];
	/** states[k] is the state after k steps (states[0] is the input). */
	states: State[];
}

export function runCircuit(n: number, steps: string[], init = ""): Run {
	const parsed = steps.map((s) => parseStep(s, n));
	let s = basisState(n, init);
	const states = [s];
	for (const st of parsed) {
		s = { re: s.re.slice(), im: s.im.slice() };
		for (const op of st.ops) applyOp(s, n, op);
		states.push(s);
	}
	return { n, steps: parsed, states };
}

// ---------------------------------------------------------------------------------------------
// Read-outs

export const prob = (s: State, i: number) => s.re[i] ** 2 + s.im[i] ** 2;

/** Probability that qubit q reads 1. */
export function probOne(s: State, n: number, q: number): number {
	let p = 0;
	for (let i = 0; i < s.re.length; i++) if (i & bit(n, q)) p += prob(s, i);
	return p;
}

/** Bloch vector (x, y, z) of qubit q's reduced state; its length is 1 only if q is unentangled. */
export function blochOf(
	s: State,
	n: number,
	q: number,
): [number, number, number] {
	const tb = bit(n, q);
	let p0 = 0;
	let p1 = 0;
	let cr = 0;
	let ci = 0; // rho01 = sum a_i0 conj(a_i1)
	for (let i = 0; i < s.re.length; i++) {
		if (i & tb) continue;
		const j = i | tb;
		p0 += prob(s, i);
		p1 += prob(s, j);
		cr += s.re[i] * s.re[j] + s.im[i] * s.im[j];
		ci += s.im[i] * s.re[j] - s.re[i] * s.im[j];
	}
	// rho = (I + x X + y Y + z Z) / 2: x = 2 Re rho01, y = -2 Im rho01, z = p0 - p1.
	return [2 * cr, -2 * ci, p0 - p1];
}

export const ket = (i: number, n: number) => i.toString(2).padStart(n, "0");

/** Short text for a complex number: 0.707, -0.5i, 0.354+0.354i. */
export function fmtComplex(re: number, im: number, d = 3): string {
	const z = (v: number) => (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v);
	const r = z(re);
	const m = z(im);
	const f = (v: number) => Number(v.toFixed(d)).toString();
	if (!m) return f(r);
	if (!r) return `${f(m)}i`;
	return `${f(r)}${m < 0 ? "−" : "+"}${f(Math.abs(m))}i`.replace(/^-/, "−");
}

// ---------------------------------------------------------------------------------------------
// Bloch-sphere helpers for the single-qubit figure

/** Rotation axis and angle of each single-qubit gate on the Bloch sphere. */
export const GATE_ROTATIONS: Record<
	string,
	{ axis: [number, number, number]; angle: number }
> = {
	X: { axis: [1, 0, 0], angle: Math.PI },
	Y: { axis: [0, 1, 0], angle: Math.PI },
	Z: { axis: [0, 0, 1], angle: Math.PI },
	H: { axis: [R, 0, R], angle: Math.PI },
	S: { axis: [0, 0, 1], angle: Math.PI / 2 },
	SDG: { axis: [0, 0, 1], angle: -Math.PI / 2 },
	T: { axis: [0, 0, 1], angle: Math.PI / 4 },
	TDG: { axis: [0, 0, 1], angle: -Math.PI / 4 },
	"RX(π/4)": { axis: [1, 0, 0], angle: Math.PI / 4 },
	"RY(π/4)": { axis: [0, 1, 0], angle: Math.PI / 4 },
};

/** Rotate v about unit axis k by angle a (Rodrigues' formula). */
export function rotate(
	v: number[],
	k: number[],
	a: number,
): [number, number, number] {
	const c = Math.cos(a);
	const s = Math.sin(a);
	const dot = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
	const cr = [
		k[1] * v[2] - k[2] * v[1],
		k[2] * v[0] - k[0] * v[2],
		k[0] * v[1] - k[1] * v[0],
	];
	return [0, 1, 2].map((i) => v[i] * c + cr[i] * s + k[i] * dot * (1 - c)) as [
		number,
		number,
		number,
	];
}

/** Bloch vector of cos(θ/2)|0> + e^{iφ} sin(θ/2)|1>. */
export const blochFromAngles = (
	theta: number,
	phi: number,
): [number, number, number] => [
	Math.sin(theta) * Math.cos(phi),
	Math.sin(theta) * Math.sin(phi),
	Math.cos(theta),
];

export function anglesFromBloch(v: number[]): { theta: number; phi: number } {
	const theta = Math.acos(Math.max(-1, Math.min(1, v[2])));
	let phi = Math.atan2(v[1], v[0]);
	if (Math.sin(theta) < 1e-9) phi = 0;
	if (phi < -1e-12) phi += 2 * Math.PI;
	return { theta, phi };
}

// ---------------------------------------------------------------------------------------------
// Measuring a qubit along an arbitrary axis (BasisMeasurement figure)

/**
 * State and measurement axis in one great circle of the Bloch sphere, at angles theta and alpha
 * from |0>. The observable A = n·σ has eigenvalues ±1; P(+1) = cos²((θ − α)/2),
 * <A> = cos(θ − α) and ΔA = |sin(θ − α)|.
 */
export function axisMeasurement(theta: number, alpha: number) {
	const d = theta - alpha;
	return {
		pPlus: Math.cos(d / 2) ** 2,
		mean: Math.cos(d),
		sd: Math.abs(Math.sin(d)),
	};
}

/** Small seeded PRNG (mulberry32) so figures replay the same shots. */
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
