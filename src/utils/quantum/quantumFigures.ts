// Circuits and chart data for the quantum computing post. Every circuit here is run by
// quantum.ts at build time; the same lists are cross-checked against Qiskit's Statevector by
// the script described in the post's verification notes.

export interface CircuitSpec {
	n: number;
	steps: string[];
	init?: string;
	labels?: string[];
}

export const interference: CircuitSpec = {
	n: 1,
	steps: [
		"H 0 | One H gate: |0> becomes (|0> + |1>)/√2. Measuring now gives 0 or 1 with probability 1/2 each.",
		"H 0 | A second H: the two paths into |1> carry amplitudes +1/2 and −1/2 and cancel; the paths into |0> add to 1. The qubit is back in |0> with certainty.",
	],
};

export const bell: CircuitSpec = {
	n: 2,
	steps: [
		"H 0 | H puts q0 in (|0> + |1>)/√2. The state is (|00> + |10>)/√2: still a product, q1 is |0>.",
		"CX 0 1 | CNOT flips q1 when q0 is 1, so |10> becomes |11>. The state (|00> + |11>)/√2 cannot be written as (state of q0) ⊗ (state of q1): the qubits are entangled.",
		"M 0 1 | Measuring gives 00 or 11, each with probability 1/2, never 01 or 10. Each qubit's Bloch vector has length 0: alone, each qubit is a fair coin; together they always agree.",
	],
};

export const deutsch: CircuitSpec = {
	n: 2,
	labels: ["x", "y"],
	steps: [
		"X 1 | Put the output qubit y in |1>.",
		"H 0; H 1 | H on both: x is (|0> + |1>)/√2, y is |−> = (|0> − |1>)/√2. Note the minus signs on |01> and |11>.",
		"CX 0 1 @U_f | The oracle U_f maps |x, y> to |x, y ⊕ f(x)>, here with f(x) = x (balanced). With y in |−>, flipping y only multiplies by −1, so the phase (−1)^f(x) is kicked back onto x: the amplitudes of |10> and |11> change sign.",
		"H 0 | H turns x = (|0> − |1>)/√2 into |1>. x reads 1 because f is balanced; a constant f would leave it 0. One call to f decided it.",
		"M 0 | Measure x: 1 with probability 1.",
	],
};

export const bernsteinVazirani: CircuitSpec = {
	n: 4,
	labels: ["x0", "x1", "x2", "y"],
	steps: [
		"X 3 | Ancilla y to |1>.",
		"H 0; H 1; H 2; H 3 | Superpose all 8 inputs x; y becomes |−>.",
		"CX 0 3 @U_f; CX 2 3 @U_f | Oracle for f(x) = s·x mod 2 with hidden s = 101: one CNOT from each qubit where s has a 1. Phase kickback gives each |x> the sign (−1)^(s·x).",
		"H 0; H 1; H 2 | H on the inputs undoes the superposition and leaves exactly |s> = |101>.",
		"M 0 1 2 | Measuring the inputs gives 101 with probability 1, after one oracle call. Classically you need 3 calls, one per bit.",
	],
};

export const grover: CircuitSpec = {
	n: 3,
	steps: [
		"H 0; H 1; H 2 | Uniform superposition: each of the 8 amplitudes is 1/√8 ≈ 0.354, so P(101) = 1/8.",
		"ORACLE 5 | The oracle flips the sign of the marked state |101>. Probabilities do not change yet, but the mean amplitude drops to 0.265.",
		"DIFF | Diffusion reflects every amplitude about the mean (a → 2·mean − a). The marked amplitude becomes 0.884, the others 0.177. P(101) = 0.781.",
		"ORACLE 5 | Second oracle call: flip the marked sign again.",
		"DIFF | Second reflection: the marked amplitude is 0.972, P(101) = 0.945. A third round would overshoot.",
		"M | Measure: 101 with probability 0.945. Guessing classically takes (N + 1)/2 = 4.5 queries on average for N = 8.",
	],
};

export const teleportation: CircuitSpec = {
	n: 3,
	labels: ["ψ", "a", "b"],
	steps: [
		"RY(pi/3) 0; S 0 @prepare | Alice's qubit holds |ψ> = 0.866|0> + 0.5i|1> (Bloch vector (0, 0.866, 0.5)). The circuit never reads these numbers.",
		"H 1 | Make the shared pair: H on a ...",
		"CX 1 2 | ... and CNOT a → b gives the Bell pair (|00> + |11>)/√2 between Alice (a) and Bob (b).",
		"CX 0 1 | Alice entangles ψ with her half: CNOT ψ → a.",
		"H 0 | H on ψ. Now each of Alice's four outcomes 00, 01, 10, 11 appears with probability 1/4, and Bob's qubit holds ψ up to an X and/or Z.",
		"CX 1 2 @X if a=1 | Bob applies X when Alice's a-bit is 1 (drawn as a controlled gate: the deferred-measurement form of a classically controlled X).",
		"CZ 0 2 @Z if ψ=1 | ... and Z when her ψ-bit is 1. Bob's Bloch vector is now (0, 0.866, 0.5): he holds |ψ>, and Alice's copy is gone.",
	],
};

export const bitFlipCode: CircuitSpec = {
	n: 5,
	labels: ["d0", "d1", "d2", "s1", "s2"],
	steps: [
		"RY(pi/3) 0 @prepare | The logical state to protect: 0.866|0> + 0.5|1> on d0.",
		"CX 0 1; CX 0 2 | Encode: 0.866|000> + 0.5|111>. This is not cloning: the three qubits are entangled, not three copies of the state.",
		"X 1 @error | Noise flips d1: 0.866|010> + 0.5|101>.",
		"CX 0 3; CX 1 3 | Syndrome bit s1 = d0 ⊕ d1, copied into an ancilla.",
		"CX 1 4; CX 2 4 | Syndrome bit s2 = d1 ⊕ d2. Both branches give s = 11, so measuring s says 'd1 flipped' and reveals nothing about 0.866 or 0.5.",
		"CCX 3 4 1 @fix d1 | Flip d1 back when s = 11. (The full decoder adds a Toffoli for s = 10, fix d0, and s = 01, fix d2.) The data is 0.866|000> + 0.5|111> again.",
	],
};

// ---------------------------------------------------------------------------------------------
// Chart data

/** Grover success probability after k iterations: sin²((2k + 1) θ), sin θ = 1/√N. */
export function groverSuccess(nQubits: number, k: number): number {
	const theta = Math.asin(1 / Math.sqrt(2 ** nQubits));
	return Math.sin((2 * k + 1) * theta) ** 2;
}

/** Optimal Grover iteration count floor(π / (4θ)). */
export const groverIterations = (nQubits: number) =>
	Math.floor(Math.PI / (4 * Math.asin(1 / Math.sqrt(2 ** nQubits))));

export function modPow(a: number, e: number, N: number): number {
	let r = 1;
	let b = a % N;
	let x = e;
	while (x > 0) {
		if (x & 1) r = (r * b) % N;
		b = (b * b) % N;
		x >>= 1;
	}
	return r;
}

/**
 * Probability of reading y from the t-qubit counting register in Shor's period finding for
 * a^x mod N (exact: sums over every value of the work register).
 */
export function shorDistribution(a: number, N: number, t: number): number[] {
	const Q = 2 ** t;
	const byValue = new Map<number, number[]>();
	for (let x = 0; x < Q; x++) {
		const v = modPow(a, x, N);
		if (!byValue.has(v)) byValue.set(v, []);
		byValue.get(v)?.push(x);
	}
	const p = new Array<number>(Q).fill(0);
	for (let y = 0; y < Q; y++) {
		for (const xs of byValue.values()) {
			let re = 0;
			let im = 0;
			for (const x of xs) {
				re += Math.cos((2 * Math.PI * x * y) / Q);
				im += Math.sin((2 * Math.PI * x * y) / Q);
			}
			p[y] += (re * re + im * im) / (Q * Q);
		}
	}
	return p;
}

/** Continued-fraction convergents of p / q, as [numerator, denominator]. */
export function convergents(p: number, q: number): [number, number][] {
	const out: [number, number][] = [];
	let [h0, h1, k0, k1] = [0, 1, 1, 0];
	let [a, b] = [p, q];
	while (b) {
		const c = Math.floor(a / b);
		[h0, h1] = [h1, c * h1 + h0];
		[k0, k1] = [k1, c * k1 + k0];
		out.push([h1, k1]);
		[a, b] = [b, a - c * b];
	}
	return out;
}

/** VQE toy: energy of RY(θ)|0> under H = Z + X is cos θ + sin θ. */
export const vqeEnergy = (theta: number) => Math.cos(theta) + Math.sin(theta);

/** Shor, N = 15, a = 7, 4 counting qubits: peaks at y = 0, 4, 8, 12. */
export const shor15 = shorDistribution(7, 15, 4);
/** Shor, N = 21, a = 2 (period 6), 6 counting qubits. */
export const shor21 = shorDistribution(2, 21, 6);
/** Grover on 6 qubits (N = 64), success probability after k = 0..12 iterations. */
export const grover64 = Array.from({ length: 13 }, (_, k) =>
	groverSuccess(6, k),
);

// ---------------------------------------------------------------------------------------------
// Quantum mechanics section

/** Rabi drive at Ω/2π = 25 MHz: a π pulse takes π/Ω = 20 ns. */
export const RABI_OMEGA = 2 * Math.PI * 25e6;

/**
 * P(1) after driving |0> for t seconds with Rabi frequency Ω and detuning Δ = ratio·Ω,
 * H = (ħ/2)(ΔZ + ΩX): P = Ω²/(Ω² + Δ²) · sin²(√(Ω² + Δ²) t / 2).
 */
export function rabiP1(
	tSeconds: number,
	detuningRatio: number,
	omega = RABI_OMEGA,
): number {
	const d = detuningRatio * omega;
	const eff = Math.hypot(omega, d);
	return (omega ** 2 / eff ** 2) * Math.sin((eff * tSeconds) / 2) ** 2;
}

/** Illustrative coherence times (not a specific device): T1 = 100 µs, T2 = 80 µs. */
export const T1_US = 100;
export const T2_US = 80;
/** Ramsey fringe: x-component of a qubit detuned by 25 kHz, decaying with T2. */
export const ramseyX = (tUs: number) =>
	Math.cos(2 * Math.PI * 0.025 * tUs) * Math.exp(-tUs / T2_US);

/** CHSH value |S| for the singlet with angles a = 0, b = θ, a' = 2θ, b' = 3θ: |3 cos θ − cos 3θ|. */
export const chshS = (thetaDeg: number) => {
	const t = (thetaDeg * Math.PI) / 180;
	return Math.abs(3 * Math.cos(t) - Math.cos(3 * t));
};
