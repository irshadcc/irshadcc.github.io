// Concrete numbers for the shared EP example (epDispatch.ts: 8 tokens, 8 experts, top-2), so that
// the tensors on the edges of MoeEpSteps' graphs show real values: hidden size 4, experts are
// SwiGLU MLPs of width 2, and the gate weight is applied inside the activation, as Megatron and
// DeepGEMM do. The weights are small multiples of 0.5 chosen to give readable numbers; the final
// outputs are checked against a plain PyTorch MoE.
import {
	COPIES,
	type Copy,
	EXPERTS,
	GATES,
	ROUTING,
	TOKENS,
	TOP_K,
} from "./epDispatch";

export const H = 4;
export const I = 2;

const range = (n: number) => [...Array(n).keys()];

/** Hidden state of each of the 8 tokens. */
export const XS: number[][] = range(TOKENS).map((t) =>
	range(H).map((j) => (((3 * t + 2 * j + 1) % 7) - 3) * 0.5),
);
/** W1 of expert e: [2I, H], gate rows then up rows. */
export const W1S: number[][][] = range(EXPERTS).map((e) =>
	range(2 * I).map((r) =>
		range(H).map((c) => (((2 * e + 3 * r + 3 * c + 1) % 5) - 2) * 0.5),
	),
);
/** W2 of expert e: [H, I]. */
export const W2S: number[][][] = range(EXPERTS).map((e) =>
	range(H).map((r) =>
		range(I).map((c) => (((e + 3 * r + 2 * c + 2) % 5) - 2) * 0.5),
	),
);

const silu = (v: number) => v / (1 + Math.exp(-v));
const matvec = (m: number[][], v: number[]) =>
	m.map((row) => row.reduce((a, w, j) => a + w * v[j], 0));

/** Expert e's output for token t's copy, scaled by its gate: W2 (g · silu(W1_gate x) ⊙ W1_up x). */
export function copyOutput(c: Copy): number[] {
	const z = matvec(W1S[c.expert], XS[c.t]);
	const act = range(I).map((i) => silu(z[i]) * z[I + i] * c.gate);
	return matvec(W2S[c.expert], act);
}

/** The MoE layer's output per token: the sum of its copies' outputs. */
export const YS: number[][] = range(TOKENS).map((t) =>
	COPIES.filter((c) => c.t === t)
		.map(copyOutput)
		.reduce((a, o) => a.map((v, j) => v + o[j]), new Array<number>(H).fill(0)),
);

/** A rank's partial output when it runs only its own experts: zero rows for tokens it doesn't serve. */
export function partialOutput(rank: number, t: number): number[] {
	return COPIES.filter((c) => c.t === t && c.dest === rank)
		.map(copyOutput)
		.reduce((a, o) => a.map((v, j) => v + o[j]), new Array<number>(H).fill(0));
}

/** Softmax-style routing probabilities shown on routing edges: the top-k gates, zero elsewhere. */
export function probsRow(t: number): number[] {
	return range(EXPERTS).map((e) => {
		const k = ROUTING[t].indexOf(e);
		return k < 0 ? 0 : GATES[t][k];
	});
}

export const routingMapRow = (t: number): number[] =>
	range(EXPERTS).map((e) => (ROUTING[t].includes(e) ? 1 : 0));

export const TOPK = TOP_K;
