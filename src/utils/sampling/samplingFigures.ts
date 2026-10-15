// Data for the static charts in the sampling post, computed from sampling.ts at build time.
import { mulberry32 } from "./sampling";

/** Three lazy random walks of 400 steps (eqs 14.28–14.30): stay with 1/2, move ±1 with 1/4 each. */
export const walks: number[][] = [1, 2, 3].map((seed) => {
	const rng = mulberry32(seed);
	const z = [0];
	for (let t = 0; t < 400; t++) {
		const u = rng();
		z.push(z[t] + (u < 0.25 ? 1 : u < 0.5 ? -1 : 0));
	}
	return z;
});

/** Step function lookup so a LineChart series can draw a walk. */
export const walkAt = (k: number) => (t: number) =>
	walks[k][Math.max(0, Math.min(400, Math.round(t)))];
