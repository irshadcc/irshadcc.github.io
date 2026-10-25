// Data for the static charts in the game theory post, computed from gameTheory.ts at build time.
import { auctionRevenue, games, hedgeSelfPlay, rpsMatrix } from "./gameTheory";

/** Step-function lookup so a LineChart series can draw a per-round sequence. */
const at = (xs: number[]) => (t: number) =>
	xs[Math.max(0, Math.min(xs.length - 1, Math.round(t) - 1))];

/** Hedge self-play in Rock–Paper–Scissors (η = 0.1, 2000 rounds), as in SimplexDynamics. */
export const hedge = hedgeSelfPlay(
	rpsMatrix(1, 1),
	[0.5, 0.3, 0.2],
	[0.2, 0.3, 0.5],
	0.1,
	2000,
);
export const hedgeRock = at(hedge.last.map((x) => x[0]));
export const hedgeRockAvg = at(hedge.avg.map((x) => x[0]));
export const hedgeAvgRegret = at(hedge.regretRow.map((r, t) => r / (t + 1)));
/** Hedge's average-regret bound for payoffs in [-1, 1]: ln 3 / (ηT) + η / 2, with η = 0.1. */
export const hedgeBound = (t: number) => Math.log(3) / (0.1 * t) + 0.05;

/** Revenue of the three auction formats with n uniform bidders (200,000 draws, seed 1). */
export const auctionTable = [2, 3, 5, 10].map((n) => ({
	n,
	theory: (n - 1) / (n + 1),
	...auctionRevenue(n, 200000, 1, 0.5),
}));

/** The 2 × 3 zero-sum game left after removing r3: Row's payoff against each column as a function of p = P(r1). */
const Z = games.zeroSum.A;
export const zeroSumLines = [0, 1, 2].map(
	(j) => (p: number) => p * Z[0][j] + (1 - p) * Z[1][j],
);
export const zeroSumEnvelope = (p: number) =>
	Math.min(...zeroSumLines.map((f) => f(p)));

/** p-beauty contest with p = 2/3: the bound after k rounds of eliminating dominated guesses. */
export const beautyRounds = Array.from(
	{ length: 9 },
	(_, k) => 100 * (2 / 3) ** k,
);
