// A toy model of one warp scheduler, to show how Nsight Compute's "Scheduler Statistics" numbers
// relate. Every resident warp loops forever over the same pattern: it issues `ilp` independent
// instructions on consecutive cycles, then waits `latency` cycles for a result before the next
// batch. Each cycle the scheduler issues one instruction from one eligible warp (loose
// round-robin) or, if no warp is eligible, issues nothing.
//
// Outputs are the quantities Nsight Compute reports per scheduler:
//   issueActive     smsp__issue_active.avg.per_cycle_active  ("Issued Warp Per Scheduler")
//   avgEligible     smsp__warps_eligible.avg.per_cycle_active ("Eligible Warps Per Scheduler")
//   noEligiblePct   100 - issueActive * 100                   ("No Eligible")
//   warpCyclesPerIssue  average cycles between two issued instructions of a warp
// With a single issue slot per cycle, "One or More Eligible" equals issueActive.
// The functions are pure so they can be tested with `node --experimental-strip-types`.

export interface SchedulerStats {
	warps: number;
	issueActive: number;
	avgEligible: number;
	noEligiblePct: number;
	warpCyclesPerIssue: number;
}

export function simulateScheduler(
	warps: number,
	ilp: number,
	latency: number,
	cycles = 40000,
): SchedulerStats {
	const ready = new Array<number>(warps).fill(0);
	const pos = new Array<number>(warps).fill(0);
	let rr = 0;
	let issued = 0;
	let eligibleSum = 0;
	let counted = 0;
	const warmup = Math.floor(cycles / 4);
	for (let t = 0; t < cycles; t++) {
		let pick = -1;
		let eligible = 0;
		for (let j = 0; j < warps; j++) {
			const w = (rr + j) % warps;
			if (ready[w] <= t) {
				eligible++;
				if (pick < 0) pick = w;
			}
		}
		if (t >= warmup) {
			counted++;
			eligibleSum += eligible;
			if (pick >= 0) issued++;
		}
		if (pick >= 0) {
			pos[pick]++;
			if (pos[pick] === ilp) {
				pos[pick] = 0;
				ready[pick] = t + 1 + latency;
			} else {
				ready[pick] = t + 1;
			}
			rr = (pick + 1) % warps;
		}
	}
	const issueActive = issued / counted;
	return {
		warps,
		issueActive,
		avgEligible: eligibleSum / counted,
		noEligiblePct: 100 * (1 - issueActive),
		warpCyclesPerIssue: warps / issueActive,
	};
}

/** Closed form for the same model: each warp needs ilp + latency cycles per ilp instructions. */
export function expectedIssueActive(
	warps: number,
	ilp: number,
	latency: number,
): number {
	return Math.min(1, (warps * ilp) / (ilp + latency));
}
