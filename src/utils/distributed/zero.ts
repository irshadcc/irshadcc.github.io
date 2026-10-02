// Per-GPU memory and communication of data parallelism and the three ZeRO stages, used by
// ZeroMemory.astro. Follows the ZeRO paper's accounting for mixed-precision Adam: 2 bytes per
// parameter for the 16-bit weights, 2 for their gradients, and K = 12 for the optimizer state
// (a 32-bit master copy of the weights, plus Adam's two moments). Activations are left out.

export type ZeroStage = 0 | 1 | 2 | 3;

export const STAGE_NAMES: Record<ZeroStage, string> = {
	0: "Data parallel",
	1: "ZeRO-1",
	2: "ZeRO-2",
	3: "ZeRO-3 / FSDP",
};

export const STAGE_SHARDS: Record<ZeroStage, string> = {
	0: "nothing sharded",
	1: "optimizer state",
	2: "+ gradients",
	3: "+ parameters",
};

export interface ZeroMemory {
	stage: ZeroStage;
	/** Bytes per GPU. */
	params: number;
	grads: number;
	optim: number;
	total: number;
	/** Bytes each GPU sends per step, with ring collectives on 16-bit values. */
	traffic: number;
}

export function zeroMemory(
	stage: ZeroStage,
	psi: number,
	n: number,
	k = 12,
): ZeroMemory {
	const shard = (bytes: number, sharded: boolean) =>
		sharded ? bytes / n : bytes;
	const params = shard(2 * psi, stage >= 3);
	const grads = shard(2 * psi, stage >= 2);
	const optim = shard(k * psi, stage >= 1);
	// All-reduce (or reduce-scatter + all-gather) moves 2Ψ values; ZeRO-3 gathers the
	// parameters twice, for 3Ψ. Each value is 2 bytes, and a ring sends (n-1)/n of it.
	const volume = stage === 3 ? 3 * psi : 2 * psi;
	const traffic = n > 1 ? (2 * volume * (n - 1)) / n : 0;
	return {
		stage,
		params,
		grads,
		optim,
		total: params + grads + optim,
		traffic,
	};
}

export const fmtBytes = (b: number) => {
	const gb = b / 1e9;
	if (gb >= 1000) return `${(gb / 1000).toFixed(gb >= 10000 ? 0 : 1)} TB`;
	if (gb >= 100) return `${gb.toFixed(0)} GB`;
	if (gb >= 1) return `${gb.toFixed(1)} GB`;
	return `${(gb * 1000).toFixed(0)} MB`;
};

/** The four stages as rows of HTML bars, on a shared linear scale. */
export function drawZero(psi: number, n: number, gpuBytes: number): string {
	const rows = ([0, 1, 2, 3] as ZeroStage[]).map((s) => zeroMemory(s, psi, n));
	const max = Math.max(gpuBytes, ...rows.map((r) => r.total));
	const pct = (b: number) => `${((100 * b) / max).toFixed(3)}%`;
	const line = pct(gpuBytes);
	return rows
		.map((r) => {
			const fits = r.total <= gpuBytes;
			return `<div class="row">
	<div class="name"><b>${STAGE_NAMES[r.stage]}</b><span>${STAGE_SHARDS[r.stage]}</span></div>
	<div class="track">
		<div class="seg p" style="width:${pct(r.params)}" title="16-bit parameters: ${fmtBytes(r.params)}"></div>
		<div class="seg g" style="width:${pct(r.grads)}" title="16-bit gradients: ${fmtBytes(r.grads)}"></div>
		<div class="seg o" style="width:${pct(r.optim)}" title="Optimizer state (fp32 weights, Adam moments): ${fmtBytes(r.optim)}"></div>
		<div class="cap" style="left:${line}"></div>
	</div>
	<div class="num ${fits ? "fits" : "over"}">${fmtBytes(r.total)}</div>
	<div class="num traffic">${n > 1 ? fmtBytes(r.traffic) : "—"}</div>
</div>`;
		})
		.join("");
}
