// Inside one node: how its GPUs, NICs, PCIe switches and CPU sockets connect, for the
// TopologyExplorer's node view. An illustrative, configurable model of a multi-GPU server, not a
// specific product:
//
//   - `sockets` CPU sockets, one NUMA node each, joined by the CPU interconnect (UPI / xGMI);
//   - PCIe switches with `gpusPerSwitch` GPUs each, spread evenly over the sockets (each switch
//     under its own host bridge of its socket);
//   - one NIC per GPU on the GPU's PCIe switch (NIC i is rail i, wired to ToR i in network.ts);
//   - optionally an NVSwitch fabric giving every GPU pair `nvlinks` bonded NVLinks.

export interface NodeShape {
	gpus: number;
	sockets: number;
	gpusPerSwitch: number;
	nvswitch: boolean;
	/** NVLinks per GPU pair through the NVSwitch fabric (18 on H100). */
	nvlinks: number;
}

export type Device = { kind: "gpu" | "nic"; i: number };

export class NodeTopology {
	readonly shape: NodeShape;
	readonly switches: number;

	constructor(shape: NodeShape) {
		for (const k of ["gpus", "sockets", "gpusPerSwitch", "nvlinks"] as const) {
			const v = shape[k];
			if (!Number.isInteger(v) || v < 1)
				throw new Error(
					`NodeTopology: ${k} must be a positive integer, got ${v}`,
				);
		}
		this.shape = shape;
		this.switches = Math.ceil(shape.gpus / shape.gpusPerSwitch);
	}

	/** PCIe switch of a GPU or NIC (NIC i shares GPU i's switch). */
	switchOf(d: Device) {
		return Math.floor(d.i / this.shape.gpusPerSwitch);
	}
	/** Socket (NUMA node) a PCIe switch hangs off: switches are spread evenly, in order. */
	socketOfSwitch(k: number) {
		return Math.floor((k * this.shape.sockets) / this.switches);
	}
	socketOf(d: Device) {
		return this.socketOfSwitch(this.switchOf(d));
	}

	/** The boxes a transfer between two devices passes, for drawing: ids as in NodeLayout. */
	path(a: Device, b: Device): string[] {
		const id = (d: Device) => `${d.kind}:${d.i}`;
		if (a.kind === b.kind && a.i === b.i) return [id(a)];
		if (a.kind === "gpu" && b.kind === "gpu" && this.shape.nvswitch)
			return [id(a), "nvswitch", id(b)];
		const sa = this.switchOf(a);
		const sb = this.switchOf(b);
		if (sa === sb) return [id(a), `sw:${sa}`, id(b)];
		const ca = this.socketOfSwitch(sa);
		const cb = this.socketOfSwitch(sb);
		const cpus = ca === cb ? [`cpu:${ca}`] : [`cpu:${ca}`, `cpu:${cb}`];
		return [id(a), `sw:${sa}`, ...cpus, `sw:${sb}`, id(b)];
	}
}

export interface Box {
	x: number;
	y: number;
	w: number;
	h: number;
}

/**
 * Where each part of a node is drawn, top to bottom: the NVSwitch bar, the GPUs (grouped by PCIe
 * switch, each with its NIC beside it), the PCIe switches and the CPU sockets, so every wire runs
 * straight down. Ids: "nvswitch", "gpu:i", "nic:i", "sw:k", "cpu:s".
 */
export class NodeLayout {
	readonly boxes = new Map<string, Box>();
	readonly width: number;
	readonly height: number;
	readonly node: NodeTopology;
	readonly sq: number;
	readonly nicW = 30;

	constructor(node: NodeTopology, sq = 26) {
		this.node = node;
		this.sq = sq;
		const { gpus, gpusPerSwitch, nvswitch, sockets } = node.shape;
		const M = 10;
		const pairW = sq + 3 + this.nicW; // a GPU and its NIC
		const gap = 10; // between pairs on one switch
		const colGap = 22; // between switches
		const colW = gpusPerSwitch * pairW + (gpusPerSwitch - 1) * gap;
		const nvY = M + 4;
		const gpuY = nvswitch ? nvY + 20 + 34 : M + 14;
		const swY = gpuY + sq + 30;
		const cpuY = swY + 22 + 36;
		const colX = (k: number) => M + k * (colW + colGap);
		for (let k = 0; k < node.switches; k++) {
			const x = colX(k);
			this.boxes.set(`sw:${k}`, { x, y: swY, w: colW, h: 22 });
			for (let j = 0; j < gpusPerSwitch; j++) {
				const i = k * gpusPerSwitch + j;
				if (i >= gpus) break;
				const gx = x + j * (pairW + gap);
				this.boxes.set(`gpu:${i}`, { x: gx, y: gpuY, w: sq, h: sq });
				this.boxes.set(`nic:${i}`, {
					x: gx + sq + 3,
					y: gpuY,
					w: this.nicW,
					h: sq,
				});
			}
		}
		const right = colX(node.switches - 1) + colW;
		if (nvswitch)
			this.boxes.set("nvswitch", { x: M, y: nvY, w: right - M, h: 20 });
		// A socket spans the switches under it; one with none gets a slot after the last.
		let spare = right + colGap;
		for (let s = 0; s < sockets; s++) {
			const ks = Array.from({ length: node.switches }, (_, k) => k).filter(
				(k) => node.socketOfSwitch(k) === s,
			);
			if (ks.length) {
				const x0 = colX(ks[0]);
				const x1 = colX(ks[ks.length - 1]) + colW;
				this.boxes.set(`cpu:${s}`, { x: x0, y: cpuY, w: x1 - x0, h: 26 });
			} else {
				this.boxes.set(`cpu:${s}`, { x: spare, y: cpuY, w: colW, h: 26 });
				spare += colW + colGap;
			}
		}
		let w = 0;
		let h = 0;
		for (const b of this.boxes.values()) {
			w = Math.max(w, b.x + b.w);
			h = Math.max(h, b.y + b.h);
		}
		this.width = w + M;
		this.height = h + M;
	}

	/** The GPU under (x, y), if any. */
	hit(x: number, y: number): number | null {
		for (let i = 0; i < this.node.shape.gpus; i++) {
			const b = this.boxes.get(`gpu:${i}`);
			if (b && x >= b.x && x < b.x + b.w && y >= b.y && y < b.y + b.h) return i;
		}
		return null;
	}
}
