// A three-tier CLOS GPU cluster, after MegaScale §3.4 (arXiv:2402.15627), and how traffic
// between two GPUs is routed through it. Shared by PhysicalClusterTopology (draws it) and
// LogicalDistributedTopology (labels parallel groups with the hops their traffic takes).
//
//   server: `gpusPerServer` GPUs, each with its own NIC on its own rail
//   pod:    `serversPerPod` servers; NIC i of every server goes to ToR i of the pod (multi-rail)
//   spine:  every ToR in a pod connects to all `spinesPerPod` spines of that pod
//   core:   spine s of every pod connects to core s (one core per spine plane)
//
// GPUs are numbered server by server: GPU g sits in server floor(g / gpusPerServer).

export interface NetworkShape {
	gpusPerServer: number;
	serversPerPod: number;
	spinesPerPod: number;
}

/** How far apart two GPUs are, by the switches their traffic crosses. */
export type HopClass = "nvlink" | "tor" | "spine" | "core";

export const HOP_CLASSES: { id: HopClass; switches: number; name: string }[] = [
	{ id: "nvlink", switches: 0, name: "NVLink (same server)" },
	{ id: "tor", switches: 1, name: "1 switch (same ToR)" },
	{ id: "spine", switches: 3, name: "3 switches (via spine)" },
	{ id: "core", switches: 5, name: "5 switches (via core)" },
];

export const torId = (pod: number, rail: number) => `tor-${pod}-${rail}`;
export const spineId = (pod: number, s: number) => `spine-${pod}-${s}`;
export const coreId = (s: number) => `core-${s}`;

// Integer hash (murmur3 finaliser), used to spread flows over equal-cost spines. A plain
// (a + b) % spines would send every flow between ranks that are multiples of the spine count
// through the same spine.
const mix = (x: number) => {
	let h = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
	h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
	return (h ^ (h >>> 16)) >>> 0;
};

export class ClusterNetwork {
	constructor(readonly shape: NetworkShape) {
		for (const [k, v] of Object.entries(shape)) {
			if (!Number.isInteger(v) || v < 1) throw new Error(`ClusterNetwork: ${k} must be a positive integer, got ${v}`);
		}
	}

	/** Server, local GPU index (= rail) and pod of a GPU. */
	place(gpu: number) {
		const server = Math.floor(gpu / this.shape.gpusPerServer);
		return { server, rail: gpu % this.shape.gpusPerServer, pod: Math.floor(server / this.shape.serversPerPod) };
	}

	/**
	 * The switches a flow between GPUs a and b crosses, in order. Where several spines would do
	 * (ECMP), a hash of the unordered pair picks one, so a -> b and b -> a match.
	 */
	route(a: number, b: number): { cls: HopClass; switches: string[] } {
		const pa = this.place(a);
		const pb = this.place(b);
		if (pa.server === pb.server) return { cls: "nvlink", switches: [] };
		const s = mix(Math.min(a, b) * 65537 + Math.max(a, b)) % this.shape.spinesPerPod;
		if (pa.pod === pb.pod) {
			if (pa.rail === pb.rail) return { cls: "tor", switches: [torId(pa.pod, pa.rail)] };
			return { cls: "spine", switches: [torId(pa.pod, pa.rail), spineId(pa.pod, s), torId(pb.pod, pb.rail)] };
		}
		return {
			cls: "core",
			switches: [torId(pa.pod, pa.rail), spineId(pa.pod, s), coreId(s), spineId(pb.pod, s), torId(pb.pod, pb.rail)],
		};
	}
}
