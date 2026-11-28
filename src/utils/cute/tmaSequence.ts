// One TMA load synchronised by an mbarrier, as a SequenceSteps diagram for "The CuTe DSL API,
// Function by Function" (src/content/posts/cute-dsl-api.mdx). The steps follow the kernel that was
// run for the post (one CTA, one warp, a 4 × 4 Float32 box) and the PTX the DSL emitted for it.

import type { Message, Participant } from "../payments/sequence";

export const tmaParticipants: Participant[] = [
	{ id: "warp", label: "Warp", sub: "32 threads" },
	{ id: "t0", label: "Thread 0", sub: "tidx == 0" },
	{ id: "bar", label: "mbarrier", sub: "8 bytes of smem" },
	{ id: "tma", label: "TMA unit", sub: "copy engine" },
	{ id: "gmem", label: "Global", sub: "6 × 10 tensor" },
	{ id: "smem", label: "Shared", sub: "4 × 4 tile" },
];

export const tmaMessages: Message[] = [
	{
		from: "t0",
		to: "bar",
		label: "mbarrier_init(bar, 1)",
		step: "init",
		note: "One thread initialises the barrier with an arrival count of 1: one thread will arrive per phase. The barrier starts in phase 0 with no bytes expected.",
	},
	{
		from: "warp",
		to: "bar",
		label: "init_fence + barrier",
		step: "fence",
		note: "<code>mbarrier_init_fence()</code> makes the initialisation visible to the TMA unit (it is <code>fence.mbarrier_init.release.cluster</code>), and <code>barrier()</code> makes every thread wait until it has happened.",
	},
	{
		from: "t0",
		to: "bar",
		label: "arrive_and_expect_tx(bar, 64)",
		step: "expect",
		note: "Thread 0 arrives (pending arrivals 1 → 0) and announces 64 bytes of transactions: 16 Float32. The phase cannot complete until those 64 bytes are counted off.",
	},
	{
		from: "warp",
		to: "tma",
		label: "cute.copy(atom, tAgA, tAsA, tma_bar_ptr=bar)",
		step: "issue",
		note: "The whole warp calls <code>cute.copy</code>; the generated code runs <code>elect.sync</code> and one lane issues <code>cp.async.bulk.tensor.2d…mbarrier::complete_tx::bytes</code> with the descriptor, the box corner as (column, row) = (4·bn, 4·bm) and the barrier. Calling it from inside <code>if tidx == 0</code> hung the kernel.",
	},
	{
		from: "tma",
		to: "gmem",
		label: "read box, clip at (6, 10)",
		step: "read",
		tone: 3,
		note: "The TMA unit reads the box using the shape and strides in the descriptor. Elements outside the tensor are not read: the padding columns of the allocation are never touched.",
	},
	{
		from: "tma",
		to: "smem",
		label: "write 16 values, OOB → 0",
		step: "write",
		tone: 3,
		note: "The box lands in shared memory in the layout the atom was built for; out-of-bounds elements are written as zero.",
	},
	{
		from: "tma",
		to: "bar",
		label: "complete_tx 64 bytes",
		step: "complete",
		tone: 1,
		note: "As the bytes land, the TMA unit subtracts them from the barrier's transaction count. With 0 pending arrivals and 0 pending bytes, phase 0 completes and the barrier moves to phase 1. The byte count is always the full box, 64, even when part of it was zero-filled.",
	},
	{
		from: "bar",
		to: "warp",
		label: "mbarrier_wait(bar, 0) returns",
		step: "wait",
		reply: true,
		tone: 1,
		note: "Every thread spins in <code>mbarrier.try_wait.parity</code> on phase 0 and leaves the loop once that phase has completed. Only now may it read the tile.",
	},
];
