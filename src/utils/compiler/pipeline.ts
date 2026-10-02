// Stages of TorchInductor on the running example, for InductorPipeline.astro.
//
// The artifacts (FX nodes, IR bodies, scheduler dependencies, fusion order and scores, the
// generated kernel) come verbatim from inductorExample.ts. This file only decides what each
// stage shows: one row per FX node, a tag on the right of each row, the edges between rows,
// boxes around fused groups, and a detail text per node. The reasons given for each
// realize-or-inline decision follow GraphLowering.run_node and StorageBox.should_realize_on_reuse
// in torch 2.13.0.

import {
	fusionScores,
	fxNodes,
	schedNodes,
	tritonKernel,
	wrapperCall,
} from "./inductorExample";

export type NodeStyle =
	| "plain"
	| "extern"
	| "pointwise"
	| "reduction"
	| "ghost"
	| "kernel";

export interface StageNode {
	row: number;
	label: string;
	tag: string;
	style: NodeStyle;
	/** Key into Stage.details. */
	detail: string;
}

export interface StageGroup {
	from: number;
	to: number;
	label: string;
}

export interface Stage {
	key: string;
	title: string;
	body: string;
	nodes: StageNode[];
	/** [producer row, consumer row]. */
	edges: [number, number][];
	groups: StageGroup[];
	details: Record<string, string>;
	focus: number;
}

const rowOf = new Map(fxNodes.map((n, i) => [n.name, i]));
const short = (target: string) =>
	target.replace(/\.(default|Tensor|dim_IntList)$/, "");
const styleOf = (kind: string): NodeStyle =>
	kind === "Reduction"
		? "reduction"
		: kind === "Pointwise"
			? "pointwise"
			: "extern";

/** Edges of the FX graph: every argument that is another node in the graph. */
export function fxEdges(): [number, number][] {
	const edges: [number, number][] = [];
	fxNodes.forEach((n, i) => {
		for (const a of n.args) {
			const j = rowOf.get(a);
			if (j !== undefined) edges.push([j, i]);
		}
	});
	return edges;
}

// Scheduler node -> FX row of the node that produced its buffer.
const bufRow = new Map(
	fxNodes
		.filter((n) => n.buffer)
		.map((n) => [n.buffer as string, rowOf.get(n.name) as number]),
);
const opRow = new Map(
	schedNodes.map((s) => [s.name, bufRow.get(s.buffers[0]) as number]),
);

/** Edges between scheduler nodes: a read of a buffer another node writes. */
export function bufferEdges(): [number, number][] {
	const edges: [number, number][] = [];
	for (const s of schedNodes) {
		const to = opRow.get(s.name) as number;
		for (const r of s.reads) {
			const m = r.match(/^(?:MemoryDep|StarDep)\(\s*(?:name=)?'(\w+)'/);
			const from = m ? bufRow.get(m[1]) : undefined;
			if (from !== undefined) edges.push([from, to]);
		}
	}
	return edges;
}

const LOWERING: Record<string, string> = {
	mm: "tuned_mm (kernel/mm.py): only the ATen choice without max-autotune",
	relu: "register_pointwise(aten.relu) -> make_pointwise(ops.relu)",
	amax: 'make_reduction("max")',
	sub: "register_pointwise(aten.sub, allow_alpha=True)",
	exp: "register_pointwise_numeric_ldf64(aten.exp)",
	sum_1: 'sum_ -> make_reduction("sum")',
	div: "div -> div_prim -> make_pointwise(ops.truediv)",
	mul: "mul -> make_pointwise(ops.mul)",
};

const REALIZE: Record<string, string> = {
	mm: "An ExternKernelOut always owns its output: buf0.",
	relu: "2 users (amax, sub), but one read and no expensive op, so mark_reuse keeps it a Pointwise. Each user calls its inner_fn and recomputes relu.",
	amax: "make_reduction realizes every Reduction it creates: buf1.",
	sub: "1 user (exp), so there is nothing to share: inlined into exp.",
	exp: "2 users (sum_1, div) and the body calls exp. On CPU, should_realize_on_reuse treats exp as expensive, so it becomes buf2.",
	sum_1: "A Reduction, realized as buf3.",
	div: "1 user (mul): inlined into mul.",
	mul: "A graph output. run_node fixes its strides to match eager (require_exact_strides), which realizes it as buf4 with a FixedLayout.",
};

const fxLine = (n: (typeof fxNodes)[number]) =>
	`${n.name}: "f32[${n.shape.join(", ")}]" = torch.ops.${n.target}(${n.args.join(", ")})`;

const DECOMPOSED = new Set(["amax", "sub", "exp", "sum_1", "div"]);

export function buildStages(): Stage[] {
	const fx = fxEdges();
	const buf = bufferEdges();
	const realized = fxNodes.map((n) => n.buffer !== null);
	const sched = schedNodes.map((s) => opRow.get(s.name) as number);

	const stages: Stage[] = [];

	stages.push({
		key: "fx",
		title: "1. The FX graph Inductor receives",
		body: "Dynamo captured the Python function and AOTAutograd traced it into ATen ops. softmax has already been decomposed into amax, sub, exp, sum and div. Click a node to see its line in the graph.",
		nodes: fxNodes.map((n, i) => ({
			row: i,
			label: n.name,
			tag: short(n.target),
			style: "plain",
			detail: n.name,
		})),
		edges: fx,
		groups: [],
		details: Object.fromEntries(
			fxNodes.map((n) => [
				n.name,
				`${fxLine(n)}${DECOMPOSED.has(n.name) ? "\n\n# one of the five ops aten._softmax decomposes into" : ""}`,
			]),
		),
		focus: 1,
	});

	stages.push({
		key: "lower",
		title: "2. Lowering to IR",
		body: "GraphLowering runs the graph like an interpreter and calls the lowering registered for each ATen op. Pointwise and Reduction nodes hold an inner_fn that computes one element; the matmul becomes a call to an external kernel.",
		nodes: fxNodes.map((n, i) => ({
			row: i,
			label: n.name,
			tag: n.kind,
			style: styleOf(n.kind),
			detail: n.name,
		})),
		edges: fx,
		groups: [],
		details: Object.fromEntries(
			fxNodes.map((n) => [
				n.name,
				`# lowering: ${LOWERING[n.name]}\n${n.body}`,
			]),
		),
		focus: 3,
	});

	stages.push({
		key: "realize",
		title: "3. Realize or inline",
		body: "A node is realized when it gets a buffer in memory. The rest stay as functions and are inlined into their users: relu, sub and div vanish, and exp's body now starts from a load of buf0. Edges now follow buffer reads.",
		nodes: fxNodes.map((n, i) => ({
			row: i,
			label: n.name,
			tag: n.buffer ?? "inlined",
			style: realized[i] ? styleOf(n.kind) : "ghost",
			detail: n.name,
		})),
		edges: buf,
		groups: [],
		details: Object.fromEntries(
			fxNodes.map((n) => [
				n.name,
				`# ${REALIZE[n.name]}${n.buffer ? `\n${n.body}` : ""}`,
			]),
		),
		focus: 4,
	});

	stages.push({
		key: "sched",
		title: "4. Scheduler nodes and dependencies",
		body: "The scheduler wraps each realized buffer in a node and records which buffers it reads and writes. group is (numel, rnumel): the size of the loop over outputs and of the reduction loop.",
		nodes: sched.map((i, k) => ({
			row: i,
			label: schedNodes[k].name,
			tag: schedNodes[k].group.includes("(")
				? schedNodes[k].group.replace(/^.*\), /, "").replace(/\)$/, "")
				: "extern",
			style: styleOf(fxNodes[i].kind),
			detail: schedNodes[k].name,
		})),
		edges: buf,
		groups: [],
		details: Object.fromEntries(
			schedNodes.map((s) => [
				s.name,
				[
					`${s.name}: ${s.kind}  # from ${s.origins.join(", ")}`,
					...s.reads.map((r) => `  reads  ${r}`),
					...s.writes.map((w) => `  writes ${w}`),
				].join("\n"),
			]),
		),
		focus: 4,
	});

	const scoreText = [
		"# score_fusion for each legal pair (higher fuses first)",
		...fusionScores.map(
			(s) =>
				`${s.pair.join(" + ").padEnd(10)} memory_score=${s.memory}  template=${s.template}  node_type=${s.node_type}  proximity=${s.proximity}`,
		),
		"",
		"# op2 + op4 share buf2 but are illegal: op3 sits between them",
		"# op0 is an extern kernel and cannot fuse",
	].join("\n");
	const [, r1, , r2] = sched; // rows of op1 and op3
	const r4 = sched[sched.length - 1];

	stages.push({
		key: "fuse1",
		title: "5. Fusion: the best-scoring pairs",
		body: "The scheduler lists every legal pair of nodes that touch a common buffer, scores each by the bytes they share, and fuses greedily. op1+op2 and op3+op4 tie at 1056 bytes and go first.",
		nodes: sched.map((i, k) => ({
			row: i,
			label: schedNodes[k].name,
			tag: k === 0 ? "extern" : "",
			style: styleOf(fxNodes[i].kind),
			detail: "scores",
		})),
		edges: buf,
		groups: [
			{ from: r1, to: opRow.get("op2") as number, label: "op1_op2" },
			{ from: r2, to: r4, label: "op3_op4" },
		],
		details: { scores: scoreText },
		focus: r1,
	});

	stages.push({
		key: "fuse2",
		title: "6. Fusion: one node",
		body: "The remaining pair, op2 + op3, now joins the two fused nodes, so four scheduler nodes become one. Only buf0 crosses the boundary between the two kernels.",
		nodes: sched.map((i, k) => ({
			row: i,
			label: schedNodes[k].name,
			tag: k === 0 ? "extern" : "",
			style: styleOf(fxNodes[i].kind),
			detail: k === 0 ? "op0" : "fused",
		})),
		edges: buf,
		groups: [{ from: r1, to: r4, label: "op1_op2_op3_op4" }],
		details: {
			op0: "op0: ExternKernelSchedulerNode  # extern_kernels.mm, never fused",
			fused:
				"op1_op2_op3_op4: FusedSchedulerNode(op1, op2, op3, op4)\n  reads  buf0\n  writes buf1, buf2, buf3, buf4\n# buf1, buf2 and buf3 are created and last used inside it",
		},
		focus: r1,
	});

	stages.push({
		key: "code",
		title: "7. Code generation",
		body: "Each scheduler node becomes one kernel: a call to extern_kernels.mm and one Triton kernel for the fused node. buf1, buf2 and buf3 stay in registers, and the kernel overwrites buf0 in place to produce the output.",
		nodes: sched.map((i, k) => ({
			row: i,
			label: k === 0 ? "extern_kernels.mm" : schedNodes[k].name,
			tag: "",
			style: k === 0 ? "extern" : "kernel",
			detail: k === 0 ? "call" : "kernel",
		})),
		edges: buf.filter(([a]) => a === 0).slice(0, 1),
		groups: [
			{ from: r1, to: r4, label: "triton_per_fused__softmax_mul_relu_0" },
		],
		details: { call: wrapperCall, kernel: tritonKernel },
		focus: r1,
	});

	return stages;
}

export const STAGE_ROWS = fxNodes.length;
