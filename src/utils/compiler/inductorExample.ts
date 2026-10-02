// Captured Inductor artifacts for the running example of the "Inside TorchInductor" post
// (src/content/posts/torch-inductor.mdx):
//
//   def f(x, w):                      # x: f32[8, 16], w: f32[16, 32]
//       y = torch.relu(x @ w)
//       return torch.softmax(y, dim=-1) * 2
//
// Generated, not hand-written: torch.compile was run on torch 2.13.0 (git cf30153c) with
// cpu_backend="triton", hooks on GraphLowering.run_node, Scheduler.fuse_nodes,
// BaseScheduling.fuse and InductorChoices.score_fusion, and Triton's compile step stubbed out
// (Triton is not installable on the macOS machine that produced this). Strings are verbatim.
// Used by pipeline.ts and InductorPipeline.astro.

export interface FxNode {
	name: string;
	target: string;
	shape: number[];
	args: string[];
	/** IR class GraphLowering produced for this node. */
	kind: string;
	/** Buffer name once realized, or null when the node was inlined into its users. */
	buffer: string | null;
	/** inner_fn of the Pointwise/Reduction, as Inductor prints it. */
	body: string;
	reduction: string | null;
}

export interface SchedNode {
	name: string;
	buffers: string[];
	reads: string[];
	writes: string[];
	/** (device, (numel, rnumel)) as the SIMD backend groups it. */
	group: string;
	kind: string;
	origins: string[];
}

export interface FusionScoreRow {
	pair: [string, string];
	memory: number;
	template: number;
	node_type: number;
	proximity: number;
}

// biome-ignore format: generated data
export const fxNodes: FxNode[] = [{"name": "mm", "target": "aten.mm.default", "shape": [8, 32], "args": ["arg0_1", "arg1_1"], "kind": "ExternKernelOut", "buffer": "buf0", "body": "extern_kernels.mm(...)", "reduction": null}, {"name": "relu", "target": "aten.relu.default", "shape": [8, 32], "args": ["mm"], "kind": "Pointwise", "buffer": null, "body": "def inner_fn(index):\n    i0, i1 = index\n    tmp0 = ops.load(buf0, i1 + 32 * i0)\n    tmp1 = ops.relu(tmp0)\n    return tmp1", "reduction": null}, {"name": "amax", "target": "aten.amax.default", "shape": [8, 1], "args": ["relu", "[-1]", "True"], "kind": "Reduction", "buffer": "buf1", "body": "def inner_fn(index, rindex):\n    i0, _ = index\n    r0_0 = rindex\n    tmp0 = ops.load(buf0, r0_0 + 32 * i0)\n    tmp1 = ops.relu(tmp0)\n    return tmp1", "reduction": "max"}, {"name": "sub", "target": "aten.sub.Tensor", "shape": [8, 32], "args": ["relu", "amax"], "kind": "Pointwise", "buffer": null, "body": "def inner_fn(index):\n    i0, i1 = index\n    tmp0 = ops.load(buf0, i1 + 32 * i0)\n    tmp1 = ops.relu(tmp0)\n    tmp2 = ops.load(buf1, i0)\n    tmp3 = tmp1 - tmp2\n    return tmp3", "reduction": null}, {"name": "exp", "target": "aten.exp.default", "shape": [8, 32], "args": ["sub"], "kind": "Pointwise", "buffer": "buf2", "body": "def inner_fn(index):\n    i0, i1 = index\n    tmp0 = ops.load(buf0, i1 + 32 * i0)\n    tmp1 = ops.relu(tmp0)\n    tmp2 = ops.load(buf1, i0)\n    tmp3 = tmp1 - tmp2\n    tmp4 = ops.exp(tmp3)\n    return tmp4", "reduction": null}, {"name": "sum_1", "target": "aten.sum.dim_IntList", "shape": [8, 1], "args": ["exp", "[-1]", "True"], "kind": "Reduction", "buffer": "buf3", "body": "def inner_fn(index, rindex):\n    i0, _ = index\n    r0_0 = rindex\n    tmp0 = ops.load(buf2, r0_0 + 32 * i0)\n    return tmp0", "reduction": "sum"}, {"name": "div", "target": "aten.div.Tensor", "shape": [8, 32], "args": ["exp", "sum_1"], "kind": "Pointwise", "buffer": null, "body": "def inner_fn(index):\n    i0, i1 = index\n    tmp0 = ops.load(buf2, i1 + 32 * i0)\n    tmp1 = ops.load(buf3, i0)\n    tmp2 = tmp0 / tmp1\n    return tmp2", "reduction": null}, {"name": "mul", "target": "aten.mul.Tensor", "shape": [8, 32], "args": ["div", "2"], "kind": "Pointwise", "buffer": "buf4", "body": "def inner_fn(index):\n    i0, i1 = index\n    tmp0 = ops.load(buf2, i1 + 32 * i0)\n    tmp1 = ops.load(buf3, i0)\n    tmp2 = tmp0 / tmp1\n    tmp3 = ops.constant(2, torch.float32)\n    tmp4 = tmp2 * tmp3\n    return tmp4", "reduction": null}];

// biome-ignore format: generated data
export const schedNodes: SchedNode[] = [{"name": "op0", "buffers": ["buf0"], "reads": ["StarDep(name='arg0_1', mode=None)", "StarDep(name='arg1_1', mode=None)"], "writes": ["StarDep(name='buf0', mode=None)"], "group": "None", "kind": "ExternKernelSchedulerNode", "origins": ["mm"]}, {"name": "op1", "buffers": ["buf1"], "reads": ["MemoryDep('buf0', c0, {c0: 256})"], "writes": ["MemoryDep('buf1', c0, {c0: 8})"], "group": "(device(type='cpu'), (8, 32))", "kind": "SchedulerNode", "origins": ["amax", "relu"]}, {"name": "op2", "buffers": ["buf2"], "reads": ["MemoryDep('buf0', c0, {c0: 256})", "MemoryDep('buf1', c0, {c0: 8})"], "writes": ["MemoryDep('buf2', c0, {c0: 256})"], "group": "(device(type='cpu'), (256, 1))", "kind": "SchedulerNode", "origins": ["exp", "relu", "sub"]}, {"name": "op3", "buffers": ["buf3"], "reads": ["MemoryDep('buf2', c0, {c0: 256})"], "writes": ["MemoryDep('buf3', c0, {c0: 8})"], "group": "(device(type='cpu'), (8, 32))", "kind": "SchedulerNode", "origins": ["sum_1"]}, {"name": "op4", "buffers": ["buf4"], "reads": ["MemoryDep('buf2', c0, {c0: 256})", "MemoryDep('buf3', c0, {c0: 8})"], "writes": ["MemoryDep('buf4', c0, {c0: 256})"], "group": "(device(type='cpu'), (256, 1))", "kind": "SchedulerNode", "origins": ["div", "mul"]}];

/** Fusions in the order Scheduler.fuse_nodes made them (one round). */
// biome-ignore format: generated data
export const fusions: [string, string][] = [["op1", "op2"], ["op3", "op4"], ["op1_op2", "op3_op4"]];

// biome-ignore format: generated data
export const fusionScores: FusionScoreRow[] = [{"pair": ["op1", "op2"], "memory": 1056, "template": 2, "node_type": 0, "proximity": -1}, {"pair": ["op2", "op3"], "memory": 1024, "template": 2, "node_type": 0, "proximity": -1}, {"pair": ["op3", "op4"], "memory": 1056, "template": 2, "node_type": 0, "proximity": -1}];

// biome-ignore format: generated data
export const postFusion: string[] = ["op0", "op1_op2_op3_op4"];

export const tritonKernel =
	"def triton_per_fused__softmax_mul_relu_0(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):\n    xnumel = 8\n    r0_numel = 32\n    R0_BLOCK: tl.constexpr = 32\n    rnumel = r0_numel\n    RBLOCK: tl.constexpr = R0_BLOCK\n    xoffset = tl.program_id(0) * XBLOCK\n    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]\n    xmask = xindex < xnumel\n    r0_index = tl.arange(0, R0_BLOCK)[None, :]\n    r0_offset = 0\n    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]\n    roffset = r0_offset\n    rindex = r0_index\n    r0_1 = r0_index\n    x0 = xindex\n    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 32*x0), xmask, eviction_policy='evict_first', other=0.0)\n    tmp1 = tl.full([1, 1], 0, tl.int32)\n    tmp2 = triton_helpers.maximum(tmp1, tmp0)\n    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])\n    tmp5 = tl.where(xmask, tmp3, float(\"-inf\"))\n    tmp6 = triton_helpers.max2(tmp5, 1)[:, None].to(tl.float32)\n    tmp7 = tmp2 - tmp6\n    tmp8 = libdevice.exp(tmp7)\n    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])\n    tmp11 = tl.where(xmask, tmp9, 0)\n    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)\n    tmp13 = (tmp8 / tmp12)\n    tmp14 = tl.full([1, 1], 2.0, tl.float32)\n    tmp15 = tmp13 * tmp14\n    tl.store(in_out_ptr0 + (r0_1 + 32*x0), tmp15, xmask)";

export const wrapperCall =
	"def call(self, args):\n    arg0_1, arg1_1 = args\n    args.clear()\n    assert_size_stride(arg0_1, (8, 16), (16, 1), 'input')\n    assert_size_stride(arg1_1, (16, 32), (32, 1), 'input')\n    buf0 = empty_strided_cpu((8, 32), (32, 1), torch.float32)\n    # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]\n    extern_kernels.mm(arg0_1, arg1_1, out=buf0)\n    del arg0_1\n    del arg1_1\n    buf2 = buf0; del buf0  # reuse\n    buf4 = buf2; del buf2  # reuse\n    # Topologically Sorted Source Nodes: [y, softmax, mul], Original ATen: [aten.relu, aten._softmax, aten.mul]\n    raw_streamNone = get_raw_stream(None)\n    triton_per_fused__softmax_mul_relu_0.run(buf4, 8, 32, stream=raw_streamNone)\n    return (buf4, )";
