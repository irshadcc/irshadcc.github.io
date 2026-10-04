// Figure data for the AOTAutograd post (src/content/posts/aot-autograd.mdx): the pipeline diagram
// and the graph listings. The listings were captured from torch 2.13.0 (git cf30153c) by a
// scratch script; see the comments above each export.
import type { BoxEdge, BoxNode } from "../payments/boxDiagram";
import type { CodeStage, StageTag } from "./llvmData";

const src = (file: string, line: number) =>
	`https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/${file}#L${line}`;
const link = (file: string, line: number, text: string) =>
	`<a href="${src(file, line)}"><code>${text}</code></a>`;

export const aotPipelineNodes: BoxNode[] = [
	{
		id: "dynamo",
		label: "Dynamo graph",
		sub: "torch-level FX",
		col: 0,
		row: 0,
		group: 5,
		note: `Dynamo hands Inductor's ${link("_inductor/compile_fx.py", 2685, "compile_fx")} an FX graph of torch calls plus example inputs. Inductor passes it to ${link("_functorch/aot_autograd.py", 1131, "aot_module_simplified")} with its own forward, backward and partition functions.`,
	},
	{
		id: "meta",
		label: "metadata pass",
		sub: "ViewAndMutationMeta",
		col: 1,
		row: 0,
		group: 0,
		note: `${link("_functorch/_aot_autograd/collect_metadata_analysis.py", 166, "run_functionalized_fw_and_collect_metadata")} runs the forward once on fake tensors under functionalization and records which inputs are mutated and which outputs alias inputs or each other. Nothing is traced yet.`,
	},
	{
		id: "joint",
		label: "joint graph",
		sub: "make_fx(create_joint)",
		col: 2,
		row: 0,
		group: 1,
		note: `${link("_functorch/_aot_autograd/graph_capture.py", 471, "aot_dispatch_autograd_graph")} traces one function that runs the forward and then calls <code>torch.autograd.grad</code>. The result is a single functional ATen graph with primals and tangents as inputs.`,
	},
	{
		id: "partition",
		label: "partitioner",
		sub: "min-cut or default",
		col: 3,
		row: 0,
		group: 2,
		note: `${link("_functorch/partitioners.py", 3725, "min_cut_rematerialization_partition")} (Inductor's choice) or ${link("_functorch/partitioners.py", 1248, "default_partition")} splits the joint graph into a forward graph and a backward graph and decides which values cross between them.`,
	},
	{
		id: "fw",
		label: "forward graph",
		sub: "fw_compiler",
		col: 4,
		row: 0,
		group: 3,
		note: "Compiled right away by the forward compiler (Inductor's <code>compile_fx_inner</code>). It returns the user outputs followed by the values saved for the backward.",
	},
	{
		id: "bw",
		label: "backward graph",
		sub: "bw_compiler",
		col: 4,
		row: 1,
		group: 3,
		note: "Compiled by the backward compiler, by default lazily on the first call to backward. It takes the saved values and the tangents and returns one gradient per input.",
	},
	{
		id: "fn",
		label: "CompiledFunction",
		sub: "torch.autograd.Function",
		col: 5,
		row: 0,
		h: 2,
		group: 4,
		note: `${link("_functorch/_aot_autograd/runtime_wrappers.py", 3441, "CompiledFunction")} wraps both: its <code>forward</code> calls the compiled forward and saves the extra outputs with <code>ctx.save_for_backward</code>; its <code>backward</code> calls the compiled backward. Runtime wrappers around it replay input mutations and regenerate aliased outputs.`,
	},
];

export const aotPipelineEdges: BoxEdge[] = [
	{ from: "dynamo", to: "meta" },
	{ from: "meta", to: "joint" },
	{ from: "joint", to: "partition" },
	{ from: "partition", to: "fw" },
	{ from: "partition", to: "bw" },
	{ from: "fw", to: "fn" },
	{ from: "bw", to: "fn" },
];

// Graphs for the running example, captured with TORCH_LOGS on torch 2.13.0 and with
// aot_function(..., partition_fn=default_partition, decompositions=select_decomp_table()) for the
// default partitioner. `torch.ops.aten.` is shortened to `aten.` and `x = None` frees are dropped.
export const rmsStages: CodeStage[] = [
	{
		id: "user",
		label: "user code",
		tool: "rms_silu.py",
		lang: "python",
		code: "import torch\nimport torch.nn.functional as F\n\ndef rms_silu(x, g, w):\n    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)\n    h = x * rstd * g            # RMSNorm\n    return F.silu(h @ w)        # linear layer, then SiLU\n\nx = torch.randn(4, 16, requires_grad=True)\ng = torch.randn(16, requires_grad=True)\nw = torch.randn(16, 8, requires_grad=True)\ntorch.compile(rms_silu)(x, g, w).sum().backward()",
		marks: [],
		note: "RMSNorm, a linear layer and SiLU. All three inputs require gradients, so AOTAutograd has to produce a forward and a backward.",
	},
	{
		id: "joint",
		label: "joint graph",
		tool: "TORCH_LOGS=aot_joint_graph",
		lang: "python",
		code: 'def forward(self, primals, tangents):\n    # primals_1: f32[4, 16] (x), primals_2: f32[16] (g), primals_3: f32[16, 8] (w), tangents_1: f32[4, 8]\n    pow_1: "f32[4, 16]" = aten.pow.Tensor_Scalar(primals_1, 2)\n    mean: "f32[4, 1]" = aten.mean.dim(pow_1, [-1], True)\n    add: "f32[4, 1]" = aten.add.Tensor(mean, 1e-06)\n    rsqrt: "f32[4, 1]" = aten.rsqrt.default(add)\n    alias: "f32[4, 1]" = aten.alias.default(rsqrt)\n    mul: "f32[4, 16]" = aten.mul.Tensor(primals_1, rsqrt)\n    mul_1: "f32[4, 16]" = aten.mul.Tensor(mul, primals_2)\n    mm: "f32[4, 8]" = aten.mm.default(mul_1, primals_3)\n    neg: "f32[4, 8]" = aten.neg.default(mm)\n    exp: "f32[4, 8]" = aten.exp.default(neg)\n    add_1: "f32[4, 8]" = aten.add.Tensor(exp, 1)\n    div: "f32[4, 8]" = aten.div.Tensor(mm, add_1)\n    sigmoid: "f32[4, 8]" = aten.sigmoid.default(mm)\n    mul_2: "f32[4, 8]" = aten.mul.Tensor(tangents_1, sigmoid)\n    sub: "f32[4, 8]" = aten.sub.Tensor(1, sigmoid)\n    mul_3: "f32[4, 8]" = aten.mul.Tensor(mm, sub)\n    add_2: "f32[4, 8]" = aten.add.Tensor(mul_3, 1)\n    mul_4: "f32[4, 8]" = aten.mul.Tensor(mul_2, add_2)\n    permute: "f32[16, 4]" = aten.permute.default(mul_1, [1, 0])\n    mm_1: "f32[16, 8]" = aten.mm.default(permute, mul_4)\n    permute_1: "f32[8, 16]" = aten.permute.default(primals_3, [1, 0])\n    mm_2: "f32[4, 16]" = aten.mm.default(mul_4, permute_1)\n    mul_5: "f32[4, 16]" = aten.mul.Tensor(mm_2, mul)\n    mul_6: "f32[4, 16]" = aten.mul.Tensor(mm_2, primals_2)\n    sum_1: "f32[1, 16]" = aten.sum.dim_IntList(mul_5, [0], True)\n    view: "f32[16]" = aten.view.default(sum_1, [16])\n    mul_7: "f32[4, 16]" = aten.mul.Tensor(mul_6, primals_1)\n    mul_8: "f32[4, 16]" = aten.mul.Tensor(mul_6, rsqrt)\n    sum_2: "f32[4, 1]" = aten.sum.dim_IntList(mul_7, [1], True)\n    alias_1: "f32[4, 1]" = aten.alias.default(alias)\n    mul_9: "f32[4, 1]" = aten.mul.Scalar(sum_2, -0.5)\n    pow_2: "f32[4, 1]" = aten.pow.Tensor_Scalar(alias_1, 3)\n    mul_10: "f32[4, 1]" = aten.mul.Tensor(mul_9, pow_2)\n    expand: "f32[4, 16]" = aten.expand.default(mul_10, [4, 16])\n    div_1: "f32[4, 16]" = aten.div.Scalar(expand, 16)\n    pow_3: "f32[4, 16]" = aten.pow.Tensor_Scalar(primals_1, 1.0)\n    mul_11: "f32[4, 16]" = aten.mul.Scalar(pow_3, 2.0)\n    mul_12: "f32[4, 16]" = aten.mul.Tensor(div_1, mul_11)\n    add_3: "f32[4, 16]" = aten.add.Tensor(mul_8, mul_12)\n    return pytree.tree_unflatten([div, add_3, view, mm_1], self._out_spec)',
		marks: [
			[2, "fwd"],
			[3, "fwd"],
			[4, "fwd"],
			[5, "fwd"],
			[6, "fwd"],
			[7, "fwd"],
			[8, "fwd"],
			[9, "fwd"],
			[10, "fwd"],
			[11, "fwd"],
			[12, "fwd"],
			[13, "fwd"],
			[14, "bwd"],
			[15, "bwd"],
			[16, "bwd"],
			[17, "bwd"],
			[18, "bwd"],
			[19, "bwd"],
			[20, "bwd"],
			[21, "bwd"],
			[22, "bwd"],
			[23, "bwd"],
			[24, "bwd"],
			[25, "bwd"],
			[26, "bwd"],
			[27, "bwd"],
			[28, "bwd"],
			[29, "bwd"],
			[30, "bwd"],
			[31, "bwd"],
			[32, "bwd"],
			[33, "bwd"],
			[34, "bwd"],
			[35, "bwd"],
			[36, "bwd"],
			[37, "bwd"],
			[38, "bwd"],
			[39, "bwd"],
			[40, "bwd"],
			[41, "out"],
		],
		note: "One graph that takes the primals (inputs) and tangents (gradients of the outputs) and returns the outputs followed by the input gradients. Lines tagged forward op are the ops the original forward runs; the rest is the backward that autograd traced. silu became neg, exp, add, div in the forward and sigmoid-based math in the backward.",
	},
	{
		id: "dfw",
		label: "default: forward",
		tool: "partition_fn=default_partition",
		lang: "python",
		code: 'def forward(self, primals_1: "f32[4, 16]", primals_2: "f32[16]", primals_3: "f32[16, 8]"):\n    pow_1: "f32[4, 16]" = aten.pow.Tensor_Scalar(primals_1, 2)\n    mean: "f32[4, 1]" = aten.mean.dim(pow_1, [-1], True)\n    add: "f32[4, 1]" = aten.add.Tensor(mean, 1e-06)\n    rsqrt: "f32[4, 1]" = aten.rsqrt.default(add)\n    alias: "f32[4, 1]" = aten.alias.default(rsqrt)\n    mul: "f32[4, 16]" = aten.mul.Tensor(primals_1, rsqrt)\n    mul_1: "f32[4, 16]" = aten.mul.Tensor(mul, primals_2)\n    mm: "f32[4, 8]" = aten.mm.default(mul_1, primals_3)\n    neg: "f32[4, 8]" = aten.neg.default(mm)\n    exp: "f32[4, 8]" = aten.exp.default(neg)\n    add_1: "f32[4, 8]" = aten.add.Tensor(exp, 1)\n    div: "f32[4, 8]" = aten.div.Tensor(mm, add_1)\n    return (div, primals_1, primals_2, primals_3, rsqrt, alias, mul, mul_1, mm)',
		marks: [[13, "saved"]],
		note: "The default partitioner keeps exactly the original forward ops and returns every forward value the backward reads: the three inputs and five computed values: rsqrt, alias (a view of rsqrt), mul, mul_1 and mm.",
	},
	{
		id: "dbw",
		label: "default: backward",
		tool: "partition_fn=default_partition",
		lang: "python",
		code: 'def forward(self, primals_1: "f32[4, 16]", primals_2: "f32[16]", primals_3: "f32[16, 8]", rsqrt: "f32[4, 1]", alias: "f32[4, 1]", mul: "f32[4, 16]", mul_1: "f32[4, 16]", mm: "f32[4, 8]", tangents_1: "f32[4, 8]"):\n    sigmoid: "f32[4, 8]" = aten.sigmoid.default(mm)\n    mul_2: "f32[4, 8]" = aten.mul.Tensor(tangents_1, sigmoid)\n    sub: "f32[4, 8]" = aten.sub.Tensor(1, sigmoid)\n    mul_3: "f32[4, 8]" = aten.mul.Tensor(mm, sub)\n    add_2: "f32[4, 8]" = aten.add.Tensor(mul_3, 1)\n    mul_4: "f32[4, 8]" = aten.mul.Tensor(mul_2, add_2)\n    permute: "f32[16, 4]" = aten.permute.default(mul_1, [1, 0])\n    mm_1: "f32[16, 8]" = aten.mm.default(permute, mul_4)\n    permute_1: "f32[8, 16]" = aten.permute.default(primals_3, [1, 0])\n    mm_2: "f32[4, 16]" = aten.mm.default(mul_4, permute_1)\n    mul_5: "f32[4, 16]" = aten.mul.Tensor(mm_2, mul)\n    mul_6: "f32[4, 16]" = aten.mul.Tensor(mm_2, primals_2)\n    sum_1: "f32[1, 16]" = aten.sum.dim_IntList(mul_5, [0], True)\n    view: "f32[16]" = aten.view.default(sum_1, [16])\n    mul_7: "f32[4, 16]" = aten.mul.Tensor(mul_6, primals_1)\n    mul_8: "f32[4, 16]" = aten.mul.Tensor(mul_6, rsqrt)\n    sum_2: "f32[4, 1]" = aten.sum.dim_IntList(mul_7, [1], True)\n    alias_1: "f32[4, 1]" = aten.alias.default(alias)\n    mul_9: "f32[4, 1]" = aten.mul.Scalar(sum_2, -0.5)\n    pow_2: "f32[4, 1]" = aten.pow.Tensor_Scalar(alias_1, 3)\n    mul_10: "f32[4, 1]" = aten.mul.Tensor(mul_9, pow_2)\n    expand: "f32[4, 16]" = aten.expand.default(mul_10, [4, 16])\n    div_1: "f32[4, 16]" = aten.div.Scalar(expand, 16)\n    pow_3: "f32[4, 16]" = aten.pow.Tensor_Scalar(primals_1, 1.0)\n    mul_11: "f32[4, 16]" = aten.mul.Scalar(pow_3, 2.0)\n    mul_12: "f32[4, 16]" = aten.mul.Tensor(div_1, mul_11)\n    add_3: "f32[4, 16]" = aten.add.Tensor(mul_8, mul_12)\n    return (add_3, view, mm_1)',
		marks: [[0, "saved"]],
		note: "The backward receives all eight saved values plus the tangent.",
	},
	{
		id: "mfw",
		label: "min-cut: forward",
		tool: "torch.compile (Inductor)",
		lang: "python",
		code: 'def forward(self, primals_1: "f32[4, 16]", primals_2: "f32[16]", primals_3: "f32[16, 8]"):\n    pow_1: "f32[4, 16]" = aten.pow.Tensor_Scalar(primals_1, 2)\n    mean: "f32[4, 1]" = aten.mean.dim(pow_1, [-1], True)\n    add: "f32[4, 1]" = aten.add.Tensor(mean, 1e-06)\n    rsqrt: "f32[4, 1]" = aten.rsqrt.default(add)\n    mul: "f32[4, 16]" = aten.mul.Tensor(primals_1, rsqrt)\n    mul_1: "f32[4, 16]" = aten.mul.Tensor(mul, primals_2)\n    mm: "f32[4, 8]" = aten.mm.default(mul_1, primals_3)\n    neg: "f32[4, 8]" = aten.neg.default(mm)\n    exp: "f32[4, 8]" = aten.exp.default(neg)\n    add_1: "f32[4, 8]" = aten.add.Tensor(exp, 1)\n    div: "f32[4, 8]" = aten.div.Tensor(mm, add_1)\n    permute: "f32[16, 4]" = aten.permute.default(mul_1, [1, 0])\n    permute_1: "f32[8, 16]" = aten.permute.default(primals_3, [1, 0])\n    return (div, primals_1, primals_2, rsqrt, mm, permute, permute_1)',
		marks: [[14, "saved"]],
		note: "The min-cut partitioner returns rsqrt, mm and two transposes: permute is h transposed (the input of the second matmul in the backward) and permute_1 is w transposed. mul (x · rstd) is no longer saved.",
	},
	{
		id: "mbw",
		label: "min-cut: backward",
		tool: "torch.compile (Inductor)",
		lang: "python",
		code: 'def forward(self, primals_1: "f32[4, 16]", primals_2: "f32[16]", rsqrt: "f32[4, 1]", mm: "f32[4, 8]", permute: "f32[16, 4]", permute_1: "f32[8, 16]", tangents_1: "f32[4, 8]"):\n    sigmoid: "f32[4, 8]" = aten.sigmoid.default(mm)\n    mul_2: "f32[4, 8]" = aten.mul.Tensor(tangents_1, sigmoid)\n    sub: "f32[4, 8]" = aten.sub.Tensor(1, sigmoid)\n    mul_3: "f32[4, 8]" = aten.mul.Tensor(mm, sub)\n    add_2: "f32[4, 8]" = aten.add.Tensor(mul_3, 1)\n    mul_4: "f32[4, 8]" = aten.mul.Tensor(mul_2, add_2)\n    mm_1: "f32[16, 8]" = aten.mm.default(permute, mul_4)\n    mm_2: "f32[4, 16]" = aten.mm.default(mul_4, permute_1)\n    mul: "f32[4, 16]" = aten.mul.Tensor(primals_1, rsqrt)\n    mul_5: "f32[4, 16]" = aten.mul.Tensor(mm_2, mul)\n    mul_6: "f32[4, 16]" = aten.mul.Tensor(mm_2, primals_2)\n    sum_1: "f32[1, 16]" = aten.sum.dim_IntList(mul_5, [0], True)\n    view: "f32[16]" = aten.view.default(sum_1, [16])\n    mul_7: "f32[4, 16]" = aten.mul.Tensor(mul_6, primals_1)\n    mul_8: "f32[4, 16]" = aten.mul.Tensor(mul_6, rsqrt)\n    sum_2: "f32[4, 1]" = aten.sum.dim_IntList(mul_7, [1], True)\n    mul_9: "f32[4, 1]" = aten.mul.Scalar(sum_2, -0.5)\n    pow_2: "f32[4, 1]" = aten.pow.Tensor_Scalar(rsqrt, 3)\n    mul_10: "f32[4, 1]" = aten.mul.Tensor(mul_9, pow_2)\n    expand: "f32[4, 16]" = aten.expand.default(mul_10, [4, 16])\n    div_1: "f32[4, 16]" = aten.div.Scalar(expand, 16)\n    pow_3: "f32[4, 16]" = aten.pow.Tensor_Scalar(primals_1, 1.0)\n    mul_11: "f32[4, 16]" = aten.mul.Scalar(pow_3, 2.0)\n    mul_12: "f32[4, 16]" = aten.mul.Tensor(div_1, mul_11)\n    add_3: "f32[4, 16]" = aten.add.Tensor(mul_8, mul_12)\n    return (add_3, view, mm_1)',
		marks: [
			[0, "saved"],
			[9, "recomp"],
		],
		note: "The backward recomputes mul = x · rstd from two saved values. It is one pointwise multiply that Inductor fuses into the kernel that consumes it, so it costs almost nothing.",
	},
];
export const rmsTags: StageTag[] = [
	{ tag: "fwd", label: "forward op" },
	{ tag: "bwd", label: "backward op" },
	{ tag: "out", label: "outputs" },
	{ tag: "saved", label: "saved for backward" },
	{ tag: "recomp", label: "recomputed" },
];

// Functionalization of input mutations, intermediate mutations and aliased outputs.
export const mutStages: CodeStage[] = [
	{
		id: "user",
		label: "user code",
		tool: "f.py",
		lang: "python",
		code: "def f(x, buf):\n    buf.add_(1)          # mutates an input\n    t = x.sin()\n    t.mul_(2)            # mutates an intermediate\n    return t + buf, x.view(-1)   # 2nd output aliases x\n\nx = torch.randn(2, 3, requires_grad=True)\nbuf = torch.zeros(3)\nout, v = torch.compile(f)(x, buf)",
		marks: [
			[1, "mut"],
			[3, "func"],
			[4, "alias"],
		],
		note: "Two in-place ops and an output that is a view of an input. The compiled graph must not contain any of them as mutations, yet the caller must see buf change and must get a real view of x back.",
	},
	{
		id: "joint",
		label: "joint graph",
		tool: "TORCH_LOGS=aot_joint_graph",
		lang: "python",
		code: 'def forward(self, primals, tangents):\n    # primals_1: f32[3] (buf), primals_2: f32[2, 3] (x), tangents_1: f32[2, 3]\n    add: "f32[3]" = aten.add.Tensor(primals_1, 1)\n    sin: "f32[2, 3]" = aten.sin.default(primals_2)\n    mul: "f32[2, 3]" = aten.mul.Tensor(sin, 2)\n    add_1: "f32[2, 3]" = aten.add.Tensor(mul, add)\n    view: "f32[6]" = aten.view.default(primals_2, [-1])\n    mul_1: "f32[2, 3]" = aten.mul.Tensor(tangents_1, 2)\n    cos: "f32[2, 3]" = aten.cos.default(primals_2)\n    mul_2: "f32[2, 3]" = aten.mul.Tensor(mul_1, cos)\n    copy_: "f32[3]" = aten.copy_.default(primals_1, add)\n    return pytree.tree_unflatten([add_1, view, None, mul_2], self._out_spec)',
		marks: [
			[2, "func"],
			[4, "func"],
			[6, "alias"],
			[10, "mut"],
		],
		note: "buf.add_(1) became add = buf + 1, t.mul_(2) became mul = sin * 2, and the write back to buf is a single copy_ at the very end. The second output is computed as a view, but the runtime will throw it away and regenerate it from x.",
	},
	{
		id: "fw",
		label: "forward graph",
		tool: "TORCH_LOGS=aot_graphs",
		lang: "python",
		code: 'def forward(self, primals_1: "f32[3]", primals_2: "f32[2, 3]"):\n    add: "f32[3]" = aten.add.Tensor(primals_1, 1)\n    sin: "f32[2, 3]" = aten.sin.default(primals_2)\n    mul: "f32[2, 3]" = aten.mul.Tensor(sin, 2)\n    add_1: "f32[2, 3]" = aten.add.Tensor(mul, add)\n    view: "f32[6]" = aten.view.default(primals_2, [-1])\n    copy_: "f32[3]" = aten.copy_.default(primals_1, add)\n    return (add_1, view, primals_2)',
		marks: [
			[1, "func"],
			[3, "func"],
			[5, "alias"],
			[6, "mut"],
		],
		note: "buf does not require grad, so with keep_input_mutations=True the copy_ stays in the graph for Inductor to fuse with the add. x (primals_2) is the only value saved for the backward, which needs cos(x).",
	},
];
export const mutTags: StageTag[] = [
	{ tag: "mut", label: "input mutation" },
	{ tag: "func", label: "functionalized in-place op" },
	{ tag: "alias", label: "output aliases an input" },
];
