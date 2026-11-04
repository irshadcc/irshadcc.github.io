// Figure data for the FlexAttention lowering post. The CodeStages listings are copied from what
// torch 2.13.0 (git cf30153c) printed for the post's running example: the Dynamo graph and the
// AOTAutograd forward / backward graphs (custom backend printing each GraphModule), and the
// Triton template Inductor rendered (captured on macOS with a stubbed Triton backend; the kernels
// were generated, not run). Only `; x = None` clean-up statements were dropped. The BoxDiagram
// data is written by hand from torch/_inductor/kernel/flex and torch/_higher_order_ops at the
// same commit.
import type { CodeStage, StageTag } from "../compiler/llvmData";
import type { BoxEdge, BoxNode } from "../payments/boxDiagram";

export const scoreModTags: StageTag[] = [
	{ tag: "idx", label: "slopes[h]" },
	{ tag: "sub", label: "kv_idx − q_idx" },
	{ tag: "mul", label: "×" },
	{ tag: "add", label: "score +" },
	{ tag: "mask", label: "mask_mod" },
	{ tag: "dscore", label: "∂out/∂score" },
	{ tag: "dslope", label: "∂out/∂slopes" },
];

export const scoreModStages: CodeStage[] = [
	{
		id: "python",
		label: "Python",
		tool: "the user's code",
		lang: "python",
		code: `slopes = torch.tensor([0.5, 0.25], device="cuda", requires_grad=True)

def alibi(score, b, h, q_idx, kv_idx):
    return score + slopes[h] * (kv_idx - q_idx)

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

out = torch.compile(flex_attention)(q, k, v, score_mod=alibi, block_mask=block_mask)`,
		marks: [
			[3, "add"],
			[6, "mask"],
		],
		note: "Two scalar functions. score_mod gets one score and its four indices and returns a new score; mask_mod returns a bool. alibi reads slopes, a tensor that is not an argument: Dynamo has to notice it and pass it in.",
	},
	{
		id: "dynamo",
		label: "Dynamo",
		tool: "GraphModule given to the torch.compile backend",
		lang: "python",
		code: `class score_mod_0(torch.nn.Module):
    def forward(self, child: "f32[]", child_1: "i32[]", child_2: "i32[]", child_3: "i32[]", child_4: "i32[]", g_import_main_slopes: "f32[2]"):
        getitem: "f32[]" = g_import_main_slopes[child_2]
        sub: "i32[]" = child_4 - child_3
        mul: "f32[]" = getitem * sub
        add: "f32[]" = child + mul
        return add

class mask_fn_0(torch.nn.Module):
    def forward(self, child: "i32[]", child_1: "i32[]", child_2: "i32[]", child_3: "i32[]"):
        ge: "b8[]" = child_2 >= child_3
        return ge`,
		marks: [
			[2, "idx"],
			[3, "sub"],
			[4, "mul"],
			[5, "add"],
			[10, "mask"],
		],
		note: "FlexAttentionHigherOrderVariable traced each function with 0-d tensors (f32[] for the score, i32[] for b, h, q_idx, kv_idx) and stored them as submodules of the graph. The global slopes became a sixth input, g_import_main_slopes; in the main graph it appears in the flex_attention call's score_mod_other_buffers tuple.",
	},
	{
		id: "aot",
		label: "AOT forward",
		tool: "forward graph from aot_module_simplified",
		lang: "python",
		code: `class sdpa_score0(torch.nn.Module):
    def forward(self, arg0_1: "f32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "f32[2]"):
        index: "f32[]" = torch.ops.aten.index.Tensor(arg5_1, [arg2_1])
        sub: "i32[]" = torch.ops.aten.sub.Tensor(arg4_1, arg3_1)
        mul: "f32[]" = torch.ops.aten.mul.Tensor(index, sub)
        add: "f32[]" = torch.ops.aten.add.Tensor(arg0_1, mul)
        return add

class sdpa_mask0(torch.nn.Module):
    def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]"):
        ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)
        return ge`,
		marks: [
			[2, "idx"],
			[3, "sub"],
			[4, "mul"],
			[5, "add"],
			[10, "mask"],
		],
		note: "trace_flex_attention re-traced both functions with make_fx, so they are now ATen ops. TransformGetItemToIndex turned slopes[h] into aten.index with a tensor index instead of a Python int, which would have specialised the graph on one head.",
	},
	{
		id: "triton-fwd",
		label: "Triton forward",
		tool: "forward_block_mn in the rendered flex_attention template",
		lang: "python",
		code: `# modification(subgraph_number=0, score="qk", b="off_z", h="off_h", m="m", n="n")
tmp0 = (qk)
tmp1 = tmp0.to(tl.float32)
tmp2 = (off_h)
tmp3 = tl.load(in_ptr9 + tmp2)
tmp4 = (n)
tmp5 = (m)
tmp6 = tmp4 - tmp5
tmp7 = tmp6.to(tl.float32)
tmp8 = tmp3 * tmp7
tmp9 = tmp1 + tmp8
post_mod_scores = tmp9

# modification(subgraph_number=1, ...), inside "if not IS_FULL_BLOCKS:"
tmp10 = (m)
tmp11 = (n)
tmp12 = tmp10 >= tmp11
mask_mod_output = tmp12`,
		marks: [
			[3, "idx"],
			[4, "idx"],
			[7, "sub"],
			[8, "sub"],
			[9, "mul"],
			[10, "add"],
			[16, "mask"],
		],
		note: "Each placeholder became the name of a template variable: score → qk (a [BLOCK_M, BLOCK_N] tile), h → off_h, q_idx → m, kv_idx → n. The scalar function now runs on a whole tile. slopes is no placeholder, so its load became a real tl.load from a new kernel argument, in_ptr9.",
	},
	{
		id: "joint",
		label: "AOT joint",
		tool: "joint_graph0 in the backward graph",
		lang: "python",
		code: `class joint_graph0(torch.nn.Module):
    def forward(self, arg0_1: "f32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "f32[]", arg6_1: "f32[2]"):
        sub: "i32[]" = torch.ops.aten.sub.Tensor(arg4_1, arg3_1)
        mul_1: "f32[]" = torch.ops.aten.mul.Tensor(arg5_1, sub)
        zeros_and_scatter: "f32[2]" = torch.ops.flex_lib.zeros_and_scatter.default([2], [arg2_1], mul_1)
        return [arg5_1, None, None, None, None, zeros_and_scatter]`,
		marks: [
			[2, "sub"],
			[3, "dslope"],
			[4, "dslope"],
			[5, "dscore"],
		],
		note: "create_fw_bw_graph differentiated score_mod as a scalar function. arg5_1 is the incoming gradient for the new score. Since score_mod adds the score unchanged, its gradient passes straight through (the first output). The gradient for slopes[h] is grad × (kv_idx − q_idx), scattered into a zero vector of slopes' shape at index h.",
	},
	{
		id: "triton-bwd",
		label: "Triton backward",
		tool: "bwd_dq_block_mn in the rendered flex_attention_backward template",
		lang: "python",
		code: `# modification(subgraph_number=1, grad_score_mod="ds", ...)
tmp12 = (ds)
grad_scores = tmp12

# modification(subgraph_number=3, mask="scatter_mask", ...), under "if WRITE_DQ:"
scatter_mask = (offs_m2[:, None] < Q_LEN ) & (offs_n2[None, :] < KV_LEN)
tmp13 = (off_hq)
tmp14 = (ds)
tmp15 = (n)
tmp16 = (m)
tmp17 = tmp15 - tmp16
tmp18 = tmp17.to(tl.float32)
tmp19 = tmp14 * tmp18
tl.atomic_add(in_ptr17 + tl.broadcast_to(tmp13, tmp19.shape), tmp19, scatter_mask, sem='relaxed')`,
		marks: [
			[1, "dscore"],
			[2, "dscore"],
			[9, "sub"],
			[10, "sub"],
			[11, "dslope"],
			[12, "dslope"],
		],
		note: "The joint graph is lowered twice into the backward template. Its first output becomes grad_scores (here just ds). The scatter becomes a store, and ModificationWrapper.store writes every store as a tl.atomic_add, because many programs add into the same slopes entry at once.",
	},
];

export const pipelineNodes: BoxNode[] = [
	{
		id: "user",
		label: "flex_attention(...)",
		sub: "torch.nn.attention.flex_attention",
		col: 0,
		row: 0,
		group: 0,
		note: "Checks shapes and devices, builds an empty BlockMask if none was given, fills in the kernel options (BACKEND, PRESCALE_QK, OUTPUT_LOGSUMEXP, ...) and calls the <code>flex_attention</code> higher-order operator, <code>flex_attention_hop</code>.",
	},
	{
		id: "eager",
		label: "no torch.compile",
		sub: "math_attention",
		col: 0,
		row: 1,
		group: 5,
		note: "Called outside <code>torch.compile</code>, <code>flex_attention</code> warns once and compiles just the operator with an eager backend. The operator's <code>CompositeExplicitAutograd</code> rule then runs <code>math_attention</code>: it vmaps score_mod over the full B × H × Q × KV score tensor. Correct, but it materializes that tensor.",
	},
	{
		id: "dynamo",
		label: "Dynamo",
		sub: "FlexAttentionHigherOrderVariable",
		col: 1,
		row: 0,
		group: 1,
		note: "Traces <code>score_mod</code> and <code>mask_mod</code> with <code>speculate_subgraph</code>, feeding them 0-d tensors. Each becomes a subgraph module; tensors they read from outside (closures, globals) are lifted into extra inputs.",
	},
	{
		id: "hop",
		label: "FX graph",
		sub: "higher_order.flex_attention",
		col: 2,
		row: 0,
		group: 1,
		note: "One node, <code>torch.ops.higher_order.flex_attention(q, k, v, score_mod_0, block_mask_tuple, scale, kernel_options, score_mod_other_buffers, mask_mod_other_buffers)</code>. The block mask is a tuple of its tensors plus <code>mask_fn_0</code>.",
	},
	{
		id: "aot",
		label: "AOTAutograd",
		sub: "FlexAttentionAutogradOp",
		col: 2,
		row: 1,
		group: 2,
		note: "The operator's Autograd rule wraps it in an <code>autograd.Function</code>. <code>create_fw_bw_graph</code> builds a joint graph of score_mod (its scalar derivative). The forward graph keeps <code>flex_attention</code>; the backward graph gets <code>flex_attention_backward</code>, with the out and LSE saved from the forward.",
	},
	{
		id: "lowering",
		label: "Inductor lowering",
		sub: "register_lowering(flex_attention)",
		col: 1,
		row: 2,
		group: 3,
		note: "<code>torch/_inductor/kernel/flex/flex_attention.py</code>. Picks the implementation by device and BACKEND, lowers the two subgraphs, sets the template's constants and builds one template choice per config.",
	},
	{
		id: "cpu",
		label: "CPU / MPS",
		sub: "C++ template, Metal shader",
		col: 0,
		row: 2,
		group: 5,
		note: "<code>lower_cpu</code> (a C++ template for x86 CPUs) and <code>lower_mps</code> (a Metal shader). Both are inference only: <code>_validate_device</code> rejects inputs that require grad.",
	},
	{
		id: "flash",
		label: "FLASH",
		sub: "CuTe DSL template",
		col: 2,
		row: 2,
		group: 5,
		note: "Only when <code>kernel_options={'BACKEND': 'FLASH'}</code>. Renders a CuTe DSL template that calls the <code>flash_attn.cute</code> kernels; needs that package installed and fails if a captured buffer requires grad.",
	},
	{
		id: "decode",
		label: "flex_decoding",
		sub: "split-KV Triton template",
		col: 0,
		row: 3,
		group: 4,
		note: "Chosen automatically for fewer than 128 queries (with static batch and heads, a power-of-two GQA ratio). Splits the keys across programs and merges the partial softmaxes with ordinary Inductor ops.",
	},
	{
		id: "subgraph",
		label: "subgraph → buffer",
		sub: "build_subgraph_buffer",
		col: 1,
		row: 3,
		group: 3,
		note: "Runs each subgraph through <code>PointwiseSubgraphLowering</code> with placeholder inputs named score, b, h, m, n. The result is a <code>ComputedBuffer</code> whose <code>inner_fn</code> is the scalar function in Inductor's define-by-run IR. It is never realized; the template inlines it.",
	},
	{
		id: "template",
		label: "TritonTemplate",
		sub: "flex_attention.py.jinja",
		col: 1,
		row: 4,
		group: 4,
		note: "The Jinja template is rendered once per config. Its <code>{{ modification(...) }}</code> calls run the subgraph's <code>inner_fn</code> under <code>ModificationWrapper</code>, which turns each placeholder load into a template variable (qk, off_h, m, n).",
	},
	{
		id: "autotune",
		label: "autotune_select_algorithm",
		sub: "pick a config",
		col: 2,
		row: 4,
		group: 3,
		note: "With one config there is nothing to benchmark. Under <code>max_autotune</code> each choice is compiled and timed on fake block-mask inputs (<code>create_num_blocks_fake_generator</code>). The winner becomes a <code>TritonTemplateBuffer</code>, scheduled and called from the wrapper like any kernel.",
	},
];

export const pipelineEdges: BoxEdge[] = [
	{ from: "user", to: "dynamo", label: "compiled" },
	{ from: "user", to: "eager", dashed: true },
	{ from: "dynamo", to: "hop" },
	{ from: "hop", to: "aot" },
	{ from: "aot", to: "lowering" },
	{ from: "lowering", to: "cpu", dashed: true },
	{ from: "lowering", to: "flash", dashed: true },
	{ from: "lowering", to: "decode", dashed: true },
	{ from: "lowering", to: "subgraph" },
	{ from: "subgraph", to: "template" },
	{ from: "template", to: "autotune" },
];

export const decodeNodes: BoxNode[] = [
	{
		id: "q",
		label: "query as GQA groups",
		sub: "[1, 2, 4, 1, 64]",
		col: 0,
		row: 0,
		group: 0,
		note: "The query <code>[B=1, Hq=8, 1, 64]</code> is viewed as <code>[B, Hkv=2, G=4, 1, 64]</code> with <code>as_strided</code>. The 4 query heads that share a KV head go into one program as its BLOCK_M = 16 rows (4 real, the rest masked), so each K/V tile is loaded once for all four.",
	},
	{
		id: "split",
		label: "264 programs",
		sub: "grid (1, B·Hkv = 2, SPLIT_KV = 132)",
		col: 1,
		row: 0,
		group: 1,
		note: "<code>get_split_k</code> returns <code>num_SM // (B·Hkv) * 2</code> = 132 // 2 × 2 = 132 on a 132-SM H100. Each program takes one KV head and a slice of <code>TILE_KV</code> keys, rounded up to a multiple of <code>BLOCK_N</code> = 64, and runs the same online softmax as the main kernel over it. With a BlockMask from <code>create_block_mask</code>, <code>TILE_KV</code> = cdiv(1024, 132) = 8 → 64, so 16 programs per KV head get one tile each and the rest have no work.",
	},
	{
		id: "bufs",
		label: "partial results",
		sub: "buf_M, buf_L, buf_ACC",
		col: 2,
		row: 0,
		group: 2,
		note: "Each program writes its row maximum and sum (<code>buf_M</code>, <code>buf_L</code>: <code>[1, 132, 8, 1]</code>, fp32) and its unnormalised output (<code>buf_ACC</code>: <code>[1, 132, 8, 1, 64]</code>, fp32).",
	},
	{
		id: "merge",
		label: "merge splits",
		sub: "max, exp2, sum: plain lowerings",
		col: 2,
		row: 1,
		group: 3,
		note: "<code>g_M = max(buf_M, dim=1)</code>, <code>alpha = exp2(buf_M − g_M)</code>, <code>g_L = sum(buf_L · alpha)</code>, <code>out = sum(buf_ACC · alpha) / g_L</code>. These are ordinary Inductor lowerings, which the scheduler fuses into reduction kernels after the template.",
	},
	{
		id: "out",
		label: "output",
		sub: "[1, 8, 1, 64] fp16, LSE",
		col: 1,
		row: 1,
		group: 4,
		note: "Cast back to the query's dtype. The LSE is <code>log2(g_L) + g_M</code>, still in base 2 like the main kernel's.",
	},
];

export const decodeEdges: BoxEdge[] = [
	{ from: "q", to: "split" },
	{ from: "split", to: "bufs" },
	{ from: "bufs", to: "merge" },
	{ from: "merge", to: "out" },
];
