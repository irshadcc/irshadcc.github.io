// Figure data for the higher-order operators post (src/content/posts/higher-order-operators.mdx):
// the registration tables of cond_op, two step-throughs of a cond call (eager with autograd, and
// make_fx tracing) and the graph listings. Listings and dispatch traces were captured from
// torch 2.14.1+cpu (git 5c488690) by a scratch script that wraps HigherOrderOperator.dispatch
// to print each key it is called with.
import type { BoxEdge, BoxNode } from "../payments/boxDiagram";
import type { Message, Participant } from "../payments/sequence";
import type { CodeStage, StageTag } from "./llvmData";

const src = (file: string, line: number) =>
	`https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/${file}#L${line}`;
const link = (file: string, line: number, text: string) =>
	`<a href="${src(file, line)}"><code>${text}</code></a>`;

// The three tables a HigherOrderOperator dispatches through, filled in with what cond.py
// registers (checked by printing cond_op.py_kernels, .python_key_table and .functorch_table).
export const hopTableNodes: BoxNode[] = [
	{
		id: "call",
		label: "cond_op(...)",
		sub: "__call__",
		col: 0,
		row: 0,
		group: 5,
		note: `${link("_ops.py", 540, "HigherOrderOperator.__call__")} flattens the arguments, collects every tensor, computes a dispatch key set from them and calls <code>dispatch</code> with the highest-priority key.`,
	},
	{
		id: "dispatch",
		label: "dispatch(key)",
		sub: "picks one table",
		col: 1,
		row: 0,
		group: 5,
		note: `${link("_ops.py", 406, "HigherOrderOperator.dispatch")} looks at the key. <code>FuncTorchDynamicLayerFrontMode</code> goes to the functorch table, <code>Python</code> goes to the mode table, and every other key is resolved to an entry of <code>py_kernels</code>.`,
	},
	{ id: "pk", label: "py_kernels", col: 0, row: 1, h: 3, group: -1 },
	{ id: "pt", label: "python_key_table", col: 1, row: 1, h: 3, group: -1 },
	{ id: "ft", label: "functorch_table", col: 2, row: 1, h: 2, group: -1 },
	{
		id: "pk-ag",
		label: "Autograd",
		sub: "cond_autograd",
		col: 0,
		row: 1,
		group: 1,
		note: `Registered by ${link("_higher_order_ops/cond.py", 378, "@cond_op.py_autograd_impl")}. If any tensor requires grad, it calls <code>CondAutogradOp.apply</code>; otherwise it re-calls <code>cond_op</code> below autograd. Reached from <code>AutogradCPU</code>, <code>AutogradCUDA</code>, ... through the <code>Autograd</code> alias.`,
	},
	{
		id: "pk-dense",
		label: "CompositeExplicitAutograd",
		sub: "cond_op_dense",
		col: 0,
		row: 2,
		group: 0,
		note: `${link("_higher_order_ops/cond.py", 300, "cond_op_dense")}: the real computation. It evaluates <code>pred</code> and runs one branch. Reached from <code>CPU</code>, <code>CUDA</code> and every other backend key.`,
	},
	{
		id: "pk-func",
		label: "Functionalize",
		sub: "cond_func (C++ API)",
		col: 0,
		row: 3,
		group: 3,
		note: `One of the three entries that ${link("_ops.py", 171, "py_functionalize_impl")} creates for ${link("_higher_order_ops/cond.py", 710, "cond_func")}: this one serves the C++ <code>Functionalize</code> dispatch key.`,
	},
	{
		id: "pt-proxy",
		label: "ProxyTorchDispatchMode",
		sub: "trace_cond",
		col: 1,
		row: 1,
		group: 2,
		note: `${link("_higher_order_ops/cond.py", 248, "trace_cond")}: active while <code>make_fx</code>, <code>torch.export</code> or AOTAutograd traces. It traces each branch into its own graph and records one <code>cond</code> node in the outer graph.`,
	},
	{
		id: "pt-fake",
		label: "FakeTensorMode",
		sub: "cond_fake_tensor_mode",
		col: 1,
		row: 2,
		group: 2,
		note: `${link("_higher_order_ops/cond.py", 393, "cond_fake_tensor_mode")}: runs both branches on fake tensors and merges their outputs into one fake tensor whose sizes cover both.`,
	},
	{
		id: "pt-func",
		label: "FunctionalTensorMode",
		sub: "cond_func (Python API)",
		col: 1,
		row: 3,
		group: 3,
		note: "The second entry made by <code>py_functionalize_impl</code>: functionalization implemented as a Python dispatch mode, which is what <code>torch.export</code> and AOTAutograd use.",
	},
	{
		id: "ft-vmap",
		label: "Vmap",
		sub: "cond_batch_rule",
		col: 2,
		row: 1,
		group: 4,
		note: `${link("_higher_order_ops/cond.py", 746, "cond_batch_rule")}: the batching rule for <code>torch.vmap</code>. A batched predicate turns cond into <code>torch.where</code>.`,
	},
	{
		id: "ft-func",
		label: "Functionalize",
		sub: "cond_func (functorch)",
		col: 2,
		row: 2,
		group: 3,
		note: "The third entry made by <code>py_functionalize_impl</code>: for <code>torch.func.functionalize</code>, a functorch transform.",
	},
];

export const hopTableEdges: BoxEdge[] = [
	{ from: "call", to: "dispatch", label: "highest key" },
	{ from: "dispatch", to: "pk" },
	{ from: "dispatch", to: "pt", label: "Python" },
	{ from: "dispatch", to: "ft", label: "functorch" },
];

// Eager call with a tensor that requires grad. The dispatch keys were printed by the scratch
// script: AutogradCPU -> CPU for the forward, AutogradCPU -> CPU again for the backward's cond.
export const eagerParticipants: Participant[] = [
	{ id: "user", label: "user code" },
	{ id: "call", label: "__call__", sub: "HigherOrderOperator" },
	{ id: "disp", label: "dispatch", sub: "+ resolve_key" },
	{ id: "ag", label: "CondAutogradOp", sub: "autograd.Function" },
	{ id: "dense", label: "cond_op_dense", sub: "CompositeExplicit…" },
	{ id: "br", label: "true_fn", sub: "x.cos()" },
];

export const eagerMessages: Message[] = [
	{
		from: "user",
		to: "call",
		label: "cond_op(p, t, f, (x,))",
		step: "call",
		phase: "forward",
		note: "<code>p = tensor(True)</code>, <code>x = tensor([0., 1., 2.], requires_grad=True)</code>. <code>CondOp.__call__</code> checks that every operand is a tensor, int or SymInt and calls <code>HigherOrderOperator.__call__</code>.",
	},
	{
		from: "call",
		to: "disp",
		label: "dispatch(AutogradCPU)",
		step: "key set",
		note: "The key set is the union of the tensors' keys, <code>{CPU, ADInplaceOrView, AutogradCPU, AutocastCPU}</code>, masked by <code>non_fallthrough_keys</code>, which drops <code>ADInplaceOrView</code> and <code>AutocastCPU</code>. What is left is <code>{CPU, AutogradCPU}</code>, and the highest-priority key is <code>AutogradCPU</code>.",
	},
	{
		from: "disp",
		to: "ag",
		label: "maybe_run_autograd",
		step: "autograd",
		note: "<code>resolve_key(AutogradCPU)</code> finds no kernel for <code>AutogradCPU</code> itself and falls to the <code>Autograd</code> alias, where <code>py_autograd_impl</code> put <code>maybe_run_autograd</code>. Grad mode is on and <code>x</code> requires grad, so it calls <code>CondAutogradOp.apply(p, t, f, x)</code>.",
	},
	{
		from: "ag",
		to: "ag",
		label: "create_bw_fn × 2, save x",
		step: "fwd setup",
		note: "<code>forward</code> stores <code>pred</code>, builds a backward function for each branch with <code>create_bw_fn</code> (not traced yet), snapshots the thread's include/exclude key sets and saves the operands for backward.",
	},
	{
		from: "ag",
		to: "call",
		label: "cond_op(...) below autograd",
		step: "re-enter",
		note: "Inside <code>_AutoDispatchBelowAutograd</code> the autograd keys are in the thread's exclude set, so calling <code>cond_op</code> again skips this layer.",
	},
	{
		from: "call",
		to: "disp",
		label: "dispatch(CPU)",
		step: "key set 2",
		note: "Same tensors, but the exclude set now removes <code>AutogradCPU</code>. The highest key left is <code>CPU</code>.",
	},
	{
		from: "disp",
		to: "dense",
		label: "resolve_key → CEA",
		step: "dense",
		note: "There is no kernel registered for <code>CPU</code>, so <code>resolve_key</code> takes the <code>CompositeExplicitAutograd</code> kernel, <code>cond_op_dense</code>. The result is stored in <code>_dispatch_cache[CPU]</code> so the next call skips <code>resolve_key</code>.",
	},
	{
		from: "dense",
		to: "br",
		label: "true_fn(x)",
		step: "branch",
		note: "<code>if pred:</code> calls <code>bool()</code> on the one-element tensor, which is <code>True</code>, so only the true branch runs.",
	},
	{
		from: "br",
		to: "user",
		label: "(cos(x),)",
		step: "result",
		reply: true,
		tone: 1,
		note: "<code>cos([0, 1, 2]) = [1.0000, 0.5403, -0.4161]</code>. <code>autograd.Function.apply</code> attaches <code>grad_fn=CondAutogradOpBackward</code> to the output.",
	},
	{
		from: "user",
		to: "ag",
		label: "backward(ones)",
		step: "backward",
		phase: "backward",
		note: "<code>y[0].sum().backward()</code> sends a gradient of ones into <code>CondAutogradOp.backward</code>.",
	},
	{
		from: "ag",
		to: "ag",
		label: "materialize bw graphs",
		step: "bw graphs",
		note: "<code>materialize_as_graph</code> traces <em>both</em> backward functions with <code>make_fx</code>, under grad mode, into two graph modules. The predicate is not looked at yet.",
	},
	{
		from: "ag",
		to: "call",
		label: "cond_op(p, t_bw, f_bw, …)",
		step: "bw cond",
		note: "The backward is itself a <code>cond</code>: the same predicate picks between the two backward graphs. Grad mode is off during backward, so this call goes <code>AutogradCPU</code> → below autograd → <code>CPU</code> → <code>cond_op_dense</code>.",
	},
	{
		from: "ag",
		to: "user",
		label: "x.grad = -sin(x)",
		step: "grads",
		reply: true,
		tone: 1,
		note: "The true backward graph computes <code>-sin(x) · 1 = [-0.0000, -0.8415, -0.9093]</code>, the same as differentiating <code>x.cos().sum()</code> directly.",
	},
];

// make_fx tracing with real tensors: AutogradCPU -> Python -> CPU in the dispatch trace.
export const traceParticipants: Participant[] = [
	{ id: "mfx", label: "make_fx", sub: "ProxyTorchDispatchMode" },
	{ id: "call", label: "__call__ +", sub: "dispatch" },
	{ id: "tc", label: "trace_cond", sub: "python_key_table" },
	{ id: "sub", label: "sub-tracer", sub: "reenter_make_fx" },
	{ id: "root", label: "outer graph", sub: "tracer.root" },
	{ id: "dense", label: "cond_op_dense", sub: "real values" },
];

export const traceMessages: Message[] = [
	{
		from: "mfx",
		to: "call",
		label: "cond_op(p, t, f, (x,))",
		step: "call",
		note: "<code>make_fx(fn)(p, x)</code> runs <code>fn</code> with <code>ProxyTorchDispatchMode</code> on the mode stack. That mode adds the <code>Python</code> key to every dispatch.",
	},
	{
		from: "call",
		to: "call",
		label: "AutogradCPU → Python",
		step: "keys",
		note: "First <code>AutogradCPU</code>: nothing requires grad, so <code>maybe_run_autograd</code> re-calls below autograd. Then the highest key is <code>Python</code>.",
	},
	{
		from: "call",
		to: "tc",
		label: "python_key_table[mode]",
		step: "mode",
		note: "<code>dispatch(Python)</code> finds the current mode, <code>ProxyTorchDispatchMode</code>, in <code>python_key_table</code>, pops it off the stack while the handler runs and calls <code>inner(mode, p, t, f, (x,))</code>, which calls <code>trace_cond</code>.",
	},
	{
		from: "tc",
		to: "sub",
		label: "trace true_fn",
		step: "true",
		note: "<code>reenter_make_fx(true_fn)(x)</code> asks the current tracer for a sub-tracer that shares its fake mode and symbolic shapes, and traces <code>true_fn</code> into a graph: <code>cos = aten.cos(x)</code>.",
	},
	{
		from: "tc",
		to: "sub",
		label: "trace false_fn",
		step: "false",
		note: "The same for <code>false_fn</code>: <code>sin = aten.sin(x)</code>. Both branches are traced, whatever the predicate's value is.",
	},
	{
		from: "tc",
		to: "root",
		label: "register_module × 2",
		step: "install",
		note: "<code>unique_graph_id</code> picks the first free name, <code>true_graph_0</code>, and <code>false_graph_0</code> follows. Both graphs become submodules of the outer graph's root module, so they can be referenced as attributes.",
	},
	{
		from: "tc",
		to: "root",
		label: "create_proxy(cond)",
		step: "node",
		tone: 1,
		note: "One <code>call_function</code> node with target <code>torch.ops.higher_order.cond</code> and arguments <code>(p_1, true_graph_0, false_graph_0, (x_1,))</code>. The proxies of <code>p</code> and <code>x</code> come from <code>unwrap_proxy</code>.",
	},
	{
		from: "tc",
		to: "dense",
		label: "cond_op(p, gm_t, gm_f, …)",
		step: "run",
		note: 'To know the output\'s value and metadata, <code>trace_cond</code> calls the operator again on the real arguments. The mode is popped, so dispatch reaches <code>CPU</code> and <code>cond_op_dense</code> runs <code>true_graph_0</code>. Under <code>tracing_mode="fake"</code> this call reaches <code>FakeTensorMode</code> instead.',
	},
	{
		from: "tc",
		to: "mfx",
		label: "track_tensor_tree",
		step: "track",
		reply: true,
		tone: 1,
		note: "<code>track_tensor_tree</code> links each output tensor to a proxy: the tuple's element 0 becomes <code>getitem = cond[0]</code>. Later ops on the output are recorded against that node.",
	},
];

// One function with torch.cond, captured four ways. Line indices are 0-based into `code`.
export const captureStages: CodeStage[] = [
	{
		id: "user",
		label: "user code",
		tool: "model.py",
		lang: "python",
		code: `class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

    def forward(self, x):
        def true_fn(x):
            return x * self.w     # reads self.w from the closure
        def false_fn(x):
            return x + 1
        return torch.cond(x.sum() > 0, true_fn, false_fn, (x,))

x = torch.tensor([0.0, 1.0, 2.0])   # sum is 3 > 0, so the true branch: [0, 2, 6]`,
		marks: [
			[7, "lift"],
			[10, "cond"],
		],
		note: "The true branch reads the parameter self.w without receiving it as an operand. Every capture below has to turn that closure read into an explicit input.",
	},
	{
		id: "dynamo",
		label: "Dynamo",
		tool: "torch.compile(m, backend=print_graph, fullgraph=True)",
		lang: "python",
		code: `def forward(self, L_x_, L_self_parameters_w_):
    l_x_ = L_x_
    l_self_parameters_w_ = L_self_parameters_w_
    sum_1 = l_x_.sum()
    gt = sum_1 > 0
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    cond = torch.ops.higher_order.cond(gt, cond_true_0, cond_false_0, (l_x_, l_self_parameters_w_))
    getitem = cond[0]
    return (getitem,)

# cond_true_0
def forward(self, l_x_, l_self_parameters_w__true_branch):
    l_x__1 = l_x_
    mul = l_x__1 * l_self_parameters_w__true_branch
    return (mul,)

# cond_false_0
def forward(self, l_x_, l_self_parameters_w__true_branch):
    l_x__1 = l_x_
    add = l_x__1 + 1
    return (add,)`,
		marks: [
			[4, "pred"],
			[7, "cond"],
			[12, "lift"],
			[14, "lift"],
			[18, "lift"],
		],
		note: "Dynamo's graph is still at the torch level (l_x_.sum(), not aten.sum). self.w was lifted into an operand, and the false branch takes it too (and ignores it) because both branches must have the same signature.",
	},
	{
		id: "export",
		label: "torch.export",
		tool: "torch.export.export(m, (x,))",
		lang: "python",
		code: `def forward(self, p_w, x):
    sum_1 = torch.ops.aten.sum.default(x)
    gt = torch.ops.aten.gt.Scalar(sum_1, 0)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (x, p_w))
    getitem = cond[0]
    return (getitem,)

# true_graph_0
def forward(self, x, p_w):
    mul = torch.ops.aten.mul.Tensor(x, p_w)
    return (mul,)

# false_graph_0
def forward(self, x, p_w):
    add = torch.ops.aten.add.Tensor(x, 1)
    return (add,)`,
		marks: [
			[2, "pred"],
			[5, "cond"],
			[11, "lift"],
		],
		note: "Export traces with make_fx after Dynamo, so the branches are ATen graphs and the names come from trace_cond: true_graph_0, false_graph_0. The parameter is the input p_w.",
	},
	{
		id: "makefx",
		label: "make_fx",
		tool: "make_fx(lambda p, x: cond_op(p, t, f, (x,)))(p, x)",
		lang: "python",
		code: `def forward(self, p_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(p_1, true_graph_0, false_graph_0, (x_1,))
    getitem = cond[0]
    return (getitem,)

# true_graph_0
def forward(self, x_1):
    cos = torch.ops.aten.cos.default(x_1)
    return (cos,)

# false_graph_0
def forward(self, x_1):
    sin = torch.ops.aten.sin.default(x_1)
    return (sin,)`,
		marks: [[3, "cond"]],
		note: "Calling cond_op directly under make_fx, with no Dynamo at all, gives the same shape of graph: this is the graph the step-through above builds. The branches here are the running example's cos and sin.",
	},
];

export const captureTags: StageTag[] = [
	{ tag: "pred", label: "predicate" },
	{ tag: "cond", label: "the cond node" },
	{ tag: "lift", label: "lifted closure variable" },
];

// invoke_once under make_fx and AOTAutograd (aot_function with printing compilers).
export const invokeStages: CodeStage[] = [
	{
		id: "makefx",
		label: "make_fx",
		tool: "make_fx(lambda x, w: invoke_once(body, (x, w)))(x, w)",
		lang: "python",
		code: `def forward(self, x_1: "f32[2, 2]", w_1: "f32[2, 2]"):
    body_graph_0 = self.body_graph_0
    invoke_once = torch.ops.higher_order.invoke_once(body_graph_0, (x_1, w_1))
    getitem: "f32[2, 2]" = invoke_once[0]
    return (getitem,)

class body_graph_0(torch.nn.Module):
    def forward(self, x_1: "f32[2, 2]", w_1: "f32[2, 2]"):
        mm: "f32[2, 2]" = torch.ops.aten.mm.default(x_1, w_1)
        relu: "f32[2, 2]" = torch.ops.aten.relu.default(mm)
        return (relu,)`,
		marks: [
			[2, "hop"],
			[8, "body"],
			[9, "body"],
		],
		note: "The proxy kernel turned the Python function body into the submodule body_graph_0 and recorded one invoke_once node that points at it.",
	},
	{
		id: "fw",
		label: "AOT forward",
		tool: "aot_function(fn, fw_compiler=print_graph, ...)",
		lang: "python",
		code: `def forward(self, primals_1: "f32[2, 2]", primals_2: "f32[2, 2]"):
    body_graph_0 = self.body_graph_0
    invoke_once = torch.ops.higher_order.invoke_once(body_graph_0, (primals_1, primals_2))
    getitem: "f32[2, 2]" = invoke_once[0]
    return (getitem, primals_1, primals_2)

class body_graph_0(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        mm: "f32[2, 2]" = torch.ops.aten.mm.default(arg0_1, arg1_1)
        relu: "f32[2, 2]" = torch.ops.aten.relu.default(mm)
        return (relu,)`,
		marks: [
			[2, "hop"],
			[4, "saved"],
		],
		note: "AOTAutograd traced the joint graph through the autograd kernel. The forward returns the output and saves both inputs, because the backward re-runs the body (see the next stage).",
	},
	{
		id: "bw",
		label: "AOT backward",
		tool: "aot_function(fn, bw_compiler=print_graph, ...)",
		lang: "python",
		code: `def forward(self, primals_1: "f32[2, 2]", primals_2: "f32[2, 2]", tangents_1: "f32[2, 2]"):
    body_graph_1 = self.body_graph_1
    invoke_once_1 = torch.ops.higher_order.invoke_once(body_graph_1, (primals_1, primals_2, tangents_1))
    getitem_2: "f32[2, 2]" = invoke_once_1[1]
    return (None, getitem_2)

class body_graph_1(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]", arg2_1: "f32[2, 2]"):
        mm: "f32[2, 2]" = torch.ops.aten.mm.default(arg0_1, arg1_1)
        relu: "f32[2, 2]" = torch.ops.aten.relu.default(mm)
        detach: "f32[2, 2]" = torch.ops.aten.detach.default(relu)
        detach_1: "f32[2, 2]" = torch.ops.aten.detach.default(detach)
        threshold_backward: "f32[2, 2]" = torch.ops.aten.threshold_backward.default(arg2_1, detach_1, 0)
        t: "f32[2, 2]" = torch.ops.aten.t.default(arg0_1)
        mm_1: "f32[2, 2]" = torch.ops.aten.mm.default(t, threshold_backward)
        zeros_like: "f32[2, 2]" = torch.ops.aten.zeros_like.default(arg0_1, pin_memory = False)
        return [zeros_like, mm_1]`,
		marks: [
			[2, "hop"],
			[8, "body"],
			[9, "body"],
			[12, "grad"],
			[14, "grad"],
			[15, "grad"],
		],
		note: "The backward is another invoke_once, whose body is the joint function traced by create_bw_fn: it recomputes relu(x @ w), masks the incoming gradient with threshold_backward and multiplies by xᵀ. x does not require grad, so its slot is zeros_like inside the body and None outside.",
	},
];

export const invokeTags: StageTag[] = [
	{ tag: "hop", label: "invoke_once node" },
	{ tag: "body", label: "forward body" },
	{ tag: "saved", label: "saved for backward" },
	{ tag: "grad", label: "gradient ops" },
];
