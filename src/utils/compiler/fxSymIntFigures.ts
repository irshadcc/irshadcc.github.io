// Hand-written figure data for the SymInt / SymNode section of the Torch FX post. The calls follow
// a logging hook on SymNode methods while make_fx(tracing_mode="symbolic") traced
// `torch.empty(x.shape[0] * 2)` on torch 2.13.0; file and line arguments are as observed.
import type { Message, Participant } from "../payments/sequence";

export const symIntParticipants: Participant[] = [
	{ id: "user", label: "User code", sub: "Python" },
	{ id: "symint", label: "torch.SymInt", sub: "Python" },
	{ id: "node", label: "SymNode", sub: "Python, sympy" },
	{ id: "env", label: "ShapeEnv, tracer", sub: "Python" },
	{ id: "cpp", label: "ATen kernel", sub: "C++, c10::SymInt" },
	{ id: "bridge", label: "PythonSymNodeImpl", sub: "C++ → Python" },
];

export const symIntMessages: Message[] = [
	{
		from: "user",
		to: "symint",
		label: "x.shape[0] * 2",
		step: "__mul__",
		phase: "Arithmetic in Python",
		note: "<code>x.shape[0]</code> is a <code>torch.SymInt</code> whose node has expression <code>s75</code> and hint 6. <code>*</code> calls <code>SymInt.__mul__</code>, which <code>sym_node.py</code> installed at import time.",
	},
	{
		from: "symint",
		to: "node",
		label: "node.mul(wrap_int(2))",
		step: "SymNode.mul",
		note: "The user-level magic method promotes the plain int 2 to a constant SymNode with <code>wrap_int</code>, then calls the node-level <code>mul</code>, so the node code only ever sees two SymNodes.",
	},
	{
		from: "node",
		to: "env",
		label: "handle_sym_dispatch(operator.mul)",
		step: "record",
		note: "A proxy mode is active (make_fx is tracing), so the operation is also handed to the tracer, which registers a lazy FX node <code>operator.mul</code>. The node is materialized only if a graph value ends up depending on it.",
	},
	{
		from: "node",
		to: "symint",
		label: "SymNode(2*s75, hint=12)",
		step: "result",
		reply: true,
		tone: 1,
		note: "The new SymNode holds the sympy expression <code>Mul(2, s75)</code>, the hint 6 × 2 = 12 and the same ShapeEnv. It is wrapped in a new <code>torch.SymInt</code> and returned to user code.",
	},
	{
		from: "user",
		to: "cpp",
		label: "torch.empty([2*s75])",
		step: "into C++",
		phase: "Crossing into C++",
		note: "The pybind <code>type_caster&lt;c10::SymInt&gt;::load</code> sees a <code>torch.SymInt</code>, reads its <code>.node</code>, wraps it in a <code>PythonSymNodeImpl</code> and stores that pointer in a one-word <code>c10::SymInt</code>.",
	},
	{
		from: "cpp",
		to: "bridge",
		label: "size.sym_ge(0)",
		step: "sym_ge",
		note: "<code>check_size_nonnegative</code> (EmptyTensor.h) runs <code>TORCH_SYM_CHECK(x.sym_ge(0), ...)</code>. The size is heap-allocated, so the fast path fails and the slow path calls the virtual <code>SymNodeImpl::ge</code>.",
	},
	{
		from: "bridge",
		to: "node",
		label: "pyobj.ge(other)",
		step: "back to Python",
		note: "<code>PythonSymNodeImpl::ge</code> takes the GIL and calls the Python method of the same name on the wrapped SymNode. Sympy simplifies <code>2*s75 &gt;= 0</code> to <code>True</code>, because size symbols are declared positive integers.",
	},
	{
		from: "cpp",
		to: "bridge",
		label: 'expect_true("EmptyTensor.h", 24)',
		step: "expect_true",
		note: "<code>TORCH_SYM_CHECK</code> needs a C++ <code>bool</code>, so it calls <code>SymBool::expect_true(__FILE__, __LINE__)</code>. The logging hook saw these exact arguments arrive in Python.",
	},
	{
		from: "bridge",
		to: "env",
		label: "SymNode.expect_true → guard_bool",
		step: "no guard",
		tone: 1,
		note: "The expression is already <code>True</code>, so <code>ShapeEnv</code> answers without adding a guard. Had it been undecided, this is where a guard (or, for unbacked symbols, a runtime assert) would be recorded, tagged with the C++ file and line.",
	},
	{
		from: "cpp",
		to: "user",
		label: "FakeTensor f32[2*s75]",
		step: "return",
		reply: true,
		phase: "Back to Python",
		note: "The kernel stores the SymInt sizes in the tensor's <code>SymbolicShapeMeta</code>. When <code>e.shape</code> is read, <code>type_caster::cast</code> unwraps each <code>PythonSymNodeImpl</code> and returns <code>torch.SymInt</code> around the very same Python SymNode.",
	},
];
