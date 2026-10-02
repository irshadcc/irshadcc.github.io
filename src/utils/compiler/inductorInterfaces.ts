// Figure data for the TorchInductor post: the objects that take part in code generation and who
// calls whom, drawn with BoxDiagram.astro. Written by hand from torch/_inductor at git cf30153c
// (scheduler.py, codegen/common.py, codegen/simd.py, codegen/triton.py, codegen/wrapper.py).

import type { BoxEdge, BoxNode } from "../payments/boxDiagram";

export const codegenNodes: BoxNode[] = [
	{
		id: "registry",
		label: "device registry",
		sub: "register_backend_for_device",
		col: 0,
		row: 0,
		group: 5,
		note: "Maps a device type to its classes: a <code>BaseScheduling</code> subclass and the wrapper generators. Filled by <code>init_backend_registration</code> for built-in devices, unless something was registered first.",
	},
	{
		id: "scheduler",
		label: "Scheduler",
		sub: "scheduler.py",
		col: 1,
		row: 0,
		group: 0,
		note: "Walks the fused nodes in order (<code>Scheduler._codegen</code>). Generated nodes go to the device's backend; extern calls it writes into the wrapper itself; it also emits buffer allocations and frees as their users finish.",
	},
	{
		id: "wrapper",
		label: "PythonWrapperCodegen",
		sub: "wrapper lines",
		col: 2,
		row: 0,
		group: 4,
		note: "Collects line objects: kernel definitions, kernel calls, allocations, frees, reuses. <code>memory_plan_reuse</code> then plans buffer reuse over the whole list and prints the <code>call</code> function.",
	},
	{
		id: "backend",
		label: "BaseScheduling",
		sub: "TritonScheduling, CppScheduling",
		col: 1,
		row: 1,
		group: 1,
		note: "One per device. <code>codegen_node</code> orders the nodes of a fused node, picks a tiling, creates a kernel object, runs each node's loop body inside it, and asks the kernel for its source.",
	},
	{
		id: "kernel",
		label: "Kernel",
		sub: "TritonKernel: loads, compute, stores",
		col: 0,
		row: 2,
		group: 2,
		note: "Holds the code being generated in <code>IndentedBuffer</code>s, the kernel's arguments (<code>KernelArgs</code>) and the CSE cache. Entering it installs <code>CSEProxy</code> as the ops handler and sets <code>V.kernel</code>. Its <code>load</code>, <code>store</code> and <code>reduction</code> methods write memory and reduction code.",
	},
	{
		id: "body",
		label: "loop body",
		sub: "SchedulerNode._body",
		col: 2,
		row: 2,
		group: 3,
		note: "The node's <code>LoopBody</code>, built from <code>inner_fn</code>. <code>node.codegen(index_vars)</code> calls it with the kernel's index variables, under <code>SimplifyIndexing</code>. It knows nothing about Triton or C++: it only calls <code>ops.*</code>.",
	},
	{
		id: "cse",
		label: "CSEProxy",
		sub: "ops handler",
		col: 1,
		row: 3,
		group: 2,
		note: "Receives every <code>ops</code> call while the kernel is active. Computes bounds, dtype and shape, deduplicates expressions through <code>kernel.cse</code>, and answers loads of buffers this kernel already stored from its store cache.",
	},
	{
		id: "overrides",
		label: "TritonKernelOverrides",
		sub: "ops → source text",
		col: 2,
		row: 3,
		group: 3,
		note: "Turns one math op into an expression string, for example <code>exp(tmp7)</code> into <code>libdevice.exp(tmp7)</code>. The C++ backend has <code>CppOverrides</code> and <code>CppVecOverrides</code> in its place.",
	},
];

export const codegenEdges: BoxEdge[] = [
	{ from: "registry", to: "backend", label: "constructs", dashed: true },
	{ from: "scheduler", to: "backend", label: "codegen_node" },
	{ from: "scheduler", to: "wrapper", label: "alloc, free" },
	{ from: "backend", to: "wrapper", label: "call_kernel" },
	{ from: "backend", to: "kernel", label: "with kernel:" },
	{ from: "backend", to: "body", label: "node.codegen" },
	{ from: "body", to: "cse", label: "ops.*" },
	{ from: "cse", to: "overrides", label: "math ops" },
	{ from: "cse", to: "kernel", label: "load, store" },
];
