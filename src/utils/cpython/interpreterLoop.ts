// Data and logic for the CPython interpreter loop architecture diagram
// Used by InterpreterLoopDiagram.astro

export interface LoopStage {
	id: string;
	title: string;
	shortName: string;
	macro: string;
	cFunction: string;
	hardwareDetail: string;
	description: string;
	activeNodes: string[];
}

export const loopStages: LoopStage[] = [
	{
		id: "frame_init",
		title: "1. Frame Initialization & Registers",
		shortName: "Frame Init",
		macro: "SET_LOCALS_FROM_FRAME()",
		cFunction: "_PyEval_EvalFrameDefault()",
		hardwareDetail:
			"Pointers pinned to CPU registers (next_instr, stack_pointer)",
		description:
			"When execution enters _PyEval_EvalFrameDefault, it establishes the CFrame and extracts two high-frequency pointers from _PyInterpreterFrame into CPU registers: next_instr (instruction pointer) and stack_pointer (top of evaluation stack). Frames are allocated contiguously on the per-thread data stack.",
		activeNodes: ["tstate", "frame", "registers"],
	},
	{
		id: "fetch_decode",
		title: "2. Fetch & Decode Code Units",
		shortName: "Fetch & Decode",
		macro: "NEXTOPARG()",
		cFunction: "next_instr++",
		hardwareDetail:
			"16-bit word fetch with 8-bit split; sequential memory read in L1I/L1D cache",
		description:
			"Every instruction in Python 3.11+ is a 16-bit code unit (_Py_CODEUNIT). NEXTOPARG() reads the next 16 bits, extracting the 8-bit opcode into a register and the 8-bit oparg. If an argument exceeds 255, an EXTENDED_ARG prefix left-shifts the argument by 8 bits.",
		activeNodes: ["registers", "fetch", "decode"],
	},
	{
		id: "threaded_dispatch",
		title: "3. Direct Threaded Code Dispatch",
		shortName: "Computed GOTO",
		macro: "DISPATCH() / goto *opcode_targets[opcode]",
		cFunction: "static void * const opcode_targets[256]",
		hardwareDetail:
			"Direct indirect branch per opcode; CPU Branch Target Buffer (BTB) predicts pairs",
		description:
			"Instead of a centralized switch statement where all opcodes share one indirect jump (causing frequent CPU branch mispredictions), CPython uses GCC computed gotos. Each opcode handler ends with an independent indirect jump to *opcode_targets[opcode]. The CPU BTB learns opcode transition pairs, boosting execution speed by 15-20%.",
		activeNodes: ["dispatch", "targets"],
	},
	{
		id: "stack_exec",
		title: "4. Opcode Execution & Stack Effects",
		shortName: "Opcode Exec",
		macro: "STACK_GROW / STACK_SHRINK / GETLOCAL",
		cFunction: "generated_cases.c.h",
		hardwareDetail:
			"Operands hot in L1 cache inside contiguous frame->localsplus array",
		description:
			"The execution jumps to TARGET(op). Inputs are retrieved directly from stack_pointer[-1] (TOS) or localsplus[oparg]. The generated code performs the operation, adjusts stack_pointer via STACK_GROW or STACK_SHRINK, stores the result, and immediately dispatches the next instruction.",
		activeNodes: ["targets", "stack", "locals"],
	},
	{
		id: "adaptive_spec",
		title: "5. Adaptive Specialization (PEP 659)",
		shortName: "Specialization",
		macro: "DEOPT_IF(type_mismatch, GENERIC_OP)",
		cFunction: "_Py_Specialize_BinaryOp()",
		hardwareDetail:
			"Inline cache words follow bytecode unit; zero-overhead type checks",
		description:
			"Generic opcodes (like BINARY_OP or LOAD_ATTR) count executions in their trailing inline CACHE entries. After a warmup threshold (typically 8 executions), the opcode is dynamically overwritten in-place with a monomorphic fast path (e.g. BINARY_OP_ADD_INT). If types subsequently change, DEOPT_IF reverts to the generic handler.",
		activeNodes: ["targets", "cache", "specialize"],
	},
	{
		id: "frame_inline",
		title: "6. Python-to-Python Frame Inlining",
		shortName: "Frame Inlining",
		macro: "DISPATCH_INLINED(NEW_FRAME)",
		cFunction: "_PyEvalFramePushAndInit()",
		hardwareDetail:
			"Zero C-level call overhead; prevents C stack overflow on deep Python recursion",
		description:
			"When a Python function calls another Python function via CALL, CPython avoids invoking _PyEval_EvalFrameDefault recursively. Instead, DISPATCH_INLINED links NEW_FRAME->previous = frame, re-points frame to the new frame, and jumps straight back to start_frame: within the same C function execution.",
		activeNodes: ["frame", "dispatch", "inline_call"],
	},
	{
		id: "eval_breaker",
		title: "7. Eval Breaker & Asynchronous Events",
		shortName: "Eval Breaker",
		macro: "CHECK_EVAL_BREAKER()",
		cFunction: "_Py_HandlePending(tstate)",
		hardwareDetail:
			"Single atomic relaxed int load checked only at loop back-edges and returns",
		description:
			"Out-of-band events (GIL release, OS signal dispatching, garbage collection, async exceptions) are consolidated into a single atomic integer eval_breaker. Rather than checking on every cycle, CPython checks eval_breaker only on loop back edges (JUMP_BACKWARD) and call returns, avoiding pipeline stalls during straight-line execution.",
		activeNodes: ["tstate", "breaker", "signals"],
	},
];

export function validateLoopStages(stages: LoopStage[]): string | null {
	if (!stages.length) return "Stages cannot be empty";
	for (const s of stages) {
		if (!s.id || !s.title || !s.macro) return `Stage ${s.id} is missing fields`;
	}
	return null;
}
