// Step-through data and logic for bytecode execution of a Python function
// compute(a=10, b=20) -> a + b * 2
// Used by BytecodeStepper.astro

export interface BytecodeStep {
	stepIndex: number;
	offset: number;
	label: string;
	opcode: number;
	opname: string;
	oparg: number;
	argval?: string;
	stack: string[];
	locals: { name: string; val: string; slot: number }[];
	macros: string[];
	cCode: string;
	explanation: string;
}

export const bytecodeSteps: BytecodeStep[] = [
	{
		stepIndex: 0,
		offset: 0,
		label: "0: RESUME",
		opcode: 151,
		opname: "RESUME",
		oparg: 0,
		stack: [],
		locals: [
			{ name: "a", val: "10", slot: 0 },
			{ name: "b", val: "20", slot: 1 },
		],
		macros: ["TARGET(RESUME)", "CHECK_EVAL_BREAKER()", "DISPATCH()"],
		cCode: `TARGET(RESUME) {
    if (frame->f_code->_co_instrumentation_version != tstate->interp->monitoring_version) {
        _Py_Instrument(frame->f_code, tstate->interp);
    }
    else if (_Py_atomic_load_relaxed_int32(&tstate->interp->ceval.eval_breaker) && oparg < 2) {
        goto handle_eval_breaker;
    }
    DISPATCH();
}`,
		explanation:
			"The frame begins execution. RESUME checks the monitoring instrumentation version (PEP 669) and tests eval_breaker for pending OS signals or GIL drop requests before dispatching the first real instruction.",
	},
	{
		stepIndex: 1,
		offset: 2,
		label: "2: LOAD_FAST a",
		opcode: 124,
		opname: "LOAD_FAST",
		oparg: 0,
		argval: "a",
		stack: ["10"],
		locals: [
			{ name: "a", val: "10", slot: 0 },
			{ name: "b", val: "20", slot: 1 },
		],
		macros: ["GETLOCAL(0)", "Py_INCREF", "STACK_GROW(1)", "DISPATCH()"],
		cCode: `TARGET(LOAD_FAST) {
    PyObject *value = GETLOCAL(oparg); /* frame->localsplus[0] */
    assert(value != NULL);
    Py_INCREF(value);
    STACK_GROW(1);
    stack_pointer[-1] = value;
    DISPATCH();
}`,
		explanation:
			"Fast local variable access index 0 directly offsets into the frame's contiguous localsplus array without dictionary hashing. Reference count increments, stack grows by 1, and value 10 becomes the Top of Stack (TOS).",
	},
	{
		stepIndex: 2,
		offset: 4,
		label: "4: LOAD_FAST b",
		opcode: 124,
		opname: "LOAD_FAST",
		oparg: 1,
		argval: "b",
		stack: ["10", "20"],
		locals: [
			{ name: "a", val: "10", slot: 0 },
			{ name: "b", val: "20", slot: 1 },
		],
		macros: ["GETLOCAL(1)", "Py_INCREF", "STACK_GROW(1)", "DISPATCH()"],
		cCode: `TARGET(LOAD_FAST) {
    PyObject *value = GETLOCAL(oparg); /* frame->localsplus[1] */
    assert(value != NULL);
    Py_INCREF(value);
    STACK_GROW(1);
    stack_pointer[-1] = value;
    DISPATCH();
}`,
		explanation:
			"Fast local variable b (slot 1) is read from localsplus[1]. The evaluation stack grows again, pushing 20 as the new Top of Stack (TOS), with 10 immediately underneath it.",
	},
	{
		stepIndex: 3,
		offset: 6,
		label: "6: LOAD_CONST 2",
		opcode: 100,
		opname: "LOAD_CONST",
		oparg: 1,
		argval: "2",
		stack: ["10", "20", "2"],
		locals: [
			{ name: "a", val: "10", slot: 0 },
			{ name: "b", val: "20", slot: 1 },
		],
		macros: ["GETITEM(co_consts, 1)", "STACK_GROW(1)", "DISPATCH()"],
		cCode: `TARGET(LOAD_CONST) {
    PyObject *value = GETITEM(frame->f_code->co_consts, oparg);
    Py_INCREF(value);
    STACK_GROW(1);
    stack_pointer[-1] = value;
    DISPATCH();
}`,
		explanation:
			"Fetches literal constant 2 from index 1 of the code object's co_consts tuple. Small integers and constants are immortal in modern CPython (3.12+), saving refcount modification overhead. Stack depth is now 3.",
	},
	{
		stepIndex: 4,
		offset: 8,
		label: "8: BINARY_OP *",
		opcode: 122,
		opname: "BINARY_OP_MULTIPLY_INT",
		oparg: 5,
		argval: "nb_multiply",
		stack: ["10", "40"],
		locals: [
			{ name: "a", val: "10", slot: 0 },
			{ name: "b", val: "20", slot: 1 },
		],
		macros: [
			"DEOPT_IF()",
			"_PyLong_Multiply()",
			"STACK_SHRINK(1)",
			"DISPATCH()",
		],
		cCode: `TARGET(BINARY_OP_MULTIPLY_INT) {
    PyObject *right = stack_pointer[-1]; /* 2 */
    PyObject *left = stack_pointer[-2];  /* 20 */
    DEOPT_IF(!PyLong_CheckExact(left), BINARY_OP);
    DEOPT_IF(!PyLong_CheckExact(right), BINARY_OP);
    PyObject *prod = _PyLong_Multiply((PyLongObject *)left, (PyLongObject *)right);
    _Py_DECREF_SPECIALIZED(right, (destructor)PyObject_Free);
    _Py_DECREF_SPECIALIZED(left, (destructor)PyObject_Free);
    STACK_SHRINK(1);
    stack_pointer[-1] = prod; /* 40 */
    next_instr += 1; /* skip inline CACHE entry */
    DISPATCH();
}`,
		explanation:
			"Specialized multiplication executes directly without generic slot lookup. It verifies both operands are exact PyLong instances, multiplies them directly via _PyLong_Multiply, drops operand references, shrinks stack depth from 3 to 2, writes 40, and skips the 2-byte inline cache entry.",
	},
	{
		stepIndex: 5,
		offset: 12,
		label: "12: BINARY_OP +",
		opcode: 122,
		opname: "BINARY_OP_ADD_INT",
		oparg: 0,
		argval: "nb_add",
		stack: ["50"],
		locals: [
			{ name: "a", val: "10", slot: 0 },
			{ name: "b", val: "20", slot: 1 },
		],
		macros: ["DEOPT_IF()", "_PyLong_Add()", "STACK_SHRINK(1)", "DISPATCH()"],
		cCode: `TARGET(BINARY_OP_ADD_INT) {
    PyObject *right = stack_pointer[-1]; /* 40 */
    PyObject *left = stack_pointer[-2];  /* 10 */
    DEOPT_IF(!PyLong_CheckExact(left), BINARY_OP);
    DEOPT_IF(!PyLong_CheckExact(right), BINARY_OP);
    PyObject *res = _PyLong_Add((PyLongObject *)left, (PyLongObject *)right);
    _Py_DECREF_SPECIALIZED(right, (destructor)PyObject_Free);
    _Py_DECREF_SPECIALIZED(left, (destructor)PyObject_Free);
    STACK_SHRINK(1);
    stack_pointer[-1] = res; /* 50 */
    next_instr += 1; /* skip inline CACHE entry */
    DISPATCH();
}`,
		explanation:
			"Specialized addition pops 40 and 10 from the stack. Fast C addition computes 50. Stack depth shrinks from 2 to 1, leaving the final evaluated sum 50 at top of stack.",
	},
	{
		stepIndex: 6,
		offset: 16,
		label: "16: RETURN_VALUE",
		opcode: 83,
		opname: "RETURN_VALUE",
		oparg: 0,
		stack: [],
		locals: [
			{ name: "a", val: "(cleared)", slot: 0 },
			{ name: "b", val: "(cleared)", slot: 1 },
		],
		macros: [
			"_PyEvalFrameClearAndPop()",
			"cframe.current_frame = prev",
			"DISPATCH()",
		],
		cCode: `TARGET(RETURN_VALUE) {
    PyObject *retval = stack_pointer[-1]; /* 50 */
    _PyFrame_SetStackPointer(frame, stack_pointer - 1);
    _PyEvalFrameClearAndPop(tstate, frame);
    _PyInterpreterFrame *prev = frame->previous;
    if (prev->owner == FRAME_OWNED_BY_CSTACK) {
        return retval;
    }
    frame = cframe.current_frame = prev;
    SET_LOCALS_FROM_FRAME();
    stack_pointer[0] = retval;
    stack_pointer++;
    DISPATCH();
}`,
		explanation:
			"Pops 50 as retval. Clears frame locals and pops the interpreter frame from tstate->datastack_chunk. If called by another Python frame, unlinks and pushes retval onto caller's stack directly without C stack returns.",
	},
];

export function validateSteps(steps: BytecodeStep[]): string | null {
	if (!steps.length) return "Steps list cannot be empty";
	for (let i = 0; i < steps.length; i++) {
		if (steps[i].stepIndex !== i) return `Step ${i} index mismatch`;
	}
	return null;
}
