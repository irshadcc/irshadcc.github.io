// Figure data for the JAX post (src/content/posts/jax-internals.mdx): the pipeline diagram, the
// four step-through figures (tracing, JVP, linearize, transpose) and the compiler-stage listings.
// Every listing and number comes from jaxData.ts, which a scratch script generated on jax 0.11.2
// and checked against jax.jvp, jax.linearize, jax.vjp and jax.grad. Notes are written by hand.
import type { BoxEdge, BoxNode } from "../payments/boxDiagram";
import {
	gradJaxpr,
	hloGrad,
	hloLoss,
	jvpJaxpr,
	linJaxpr,
	lossJaxpr,
	modelSource,
	stablehloGrad,
	stablehloLoss,
	traceLog,
	values as v,
	vjpJaxpr,
} from "./jaxData";
import type { JaxprRow, JaxprStepsData } from "./jaxprSteps";
import { upTo } from "./jaxprSteps";
import type { CodeStage, StageTag } from "./llvmData";

const COMMIT = "32544801e26115ac1794926d027148abf3baf009";
export const jaxSrc = (file: string, line: number) =>
	`https://github.com/jax-ml/jax/blob/${COMMIT}/jax/_src/${file}#L${line}`;
const link = (file: string, line: number, text: string) =>
	`<a href="${jaxSrc(file, line)}"><code>${text}</code></a>`;

// ---------------------------------------------------------------------------------------------
// The pipeline: from a Python function to an XLA executable.

export const jaxPipelineNodes: BoxNode[] = [
	{
		id: "py",
		label: "Python function",
		sub: "loss(w, x, y)",
		col: 0,
		row: 0,
		group: 5,
		note: "Ordinary Python that calls <code>jax.numpy</code> and <code>jax.lax</code>. JAX never reads its source or bytecode; it only runs it, with tracers in place of arrays.",
	},
	{
		id: "trace",
		label: "trace",
		sub: "DynamicJaxprTrace",
		col: 1,
		row: 0,
		group: 0,
		note: `${link("interpreters/partial_eval.py", 2048, "trace_to_jaxpr_nocache")} creates a ${link("interpreters/partial_eval.py", 1623, "DynamicJaxprTrace")}, makes one tracer per argument with ${link("interpreters/partial_eval.py", 1668, "new_arg")} and calls the function. Every primitive the function binds becomes one equation.`,
	},
	{
		id: "jaxpr",
		label: "jaxpr",
		sub: "typed, functional IR",
		col: 2,
		row: 0,
		group: 1,
		note: `A ${link("core.py", 100, "Jaxpr")}: input variables, a flat list of equations (primitive, inputs, outputs, parameters) and outputs. Every variable has a shape and dtype; there is no control flow except through primitives like <code>cond</code> and <code>scan</code>.`,
	},
	{
		id: "transform",
		label: "transformations",
		sub: "jvp, grad, vmap",
		col: 2,
		row: 1,
		group: 2,
		note: `<code>grad</code>, <code>jvp</code> and <code>vmap</code> are traces too (${link("interpreters/ad.py", 520, "JVPTrace")}, ${link("interpreters/ad.py", 670, "LinearizeTrace")}, BatchTrace). They run on the way into the jaxpr: their tracers intercept each primitive and bind new primitives on the trace below, so the jaxpr that comes out already contains the derivative.`,
	},
	{
		id: "lower",
		label: "lowering",
		sub: "jaxpr_subcomp",
		col: 3,
		row: 0,
		group: 3,
		note: `${link("interpreters/mlir.py", 1374, "lower_jaxpr_to_module")} builds an MLIR module; ${link("interpreters/mlir.py", 2191, "jaxpr_subcomp")} walks the equations and calls each primitive's lowering rule, registered with ${link("interpreters/mlir.py", 1050, "register_lowering")}. The result is StableHLO.`,
	},
	{
		id: "xla",
		label: "XLA compile",
		sub: "PJRT client",
		col: 4,
		row: 0,
		group: 4,
		note: `${link("compiler.py", 429, "compile_or_get_cached")} checks the persistent compilation cache, then ${link("compiler.py", 335, "backend_compile_and_load")} hands the module to the backend (CPU, GPU or TPU) through PJRT. XLA optimizes it (fusion, layout, scheduling) and returns a loaded executable.`,
	},
	{
		id: "run",
		label: "executable",
		sub: "cached by signature",
		col: 4,
		row: 1,
		group: 5,
		note: `The executable is stored in the jitted function's C++ cache, keyed by the arguments' shapes, dtypes, shardings and static arguments. The next call with the same signature skips tracing, lowering and compiling and goes straight here.`,
	},
];

export const jaxPipelineEdges: BoxEdge[] = [
	{ from: "py", to: "trace" },
	{ from: "trace", to: "jaxpr" },
	{ from: "transform", to: "trace", dashed: true },
	{ from: "jaxpr", to: "lower" },
	{ from: "lower", to: "xla" },
	{ from: "xla", to: "run" },
];

// ---------------------------------------------------------------------------------------------
// Tracing loss into a jaxpr, one primitive at a time.

const emitted = traceLog.filter((l) => l.added > 0);
const tracerRows: [string, string, string][] = [
	["w", "a", "f32[2]"],
	["x", "b", "f32[3,2]"],
	["y", "c", "f32[3]"],
	["x @ w", "d", "f32[3]"],
	["jnp.tanh(x @ w)", "e", "f32[3]"],
	["err = predict(w, x) - y", "f", "f32[3]"],
	["err ** 2", "g", "f32[3]"],
	["jnp.mean: sum", "h", "f32[]"],
	["jnp.mean: / 3", "i", "f32[]"],
];
const tracerTable = (n: number, fresh: number[]): JaxprRow[] =>
	tracerRows.slice(0, n).map((cells, i) => ({ cells, hi: fresh.includes(i) }));

const traceNotes = [
	`<code>make_jaxpr(loss)</code> traces through <code>jit(loss).trace</code>, which reaches ${link("interpreters/partial_eval.py", 2048, "trace_to_jaxpr_nocache")}. It creates a <code>DynamicJaxprTrace</code> and calls ${link("interpreters/partial_eval.py", 1668, "new_arg")} once per input: each call makes a fresh variable (<code>a</code>, <code>b</code>, <code>c</code>) and a tracer that holds it and its abstract value (shape and dtype, no data). The trace is made current with <code>set_current_trace</code> and <code>loss</code> is called with the three tracers.`,
	`<code>x @ w</code> calls <code>Tracer.__matmul__</code>, which JAX forwards to the abstract value (${link("numpy/array_methods.py", 1530, "_forward_operator_to_aval")}), then <code>jnp.matmul</code>, <code>lax.dot_general</code> and finally ${link("lax/lax.py", 2652, "dot_general_p.bind(b, a)")}. ${link("core.py", 686, "Primitive.bind")} reads the current trace and calls its ${link("interpreters/partial_eval.py", 1736, "process_primitive")}. That runs only the <em>shape rule</em>: f32[3,2] times f32[2] gives f32[3]. It makes variable <code>d</code>, appends the equation and returns a new tracer. Nothing is multiplied.`,
	"<code>jnp.tanh</code> is itself a <code>jit</code> with <code>inline=JAX_EARLY</code>, so its own small jaxpr is traced and inlined here, leaving one <code>tanh</code> equation. Back in <code>loss</code>, the returned tracer <code>e</code> is what Python sees as <code>predict(w, x)</code>.",
	"<code>predict(w, x) - y</code> goes through <code>Tracer.__sub__</code> and <code>jnp.subtract</code> to <code>sub_p</code>. Both operands are tracers of this trace, so the equation reads <code>sub e c</code>. Python binds the result to the name <code>err</code>; the jaxpr never sees that name.",
	"<code>err ** 2</code> becomes <code>integer_pow[y=2]</code>. The exponent is a Python int, so it is a <em>parameter</em> of the equation (inside the brackets), not an input: the jaxpr is specialised to squaring.",
	"<code>jnp.mean</code> first converts the count 3 to float32. That <code>convert_element_type</code> has no tracer inputs, so it is folded while tracing and leaves no equation. Then it sums over axis 0 with <code>reduce_sum</code>, giving a scalar f32[].",
	"It divides by the count. The constant 3.0 is small, so it is written inline as a <em>literal</em> (<code>3.0:f32[]</code>) rather than becoming an input variable.",
	`<code>loss</code> returns tracer <code>i</code>. ${link("interpreters/partial_eval.py", 1517, "JaxprStackFrame.to_jaxpr")} collects the recorded equations, sets the outputs to <code>(i,)</code> and the trace is discarded. Six equations, all typed; no Python left.`,
];

export const traceSteps: JaxprStepsData = {
	top: { title: "model.py (being run with tracers)", lines: modelSource },
	bottom: {
		title: "jaxpr being recorded by DynamicJaxprTrace",
		lines: lossJaxpr,
	},
	columns: ["Python value", "tracer's variable", "abstract value"],
	steps: [
		{
			label: "inputs",
			top: [5],
			shown: [0],
			bottom: [0],
			rows: tracerTable(3, [0, 1, 2]),
			note: traceNotes[0],
		},
		...emitted.map((l, k) => ({
			label: ["x @ w", "tanh", "- y", "** 2", "sum", "/ 3"][k],
			top: l.src,
			shown: upTo(k + 2),
			bottom: [k + 1],
			rows: tracerTable(k + 4, [k + 3]),
			note: traceNotes[k + 1],
		})),
		{
			label: "return",
			top: [7],
			shown: upTo(lossJaxpr.length),
			bottom: [lossJaxpr.length - 1],
			rows: tracerTable(tracerRows.length, []),
			note: traceNotes[7],
		},
	],
};

// ---------------------------------------------------------------------------------------------
// Forward mode: jax.jvp(loss, (w,), (t,)) with t = [1, 0], primitive by primitive.

const jvpRows: [string, string, string, string][] = [
	["a (w)", v.w, v.t, "c, d"],
	["d", v.d, v.dDot, "e, f"],
	["e", v.e, v.eDot, "g, i"],
	["f", v.f, v.fDot, "j, i"],
	["g", v.g, v.gDot, "k, m"],
	["h", v.h, v.hDot, "n, o"],
	["i", v.i, v.iDot, "p, q"],
];
const jvpTable = (n: number): JaxprRow[] =>
	jvpRows.slice(0, n).map((cells, i) => ({ cells, hi: i === n - 1 }));
const jvpGroups = [[1, 2], [3, 4, 5], [6], [7, 8, 9], [10, 11], [12, 13]];
const jvpNotes = [
	`<code>jax.jvp</code> (${link("interpreters/ad.py", 50, "ad.jvp")}) creates a <code>JVPTrace</code> and wraps <code>w</code> in a <code>JVPTracer</code> holding the primal <code>w</code> and the tangent <code>t = [1, 0]</code>. <code>x</code> and <code>y</code> are closed over, so they stay plain arrays: their tangent is a <em>symbolic zero</em>, which costs nothing. In the JVP jaxpr below they become constants <code>a</code> and <code>b</code>; <code>w</code> and <code>t</code> are inputs <code>c</code> and <code>d</code>.`,
	`${link("interpreters/ad.py", 543, "JVPTrace.process_primitive")} splits each input into (primal, tangent) and calls dot_general's JVP rule on the trace <em>below</em> it. dot_general is bilinear, so the rule (${link("interpreters/ad.py", 996, "defbilinear")}) binds dot_general twice: once on the primals, <code>x·w = ${v.d}</code>, and once with the tangent in place of <code>w</code>, <code>x·t = ${v.dDot}</code>. The tangent for <code>x</code> is zero, so its term is skipped.`,
	`tanh's rule (${link("lax/lax.py", 4666, "defjvp2(tanh_p, …)")}) reuses the output: <code>ė = ḋ · (1 − e²)</code>. It binds <code>tanh</code>, then <code>one_minus_square</code> on the result, then <code>mul</code>. Check one entry: <code>1 · (1 − 0.2913²) = 0.9151</code>.`,
	"<code>sub</code> with a zero tangent on <code>y</code>: the rule (<code>_sub_jvp</code>) returns the primal difference and passes <code>ė</code> through unchanged, so the tangent costs no equation.",
	`integer_pow's rule for <code>y=2</code> is <code>ġ = ḟ · (2f)</code>: it binds <code>integer_pow</code> for the primal, <code>mul 2.0 j</code> for <code>2f</code> and <code>mul</code> for the product.`,
	`<code>reduce_sum</code> is linear, so its JVP is the same primitive applied to the tangent: <code>ḣ = Σ ġ = ${v.hDot}</code>.`,
	`<code>div</code> by the constant 3 is linear in its first input: the tangent of <code>i</code> is <code>ḣ / 3 = ${v.iDot}</code>. This is the directional derivative of the loss along <code>t = [1, 0]</code>, i.e. ∂loss/∂w₀. One forward pass gives one column of the Jacobian.`,
];

export const jvpSteps: JaxprStepsData = {
	top: { title: "jaxpr of loss (being evaluated)", lines: lossJaxpr },
	bottom: {
		title: "jaxpr of jax.jvp(loss) (what JVPTrace emits)",
		lines: jvpJaxpr,
	},
	columns: ["loss variable", "primal", "tangent", "JVP jaxpr variables"],
	steps: [
		{
			label: "inputs",
			top: [0],
			shown: [0],
			bottom: [0],
			rows: jvpTable(1),
			note: jvpNotes[0],
		},
		...jvpGroups.map((g, k) => ({
			label: ["dot", "tanh", "sub", "pow", "sum", "div"][k],
			top: [k + 1],
			shown: [
				0,
				...jvpGroups.slice(0, k + 1).flat(),
				...(k === 5 ? [jvpJaxpr.length - 1] : []),
			],
			bottom: g,
			rows: jvpTable(k + 2),
			note: jvpNotes[k + 1],
		})),
	],
};

// ---------------------------------------------------------------------------------------------
// Linearize: primal values are computed now, tangent equations are recorded for later.

type LinRow = [string, string, string];
const linRows: LinRow[][] = [
	[],
	[
		["d", "primal, computed now", v.d],
		["a", "residual: x", v.resA],
	],
	[
		["e", "primal, computed now", v.e],
		["b", "residual: 1 − e²", v.resB],
	],
	[["f", "primal, computed now", v.f]],
	[
		["g", "primal, computed now", v.g],
		["c", "residual: 2f", v.resC],
	],
	[["h", "primal, computed now", v.h]],
	[["i", "primal, computed now", v.i]],
];
const linTable = (k: number): JaxprRow[] =>
	linRows
		.slice(0, k + 1)
		.flatMap((rows, j) => rows.map((cells) => ({ cells, hi: j === k })));
const linEmit = [[], [1], [2], [], [3], [4], [5, linJaxpr.length - 1]];
const linNotes = [
	`<code>jax.vjp</code> and <code>jax.grad</code> start with ${link("interpreters/ad.py", 226, "ad.linearize")}. It wraps <code>w</code> in a <code>LinearizeTracer</code> whose primal is the real array and whose tangent is a tracer of a separate <code>DynamicJaxprTrace</code>, the <em>tangent trace</em>. Input <code>d</code> of the linear jaxpr is that tangent.`,
	`${link("interpreters/ad.py", 713, "LinearizeTrace.process_primitive")} has no special rule for dot_general, so it falls back to the JVP rule (${link("interpreters/ad.py", 821, "linearize_from_jvp")}) run under partial evaluation: inputs with known values are computed now, anything that touches the tangent is recorded. The primal <code>x·w</code> is computed; <code>x·ḋ</code> becomes equation <code>e</code>, and <code>x</code> is captured as residual <code>a</code>.`,
	`tanh's JVP needs <code>1 − e²</code>. That depends only on primals, so it is <em>computed now</em> (${v.resB}) and saved as residual <code>b</code>. Only the multiplication by the tangent is recorded: <code>f = mul e b</code>.`,
	"<code>sub</code> with a zero tangent on <code>y</code> passes the tangent through, so no equation is recorded. The primal <code>err</code> is computed.",
	"For <code>err ** 2</code> the factor <code>2f</code> is known now and saved as residual <code>c</code>; the tangent product <code>g = mul f c</code> is recorded.",
	"<code>reduce_sum</code> is linear: the same primitive is recorded on the tangent.",
	`The division is recorded too, and the primal loss ${v.i} is returned. The linear jaxpr contains only linear operations on <code>d</code> and three residuals. Calling it with <code>t = [1, 0]</code> gives ${v.iDot}, the same as the JVP.`,
];

export const linSteps: JaxprStepsData = {
	top: { title: "jaxpr of loss (being evaluated)", lines: lossJaxpr },
	bottom: {
		title:
			"linear jaxpr recorded on the tangent trace (residuals a, b, c; tangent input d)",
		lines: linJaxpr,
	},
	columns: ["name", "what it is", "value"],
	steps: linEmit.map((b, k) => ({
		label: ["inputs", "dot", "tanh", "sub", "pow", "sum", "div"][k],
		top: k === 0 ? [0] : [k],
		shown: [0, ...linEmit.slice(0, k + 1).flat()],
		bottom: b,
		rows: linTable(k),
		note: linNotes[k],
	})),
};

// ---------------------------------------------------------------------------------------------
// Transpose: walk the linear jaxpr backwards with cotangents.

const bwdRows: [string, string, string, string][] = [
	["i", "ct_i = seed", "d", v.ctI],
	["h", "ct_h = ct_i / 3", "e", v.ctH],
	["g", "ct_g = broadcast ct_h", "f", v.ctG],
	["f", "ct_f = ct_g · c", "g", v.ctF],
	["e", "ct_e = ct_f · b", "h", v.ctE],
	["d", "ct_d = ct_eᵀ x  (= ∇w)", "i", v.ctW],
];
const bwdTable = (n: number): JaxprRow[] =>
	bwdRows.slice(0, n).map((cells, i) => ({ cells, hi: i === n - 1 }));
const bwdNotes = [
	`<code>grad</code> calls the VJP function with the seed cotangent 1.0 for the scalar output (${link("lax/lax.py", 9790, "_one_vjp")}). ${link("interpreters/ad.py", 299, "backward_pass3")} gives every variable of the linear jaxpr an accumulator and walks the equations in reverse. Input <code>d</code> of the VJP jaxpr is the seed.`,
	`Last equation first: <code>i = div h 3.0</code>. div's transpose rule (<code>_div_transpose_rule</code>) divides the cotangent by the constant: <code>ct_h = 1 / 3</code>.`,
	`<code>reduce_sum</code> summed three numbers into one, so its transpose (${link("lax/lax.py", 8610, "_reduce_sum_transpose_rule")}) copies the cotangent back to all three: <code>broadcast_in_dim</code>.`,
	`<code>g = mul f c</code> is linear in <code>f</code>; <code>c = 2·err</code> is a residual. mul's transpose (${link("lax/lax.py", 5215, "_mul_transpose_left")}) multiplies the cotangent by the other operand. In the VJP jaxpr the same residual is called <code>a</code>.`,
	"<code>f = mul e b</code> with residual <code>b = 1 − e²</code>: again multiply by the residual.",
	`<code>e = dot_general a d</code> computed <code>x·ḋ</code>. Its transpose (${link("lax/lax.py", 5994, "_dot_general_transpose_lhs")}, via the rhs variant) contracts the cotangent with <code>x</code> over the other axis: <code>ct_d = ct_eᵀ x</code>. Check one entry: −0.1777·1 − 0.7666·3 − 0.2557·5 = −3.7562, which matches <code>jax.grad</code> and the JVP along [1, 0].`,
];

export const bwdSteps: JaxprStepsData = {
	top: { title: "linear jaxpr (walked bottom-up)", lines: linJaxpr },
	bottom: {
		title: "transposed (VJP) jaxpr, residuals a = 2f, b = 1 − e², c = x",
		lines: vjpJaxpr,
	},
	columns: ["linear variable", "cotangent", "VJP jaxpr variable", "value"],
	steps: bwdRows.map((_, k) => ({
		label: ["seed", "div", "sum", "mul c", "mul b", "dot"][k],
		top: k === 0 ? [linJaxpr.length - 1] : [linJaxpr.length - 1 - k],
		shown: k === 5 ? upTo(vjpJaxpr.length) : upTo(k + 1),
		bottom: [k],
		rows: bwdTable(k + 1),
		note: bwdNotes[k],
	})),
};

// ---------------------------------------------------------------------------------------------
// The same loss at each compiler stage, and the gradient.

/** Tag every line of `code` that matches one of the rules (first match wins). */
const markLines = (
	code: string,
	rules: [RegExp, string][],
): [number, string][] =>
	code.split("\n").flatMap((line, i): [number, string][] => {
		const hit = rules.find(([re]) => re.test(line));
		return hit ? [[i, hit[1]]] : [];
	});

export const lossTags: StageTag[] = [
	{ tag: "dot", label: "x @ w" },
	{ tag: "tanh", label: "tanh" },
	{ tag: "sub", label: "− y" },
	{ tag: "sq", label: "** 2" },
	{ tag: "sum", label: "sum" },
	{ tag: "div", label: "/ 3" },
];
const lossRules: [RegExp, string][] = [
	[/dot_general|= f32\[3,1\]\{1,0\} dot\(/, "dot"],
	[/tanh/, "tanh"],
	[/= sub |subtract/, "sub"],
	[/integer_pow|stablehlo\.multiply|%integer_pow/, "sq"],
	[/reduce_sum|stablehlo\.reduce|reduce\(/, "sum"],
	[/= div |divide|%multiply\.1/, "div"],
];
// Compiled HLO renames and fuses ops, so match on the name each instruction defines.
const lossHloRules: [RegExp, string][] = [
	[/%dot = /, "dot"],
	[/%tanh\.0 = /, "tanh"],
	[/%sub\.0 = /, "sub"],
	[/%integer_pow\.0 = /, "sq"],
	[/%reduce_sum\.\d+ = |%region_0/, "sum"],
	[/%multiply\.1 = |%constant\.1 = /, "div"],
];
const lossJaxprText = lossJaxpr.join("\n");

export const lossStages: CodeStage[] = [
	{
		id: "jaxpr",
		label: "jaxpr",
		tool: "jax.make_jaxpr(loss)(w, x, y)",
		lang: "plaintext",
		code: lossJaxprText,
		marks: markLines(lossJaxprText, lossRules),
		note: "The traced program: one equation per primitive, every variable typed.",
	},
	{
		id: "stablehlo",
		label: "StableHLO",
		tool: "jax.jit(loss).lower(w, x, y).as_text()",
		lang: "mlir",
		code: stablehloLoss,
		marks: markLines(stablehloLoss, lossRules),
		note: "One StableHLO op per equation, from each primitive's lowering rule. integer_pow[y=2] became a multiply; jnp.mean's sum became a reduce with an add body.",
	},
	{
		id: "hlo",
		label: "optimized HLO",
		tool: "jax.jit(loss).lower(w, x, y).compile().as_text()",
		lang: "plaintext",
		code: hloLoss,
		marks: markLines(hloLoss, lossHloRules),
		note: "After XLA's CPU pipeline: the dot stays a library call, and tanh, subtract, square, sum and the division (now a multiply by 0.333) are fused into one loop, %fused_computation.",
	},
];

export const gradTags: StageTag[] = [
	{ tag: "fwd", label: "forward: x·w, tanh, err" },
	{ tag: "dtanh", label: "1 − e²" },
	{ tag: "scale", label: "2·err / 3" },
	{ tag: "chain", label: "products" },
	{ tag: "bwd", label: "xᵀ (dot transpose)" },
];
const gradRules: [RegExp, string][] = [
	[/contracting_dims = \[0\]|\(\(\[0\], \[0\]\)|dot\.1 = /, "bwd"],
	[
		/dot_general|tanh [a-z%]|stablehlo\.tanh|tanh\(|= sub e c|integer_pow|reduce_sum|= div j|subtract %1, %arg2|subtract\(%tanh\.0, %param_0|%dot = /,
		"fwd",
	],
	[
		/one_minus_square|stablehlo\.add|%3 = |%5 = |%6 = |add\(%tanh|subtract\(%broadcast\.1|%mul\.1 = |%cst = |%cst_0 =|%2 = |%4 = |%broadcast\.1 = |%constant\.1 = /,
		"dtanh",
	],
	[
		/mul 2\.0|div 1\.0|broadcast_in_dim k|broadcast_in_dim l|%8 = |%9 = |%10 = |%11 = |%cst_[123] =|0\.666|%broadcast\.2 = |%broadcast_in_dim\.0 = /,
		"scale",
	],
	[/= mul [lmn] |%12 = |%13 = |%mul\.0 = /, "chain"],
];
const gradJaxprText = gradJaxpr.join("\n");

export const gradStages: CodeStage[] = [
	{
		id: "jaxpr",
		label: "jaxpr",
		tool: "jax.make_jaxpr(jax.grad(loss))(w, x, y)",
		lang: "plaintext",
		code: gradJaxprText,
		marks: markLines(gradJaxprText, gradRules),
		note: "grad's jaxpr: the forward pass (with 1 − e² and 2·err computed as residuals), the unused loss value bound to _, then the transposed linear program.",
	},
	{
		id: "stablehlo",
		label: "StableHLO",
		tool: "jax.jit(jax.grad(loss)).lower(w, x, y).as_text()",
		lang: "mlir",
		code: stablehloGrad,
		marks: markLines(stablehloGrad, gradRules),
		note: "The unused loss value and its reduce are gone (jit removes dead equations). one_minus_square lowers to (1 + e)(1 − e).",
	},
	{
		id: "hlo",
		label: "optimized HLO",
		tool: "jax.jit(jax.grad(loss)).lower(w, x, y).compile().as_text()",
		lang: "plaintext",
		code: hloGrad,
		marks: markLines(hloGrad, gradRules),
		note: "XLA folded 2 · (1/3) into one constant 0.6667 and fused everything between the two dots into one loop. The whole gradient is two dots and one fused elementwise kernel.",
	},
];
