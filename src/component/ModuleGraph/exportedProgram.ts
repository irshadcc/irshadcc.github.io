// ModuleGraph's input (moduleGraph.ts's ModuleSpec) from a real model: a torch.export program
// (the JSON graph inside a .pt2 archive, models/model.json) and, optionally, per-parameter
// statistics stitched onto it.
//
// fromExportedProgram nests the program's ATen ops by the module each ran in (the node's
// `nn_module_stack`), so `model.layers.0.self_attn` becomes a box inside `model.layers.0`. By
// default the innermost modules (Linear, RMSNorm, the activation) become single cards, and ops
// that only change a view or a dtype (view, transpose, to, ...) are skipped: their output is
// wired straight to their input's producer. Parameters are not drawn as inputs; each is listed
// in `weights` of the card that reads it, named by its fully qualified name (FQN).
//
// withParameterStats fills those weights from a { fqn: { rank, distribution } } file (shape,
// histogram, moments and the numerical rank, e.g. torch.linalg.matrix_rank's), so the hover
// cards show them. Usage in .mdx:
//   import program from "../../component/ModuleGraph/models/qwen3GuardStream4B.json";
//   import stats from "../../component/ModuleGraph/models/qwen3GuardStream4BStats.json";
//   <ModuleGraph program={program} parameterStats={stats} statsPrefix="model." wide />
// Pure: it runs at build time.
import type {
	ModuleSpec,
	OpSpec,
	TensorValue,
	WeightSpec,
} from "./moduleGraph";

/** The parts of torch.export's serialized schema read here (torch/_export/serde/schema.py). */
type Arg = {
	as_tensor?: { name: string };
	as_tensors?: { name: string }[];
	as_optional_tensors?: ({ as_tensor?: { name: string } } | null)[];
	[k: string]: unknown;
};
interface ExportNode {
	target: string;
	name?: string;
	inputs: { name: string; arg: Arg }[];
	outputs: Arg[];
	metadata?: { nn_module_stack?: string };
}
interface ExportGraph {
	inputs: Arg[];
	outputs: Arg[];
	nodes: ExportNode[];
	tensor_values: Record<
		string,
		{
			dtype?: number;
			sizes: { as_int?: number; as_expr?: { expr_str?: string } }[];
		}
	>;
}
type InputSpec = {
	parameter?: { arg: { name: string }; parameter_name: string };
	buffer?: { arg: { name: string }; buffer_name: string };
	tensor_constant?: { arg: { name: string }; tensor_constant_name: string };
	user_input?: { arg: Arg };
	[k: string]: unknown;
};
export interface ExportedProgram {
	graph_module: {
		graph: ExportGraph;
		signature: {
			input_specs: InputSpec[];
			output_specs: { user_output?: { arg: Arg }; [k: string]: unknown }[];
		};
	};
}

export interface FromExportOptions {
	/** ATen op names (the part after "aten.", without overload) wired through instead of drawn. */
	elide?: string[];
	/** Draw the ops inside the innermost modules instead of one card per module. */
	expandLeaves?: boolean;
	/** Names of the program's outputs, in order (default: the tensors' names). */
	outputNames?: string[];
}

/** Ops that only reshape, re-view, copy or cast: nothing a reader of the dataflow needs. */
export const VIEW_OPS = [
	"view",
	"_unsafe_view",
	"reshape",
	"transpose",
	"permute",
	"unsqueeze",
	"squeeze",
	"expand",
	"contiguous",
	"clone",
	"alias",
	"detach",
	"to",
	"_to_copy",
	"type_as",
	"lift_fresh_copy",
];

/** torch.export's ScalarType codes, for the dtypes a model uses. */
const DTYPES: Record<number, string> = {
	1: "uint8",
	2: "int8",
	4: "int32",
	5: "int64",
	6: "float16",
	7: "float32",
	8: "float64",
	12: "bool",
	13: "bfloat16",
};

/** "torch.ops.aten.mul.Tensor" → "aten.mul"; other targets (higher-order ops) keep their last part. */
export function opName(target: string): string {
	const m = /^torch\.ops\.(\w+)\.(\w+)(?:\.\w+)?$/.exec(target);
	return m ? `${m[1]}.${m[2]}` : target;
}

/** The module stack as (dotted path, class) pairs, outermost first; the root has path "". */
export function moduleStack(s: string | undefined): [string, string][] {
	if (!s) return [];
	return s
		.split(";")
		.filter(Boolean)
		.map((entry) => {
			const [, path = "", cls = ""] = entry.split(",");
			return [path, cls.slice(cls.lastIndexOf(".") + 1)];
		});
}

const parentPath = (p: string) =>
	p.includes(".") ? p.slice(0, p.lastIndexOf(".")) : "";
const lastSeg = (p: string) => p.slice(p.lastIndexOf(".") + 1);
const within = (p: string, ancestor: string) =>
	ancestor === "" || p === ancestor || p.startsWith(`${ancestor}.`);

/** Tensor names among an argument (a tensor, a list of tensors, a list of optional tensors). */
function tensorsOf(arg: Arg): string[] {
	if (arg.as_tensor) return [arg.as_tensor.name];
	if (arg.as_tensors) return arg.as_tensors.map((t) => t.name);
	if (arg.as_optional_tensors)
		return arg.as_optional_tensors.flatMap((t) =>
			t?.as_tensor ? [t.as_tensor.name] : [],
		);
	return [];
}

type Leaf = ModuleSpec & {
	inputs: Record<string, string>;
	outputs: Record<string, TensorValue>;
};
/** Where a tensor is made: the box it is visible in, and the key of the card (op or leaf) that makes it. */
type Producer = { scope: string; key: string; leaf: boolean } | { input: true };

/**
 * A ModuleSpec from a torch.export program: boxes for the modules, cards for the ops and the
 * innermost modules, each parameter listed under the card that reads it (WeightSpec.name is
 * its FQN; see withParameterStats). Throws if the program names a tensor nothing produces.
 */
export function fromExportedProgram(
	program: ExportedProgram,
	options: FromExportOptions = {},
): ModuleSpec {
	const { graph, signature } = program.graph_module;
	const elide = new Set(options.elide ?? VIEW_OPS);
	const value = (t: string): TensorValue => {
		const tv = graph.tensor_values[t];
		if (!tv) return { name: t, shape: [] };
		const symbolic = tv.sizes.some((s) => s.as_int === undefined);
		return {
			name: t,
			shape: tv.sizes.map((s) => s.as_int ?? Number.NaN),
			symbolic_shape: symbolic
				? `(${tv.sizes.map((s) => s.as_int ?? s.as_expr?.expr_str ?? "?").join(", ")})`
				: undefined,
			note: tv.dtype !== undefined ? DTYPES[tv.dtype] : undefined,
		};
	};

	// Graph inputs: parameters (FQN), buffers and constants (dropped), user inputs (drawn).
	const params = new Map<string, string>();
	const hidden = new Set<string>();
	const userInputs: string[] = [];
	for (const s of signature.input_specs) {
		if (s.parameter)
			params.set(s.parameter.arg.name, s.parameter.parameter_name);
		else if (s.buffer) hidden.add(s.buffer.arg.name);
		else if (s.tensor_constant) hidden.add(s.tensor_constant.arg.name);
		else if (s.user_input) userInputs.push(...tensorsOf(s.user_input.arg));
	}

	// Classes and the module tree, from every node's stack.
	const classes = new Map<string, string>([["", ""]]);
	const nodes = graph.nodes.map((n) => {
		const stack = moduleStack(n.metadata?.nn_module_stack);
		for (const [p, c] of stack) classes.set(p, c);
		return {
			n,
			op: opName(n.target),
			path: stack.length ? stack[stack.length - 1][0] : "",
			ins: n.inputs.flatMap((i) =>
				tensorsOf(i.arg).map((t, k, all) => ({
					arg: all.length > 1 ? `${i.name || "arg"}${k}` : i.name || "arg",
					t,
				})),
			),
			outs: n.outputs.flatMap(tensorsOf),
		};
	});
	// Paths missing from every stack (a ModuleList such as "layers") are boxes without a class.
	for (const p of [...classes.keys()])
		for (let q = parentPath(p); q && !classes.has(q); q = parentPath(q))
			classes.set(q, "");
	const paths = [...classes.keys()];
	const leaves = new Set(
		options.expandLeaves
			? []
			: paths.filter(
					(p) => p !== "" && !paths.some((q) => q !== p && within(q, p)),
				),
	);
	const leafOf = (p: string) => {
		for (let q = p; q; q = parentPath(q)) if (leaves.has(q)) return q;
		return undefined;
	};

	// Skipped ops: their output stands for their first tensor input.
	const alias = new Map<string, string>();
	const resolve = (t: string): string => {
		let s = t;
		while (alias.has(s)) s = alias.get(s) as string;
		return s;
	};

	const boxes = new Map<string, ModuleSpec>();
	const box = (p: string): ModuleSpec => {
		const have = boxes.get(p);
		if (have) return have;
		const spec: ModuleSpec = {
			type: "module",
			name: p === "" ? "" : lastSeg(p),
			class: classes.get(p) || undefined,
			inputs: {},
			operations: {},
			outputs: {},
		};
		boxes.set(p, spec);
		if (p !== "") {
			const parent = box(parentPath(p));
			(parent.operations ?? {})[lastSeg(p)] = spec;
		}
		return spec;
	};
	const root = box("");
	for (const t of userInputs) (root.inputs ?? {})[t] = value(t);

	const leafSpecs = new Map<string, Leaf>();
	const producers = new Map<string, Producer>();
	for (const t of userInputs) producers.set(t, { input: true });
	/** Each card's reads, wired in a second pass once every producer is known. */
	const reads: {
		scope: string;
		key: string;
		args: { arg: string; t: string }[];
	}[] = [];
	/** Lists parameter tensor t in the card's weights, named relative to the module `owner`. */
	const addWeight = (
		card: { weights?: Record<string, WeightSpec> },
		owner: string,
		t: string,
	) => {
		const fqn = params.get(t) ?? t;
		const local = fqn.startsWith(`${owner}.`)
			? fqn.slice(owner.length + 1)
			: fqn;
		card.weights = card.weights ?? {};
		card.weights[local] ??= { name: fqn, shape: value(t).shape };
	};

	for (const { n, op, path, ins, outs } of nodes) {
		if (!outs.length) continue;
		const tensorIns = ins.filter((i) => !hidden.has(resolve(i.t)));
		const short = op.slice(op.indexOf(".") + 1);
		if (elide.has(short) && op.startsWith("aten.")) {
			const first = tensorIns.find((i) => !params.has(resolve(i.t)));
			if (first) {
				for (const o of outs) alias.set(o, first.t);
				continue;
			}
		}
		const isParam = (t: string) => params.has(resolve(t));
		const leaf = leafOf(path);
		if (leaf) {
			let spec = leafSpecs.get(leaf);
			if (!spec) {
				spec = {
					type: "module",
					class: classes.get(leaf) || undefined,
					inputs: {},
					outputs: {},
				};
				leafSpecs.set(leaf, spec);
				const parent = box(parentPath(leaf));
				(parent.operations ?? {})[lastSeg(leaf)] = spec;
				reads.push({ scope: parentPath(leaf), key: lastSeg(leaf), args: [] });
			}
			const r = reads.find(
				(x) => x.scope === parentPath(leaf) && x.key === lastSeg(leaf),
			);
			for (const i of tensorIns) {
				const p = producers.get(resolve(i.t));
				// A tensor made inside the same module is not one of its inputs.
				const inside =
					p !== undefined &&
					"key" in p &&
					p.leaf &&
					p.key === lastSeg(leaf) &&
					p.scope === parentPath(leaf);
				if (isParam(i.t)) addWeight(spec, leaf, resolve(i.t));
				else if (!inside) r?.args.push({ arg: resolve(i.t), t: i.t });
			}
			for (const o of outs)
				producers.set(o, {
					scope: parentPath(leaf),
					key: lastSeg(leaf),
					leaf: true,
				});
			continue;
		}
		const key = n.name ?? outs[0];
		const spec: OpSpec = {
			type: "op",
			op,
			inputs: {},
			outputs: Object.fromEntries(outs.map((o) => [o, value(o)])),
		};
		const owner = box(path);
		(owner.operations ?? {})[key] = spec;
		const args: { arg: string; t: string }[] = [];
		for (const i of tensorIns) {
			if (isParam(i.t)) addWeight(spec, path, resolve(i.t));
			else args.push(i);
		}
		reads.push({ scope: path, key, args });
		for (const o of outs) producers.set(o, { scope: path, key, leaf: false });
	}

	/** A reference to tensor t from inside box `scope`, adding module inputs and outputs on the way. */
	const refIn = (scope: string, raw: string): string => {
		const t = resolve(raw);
		const p = producers.get(t);
		if (!p)
			throw new Error(`fromExportedProgram: no producer for tensor "${t}"`);
		const spec = box(scope);
		if ("input" in p || !within(p.scope, scope)) {
			if (scope === "") return t;
			spec.inputs = spec.inputs ?? {};
			if (!spec.inputs[t]) spec.inputs[t] = refIn(parentPath(scope), t);
			return t;
		}
		if (p.scope === scope) {
			if (p.leaf) {
				const leaf = leafSpecs.get(scope ? `${scope}.${p.key}` : p.key) as Leaf;
				leaf.outputs[t] ??= value(t);
			}
			return `${p.key}.${t}`;
		}
		const rest = scope ? p.scope.slice(scope.length + 1) : p.scope;
		const child = scope ? `${scope}.${rest.split(".")[0]}` : rest.split(".")[0];
		const childSpec = box(child);
		if (!childSpec.outputs[t]) childSpec.outputs[t] = refIn(child, t);
		return `${lastSeg(child)}.${t}`;
	};

	for (const { scope, key, args } of reads) {
		const card = box(scope).operations?.[key] as OpSpec | ModuleSpec;
		card.inputs = card.inputs ?? {};
		const inputs = card.inputs as Record<string, string>;
		for (const { arg, t } of args) {
			const name = card.type === "op" ? arg : resolve(t);
			if (inputs[name]) continue;
			inputs[name] = refIn(scope, t);
		}
	}
	graph.outputs.forEach((o, i) => {
		for (const t of tensorsOf(o)) {
			const name = options.outputNames?.[i] ?? resolve(t);
			root.outputs[name] = refIn("", t);
		}
	});
	return root;
}

/** One parameter's entry in a statistics file such as the one ParameterStatsExporter writes. */
export interface ParameterStats {
	rank: number | null;
	distribution: {
		shape: number[];
		dtype?: string;
		numel?: number;
		mean?: number;
		std?: number;
		min?: number;
		max?: number;
		histogram?: { bin_edges: number[]; counts: number[] };
	};
}

/**
 * A copy of the spec with every weight that names a parameter (WeightSpec.name) filled from
 * `stats`, looked up by the name without `prefix` (e.g. "model." when the program wraps the
 * model the statistics were taken from). Weights with no statistics are left as they are.
 */
export function withParameterStats(
	spec: ModuleSpec,
	stats: Record<string, ParameterStats>,
	prefix = "",
): ModuleSpec {
	const fill = (w: WeightSpec): WeightSpec => {
		if (!w.name) return w;
		const key = w.name.startsWith(prefix)
			? w.name.slice(prefix.length)
			: w.name;
		const s = stats[key];
		if (!s) return w;
		const d = s.distribution;
		return {
			...w,
			shape: d.shape,
			rank: s.rank ?? undefined,
			histogram: d.histogram
				? { edges: d.histogram.bin_edges, counts: d.histogram.counts }
				: undefined,
			moments:
				d.mean !== undefined && d.std !== undefined
					? { count: d.numel, mean: d.mean, std: d.std, min: d.min, max: d.max }
					: undefined,
			dtype: d.dtype?.replace(/^torch\./, ""),
		};
	};
	const walk = <T extends ModuleSpec | OpSpec>(m: T): T => {
		const out = { ...m };
		if (m.weights)
			out.weights = Object.fromEntries(
				Object.entries(m.weights).map(([k, w]) => [k, fill(w)]),
			);
		if (m.type !== "op" && m.operations)
			(out as ModuleSpec).operations = Object.fromEntries(
				Object.entries(m.operations).map(([k, c]) => [k, walk(c)]),
			);
		return out;
	};
	return walk(spec);
}
