// The input format of ModuleGraph, a PyTorch module as nested JSON, and its lowering to a flat
// graph: nodes (ops, leaf modules, the root's inputs and outputs), each tagged with the dotted
// name of the module it runs in, and edges, one per tensor and reader.
//
// A module has inputs, operations (ops and submodules) and outputs:
//
//   {
//     "name": "", "type": "module", "class": "TopKRouter",
//     "inputs":  { "hidden_states": { "shape": [4, 4], ... } },        // root: tensor values
//     "operations": {
//       "gate":    { "type": "module", "class": "Linear",              // leaf module: one node
//                    "inputs": { "input": "hidden_states" },
//                    "outputs": { "out": { "name": "logits", "shape": [4, 4], ... } } },
//       "softmax": { "type": "op", "op": "torch.softmax",
//                    "inputs": { "input": "gate.out" },
//                    "outputs": { "out": { "name": "probs", ... } } }
//     },
//     "outputs": { "probs": "softmax.out" }                            // references
//   }
//
// References name a tensor in the current module's scope: "x" is the module's input x,
// "softmax.out" is output out of the sibling operation softmax (for a submodule, one of its
// outputs). A submodule's inputs are references in its parent's scope, which is how tensors
// cross module boundaries; the root's inputs are tensor values. Operation names contain no
// dots (a method of an object can be named "object:method", e.g. "dispatcher:permute_tokens").
//
// A function call that runs several ops (a helper, a method of a plain object, an autograd
// Function's forward) is a scope like a module, with `type: "function"` and the function's name
// in `function`, e.g. { "type": "function", "function": "_AllToAll.apply", ... }. It is drawn as
// a dashed box labelled "key: _AllToAll.apply()"; without operations it is a single node.
//
// The root may define `symbols`, the meaning of each name used in a symbolic shape, e.g.
// { "total_tokens": "tokens on this rank: batch_size · seq_len" }; hover cards explain the names
// their shapes use. Any op or module may list `weights`, its parameters (WeightSpec): hovering
// its card (or a module box's label) shows their value histogram, singular values and ranks.
//
// The listing order of operations carries no meaning: an operation depends on what its inputs
// reference, and nothing else. The optional `layout.order` lists child names left to right for
// independent branches. A module with operations is drawn as a box; one without (a leaf, such
// as Linear) is a single node, like an op.

import { type BadgeIcon, ICON_NAMES, type IconName } from "./icons";

export interface TensorValue {
	/** Name in the code; defaults to the output's key. */
	name?: string;
	/** Shape in words, e.g. "(total_tokens, num_experts)". */
	symbolic_shape?: string;
	/** Shape in this example. */
	shape: number[];
	/** Titles of the row and column axes of the matrix shown on hover. */
	axes?: [string, string];
	row_labels?: string[];
	col_labels?: string[];
	/** Values as display strings, one array per row. */
	values?: string[][];
	/** Optional tint per row (0–7), e.g. the expert a row goes to. */
	row_tones?: (number | null)[];
	note?: string;
}

/**
 * A parameter of an op or module. Give `values` and the component computes the rest at build
 * time; or give `singular_values` and/or `histogram` (e.g. from a large layer) without values.
 */
export interface WeightSpec {
	/** The parameter's shape, e.g. [out_features, in_features]. */
	shape: number[];
	/** The parameter as a matrix, one array per row; flatten a higher-rank weight to (shape[0], rest). */
	values?: number[][];
	/** Singular values (any order), when `values` is not given. */
	singular_values?: number[];
	/** A histogram of the values, when `values` is not given: n + 1 bin edges, n counts. */
	histogram?: { edges: number[]; counts: number[] };
	/** Numerical rank counts σ > rank_tol · σ₁; default max(m, n) · ε of float32. */
	rank_tol?: number;
	note?: string;
}

export interface Equation {
	title: string;
	/** LaTeX, one display line each. */
	latex: string[];
	note?: string;
}

export type Ref = string;

/** A built-in icon (icons.ts), or a small square badge with text, e.g. { "text": "A2A" }. */
export type Icon = IconName | BadgeIcon;

interface Common {
	/** Node or box title; defaults to the key. */
	label?: string;
	/** Defaults by kind: fn for ops, hexagon for leaf modules, exchange for collectives. */
	icon?: Icon;
	/** Shown when the node is hovered. */
	equation?: Equation;
	/** Parameters, by name ("weight", "bias"), shown when the node or box is hovered. */
	weights?: Record<string, WeightSpec>;
}

export interface OpSpec extends Common {
	type: "op";
	/** The PyTorch call, e.g. "torch.softmax". */
	op?: string;
	/** "collective" colours the node as communication. */
	kind?: "compute" | "collective";
	inputs?: Record<string, Ref>;
	outputs: Record<string, TensorValue>;
}

export interface ModuleSpec extends Common {
	/** "module": an nn.Module. "function": a function call that runs several ops. */
	type: "module" | "function";
	/** Attribute name in the parent; "" for the root. Defaults to the key. */
	name?: string;
	/** A module's class, e.g. "TopKRouter". */
	class?: string;
	/** A function's name, e.g. "_AllToAll.apply" or "torch.nn.functional.scaled_dot_product_attention". */
	function?: string;
	/** Root: tensor values. Submodule: references in the parent's scope. */
	inputs?: Record<string, Ref | TensorValue>;
	operations?: Record<string, OpSpec | ModuleSpec>;
	/** Module with operations: references in its own scope. Leaf module: tensor values. */
	outputs: Record<string, Ref | TensorValue>;
	layout?: { order?: string[] };
	/** Root only: what each name in a symbolic shape means, e.g. { "hidden": "model width h" }. */
	symbols?: Record<string, string>;
}

export type NodeKind = "input" | "output" | "op" | "collective" | "module";

/** A node of the lowered graph: an op, a leaf module, or one of the root's inputs or outputs. */
export interface Node {
	/** Dotted name: "router.softmax", "experts.0.w1"; "input:x" and "output:y" for the root's. */
	id: string;
	label: string;
	kind: NodeKind;
	/** Dotted name of the module it runs in ("" for the root); null for the root's inputs and outputs. */
	module: string | null;
	/** Second line: the PyTorch call ("torch.softmax"), the class ("Linear"), or the shape. */
	subtitle?: string;
	icon: Icon;
	equation?: Equation;
	/** A collapsed module or function (see collapse): its id is the module's dotted name. */
	collapsed?: boolean;
	/** The tensors it produces (for an output node, none). */
	outputs?: (TensorValue & { name: string })[];
	weights?: Record<string, WeightSpec>;
}

/** A tensor from the node that produced it to one node that reads it. */
export interface Edge {
	from: string;
	to: string;
	tensor: TensorValue & { name: string };
	/** Index in the lowered graph's edges; kept by collapse, so a tensor keeps its hover card. */
	index: number;
}

/** What the boxes need beyond the dotted names: class or function name, left-to-right order. */
export interface ModuleInfo {
	class?: string;
	/** Set for a function scope: the function's name ("" if the spec gives none). */
	function?: string;
	/** Child ids (dotted) to keep left to right. */
	order: string[];
	weights?: Record<string, WeightSpec>;
}

export interface LoweredGraph {
	nodes: Node[];
	edges: Edge[];
	/** Every module with operations, by dotted name ("" for the root). */
	modules: Record<string, ModuleInfo>;
}

const isRef = (v: Ref | TensorValue): v is Ref => typeof v === "string";
const isLeaf = (m: ModuleSpec) =>
	!m.operations || !Object.keys(m.operations).length;
/** The dotted name of child `key` of the module at `path`. */
export const join = (path: string, key: string) =>
	path ? `${path}.${key}` : key;

interface Scope {
	path: string;
	spec: ModuleSpec;
	parent?: Scope;
}

/**
 * Lowers a module tree to nodes and edges, following every reference through module inputs and
 * outputs to the node that produced the tensor. Throws with every problem found: unknown
 * references, missing outputs, tensors where references belong (or the reverse), dotted names
 * and cycles.
 */
export function lower(root: ModuleSpec): LoweredGraph {
	const problems: string[] = [];
	const nodes: Node[] = [];
	const edges: Edge[] = [];
	const modules: Record<string, ModuleInfo> = {};
	const where = (scope: Scope) => scope.path || "root";
	const iconOf = (id: string, icon: Icon | undefined, fallback: Icon): Icon => {
		if (icon === undefined) return fallback;
		if (typeof icon === "string" && !ICON_NAMES.includes(icon)) {
			problems.push(
				`${id}: unknown icon "${icon}" (one of ${ICON_NAMES.join(", ")})`,
			);
			return fallback;
		}
		return icon;
	};
	type Producer = { node: string; tensor: TensorValue & { name: string } };
	const named = (t: TensorValue, key: string) => ({
		...t,
		name: t.name ?? key,
	});
	const outputsOf = (outs: Record<string, Ref | TensorValue>) =>
		Object.entries(outs).flatMap(([k, v]) => (isRef(v) ? [] : [named(v, k)]));

	/** The node (and tensor) a reference in a scope points to. */
	const resolve = (scope: Scope, ref: Ref): Producer | undefined => {
		const dot = ref.lastIndexOf(".");
		if (dot < 0) {
			const binding = scope.spec.inputs?.[ref];
			if (binding === undefined) {
				problems.push(`${where(scope)}: unknown input "${ref}"`);
				return undefined;
			}
			if (!scope.parent) {
				if (isRef(binding)) {
					problems.push(
						`root input "${ref}" must be a tensor value, not a reference`,
					);
					return undefined;
				}
				return { node: `input:${ref}`, tensor: named(binding, ref) };
			}
			if (!isRef(binding)) {
				problems.push(
					`${where(scope)}: input "${ref}" must be a reference in the parent's scope`,
				);
				return undefined;
			}
			return resolve(scope.parent, binding);
		}
		const [key, out] = [ref.slice(0, dot), ref.slice(dot + 1)];
		const child = scope.spec.operations?.[key];
		if (!child) {
			problems.push(
				`${where(scope)}: reference "${ref}" names unknown operation "${key}"`,
			);
			return undefined;
		}
		const id = join(scope.path, key);
		const value = child.outputs?.[out];
		if (value === undefined) {
			problems.push(`${where(scope)}: "${key}" has no output "${out}"`);
			return undefined;
		}
		if (child.type === "op" || isLeaf(child as ModuleSpec)) {
			if (isRef(value)) {
				problems.push(
					`${id}: output "${out}" of a leaf must be a tensor value`,
				);
				return undefined;
			}
			return { node: id, tensor: named(value, out) };
		}
		if (!isRef(value)) {
			problems.push(
				`${id}: output "${out}" of a module with operations must be a reference`,
			);
			return undefined;
		}
		return resolve(
			{ path: id, spec: child as ModuleSpec, parent: scope },
			value,
		);
	};

	const connect = (
		scope: Scope,
		to: string,
		refs: Record<string, Ref | TensorValue>,
	) => {
		for (const [k, ref] of Object.entries(refs)) {
			if (!isRef(ref)) {
				problems.push(`${to}: input "${k}" must be a reference`);
				continue;
			}
			const p = resolve(scope, ref);
			if (!p) continue;
			if (
				!edges.some(
					(e) =>
						e.from === p.node && e.to === to && e.tensor.name === p.tensor.name,
				)
			)
				edges.push({ from: p.node, to, tensor: p.tensor, index: edges.length });
		}
	};

	const walk = (scope: Scope) => {
		modules[scope.path] = {
			class: scope.spec.class,
			function:
				scope.spec.type === "function"
					? (scope.spec.function ?? "")
					: undefined,
			order: (scope.spec.layout?.order ?? []).map((k) => join(scope.path, k)),
			weights: scope.spec.weights,
		};
		for (const [key, child] of Object.entries(scope.spec.operations ?? {})) {
			if (key.includes("."))
				problems.push(
					`${where(scope)}: operation name "${key}" contains a dot`,
				);
			const id = join(scope.path, key);
			if (child.type === "op") {
				const collective = child.kind === "collective";
				nodes.push({
					id,
					label: child.label ?? key,
					kind: collective ? "collective" : "op",
					module: scope.path,
					subtitle: child.op,
					icon: iconOf(id, child.icon, collective ? "exchange" : "fn"),
					equation: child.equation,
					outputs: outputsOf(child.outputs),
					weights: child.weights,
				});
				connect(scope, id, child.inputs ?? {});
			} else if (isLeaf(child)) {
				const fn = child.type === "function";
				nodes.push({
					id,
					label: child.label ?? key,
					kind: fn ? "op" : "module",
					module: scope.path,
					subtitle: fn ? child.function : child.class,
					icon: iconOf(id, child.icon, fn ? "fn" : "hexagon"),
					equation: child.equation,
					outputs: outputsOf(child.outputs),
					weights: child.weights,
				});
				connect(scope, id, child.inputs ?? {});
			} else walk({ path: id, spec: child, parent: scope });
		}
	};

	const rootScope: Scope = { path: "", spec: root };
	for (const [k, v] of Object.entries(root.inputs ?? {}))
		nodes.push({
			id: `input:${k}`,
			label: k,
			kind: "input",
			module: null,
			subtitle: isRef(v) ? "input" : `input (${v.shape.join(", ")})`,
			icon: "file",
			outputs: isRef(v) ? [] : [named(v, k)],
		});
	walk(rootScope);
	for (const [k, v] of Object.entries(root.outputs)) {
		const id = `output:${k}`;
		nodes.push({
			id,
			label: k,
			kind: "output",
			module: null,
			subtitle: "output",
			icon: "commit",
		});
		connect(rootScope, id, { [k]: v });
	}
	const cycle = findCycle(nodes, edges);
	if (cycle) problems.push(`cycle: ${cycle.join(" → ")}`);
	if (problems.length) throw new Error(`ModuleGraph: ${problems.join("; ")}`);
	return { nodes, edges, modules };
}

/**
 * A box's label: the qualified name and class ("experts.0: SwiGLUMLP"), the name and function
 * ("dispatch: _AllToAll.apply()"), or the root's class.
 */
export function moduleLabel(
	path: string,
	info: ModuleInfo | undefined,
): string {
	const name = path.replace(/:/g, ".");
	if (info?.function !== undefined)
		return info.function ? `${name}: ${info.function}()` : `${name}()`;
	if (!path) return info?.class ?? "";
	return info?.class ? `${name}: ${info.class}` : name;
}

/** A cycle among the nodes, as a list of ids, or undefined. */
function findCycle(nodes: Node[], edges: Edge[]): string[] | undefined {
	const out = new Map<string, string[]>(nodes.map((n) => [n.id, []]));
	for (const e of edges) out.get(e.from)?.push(e.to);
	const state = new Map<string, 1 | 2>();
	const stack: string[] = [];
	const visit = (v: string): string[] | undefined => {
		state.set(v, 1);
		stack.push(v);
		for (const w of out.get(v) ?? []) {
			if (state.get(w) === 1) return [...stack.slice(stack.indexOf(w)), w];
			if (!state.has(w)) {
				const c = visit(w);
				if (c) return c;
			}
		}
		stack.pop();
		state.set(v, 2);
		return undefined;
	};
	for (const n of nodes) {
		if (state.has(n.id)) continue;
		const c = visit(n.id);
		if (c) return c;
	}
	return undefined;
}

/** The module enclosing the module at `path` ("" for a top-level one), or null for the root. */
export const parentModule = (path: string): string | null =>
	path === ""
		? null
		: path.includes(".")
			? path.slice(0, path.lastIndexOf("."))
			: "";

/**
 * The graph with the given modules and functions (dotted names, "" for the root) folded into one
 * card each: every node inside a collapsed box becomes that box's card (the outermost collapsed
 * box wins), edges are re-pointed to the cards, and edges that now start and end on the same
 * card disappear. Edges that become identical (same ends, same tensor) are kept once.
 */
export function collapse(
	g: LoweredGraph,
	collapsed: Iterable<string>,
): LoweredGraph {
	const set = new Set(collapsed);
	if (!set.size) return g;
	/** The outermost collapsed box holding a module, or undefined. */
	const folded = (module: string | null): string | undefined => {
		if (module === null) return undefined;
		const parts = module ? module.split(".") : [];
		for (let i = 0; i <= parts.length; i++) {
			const p = parts.slice(0, i).join(".");
			if (set.has(p)) return p;
		}
		return undefined;
	};
	const target = new Map<string, string>();
	const nodes: Node[] = [];
	for (const n of g.nodes) {
		const box = folded(n.module);
		if (box === undefined) {
			nodes.push(n);
			target.set(n.id, n.id);
			continue;
		}
		target.set(n.id, box);
		if (nodes.some((m) => m.collapsed && m.id === box)) continue;
		const info = g.modules[box];
		const fn = info?.function !== undefined;
		const name = box.slice(box.lastIndexOf(".") + 1).replace(/:/g, ".");
		nodes.push({
			id: box,
			label: box === "" ? (info?.class ?? "module") : name,
			kind: fn ? "op" : "module",
			module: parentModule(box),
			subtitle: fn
				? `${info?.function || name}()`
				: box === ""
					? undefined
					: info?.class,
			icon: fn ? "fn" : "hexagon",
			collapsed: true,
		});
	}
	const edges: Edge[] = [];
	for (const e of g.edges) {
		const from = target.get(e.from) ?? e.from;
		const to = target.get(e.to) ?? e.to;
		if (from === to) continue;
		if (
			edges.some(
				(x) =>
					x.from === from && x.to === to && x.tensor.name === e.tensor.name,
			)
		)
			continue;
		edges.push({ ...e, from, to });
	}
	return { nodes, edges, modules: g.modules };
}
