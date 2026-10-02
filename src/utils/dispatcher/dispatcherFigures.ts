import {
	AFTER_AUTOGRAD,
	BACKENDS,
	DEFAULT_EXCLUDED,
	DEFAULT_INCLUDED,
	type Entry,
	FUNCTIONALITIES,
	type Kernel,
	OFFSETS_AND_MASKS,
	OperatorTable,
	computeDispatchKeySet,
	highestFunctionality,
	indexOfHighestBit,
	isPerBackend,
	keySet,
	minus,
	runtimeKeys,
	tableIndex,
	tensorKeySet,
} from "./dispatchKeys";
// Figure data for the PyTorch dispatcher post, computed with the dispatchKeys.ts model from the
// real registrations in dispatcherData.ts. Used by DispatchTable.astro and KeySetSteps.astro.
import {
	ADD_TENSOR_REGISTRATIONS,
	BACKEND_FALLBACKS,
	MM_REGISTRATIONS,
	type Registration,
} from "./dispatcherData";

const kernelOf = (r: Registration): Kernel => ({
	name: r.key,
	site: r.site,
	fallthrough: r.fallthrough,
});

const fallbackMap = () =>
	new Map(BACKEND_FALLBACKS.map((r) => [r.key, kernelOf(r)]));

/** An operator table with every kernel in `regs` registered, oldest first. */
export const buildTable = (regs: Registration[]) => {
	const op = new OperatorTable(fallbackMap());
	for (const r of regs) op.register(r.key, kernelOf(r));
	return op;
};

export const addTable = () => buildTable(ADD_TENSOR_REGISTRATIONS);
export const mmTable = () => buildTable(MM_REGISTRATIONS);

// ---------- DispatchTable: aten::add.Tensor's table, filled in one registration group at a time ----------

export interface TableStep {
	title: string;
	note: string;
	table: Entry[];
	changed: number[];
}

// The C++ Meta kernel is registered before the Python one (oldest first).
const firstMeta = ADD_TENSOR_REGISTRATIONS.findIndex((r) => r.key === "Meta");

// Static initialisers run in an unspecified order, so these groups are for exposition; the final
// table doesn't depend on the order (only on which kernel is newest per key).
const ADD_GROUPS: {
	title: string;
	note: string;
	keys: (r: Registration, i: number) => boolean;
}[] = [
	{
		title: "def: aten::add.Tensor gets a schema",
		note: "registerDef stores the schema; the DispatchKeyExtractor learns that arguments 0 and 1 (self, other) are tensors. The table is untouched.",
		keys: () => false,
	},
	{
		title: "impl: CompositeExplicitAutogradNonFunctional",
		note: "An alias key: one kernel fans out to Undefined and to every backend slot in its key set (not XLA, Lazy or Sparse), as a 'default backend kernel'.",
		keys: (r) => r.key === "CompositeExplicitAutogradNonFunctional",
	},
	{
		title: "impl: CPU",
		note: "A direct registration always wins, so slot 1 now holds wrapper_CPU_add_Tensor. registerKernel also recomputes AutogradCPU.",
		keys: (r) => r.key === "CPU",
	},
	{
		title: "impl: the other backend kernels",
		note: "MPS, Meta (C++), MkldnnCPU and the sparse and nested kernels each overwrite one slot.",
		keys: (r, i) =>
			/^(MPS|MkldnnCPU|Sparse|NestedTensor)/.test(r.key) ||
			(r.key === "Meta" && i === firstMeta),
	},
	{
		title: "impl: Autograd (alias) from VariableType",
		note: "The Autograd alias covers AutogradOther, AutogradNestedTensor and all 16 Autograd<Backend> slots, replacing the boxed autograd fallback.",
		keys: (r) => r.key === "Autograd",
	},
	{
		title: "impl: Tracer, ZeroTensor, Batched, FuncTorchBatched, Named",
		note: "Kernels for wrapper functionalities. Named is a fallthrough kernel: the slot is set, and the key is masked out of dispatch.",
		keys: (r) =>
			["Tracer", "ZeroTensor", "Batched", "FuncTorchBatched", "Named"].includes(
				r.key,
			),
	},
	{
		title: "impl: Meta again, from Python",
		note: "torch/_meta_registrations.py registers a second Meta kernel at import time. kernels_[Meta] keeps both; the table points at the newest.",
		keys: (r, i) => r.key === "Meta" && i !== firstMeta,
	},
];
export const addTableSteps = (): TableStep[] => {
	const op = new OperatorTable(fallbackMap());
	const steps: TableStep[] = [
		{
			title: "OperatorEntry created",
			note: "The constructor copies every backend fallback into the table (Python, Functionalize, the autograd fallbacks, ...). Every other slot is empty.",
			table: [...op.table],
			changed: op.table
				.filter((e) => e.source !== "missing")
				.map((e) => e.slot),
		},
	];
	const done = new Set<number>();
	for (const g of ADD_GROUPS) {
		const changed = new Set<number>();
		ADD_TENSOR_REGISTRATIONS.forEach((r, i) => {
			if (done.has(i) || !g.keys(r, i)) return;
			done.add(i);
			for (const s of op.register(r.key, kernelOf(r))) changed.add(s);
		});
		steps.push({
			title: g.title,
			note: g.note,
			table: [...op.table],
			changed: [...changed].sort((a, b) => a - b),
		});
	}
	if (done.size !== ADD_TENSOR_REGISTRATIONS.length)
		throw new Error("addTableSteps: registration left out");
	return steps;
};

/** Rows of the table figure: per-backend functionalities get a row of 16; runs of single keys are wrapped. */
export interface TableRow {
	label: string;
	slots: number[];
	perBackend: boolean;
}

export const tableRows = (): TableRow[] => {
	const rows: TableRow[] = [];
	const run: number[] = [];
	const flush = () => {
		while (run.length) {
			const part = run.splice(0, BACKENDS.length);
			const label =
				part.length === 1
					? `${part[0]}`
					: `${part[0]}–${part[part.length - 1]}`;
			rows.push({ label, slots: part, perBackend: false });
		}
	};
	for (let f = 0; f < FUNCTIONALITIES.length; f++) {
		const { offset } = OFFSETS_AND_MASKS[f];
		if (isPerBackend(f)) {
			flush();
			const name =
				FUNCTIONALITIES[f] === "AutogradFunctionality"
					? "Autograd"
					: FUNCTIONALITIES[f];
			rows.push({
				label: name,
				slots: BACKENDS.map((_, b) => offset + b),
				perBackend: true,
			});
		} else run.push(offset);
	}
	flush();
	return rows;
};

// ---------- KeySetSteps: computing the dispatch key set for one call ----------

export interface Stage {
	title: string;
	code: string;
	ks: bigint;
	/** Lookup stage only: how the slot was found. */
	lookup?: {
		f: number;
		perBackend: boolean;
		offset: number;
		backend: number;
		slot: number;
		key: string;
		source: string;
		site?: string;
	};
}

export interface Scenario {
	id: string;
	label: string;
	call: string;
	stages: Stage[];
}

const lookupStage = (
	title: string,
	code: string,
	ks: bigint,
	op: OperatorTable,
): Stage => {
	const f = highestFunctionality(ks);
	const { offset, mask } = OFFSETS_AND_MASKS[f];
	const backend = indexOfHighestBit((ks & mask) >> 1n);
	const slot = tableIndex(ks);
	const e = op.table[slot];
	return {
		title,
		code,
		ks,
		lookup: {
			f,
			perBackend: isPerBackend(f),
			offset,
			backend,
			slot,
			key: e.key,
			source: e.source,
			site: e.kernel?.site,
		},
	};
};

/** The lookup arithmetic in words, for the figure's panel. */
export const lookupText = (s: Stage) => {
	const l = s.lookup;
	if (!l) return "";
	const fname = FUNCTIONALITIES[l.f];
	const how = l.perBackend
		? `${fname} is per-backend, so add the highest backend bit's index: ${BACKENDS[l.backend]} → +${l.backend}.`
		: `${fname} is not per-backend (mask 0), so the slot is the offset.`;
	const where = l.site ? ` (${l.site})` : "";
	return `Highest functionality bit: ${fname} (f = ${l.f}) → offset ${l.offset}. ${how} Slot ${l.slot} = ${l.key}, holding the ${l.source}${where}.`;
};

/** Markers for drawBits: the highest functionality bit, and the backend bit if it was used. */
export const lookupMarks = (s: Stage) =>
	s.lookup
		? { f: s.lookup.f, b: s.lookup.perBackend ? s.lookup.backend : -1 }
		: undefined;

interface CallSpec {
	id: string;
	label: string;
	call: string;
	args: { name: string; ks: bigint; how: string }[];
	included: bigint;
	excluded: bigint;
	tlsNote: string;
	op: OperatorTable;
	opName: string;
	redispatch?: { mask: bigint; code: string };
}

const scenario = (s: CallSpec): Scenario => {
	const stages: Stage[] = [];
	let acc = 0n;
	s.args.forEach((a, i) => {
		acc |= a.ks;
		stages.push({
			title:
				i === 0
					? `Start from ${a.name}.key_set()`
					: `OR in ${a.name}.key_set()`,
			code: a.how,
			ks: acc,
		});
	});
	const withInc = acc | s.included;
	stages.push({
		title: "OR the thread-local included set",
		code: `ks | tls.included_   // ${runtimeKeys(s.included).join(", ") || "empty"}`,
		ks: withInc,
	});
	const withExc = minus(withInc, s.excluded);
	stages.push({
		title: "Remove the thread-local excluded set",
		code: `… - tls.excluded_   // ${s.tlsNote}`,
		ks: withExc,
	});
	const mask = s.op.nonFallthrough;
	const final = computeDispatchKeySet(acc, s.included, s.excluded, mask);
	const dropped = runtimeKeys(
		withExc & ~mask & ~((1n << BigInt(BACKENDS.length)) - 1n),
	);
	stages.push({
		title: `Mask out ${s.opName}'s fallthrough keys`,
		code: `… & nonFallthroughKeys_   // drops ${dropped.join(", ") || "nothing"}`,
		ks: final,
	});
	if (final !== s.op.dispatchKeySet(acc, s.included, s.excluded))
		throw new Error("scenario: mask mismatch");
	stages.push(
		lookupStage(
			"Look up the slot",
			"op.lookup(ks)  →  dispatchTable_[getDispatchTableIndexForDispatchKeySet()]",
			final,
			s.op,
		),
	);
	if (s.redispatch) {
		const ks2 = final & s.redispatch.mask;
		stages.push(
			lookupStage("The kernel redispatches", s.redispatch.code, ks2, s.op),
		);
	}
	return { id: s.id, label: s.label, call: s.call, stages };
};

export const keySetScenarios = (): Scenario[] => {
	const add = addTable();
	const mm = mmTable();
	const cpu = tensorKeySet("CPU");
	const redispatch = {
		mask: AFTER_AUTOGRAD,
		code: "at::redispatch::add(ks & c10::after_autograd_keyset, …)",
	};
	return [
		scenario({
			id: "cpu",
			label: "torch.add(x, w)",
			call: "x, w: float32 CPU tensors (w requires grad)",
			args: [
				{
					name: "x",
					ks: cpu,
					how: "CPU, plus ADInplaceOrView, AutogradCPU, AutocastCPU from the TensorImpl constructor",
				},
				{
					name: "w",
					ks: cpu,
					how: "same keys: requires_grad does not change a tensor's key set",
				},
			],
			included: DEFAULT_INCLUDED,
			excluded: DEFAULT_EXCLUDED,
			tlsNote: "every Autocast key (autocast is off)",
			op: add,
			opName: "add",
			redispatch,
		}),
		scenario({
			id: "cuda",
			label: "CUDA + CPU scalar",
			call: "torch.add(a, torch.tensor(2.0)) with a on CUDA. The table is from a macOS build without CUDA kernels, so slot 2 holds the default backend kernel.",
			args: [
				{
					name: "a",
					ks: tensorKeySet("CUDA"),
					how: "CUDA, ADInplaceOrView, AutogradCUDA, AutocastCUDA",
				},
				{
					name: "b",
					ks: cpu,
					how: "a 0-dim CPU tensor adds the CPU backend bit",
				},
			],
			included: DEFAULT_INCLUDED,
			excluded: DEFAULT_EXCLUDED,
			tlsNote: "every Autocast key (autocast is off)",
			op: add,
			opName: "add",
			redispatch,
		}),
		scenario({
			id: "autocast",
			label: "mm under autocast",
			call: 'torch.mm(a, b) inside torch.autocast("cpu")',
			args: [
				{
					name: "a",
					ks: cpu,
					how: "CPU, ADInplaceOrView, AutogradCPU, AutocastCPU",
				},
				{ name: "b", ks: cpu, how: "same keys" },
			],
			included: DEFAULT_INCLUDED,
			excluded: minus(DEFAULT_EXCLUDED, keySet("AutocastCPU")),
			tlsNote: "autocast removed AutocastCPU from it",
			op: mm,
			opName: "mm",
		}),
		scenario({
			id: "inference",
			label: "inference_mode",
			call: "torch.add(x, w) inside torch.inference_mode(), on tensors created there",
			args: [
				{
					name: "x",
					ks: tensorKeySet("CPU", true),
					how: "inference tensors get no autograd or ADInplaceOrView keys",
				},
				{ name: "w", ks: tensorKeySet("CPU", true), how: "same keys" },
			],
			included: keySet("BackendSelect"),
			excluded:
				DEFAULT_EXCLUDED |
				keySet(
					"AutogradFunctionality",
					"AutogradOther",
					"AutogradNestedTensor",
				),
			tlsNote: "Autocast keys and the autograd keys",
			op: add,
			opName: "add",
		}),
	];
};

/** The bits that are set in any stage of a scenario, for labelling the bit strip. */
export const scenarioBits = (s: Scenario) =>
	s.stages.reduce((acc, st) => acc | st.ks, 0n);
