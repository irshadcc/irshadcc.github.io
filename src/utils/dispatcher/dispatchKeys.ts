// A model of PyTorch's dispatch keys, DispatchKeySet and per-operator dispatch table, following
// c10/core/DispatchKey.h, c10/core/DispatchKeySet.{h,cpp} and
// aten/src/ATen/core/dispatch/{OperatorEntry,DispatchKeyExtractor}.{h,cpp} at PyTorch v2.9.1.
// Key sets are 64-bit masks, held here as bigint. Used by the dispatcher post's figures and
// checked against a real PyTorch 2.9.1 build (see the post's verification notes).

// BackendComponent, in enum order. InvalidBit = 0 is left out, so BACKENDS[i] has enum value i + 1
// and lives in bit i of the key set.
export const BACKENDS = [
	"CPU",
	"CUDA",
	"HIP",
	"XLA",
	"MPS",
	"IPU",
	"XPU",
	"HPU",
	"VE",
	"Lazy",
	"MTIA",
	"MAIA",
	"PrivateUse1",
	"PrivateUse2",
	"PrivateUse3",
	"Meta",
] as const;

// Functionality keys, in enum order: FUNCTIONALITIES[k] has enum value k and, for k >= 1, lives in
// bit NUM_BACKENDS + k - 1 of the key set.
export const FUNCTIONALITIES = [
	"Undefined",
	"Dense",
	"FPGA",
	"Vulkan",
	"Metal",
	"Quantized",
	"CustomRNGKeyId",
	"MkldnnCPU",
	"Sparse",
	"SparseCsr",
	"NestedTensor",
	"BackendSelect",
	"Python",
	"Fake",
	"FuncTorchDynamicLayerBackMode",
	"Functionalize",
	"Named",
	"Conjugate",
	"Negative",
	"ZeroTensor",
	"ADInplaceOrView",
	"AutogradOther",
	"AutogradFunctionality",
	"AutogradNestedTensor",
	"Tracer",
	"AutocastCPU",
	"AutocastMTIA",
	"AutocastMAIA",
	"AutocastXPU",
	"AutocastIPU",
	"AutocastHPU",
	"AutocastXLA",
	"AutocastMPS",
	"AutocastCUDA",
	"AutocastPrivateUse1",
	"FuncTorchBatched",
	"BatchedNestedTensor",
	"FuncTorchVmapMode",
	"Batched",
	"VmapMode",
	"FuncTorchGradWrapper",
	"DeferredInit",
	"PythonTLSSnapshot",
	"FuncTorchDynamicLayerFrontMode",
	"TESTING_ONLY_GenericWrapper",
	"TESTING_ONLY_GenericMode",
	"PreDispatch",
	"PythonDispatcher",
] as const;

export type Backend = (typeof BACKENDS)[number];
export type Functionality = (typeof FUNCTIONALITIES)[number];

export const NUM_BACKENDS = BACKENDS.length; // 16
export const NUM_FUNCTIONALITY_KEYS = FUNCTIONALITIES.length; // 48 = EndOfFunctionalityKeys

// Functionalities that are customisable per backend, with the prefix of their runtime keys
// (Dense + CPU is "CPU", AutogradFunctionality + CPU is "AutogradCPU").
export const PER_BACKEND: Partial<Record<Functionality, string>> = {
	Dense: "",
	Quantized: "Quantized",
	Sparse: "Sparse",
	SparseCsr: "SparseCsr",
	NestedTensor: "NestedTensor",
	AutogradFunctionality: "Autograd",
};

export const isPerBackend = (f: number) => FUNCTIONALITIES[f] in PER_BACKEND;

export const NUM_RUNTIME_ENTRIES =
	NUM_FUNCTIONALITY_KEYS + Object.keys(PER_BACKEND).length * (NUM_BACKENDS - 1); // 138

export const FULL_BACKEND_MASK = (1n << BigInt(NUM_BACKENDS)) - 1n;
export const FULL =
	(1n << BigInt(NUM_BACKENDS + NUM_FUNCTIONALITY_KEYS - 1)) - 1n;

export const functionalityBit = (f: number) =>
	f === 0 ? 0n : 1n << BigInt(NUM_BACKENDS + f - 1);
export const backendBit = (b: number) => 1n << BigInt(b); // b is the 0-based index into BACKENDS

/** A runtime key split into its functionality (enum value) and backend (index, or -1). */
export interface KeyParts {
	f: number;
	b: number;
}

const RUNTIME_KEYS = new Map<string, KeyParts>();
for (const [f, name] of FUNCTIONALITIES.entries()) {
	const prefix = PER_BACKEND[name];
	if (prefix === undefined) RUNTIME_KEYS.set(name, { f, b: -1 });
	else
		for (const [b, be] of BACKENDS.entries())
			RUNTIME_KEYS.set(prefix + be, { f, b });
}

export const parseKey = (name: string): KeyParts => {
	const parts = RUNTIME_KEYS.get(name);
	if (!parts) throw new Error(`unknown runtime dispatch key ${name}`);
	return parts;
};

export const keyName = ({ f, b }: KeyParts) => {
	const prefix = PER_BACKEND[FUNCTIONALITIES[f]];
	return prefix === undefined ? FUNCTIONALITIES[f] : prefix + BACKENDS[b];
};

/** DispatchKeySet(DispatchKey k) for a runtime or building-block key, OR'ed over all names. */
export const keySet = (...names: string[]) => {
	let ks = 0n;
	for (const n of names) {
		const bi = BACKENDS.indexOf(n.replace(/Bit$/, "") as Backend);
		if (n.endsWith("Bit") && bi >= 0) {
			ks |= backendBit(bi);
			continue;
		}
		const fi = FUNCTIONALITIES.indexOf(n as Functionality);
		if (fi >= 0) {
			ks |= functionalityBit(fi);
			continue;
		}
		const { f, b } = parseKey(n);
		ks |= functionalityBit(f) | (b >= 0 ? backendBit(b) : 0n);
	}
	return ks;
};

/** 64 - countLeadingZeros(x): one more than the index of the highest set bit, 0 if none. */
export const indexOfHighestBit = (x: bigint) =>
	x === 0n ? 0 : x.toString(2).length;

export const highestFunctionality = (ks: bigint) => {
	const i = indexOfHighestBit(ks);
	return i < NUM_BACKENDS ? 0 : i - NUM_BACKENDS;
};

/** Index into BACKENDS of the highest backend bit, or -1 if no backend bit is set. */
export const highestBackend = (ks: bigint) =>
	indexOfHighestBit(ks & FULL_BACKEND_MASK) - 1;

/** DispatchKeySet::highestPriorityTypeId(). */
export const highestPriorityKey = (ks: bigint): string => {
	const f = highestFunctionality(ks);
	if (isPerBackend(f)) {
		const b = highestBackend(ks);
		// With no backend bit, C++ lands on the placeholder StartOf<Functionality>Backends key.
		return b < 0 ? `StartOf${FUNCTIONALITIES[f]}Backends` : keyName({ f, b });
	}
	return FUNCTIONALITIES[f];
};

/** initializeFunctionalityOffsetsAndMasks(): where each functionality's slots start. */
export const OFFSETS_AND_MASKS: { offset: number; mask: bigint }[] = (() => {
	const out = [{ offset: 0, mask: 0n }];
	for (let f = 1; f < NUM_FUNCTIONALITY_KEYS; f++) {
		const prev = out[f - 1];
		out.push({
			offset: prev.offset + (prev.mask === 0n ? 1 : NUM_BACKENDS),
			mask: isPerBackend(f) ? FULL_BACKEND_MASK : 0n,
		});
	}
	return out;
})();

/** DispatchKeySet::getDispatchTableIndexForDispatchKeySet(), the hot-path version. */
export const tableIndex = (ks: bigint) => {
	const f = indexOfHighestBit(ks >> BigInt(NUM_BACKENDS));
	const { offset, mask } = OFFSETS_AND_MASKS[f];
	return offset + indexOfHighestBit((ks & mask) >> 1n);
};

export const tableIndexOfKey = (name: string) => tableIndex(keySet(name));

/** Runtime keys in a set, in the order DispatchKeySet's iterator yields them (lowest first). */
export const runtimeKeys = (ks: bigint): string[] => {
	const out: string[] = [];
	for (let f = 1; f < NUM_FUNCTIONALITY_KEYS; f++) {
		if ((ks & functionalityBit(f)) === 0n) continue;
		if (!isPerBackend(f)) out.push(FUNCTIONALITIES[f]);
		else
			for (let b = 0; b < NUM_BACKENDS; b++)
				if (ks & backendBit(b)) out.push(keyName({ f, b }));
	}
	return out;
};

/** Mirrors operator<<(DispatchKeySet): "DispatchKeySet(CPU, AutogradCPU)". */
export const keySetToString = (ks: bigint) =>
	`DispatchKeySet(${runtimeKeys(ks).join(", ")})`;

/** Table slot -> runtime key name, the inverse of tableIndexOfKey (getDispatchTableIndexToKey). */
export const SLOT_KEYS: string[] = (() => {
	const arr = new Array<string>(NUM_RUNTIME_ENTRIES).fill("Undefined");
	for (const k of runtimeKeys(FULL)) arr[tableIndexOfKey(k)] = k;
	return arr;
})();

// ---------- Set algebra used by the dispatcher (DispatchKeySet operators) ----------

/** a - b: removes functionality bits only, backend bits of a are kept. */
export const minus = (a: bigint, b: bigint) => a & (FULL_BACKEND_MASK | ~b);
/** DispatchKeySet(FULL_AFTER, k): every functionality below k, all backends, plus PythonDispatcher. */
export const fullAfter = (name: string) =>
	((1n << BigInt(NUM_BACKENDS + parseKey(name).f - 1)) - 1n) |
	keySet("PythonDispatcher");
/** ks.remove(k): clears k's functionality bit only. */
export const removeKey = (ks: bigint, name: string) =>
	ks & ~(keySet(name) & ~FULL_BACKEND_MASK);

export const AUTOGRAD_KEYSET = keySet(
	"AutogradFunctionality",
	"AutogradOther",
	"AutogradNestedTensor",
);
export const AUTOCAST_KEYSET = keySet(
	"AutocastCPU",
	"AutocastMPS",
	"AutocastCUDA",
	"AutocastXPU",
	"AutocastIPU",
	"AutocastHPU",
	"AutocastXLA",
	"AutocastPrivateUse1",
	"AutocastMTIA",
	"AutocastMAIA",
);
export const DEFAULT_INCLUDED = keySet("BackendSelect", "ADInplaceOrView");
export const DEFAULT_EXCLUDED = AUTOCAST_KEYSET;
export const AFTER_AUTOGRAD = fullAfter("AutogradOther");

/** impl::computeDispatchKeySet: ((ks | included) - excluded) & mask. */
export const computeDispatchKeySet = (
	ks: bigint,
	included: bigint,
	excluded: bigint,
	mask: bigint,
) => minus(ks | included, excluded) & mask;

// Backends with their own Autograd<Backend> key (getAutogradRelatedKeySetFromBackend); others
// share AutogradOther. And the backends with an Autocast<Backend> key.
const OWN_AUTOGRAD = [
	"CPU",
	"IPU",
	"MTIA",
	"MAIA",
	"XPU",
	"CUDA",
	"XLA",
	"Lazy",
	"Meta",
	"MPS",
	"HPU",
];
const OWN_AUTOCAST = [
	"CPU",
	"MTIA",
	"MAIA",
	"XPU",
	"IPU",
	"HPU",
	"CUDA",
	"XLA",
	"PrivateUse1",
	"MPS",
];
const autogradKeyFor = (be: string) =>
	OWN_AUTOGRAD.includes(be) || be.startsWith("PrivateUse")
		? `Autograd${be}`
		: "AutogradOther";

/** Key set a new tensor gets in the TensorImpl constructor (outside inference mode). */
export const tensorKeySet = (runtimeKey: string, inferenceMode = false) => {
	let ks = keySet(runtimeKey);
	const b = highestBackend(ks);
	const be = b < 0 ? undefined : BACKENDS[b];
	if (be && OWN_AUTOCAST.includes(be)) ks |= keySet(`Autocast${be}`);
	if (inferenceMode)
		return minus(ks, AUTOGRAD_KEYSET | keySet("ADInplaceOrView"));
	return (
		ks | keySet("ADInplaceOrView", be ? autogradKeyFor(be) : "AutogradOther")
	);
};

// ---------- Alias keys (DispatchKeySet.cpp) ----------

const AUTOGRADOTHER_BACKENDS =
	keySet(
		"FPGA",
		"Vulkan",
		"Metal",
		"CustomRNGKeyId",
		"MkldnnCPU",
		"Sparse",
		"SparseCsr",
		"Quantized",
	) | FULL_BACKEND_MASK;
const BACKEND_KEYSET = AUTOGRADOTHER_BACKENDS | keySet("Dense");
const NON_FUNCTIONAL_BACKEND_KEYSET =
	removeKey(BACKEND_KEYSET, "Sparse") & ~keySet("XLABit", "LazyBit");
const MATH_KEYSET =
	BACKEND_KEYSET | AUTOGRAD_KEYSET | keySet("NestedTensor", "Functionalize");
const NESTED_KEYSET =
	keySet("AutogradNestedTensor", "NestedTensor") | FULL_BACKEND_MASK;

const has = (set: bigint, name: string) => {
	const k = keySet(name);
	return (set & k) === k;
};

export const ALIAS_KEYS = [
	"Autograd",
	"CompositeImplicitAutograd",
	"FuncTorchBatchedDecomposition",
	"CompositeImplicitAutogradNestedTensor",
	"CompositeExplicitAutograd",
	"CompositeExplicitAutogradNonFunctional",
] as const;
export const isAlias = (k: string) =>
	(ALIAS_KEYS as readonly string[]).includes(k);

/** getRuntimeDispatchKeySet(alias): the runtime keys an alias key fans out to. */
export const aliasRuntimeKeySet = (alias: string): bigint => {
	switch (alias) {
		case "Autograd":
			return AUTOGRAD_KEYSET | FULL_BACKEND_MASK;
		case "CompositeImplicitAutograd":
			return MATH_KEYSET;
		case "CompositeImplicitAutogradNestedTensor":
			return NESTED_KEYSET;
		case "CompositeExplicitAutograd":
			return BACKEND_KEYSET;
		case "CompositeExplicitAutogradNonFunctional":
			return NON_FUNCTIONAL_BACKEND_KEYSET;
		default:
			return keySet(alias);
	}
};

/** runtimeDispatchKeySetHas / isIncludedInAlias. */
export const inAlias = (k: string, alias: string): boolean => {
	if (k === "Undefined") return false;
	switch (alias) {
		case "Autograd":
			return has(AUTOGRAD_KEYSET, FUNCTIONALITIES[parseKey(k).f]);
		case "CompositeImplicitAutograd":
			return has(MATH_KEYSET, k);
		case "CompositeImplicitAutogradNestedTensor":
			return has(NESTED_KEYSET, k);
		case "CompositeExplicitAutograd":
			return k !== "NestedTensor" && has(BACKEND_KEYSET, k);
		case "CompositeExplicitAutogradNonFunctional":
			return k !== "NestedTensor" && has(NON_FUNCTIONAL_BACKEND_KEYSET, k);
		case "FuncTorchBatchedDecomposition":
			return k === "FuncTorchBatched";
		default:
			return k === alias;
	}
};

/** getBackendKeySetFromAutograd. */
const backendKeySetFromAutograd = (k: string): bigint => {
	if (k === "AutogradNestedTensor")
		return keySet("NestedTensor") | FULL_BACKEND_MASK;
	if (k === "AutogradOther") return AUTOGRADOTHER_BACKENDS;
	const own = [
		"CPU",
		"CUDA",
		"XLA",
		"Lazy",
		"Meta",
		"MPS",
		"HPU",
		"IPU",
		"XPU",
		"MAIA",
	];
	const m = /^Autograd(.+)$/.exec(k);
	if (m && (own.includes(m[1]) || m[1].startsWith("PrivateUse")))
		return keySet(m[1]);
	return 0n;
};

/** isBackendDispatchKey + getAutogradKeyFromBackend, for the autograd refresh after a backend impl. */
const autogradKeyForBackendKey = (k: string): string | undefined => {
	if (
		k === "Undefined" ||
		isAlias(k) ||
		k === "NestedTensor" ||
		!has(BACKEND_KEYSET, k)
	)
		return undefined;
	const b = parseKey(k).b;
	return b < 0 ? "AutogradOther" : autogradKeyFor(BACKENDS[b]);
};

// ---------- The operator's dispatch table (OperatorEntry) ----------

/** Where a table entry came from, as OperatorEntry::dumpComputedTable() labels it. */
export type EntrySource =
	| "kernel"
	| "default backend kernel"
	| "nested kernel"
	| "math kernel"
	| "ambiguous autogradother"
	| "autograd kernel"
	| "batched kernel"
	| "backend fallback"
	| "missing";

export interface Kernel {
	/** Short label for the kernel, e.g. "wrapper_CPU_add_Tensor". */
	name: string;
	/** Registration site, e.g. "RegisterCPU_0.cpp". */
	site: string;
	fallthrough?: boolean;
}

export interface Entry {
	slot: number;
	key: string;
	source: EntrySource;
	kernel?: Kernel;
}

/** A simulated OperatorEntry: kernels_ (newest first per key) and the computed dispatchTable_. */
export class OperatorTable {
	readonly kernels = new Map<string, Kernel[]>();
	readonly table: Entry[];
	nonFallthrough = FULL;
	readonly nonFallthroughPerBackend: bigint[] = new Array(NUM_BACKENDS).fill(
		FULL,
	);
	requiresBitsetPerBackend = false;

	readonly fallbacks: Map<string, Kernel>;

	constructor(fallbacks: Map<string, Kernel>) {
		this.fallbacks = fallbacks;
		// The constructor copies every backend fallback that already exists into the table.
		this.table = SLOT_KEYS.map((key, slot) => {
			const fb = fallbacks.get(key);
			return fb
				? { slot, key, source: "backend fallback", kernel: fb }
				: { slot, key, source: "missing" };
		});
		for (const e of this.table)
			if (e.kernel?.fallthrough) this.setFallthrough(e.key, true);
	}

	private front(key: string) {
		return this.kernels.get(key)?.[0];
	}

	private hasAnyKernelIn(ks: bigint) {
		for (const k of this.kernels.keys())
			if (!isAlias(k) && has(ks, k)) return true;
		return false;
	}

	/** computeDispatchTableEntryWithDebug. */
	compute(key: string): { source: EntrySource; kernel?: Kernel } {
		const direct = this.front(key);
		if (direct) return { source: "kernel", kernel: direct };
		const und = key === "Undefined";
		for (const alias of [
			"CompositeExplicitAutogradNonFunctional",
			"CompositeExplicitAutograd",
		]) {
			const k = this.front(alias);
			if (k && (und || inAlias(key, alias)))
				return { source: "default backend kernel", kernel: k };
		}
		const hasBackendKernel =
			this.hasAnyKernelIn(backendKeySetFromAutograd(key)) ||
			this.kernels.has("CompositeExplicitAutograd");
		const nested = this.front("CompositeImplicitAutogradNestedTensor");
		if (!und && nested && inAlias(key, "CompositeImplicitAutogradNestedTensor"))
			return { source: "nested kernel", kernel: nested };
		const math = this.front("CompositeImplicitAutograd");
		if (math && (und || inAlias(key, "CompositeImplicitAutograd"))) {
			if (
				key === "AutogradOther" &&
				this.hasAnyKernelIn(AUTOGRADOTHER_BACKENDS)
			)
				return { source: "ambiguous autogradother" };
			if (!hasBackendKernel) return { source: "math kernel", kernel: math };
		}
		const ag = this.front("Autograd");
		if (ag && inAlias(key, "Autograd"))
			return { source: "autograd kernel", kernel: ag };
		const batched = this.front("FuncTorchBatchedDecomposition");
		if (batched && inAlias(key, "FuncTorchBatchedDecomposition"))
			return { source: "batched kernel", kernel: batched };
		const fb = this.fallbacks.get(key);
		if (fb) return { source: "backend fallback", kernel: fb };
		return { source: "missing" };
	}

	/** DispatchKeyExtractor::setOperatorHasFallthroughForKey. */
	private setFallthrough(key: string, ft: boolean) {
		const apply = (ks: bigint) => (ft ? removeKey(ks, key) : ks | keySet(key));
		this.nonFallthrough = apply(this.nonFallthrough);
		const { f, b } = parseKey(key);
		if (isPerBackend(f)) {
			this.nonFallthroughPerBackend[b] = apply(
				this.nonFallthroughPerBackend[b],
			);
			const arr = this.nonFallthroughPerBackend;
			this.requiresBitsetPerBackend = arr.some(
				(v, i) => i > 0 && v !== arr[i - 1],
			);
		} else {
			for (let i = 0; i < NUM_BACKENDS; i++)
				this.nonFallthroughPerBackend[i] = apply(
					this.nonFallthroughPerBackend[i],
				);
		}
	}

	/** updateDispatchTableEntry_. Returns the slot if its entry changed. */
	private updateEntry(key: string): number | undefined {
		const slot = tableIndexOfKey(key);
		const before = this.table[slot];
		const next = this.compute(key);
		this.table[slot] = { slot, key, ...next };
		this.setFallthrough(key, !!next.kernel?.fallthrough);
		return before.source !== next.source || before.kernel !== next.kernel
			? slot
			: undefined;
	}

	/** updateDispatchTable_. Returns the slots whose entries changed. */
	private update(key: string): number[] {
		const changed: (number | undefined)[] = [];
		if (key === "Undefined")
			return [this.updateEntry(key)].filter((s) => s !== undefined);
		for (const k of runtimeKeys(aliasRuntimeKeySet(key)))
			changed.push(this.updateEntry(k));
		if (key.startsWith("Composite") && !key.includes("NestedTensor"))
			changed.push(this.updateEntry("Undefined"));
		const ag = autogradKeyForBackendKey(key);
		if (ag) changed.push(this.updateEntry(ag));
		return [...new Set(changed.filter((s): s is number => s !== undefined))];
	}

	/** OperatorEntry::registerKernel: newest registration goes to the front of kernels_[key]. */
	register(key: string, kernel: Kernel): number[] {
		this.kernels.set(key, [kernel, ...(this.kernels.get(key) ?? [])]);
		return this.update(key);
	}

	/** DispatchKeyExtractor::getDispatchKeySetUnboxed, given the OR of the argument key sets. */
	dispatchKeySet(
		argKeys: bigint,
		included = DEFAULT_INCLUDED,
		excluded = DEFAULT_EXCLUDED,
	) {
		if (!this.requiresBitsetPerBackend)
			return computeDispatchKeySet(
				argKeys,
				included,
				excluded,
				this.nonFallthrough,
			);
		const b = Math.max(0, highestBackend(minus(argKeys | included, excluded)));
		return computeDispatchKeySet(
			argKeys,
			included,
			excluded,
			this.nonFallthroughPerBackend[b],
		);
	}

	/** OperatorEntry::lookup. */
	lookup(ks: bigint) {
		return this.table[tableIndex(ks)];
	}
}

/** Hex of a key set as PyTorch's DispatchKeySet.raw_repr() would print it. */
export const hex = (ks: bigint) => `0x${ks.toString(16)}`;
