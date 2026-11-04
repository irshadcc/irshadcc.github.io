// Linear layouts over F2 (Zhou et al., "Linear Layouts", ASPLOS '26; arXiv 2505.23819), written to
// follow Triton's C++ class mlir::triton::LinearLayout at v3.8.0
// (include/triton/Tools/LinearLayout.h, lib/Tools/LinearLayout.cpp).
//
// A layout maps labelled input dims (register, lane, warp, or offset) to labelled output dims (the
// tensor's dim0, dim1, ...). It is stored as bases: bases[inDim][i] = L(inDim = 2^i, others 0),
// one integer per output dim. Every other value is the XOR of the bases whose input bit is set.
// Output dims are listed minor-to-major, as in Triton: when a layout is flattened to one integer,
// the first output dim supplies the low bits.
//
// Also here: Triton's constructors for blocked, NVIDIA mma (v2), dot-operand, slice and swizzled
// shared layouts (lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp), the conversion layout
// and its classification (minimalCvtLayout / ConvertLayoutOpToLLVM.cpp), the paper's warp-shuffle
// rounds (Section 5.4), a port of the segment choice in Triton's optimal swizzling
// (lib/Tools/GenericSwizzling.cpp) and a bank-conflict simulator used to check it.

export interface InDim {
	name: string;
	/** bases[i][k] = output k of L(name = 2^i). */
	bases: number[][];
}

export interface OutDim {
	name: string;
	size: number;
}

export interface LinearLayout {
	ins: InDim[];
	outs: OutDim[];
}

const log2 = (n: number) => {
	let k = 0;
	while (1 << k < n) k++;
	if (1 << k !== n) throw new Error(`${n} is not a power of two`);
	return k;
};
export const isPow2 = (n: number) => n > 0 && (n & (n - 1)) === 0;
const nextPow2 = (n: number) => {
	let p = 1;
	while (p <= n) p <<= 1;
	return p;
};
export const popcount = (x: number) => {
	let c = 0;
	for (let v = x; v; v &= v - 1) c++;
	return c;
};

/** Builds a layout; output sizes default to the next power of two above the largest basis entry. */
export function makeLayout(
	ins: [string, number[][]][],
	outNames: string[],
	outSizes?: number[],
): LinearLayout {
	const sizes =
		outSizes ??
		outNames.map((_, k) => {
			let m = 0;
			for (const [, bs] of ins) for (const b of bs) m = Math.max(m, b[k]);
			return m === 0 ? 1 : nextPow2(m);
		});
	return {
		ins: ins.map(([name, bases]) => ({
			name,
			bases: bases.map((b) => [...b]),
		})),
		outs: outNames.map((name, k) => ({ name, size: sizes[k] })),
	};
}

export const inDim = (L: LinearLayout, name: string) =>
	L.ins.find((d) => d.name === name);
export const inLog2 = (L: LinearLayout, name: string) =>
	inDim(L, name)?.bases.length ?? 0;
export const outIndex = (L: LinearLayout, name: string) =>
	L.outs.findIndex((d) => d.name === name);
export const outLog2 = (L: LinearLayout, name: string) => {
	const k = outIndex(L, name);
	return k < 0 ? 0 : log2(L.outs[k].size);
};
export const totalInLog2 = (L: LinearLayout) =>
	L.ins.reduce((s, d) => s + d.bases.length, 0);
export const totalOutLog2 = (L: LinearLayout) =>
	L.outs.reduce((s, d) => s + log2(d.size), 0);

/** L(ins): XOR of the bases whose input bit is set. Missing input dims are 0. */
export function apply(
	L: LinearLayout,
	input: Record<string, number>,
): number[] {
	const out = L.outs.map(() => 0);
	for (const d of L.ins) {
		const v = input[d.name] ?? 0;
		d.bases.forEach((b, i) => {
			if ((v >> i) & 1) for (let k = 0; k < out.length; k++) out[k] ^= b[k];
		});
	}
	return out;
}

/** Packs an output vector into one integer, first output dim in the low bits. */
export function flattenOut(L: LinearLayout, v: number[]): number {
	let x = 0;
	let shift = 0;
	L.outs.forEach((d, k) => {
		x |= v[k] << shift;
		shift += log2(d.size);
	});
	return x;
}

export function unflattenOut(L: LinearLayout, x: number): number[] {
	let shift = 0;
	return L.outs.map((d) => {
		const v = (x >> shift) & (d.size - 1);
		shift += log2(d.size);
		return v;
	});
}

/** The bases of one input dim as flattened integers. */
export const flatBases = (L: LinearLayout, name: string) =>
	(inDim(L, name)?.bases ?? []).map((b) => flattenOut(L, b));

/** Reorders the output dims (and the entries of every basis). */
export function transposeOuts(L: LinearLayout, order: string[]): LinearLayout {
	const idx = order.map((n) => outIndex(L, n));
	return {
		ins: L.ins.map((d) => ({
			name: d.name,
			bases: d.bases.map((b) => idx.map((k) => b[k])),
		})),
		outs: idx.map((k) => ({ ...L.outs[k] })),
	};
}

export const identity1D = (size: number, i: string, o: string) =>
	makeLayout(
		[[i, Array.from({ length: log2(size) }, (_, k) => [1 << k])]],
		[o],
		[size],
	);
export const zeros1D = (size: number, i: string, o: string) =>
	makeLayout([[i, Array.from({ length: log2(size) }, () => [0])]], [o], [1]);

/**
 * Triton's operator*: the inner layout acts on the low bits of every shared input dim and owns the
 * low bits of every shared output dim; the outer layout's bases are shifted above them. This is
 * the paper's product (Definition 4.3), a block-diagonal matrix.
 */
export function mul(inner: LinearLayout, outer: LinearLayout): LinearLayout {
	const sup = (a: string[], b: string[]) => {
		const r = [...a];
		for (const x of b) if (!r.includes(x)) r.push(x);
		return r;
	};
	const inNames = sup(
		inner.ins.map((d) => d.name),
		outer.ins.map((d) => d.name),
	);
	const outNames = sup(
		inner.outs.map((d) => d.name),
		outer.outs.map((d) => d.name),
	);
	const outLogs = outNames.map((o) => outLog2(inner, o) + outLog2(outer, o));
	const ins: [string, number[][]][] = inNames.map((name) => {
		const nIn = inLog2(inner, name);
		const nOut = inLog2(outer, name);
		const bases = Array.from({ length: nIn + nOut }, () =>
			outNames.map(() => 0),
		);
		outNames.forEach((o, k) => {
			const ki = outIndex(inner, o);
			const ko = outIndex(outer, o);
			if (ki >= 0)
				for (let i = 0; i < nIn; i++)
					bases[i][k] = inDim(inner, name)?.bases[i][ki] ?? 0;
			if (ko >= 0) {
				const shift = ki >= 0 ? outLog2(inner, o) : 0;
				for (let i = 0; i < nOut; i++)
					bases[nIn + i][k] = (inDim(outer, name)?.bases[i][ko] ?? 0) << shift;
			}
		});
		return [name, bases];
	});
	return makeLayout(
		ins,
		outNames,
		outLogs.map((l) => 1 << l),
	);
}

export const mulAll = (...ls: LinearLayout[]) =>
	ls.reduce((a, b) => mul(a, b), makeLayout([], []));

/** outer ∘ L. L's output dims must be outer's input dims. */
export function compose(L: LinearLayout, outer: LinearLayout): LinearLayout {
	return makeLayout(
		L.ins.map((d) => [
			d.name,
			d.bases.map((b) => {
				const input: Record<string, number> = {};
				L.outs.forEach((o, k) => {
					input[o.name] = b[k];
				});
				return apply(outer, input);
			}),
		]),
		outer.outs.map((o) => o.name),
		outer.outs.map((o) => o.size),
	);
}

/** Keeps only these input dims and output dims (others' inputs become 0, outputs are dropped). */
export function sublayout(
	L: LinearLayout,
	inNames: string[],
	outNames: string[],
): LinearLayout {
	const keep = L.outs
		.map((o, k) => (outNames.includes(o.name) ? k : -1))
		.filter((k) => k >= 0);
	return {
		ins: L.ins
			.filter((d) => inNames.includes(d.name))
			.map((d) => ({
				name: d.name,
				bases: d.bases.map((b) => keep.map((k) => b[k])),
			})),
		outs: keep.map((k) => ({ ...L.outs[k] })),
	};
}

/** Reduced row echelon form over F2 of rows given as integers; returns [rows, pivots]. */
function rref(rows: number[]): { rows: number[]; pivots: number[] } {
	const m = rows.filter((r) => r !== 0);
	const pivots: number[] = [];
	let r = 0;
	for (let bit = 0; bit < 31 && r < m.length; bit++) {
		const p = m.findIndex((x, i) => i >= r && (x >> bit) & 1);
		if (p < 0) continue;
		[m[r], m[p]] = [m[p], m[r]];
		for (let i = 0; i < m.length; i++)
			if (i !== r && (m[i] >> bit) & 1) m[i] ^= m[r];
		pivots.push(bit);
		r++;
	}
	return { rows: m.slice(0, r), pivots };
}

/** Dimension of span(vectors). */
export const rank = (vectors: number[]) => rref(vectors).rows.length;

/** Every element of span(vectors) (assumes they are independent or not; duplicates removed). */
export function span(vectors: number[]): number[] {
	const basis = rref(vectors).rows;
	const out: number[] = [];
	for (let s = 0; s < 1 << basis.length; s++) {
		let x = 0;
		basis.forEach((b, i) => {
			if ((s >> i) & 1) x ^= b;
		});
		out.push(x);
	}
	return out.sort((a, b) => a - b);
}

/** Unit vectors completing span(basis) to F2^dim (Triton's complementBasis). */
export function complementBasis(basis: number[], dim: number): number[] {
	const { pivots } = rref(basis);
	const out: number[] = [];
	for (let i = 0; i < dim; i++) if (!pivots.includes(i)) out.push(1 << i);
	return out;
}

/**
 * Solves sum_i y_i cols[i] = target over F2 with free variables set to 0 (the minimum-weight
 * choice the paper describes); returns the bitmask y or null if there is no solution.
 */
export function solve(cols: number[], target: number): number | null {
	// Augment each column with its own index bit so the elimination records the combination.
	const n = cols.length;
	const rows = cols.map((c, i) => ({ v: c, comb: 1 << i }));
	const used: { v: number; comb: number; bit: number }[] = [];
	for (const r of rows) {
		let { v, comb } = r;
		for (const u of used)
			if ((v >> u.bit) & 1) {
				v ^= u.v;
				comb ^= u.comb;
			}
		if (v === 0) continue;
		const bit = 31 - Math.clz32(v & -v);
		// Keep the earlier rows reduced on this pivot.
		for (const u of used)
			if ((u.v >> bit) & 1) {
				u.v ^= v;
				u.comb ^= comb;
			}
		used.push({ v, comb, bit });
	}
	let t = target;
	let y = 0;
	for (const u of used)
		if ((t >> u.bit) & 1) {
			t ^= u.v;
			y ^= u.comb;
		}
	return t === 0 && n >= 0 ? y : null;
}

/**
 * Triton's invertAndCompose: returns C with L(x) = outer(C(x)), so C maps L's inputs to outer's
 * inputs. Input dims whose sublayouts are equal in both are mapped as the identity first; the rest
 * is a minimum-weight least-squares solve. For a conversion src -> dst Triton calls
 * dst.invertAndCompose(src): for each destination register, where the source keeps that element.
 */
export function invertAndCompose(
	L: LinearLayout,
	outer: LinearLayout,
): LinearLayout {
	const outNames = L.outs.map((o) => o.name);
	const A = transposeOuts(outer, outNames);
	const flatA = (name: string) => flatBases(A, name);
	const flatL = (name: string) => flatBases(L, name);
	const identity = L.ins
		.map((d) => d.name)
		.filter(
			(n) =>
				inDim(A, n) && JSON.stringify(flatA(n)) === JSON.stringify(flatL(n)),
		);
	const aIns = A.ins.filter((d) => !identity.includes(d.name));
	const cols: number[] = [];
	const owner: [string, number][] = [];
	for (const d of aIns)
		d.bases.forEach((_, i) => {
			cols.push(flattenOut(A, d.bases[i]));
			owner.push([d.name, i]);
		});
	const resultOuts = A.ins.map((d) => d.name);
	const ins: [string, number[][]][] = L.ins.map((d) => [
		d.name,
		d.bases.map((b, i) => {
			const v = resultOuts.map(() => 0);
			if (identity.includes(d.name)) {
				v[resultOuts.indexOf(d.name)] = 1 << i;
				return v;
			}
			const y = solve(cols, flattenOut(L, b));
			if (y === null)
				throw new Error("invertAndCompose: outer does not cover L");
			owner.forEach(([name, bit], j) => {
				if ((y >> j) & 1) v[resultOuts.indexOf(name)] ^= 1 << bit;
			});
			return v;
		}),
	]);
	return makeLayout(
		ins,
		resultOuts,
		A.ins.map((d) => 1 << d.bases.length),
	);
}

/** Inverse of an invertible layout: outputs become inputs. */
export function invert(L: LinearLayout): LinearLayout {
	const id = makeLayout(
		L.outs.map((o) => [
			o.name,
			Array.from({ length: log2(o.size) }, (_, i) =>
				L.outs.map((p) => (p.name === o.name ? 1 << i : 0)),
			),
		]),
		L.outs.map((o) => o.name),
		L.outs.map((o) => o.size),
	);
	return invertAndCompose(id, L);
}

/** True when L is the identity on these dims and does not mix them with the others. */
export function isTrivialOver(L: LinearLayout, dims: string[]): boolean {
	for (const d of L.ins) {
		const mine = dims.includes(d.name);
		for (let i = 0; i < d.bases.length; i++) {
			for (let k = 0; k < L.outs.length; k++) {
				const o = L.outs[k].name;
				const want = mine && o === d.name ? 1 << i : 0;
				if ((mine || dims.includes(o)) && d.bases[i][k] !== want) return false;
			}
		}
	}
	return true;
}

/** Triton's minimalCvtLayout: dst.invertAndCompose(src), then quotient by block, warp, lane... */
export function conversionLayout(
	src: LinearLayout,
	dst: LinearLayout,
): LinearLayout {
	let comp = invertAndCompose(dst, src);
	const s = src.ins.map((d) => d.name);
	const t = dst.ins.map((d) => d.name);
	for (let i = 0; i < Math.min(s.length, t.length); i++) {
		const a = s[s.length - 1 - i];
		if (a !== t[t.length - 1 - i] || !isTrivialOver(comp, [a])) break;
		const rest = comp.ins.map((d) => d.name).filter((n) => n !== a);
		comp = sublayout(comp, rest, rest);
	}
	return comp;
}

export type ConversionKind = "none" | "register" | "shuffle" | "shared";

/**
 * Which lowering ConvertLayoutOpToLLVM picks, from the input dims left in the conversion: warp or
 * block -> shared memory; lane -> warp shuffles (Triton falls back to shared memory when its
 * decomposition needs two or more register-lane transpositions, which this does not model);
 * register -> a permutation of the thread's own registers; nothing -> no-op.
 */
export function conversionKind(cvt: LinearLayout): ConversionKind {
	const n = cvt.ins.map((d) => d.name);
	if (n.includes("warp") || n.includes("block")) return "shared";
	if (n.includes("lane")) return "shuffle";
	if (n.includes("register")) return "register";
	return "none";
}

/**
 * Largest power of two V such that registers 0..V-1 hold consecutive elements of the first output
 * dim (Triton's getNumConsecutiveInOut on the layout as given). For vector widths, pass a layout
 * whose first output dim is the fastest-moving tensor dim.
 */
export function numConsecutive(L: LinearLayout): number {
	const first = L.ins[0];
	if (!first || L.outs.length === 0) return 1;
	let consec = 0;
	while (
		consec < first.bases.length &&
		first.bases[consec][0] === 1 << consec &&
		first.bases[consec].slice(1).every((x) => x === 0)
	)
		consec++;
	let other = 0;
	for (const d of L.ins)
		d.bases.forEach((b, i) => {
			if (d !== first || i >= consec) other |= b[0];
		});
	const tz = other ? 31 - Math.clz32(other & -other) : 31;
	return 1 << Math.min(consec, tz);
}

/** Input bits whose basis is zero: changing them does not change the element (broadcast copies). */
export function zeroBases(L: LinearLayout): Record<string, number[]> {
	const r: Record<string, number[]> = {};
	for (const d of L.ins)
		r[d.name] = d.bases
			.map((b, i) => (b.every((x) => x === 0) ? i : -1))
			.filter((i) => i >= 0);
	return r;
}

/** Definition 4.10: every basis has at most one set bit and no two non-zero bases repeat. */
export function isDistributedForm(L: LinearLayout): boolean {
	const seen = new Set<number>();
	for (const d of L.ins)
		for (const b of d.bases) {
			const x = flattenOut(L, b);
			if (x === 0) continue;
			if (popcount(x) !== 1 || seen.has(x)) return false;
			seen.add(x);
		}
	return true;
}

// ---------------------------------------------------------------------------------------------
// Triton's layouts as linear layouts (LinearLayoutConversions.cpp at v3.8.0)
// ---------------------------------------------------------------------------------------------

export const dimNames = (rank: number) =>
	Array.from({ length: rank }, (_, i) => `dim${i}`);

/** identityStandardND: identity on `shape`, most-minor dim (order[0]) first. */
export function identityND(
	i: string,
	shape: number[],
	order: number[],
): LinearLayout {
	const names = dimNames(shape.length);
	return mulAll(...order.map((d) => identity1D(shape[d], i, names[d])));
}

/**
 * ensureLayoutNotSmallerThan + ensureLayoutNotLargerThan, as combineCtaCgaWithShape applies them
 * for one CTA: a tile smaller than the tensor gets more registers (repeats); a tile larger than the
 * tensor has its out-of-range bases zeroed (broadcast), and register bases that became zero are
 * dropped.
 */
export function fitToShape(tile: LinearLayout, shape: number[]): LinearLayout {
	const names = dimNames(shape.length);
	const first = tile.ins[0].name;
	let L = tile;
	for (const o of tile.outs) {
		const want = shape[names.indexOf(o.name)];
		if (o.size < want) L = mul(L, identity1D(want / o.size, first, o.name));
	}
	const sizes = L.outs.map((o) =>
		Math.min(o.size, shape[names.indexOf(o.name)]),
	);
	const ins: [string, number[][]][] = L.ins.map((d) => {
		const bs: number[][] = [];
		for (const b of d.bases) {
			const wasZero = b.every((x) => x === 0);
			const nb = b.map((x, k) => (x >= sizes[k] ? 0 : x));
			const isZero = nb.every((x) => x === 0);
			if (d.name !== "register" || wasZero || !isZero) bs.push(nb);
		}
		return [d.name, bs];
	});
	return transposeOuts(
		makeLayout(
			ins,
			L.outs.map((o) => o.name),
			sizes,
		),
		names,
	);
}

export interface BlockedParams {
	sizePerThread: number[];
	threadsPerWarp: number[];
	warpsPerCTA: number[];
	/** Fastest-moving dim first, e.g. [1, 0] for row-major. */
	order: number[];
}

/** BlockedEncodingAttr::toLinearLayout for one CTA. */
export function blocked(p: BlockedParams, shape: number[]): LinearLayout {
	const tile = mulAll(
		identityND("register", p.sizePerThread, p.order),
		identityND("lane", p.threadsPerWarp, p.order),
		identityND("warp", p.warpsPerCTA, p.order),
	);
	return fitToShape(tile, shape);
}

/** nvidiaMmaTile for a 2-D tile [m, n], inner dim 1. */
function mmaTile(m: number, n: number, kWidth: number): LinearLayout {
	return mulAll(
		identityND("register", [1, 1], [1, 0]),
		identity1D(kWidth, "register", "dim1"),
		identity1D(4, "lane", "dim1"),
		identity1D(8, "lane", "dim0"),
		identity1D(m / 8, "register", "dim0"),
		identity1D(n / (kWidth * 4), "register", "dim1"),
	);
}

/** NvidiaMmaEncodingAttr (Ampere, version 2, instrShape [16, 8]) accumulator layout. */
export function mmaV2(warpsPerCTA: number[], shape: number[]): LinearLayout {
	const tile = mmaTile(16, 8, 2);
	const warps = transposeOuts(
		identityND("warp", warpsPerCTA, [1, 0]),
		tile.outs.map((o) => o.name),
	);
	return fitToShape(mul(tile, warps), shape);
}

/** DotOperandEncodingAttr with an Ampere mma parent: opIdx 0 (A, [16, 8 kWidth] tile) or 1 (B). */
export function mmaV2Operand(
	opIdx: 0 | 1,
	kWidth: number,
	warpsPerCTA: number[],
	shape: number[],
): LinearLayout {
	const isA = opIdx === 0;
	// Operand B is K-major: its tile is [8 kWidth, 8] with order [0, 1].
	const tile = isA
		? mmaTile(16, kWidth * 8, kWidth)
		: mulAll(
				identityND("register", [1, 1], [0, 1]),
				identity1D(kWidth, "register", "dim0"),
				identity1D(4, "lane", "dim0"),
				identity1D(8, "lane", "dim1"),
				identity1D(1, "register", "dim1"),
				identity1D((kWidth * 8) / (kWidth * 4), "register", "dim0"),
			);
	const kDim = isA ? 1 : 0;
	const names = dimNames(2);
	const warps = mulAll(
		...[1, 0].map((d) =>
			d === kDim
				? zeros1D(warpsPerCTA[d], "warp", names[d])
				: identity1D(warpsPerCTA[d], "warp", names[d]),
		),
	);
	return fitToShape(
		mul(
			tile,
			transposeOuts(
				warps,
				tile.outs.map((o) => o.name),
			),
		),
		shape,
	);
}

/** SliceEncodingAttr: drop output dim `dim` of the parent and its now-zero register bases. */
export function slice(
	parent: (shape: number[]) => LinearLayout,
	shape: number[],
	dim: number,
): LinearLayout {
	const parentShape = [...shape];
	parentShape.splice(dim, 0, 1);
	const P = parent(parentShape);
	const keepNames = P.outs.map((o) => o.name).filter((n) => n !== `dim${dim}`);
	const sub = sublayout(
		P,
		P.ins.map((d) => d.name),
		keepNames,
	);
	const rename = (n: string) => {
		const k = Number(n.slice(3));
		return `dim${k > dim ? k - 1 : k}`;
	};
	return makeLayout(
		sub.ins.map((d) => [
			d.name,
			d.name === "register"
				? d.bases.filter((b) => b.some((x) => x !== 0))
				: d.bases,
		]),
		sub.outs.map((o) => rename(o.name)),
		sub.outs.map((o) => o.size),
	);
}

/**
 * SwizzledSharedEncodingAttr for a 2-D [rows, cols] tile, order [1, 0]: offset -> (row, col).
 * Column bases are the identity; row 2^k's basis also XORs vec * ((2^k / perPhase) % maxPhase)
 * into the column. With vec = 1, perPhase = 1, maxPhase = 1 this is the unswizzled layout.
 */
export function swizzledShared(
	vec: number,
	perPhase: number,
	maxPhase: number,
	shape: number[],
): LinearLayout {
	const [rows, cols] = shape;
	const bases: number[][] = [];
	for (let c = 1; c < cols; c *= 2) bases.push([0, c]);
	for (let r = 1; r < rows; r *= 2)
		bases.push([r, (vec * (Math.floor(r / perPhase) % maxPhase)) % cols]);
	return makeLayout([["offset", bases]], ["dim0", "dim1"], [rows, cols]);
}

/** Paper's Definition 4.11 written forwards: element (i, j) -> offset (columns only). */
export function mmaSwizzleColumn(
	i: number,
	j: number,
	vec: number,
	perPhase: number,
	maxPhase: number,
) {
	return (
		((Math.floor(i / perPhase) % maxPhase) ^ Math.floor(j / vec)) * vec +
		(j % vec)
	);
}

// ---------------------------------------------------------------------------------------------
// Warp shuffles (paper, Section 5.4, "Intra-warp Data Exchange")
// ---------------------------------------------------------------------------------------------

export interface ShuffleMove {
	/** Flattened tensor element. */
	element: number;
	srcLane: number;
	srcReg: number;
	dstLane: number;
	dstReg: number;
}

export interface ShufflePlan {
	V: number[];
	I: number[];
	E: number[];
	F: number[];
	G: number[];
	R: number[];
	/** rounds[r] = the moves of round r; each lane sends and receives 2^|V| elements. */
	rounds: ShuffleMove[][];
}

/**
 * The paper's construction for a conversion inside one warp with no broadcasting: V = A_Reg ∩
 * B_Reg (capped so a shuffle moves at most 32 bits), I = A_Thr ∩ B_Thr, E = A_Thr \ I,
 * F = B_Thr \ I, G = {e_i ⊕ f_i}; R completes V ∪ I ∪ G to a basis and round r moves the
 * elements R(r) ⊕ span(V ∪ I ∪ G).
 */
export function shufflePlan(
	A: LinearLayout,
	B: LinearLayout,
	elemBits = 32,
): ShufflePlan {
	const d = totalOutLog2(A);
	const aReg = flatBases(A, "register");
	const bReg = flatBases(B, "register");
	const aThr = flatBases(A, "lane");
	const bThr = flatBases(B, "lane");
	const maxV = Math.max(0, Math.log2(32 / elemBits));
	const V = aReg.filter((x) => bReg.includes(x)).slice(0, maxV);
	const I = aThr.filter((x) => bThr.includes(x));
	const E = aThr.filter((x) => !I.includes(x)).sort((a, b) => a - b);
	const F = bThr.filter((x) => !I.includes(x)).sort((a, b) => a - b);
	const G = E.map((e, i) => e ^ F[i]);
	const R = complementBasis([...V, ...I, ...G], d);
	const base = span([...V, ...I, ...G]);
	const invA = invertAndCompose(identityOuts(A), A);
	const invB = invertAndCompose(identityOuts(B), B);
	const where = (inv: LinearLayout, x: number) => {
		const hw = apply(
			inv,
			Object.fromEntries(A.outs.map((o, k) => [o.name, unflattenOut(A, x)[k]])),
		);
		const names = inv.outs.map((o) => o.name);
		return {
			lane: hw[names.indexOf("lane")] ?? 0,
			reg: hw[names.indexOf("register")] ?? 0,
		};
	};
	const rounds: ShuffleMove[][] = [];
	for (let r = 0; r < 1 << R.length; r++) {
		let off = 0;
		R.forEach((b, i) => {
			if ((r >> i) & 1) off ^= b;
		});
		rounds.push(
			base.map((x) => {
				const element = x ^ off;
				const s = where(invA, element);
				const t = where(invB, element);
				return {
					element,
					srcLane: s.lane,
					srcReg: s.reg,
					dstLane: t.lane,
					dstReg: t.reg,
				};
			}),
		);
	}
	return { V, I, E, F, G, R, rounds };
}

/** The identity on L's output dims, as a layout (used to invert L with invertAndCompose). */
export function identityOuts(L: LinearLayout): LinearLayout {
	return makeLayout(
		L.outs.map((o) => [
			o.name,
			Array.from({ length: log2(o.size) }, (_, i) =>
				L.outs.map((p) => (p.name === o.name ? 1 << i : 0)),
			),
		]),
		L.outs.map((o) => o.name),
		L.outs.map((o) => o.size),
	);
}

// ---------------------------------------------------------------------------------------------
// Shared memory: vectorisation, optimal swizzling, bank conflicts
// ---------------------------------------------------------------------------------------------

/** Flattens a layout's outputs to one dim "x" (first output dim in the low bits). */
export function flatten(L: LinearLayout): LinearLayout {
	return makeLayout(
		L.ins.map((d) => [d.name, d.bases.map((b) => [flattenOut(L, b)])]),
		["x"],
		[1 << totalOutLog2(L)],
	);
}

/**
 * Port of computeSegment (GenericSwizzling.cpp): choose the segment bases of the shared layout.
 * First unit vectors no lane of either access touches (the complement C), then e_i ⊕ f_i for the
 * lanes that differ (H), then, if still short, plain lane bases (bank conflicts).
 */
export function computeSegment(
	bankSrc: number[],
	bankDst: number[],
	dim: number,
	lenSegment: number,
): number[] {
	const setSrc = new Set(bankSrc.filter((x) => x !== 0));
	const setDst = new Set(bankDst.filter((x) => x !== 0));
	const segment: number[] = [];
	for (let b = 0; b < dim; b++)
		if (!setSrc.has(1 << b) && !setDst.has(1 << b)) segment.push(1 << b);
	if (segment.length >= lenSegment) return segment.slice(0, lenSegment);
	let A = [...setSrc].filter((v) => !setDst.has(v));
	let B = [...setDst].filter((v) => !setSrc.has(v));
	if (A.length > B.length) [A, B] = [B, A];
	A.sort((a, b) => a - b);
	B.sort((a, b) => a - b);
	const logConflicts = Math.min(
		Math.max(0, lenSegment - A.length - segment.length),
		A.length,
	);
	for (let i = logConflicts; i < A.length; i++) segment.push(A[i] ^ B[i]);
	segment.push(...A.slice(0, logConflicts), ...B.slice(0, logConflicts));
	return segment.slice(0, lenSegment);
}

export interface SwizzleResult {
	/** Shared layout: inputs vector, bank, segment; output the flattened tensor. */
	smem: LinearLayout;
	vec: number[];
	bank: number[];
	segment: number[];
}

/**
 * The optimal shared layout for a store in layout src and a load in layout dst (one warp, 32
 * banks of 4 bytes): vectorise over the registers both share (up to 128 bits), take the lanes of
 * one wavefront, choose segments with computeSegment and banks as the complement. Skips Triton's
 * extra step that pads sub-32-bit vectors with warp bases.
 */
export function optimalSwizzle(
	src: LinearLayout,
	dst: LinearLayout,
	elemBits: number,
): SwizzleResult {
	const S = flatten(src);
	const D = flatten(dst);
	const dim = totalOutLog2(S);
	const regS = flatBases(S, "register");
	const regD = flatBases(D, "register");
	const vec = regS
		.filter((x) => x !== 0 && regD.includes(x))
		.slice(0, Math.log2(128 / elemBits));
	const vecBits = (1 << vec.length) * elemBits;
	const log2Vec = Math.log2(Math.max(1, vecBits / 32));
	const laneTile = (L: LinearLayout) => {
		const lanes = flatBases(L, "lane");
		const phase = Math.max(0, log2Vec + lanes.length - 5);
		return lanes.slice(0, lanes.length - phase);
	};
	const lenBank = Math.min(
		Math.log2((32 * 32) / ((1 << vec.length) * elemBits)),
		dim - vec.length,
	);
	const lenSeg = dim - lenBank - vec.length;
	const segment = computeSegment(
		[...vec, ...laneTile(S)],
		[...vec, ...laneTile(D)],
		dim,
		lenSeg,
	);
	const bank = complementBasis([...vec, ...segment], dim);
	const smem = makeLayout(
		[
			["vector", vec.map((x) => [x])],
			["bank", bank.map((x) => [x])],
			["segment", segment.map((x) => [x])],
		],
		["x"],
		[1 << dim],
	);
	return { smem, vec, bank, segment };
}

/** A shared layout given as (vector, bank, segment) or offset inputs, reduced to offset -> x. */
export function asOffsetLayout(smem: LinearLayout): LinearLayout {
	const F = flatten(smem);
	const bases: number[][] = [];
	for (const d of F.ins) for (const b of d.bases) bases.push(b);
	return makeLayout([["offset", bases]], ["x"], [F.outs[0].size]);
}

export interface Access {
	/** Lane -> list of element offsets (in elements) this lane touches in the instruction. */
	offsets: number[][];
	wavefronts: number;
}

export interface AccessReport {
	vec: number;
	instructions: Access[];
	wavefronts: number;
	/** Wavefronts a conflict-free access of the same width would take. */
	ideal: number;
}

/**
 * Simulates the shared-memory instructions a warp issues to store (or load) a distributed layout L
 * through shared layout smem (offset -> element), with 32 banks of 4 bytes. One instruction moves,
 * per lane, the elements x ⊕ span(vec), which must sit at consecutive offsets; registers are
 * regrouped to make that so (the paper's register permutation P_Reg). Without `vec`, the vector is
 * grown greedily: the elements at offsets 1, 2, 4, ... while each is held in the lane's registers
 * (at most 128 bits), which is what lowering a store to a fixed shared layout does. An access of
 * W > 4 bytes per lane is split into 128-byte phases of 128 / W lanes; within a phase the
 * wavefronts are the most distinct 4-byte words any one bank serves.
 */
export function simulateAccess(
	L: LinearLayout,
	smem: LinearLayout,
	elemBits: number,
	vec?: number[],
): AccessReport {
	const Lf = flatten(L);
	const M = asOffsetLayout(smem);
	const toOff = invertAndCompose(identityOuts(M), M);
	const offOf = (x: number) => apply(toOff, { x })[0];
	const regs = flatBases(Lf, "register").filter((x) => x !== 0);
	const lanes = 1 << inLog2(Lf, "lane");
	const maxV = Math.log2(128 / elemBits);
	const laneOffs = flatBases(Lf, "lane").map(offOf);
	let V = vec;
	if (!V) {
		// Grow while the element at offset 2^k is in the lane's registers and no lane basis
		// touches the low k + 1 offset bits (else lanes would see the vector in different orders).
		V = [];
		while (V.length < maxV) {
			const x = apply(M, { offset: 1 << V.length })[0];
			const low = (2 << V.length) - 1;
			if (
				rank([...regs, x]) !== rank(regs) ||
				laneOffs.some((o) => (o & low) !== 0)
			)
				break;
			V.push(x);
		}
	}
	const n = 1 << V.length;
	const bytes = (n * elemBits) / 8;
	const perPhase = bytes > 4 ? Math.max(1, 128 / bytes) : lanes;
	// Registers of lane 0, grouped into cosets of span(V).
	const nRegs = 1 << inLog2(Lf, "register");
	const seen = new Set<number>();
	const groups: number[][] = [];
	const vecElems = span(V);
	for (let r = 0; r < nRegs; r++) {
		const x0 = apply(Lf, { register: r })[0];
		if (seen.has(x0)) continue;
		// Start the vector at the element with the lowest offset in its coset.
		let x = x0;
		for (const y of vecElems) if (offOf(x0 ^ y) < offOf(x)) x = x0 ^ y;
		// Elements of the vector in offset order: coefficient c on V's bases.
		const g = Array.from({ length: n }, (_, c) => {
			let y = x;
			V.forEach((b, i) => {
				if ((c >> i) & 1) y ^= b;
			});
			return y;
		});
		for (const y of vecElems) seen.add(x ^ y);
		groups.push(g);
	}
	const instructions: Access[] = [];
	for (const g of groups) {
		const offsets: number[][] = [];
		for (let lane = 0; lane < lanes; lane++) {
			const laneBits = apply(Lf, { lane })[0];
			const row = g.map((x) => offOf(x ^ laneBits));
			if (row.some((o, c) => o !== row[0] + c))
				throw new Error(
					"simulateAccess: vector is not contiguous in shared memory",
				);
			offsets.push(row);
		}
		let wf = 0;
		for (let p = 0; p < lanes; p += perPhase) {
			const words = new Map<number, Set<number>>();
			for (let lane = p; lane < Math.min(lanes, p + perPhase); lane++)
				for (const o of offsets[lane]) {
					const word = Math.floor((o * elemBits) / 32);
					const bank = word % 32;
					if (!words.has(bank)) words.set(bank, new Set());
					words.get(bank)?.add(word);
				}
			wf += Math.max(...[...words.values()].map((s) => s.size));
		}
		instructions.push({ offsets, wavefronts: wf });
	}
	const phases = Math.ceil(lanes / perPhase);
	return {
		vec: n,
		instructions,
		wavefronts: instructions.reduce((s, a) => s + a.wavefronts, 0),
		ideal: instructions.length * phases,
	};
}

/** Formats a basis vector like Triton's toString: "(0, 1)". */
export const fmtBasis = (b: number[]) =>
	b.length === 1 ? String(b[0]) : `(${b.join(", ")})`;

/** Triton-style text: "register = [(0, 1), (1, 0)]  lane = [...]". */
export function describe(L: LinearLayout): string {
	return L.ins
		.map((d) => `${d.name} = [${d.bases.map(fmtBasis).join(", ")}]`)
		.join("\n");
}
