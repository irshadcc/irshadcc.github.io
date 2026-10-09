// A CuTe layout: a (possibly nested) shape paired with a congruent stride.
// It maps a coordinate to an offset, e.g. ((2,2),4):((1,8),2).

export type IntTuple = number | IntTuple[];

function product(t: IntTuple): number {
	return typeof t === "number"
		? t
		: t.reduce<number>((acc, m) => acc * product(m), 1);
}

export function formatTuple(t: IntTuple): string {
	return typeof t === "number" ? `${t}` : `(${t.map(formatTuple).join(",")})`;
}

function congruent(a: IntTuple, b: IntTuple): boolean {
	if (typeof a === "number" || typeof b === "number")
		return typeof a === typeof b;
	return a.length === b.length && a.every((m, i) => congruent(m, b[i]));
}

/** Column-major (left-most mode fastest) compact strides for a shape. */
function compactStride(shape: IntTuple, start = 1): IntTuple {
	if (typeof shape === "number") return start;
	let s = start;
	return shape.map((m) => {
		const d = compactStride(m, s);
		s *= product(m);
		return d;
	});
}

/** Natural (nested) coordinate of a 1-D index, split colexicographically over a shape. */
function coordOf(idx: number, shape: IntTuple): IntTuple {
	if (typeof shape === "number") return idx;
	return shape.map((m) => {
		const n = product(m);
		const c = coordOf(idx % n, m);
		idx = Math.floor(idx / n);
		return c;
	});
}

/** Colexicographic 1-D index of a nested coordinate in a shape, the inverse of coordOf. */
function colexIndex(coord: IntTuple, shape: IntTuple): number {
	if (typeof shape === "number") return coord as number;
	let idx = 0;
	let scale = 1;
	shape.forEach((m, i) => {
		idx += colexIndex((coord as IntTuple[])[i], m) * scale;
		scale *= product(m);
	});
	return idx;
}

/** Offset of a 1-D coordinate, split colexicographically over a nested shape. */
function offsetOf(idx: number, shape: IntTuple, stride: IntTuple): number {
	if (typeof shape === "number") return idx * (stride as number);
	let off = 0;
	shape.forEach((m, i) => {
		const n = product(m);
		off += offsetOf(idx % n, m, (stride as IntTuple[])[i]);
		idx = Math.floor(idx / n);
	});
	return off;
}

function flatten(t: IntTuple): number[] {
	return typeof t === "number" ? [t] : t.flatMap(flatten);
}

/** A layout from a list of modes, e.g. [(4):(1), (2):(8)] -> (4,2):(1,8). */
export function makeLayout(modes: CuteLayout[]): CuteLayout {
	return new CuteLayout(
		modes.map((m) => m.shape),
		modes.map((m) => m.stride),
	);
}

/** Flat 1-D layout from parallel shape/stride lists; a single mode stays rank-1. */
function fromFlat(shape: number[], stride: number[]): CuteLayout {
	return shape.length === 1
		? new CuteLayout(shape[0], stride[0])
		: new CuteLayout(shape, stride);
}

// The layout algebra below follows pycute (python/pycute/layout.py in CUTLASS) mode for mode.

/** Flatten, drop size-1 modes and merge modes where one continues the other. */
function coalesce(layout: CuteLayout): CuteLayout {
	const shape = [1];
	const stride = [0];
	const flatStride = flatten(layout.stride);
	flatten(layout.shape).forEach((s, i) => {
		const d = flatStride[i];
		const last = shape.length - 1;
		if (s === 1) return;
		if (shape[last] === 1) {
			shape[last] = s;
			stride[last] = d;
		} else if (shape[last] * stride[last] === d) {
			shape[last] *= s;
		} else {
			shape.push(s);
			stride.push(d);
		}
	});
	return fromFlat(shape, stride);
}

/** The layout that fills the offsets in [0, cotarget) that `layout` does not reach. */
export function complement(layout: CuteLayout, cotarget: number): CuteLayout {
	const flatShape = flatten(layout.shape);
	const modes = flatten(layout.stride)
		.map((d, i) => [d, flatShape[i]])
		.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
	const shape: number[] = [];
	const stride: number[] = [];
	let current = 1;
	for (const [d, s] of modes) {
		if (d === 0 || s === 1) continue;
		if (d % current !== 0)
			throw new Error(`complement: ${layout} is not complementable`);
		shape.push(d / current);
		stride.push(current);
		current = s * d;
	}
	shape.push(Math.ceil(cotarget / current));
	stride.push(current);
	return coalesce(fromFlat(shape, stride));
}

/** a ∘ b: the layout c with c(i) = a(b(i)). A nested b is composed mode by mode. */
export function composition(a: CuteLayout, b: CuteLayout): CuteLayout {
	if (typeof b.shape !== "number") {
		return makeLayout(b.shape.map((_, i) => composition(a, b.mode(i))));
	}
	const bStride = b.stride as number;
	if (bStride === 0) return new CuteLayout(b.shape, 0);

	const flatA = coalesce(a);
	const aShape = flatten(flatA.shape);
	const aStride = flatten(flatA.stride);
	const shape: number[] = [];
	const stride: number[] = [];
	let restShape = b.shape;
	let restStride = bStride;
	for (let i = 0; i < aShape.length - 1; i++) {
		const s = aShape[i];
		if (s % restStride !== 0 && restStride % s !== 0) {
			throw new Error(`composition: ${a} and ${b} are not divisible`);
		}
		const newShape = Math.min(
			Math.max(1, Math.floor(s / restStride)),
			restShape,
		);
		if (newShape !== 1) {
			shape.push(newShape);
			stride.push(restStride * aStride[i]);
		}
		restShape = Math.floor(restShape / newShape);
		restStride = Math.ceil(restStride / s);
	}
	if (restShape !== 1 || shape.length === 0) {
		shape.push(restShape);
		stride.push(restStride * aStride[aStride.length - 1]);
	}
	return fromFlat(shape, stride);
}

/** One tile of a by-mode tiler: an integer n is the compact layout n:1. */
export type TileMode = number | CuteLayout;
/** A single layout, or one TileMode per leading mode of the divided layout. */
export type Tiler = TileMode | TileMode[];

function asLayout(t: TileMode): CuteLayout {
	return typeof t === "number" ? new CuteLayout(t, 1) : t;
}

/** logical_divide for one tile: (tile, rest), where rest = complement(tile, size(a)). */
export function logicalDivide(a: CuteLayout, tile: TileMode): CuteLayout {
	const t = asLayout(tile);
	return composition(a, makeLayout([t, complement(t, a.size)]));
}

export class CuteLayout {
	readonly shape: IntTuple;
	readonly stride: IntTuple;

	/** Omit `stride` for the compact column-major layout of `shape`. */
	constructor(shape: IntTuple, stride: IntTuple = compactStride(shape)) {
		if (!congruent(shape, stride)) {
			throw new Error(
				`Shape ${formatTuple(shape)} and stride ${formatTuple(stride)} are not congruent`,
			);
		}
		this.shape = shape;
		this.stride = stride;
	}

	get rank(): number {
		return typeof this.shape === "number" ? 1 : this.shape.length;
	}

	get size(): number {
		return product(this.shape);
	}

	/** The i-th top-level mode as its own layout. */
	mode(i: number): CuteLayout {
		if (typeof this.shape === "number") {
			if (i !== 0) throw new Error(`Mode ${i} out of range for rank-1 layout`);
			return this;
		}
		return new CuteLayout(this.shape[i], (this.stride as IntTuple[])[i]);
	}

	/** Offset for one 1-D coordinate per top-level mode, e.g. layout.offset(row, col). */
	offset(...coord: number[]): number {
		if (coord.length === 1) return offsetOf(coord[0], this.shape, this.stride);
		if (coord.length !== this.rank) {
			throw new Error(`Expected ${this.rank} coordinates, got ${coord.length}`);
		}
		return coord.reduce((off, c, i) => off + this.mode(i).offset(c), 0);
	}

	/** Nested coordinate of a 1-D index, e.g. index 5 in shape (2,(2,2)) is (1,(0,1)). */
	coord(idx: number): IntTuple {
		return coordOf(idx, this.shape);
	}

	/**
	 * zipped_divide: split into tiles, as ((tile...),(rest...)). Mode 0 indexes inside one tile,
	 * mode 1 picks the tile; layout modes the tiler does not cover go to the end of mode 1.
	 *   new CuteLayout([8, 8]).zippedDivide([2, 4])   // ((2,4),(4,2)):((1,8),(2,32))
	 */
	zippedDivide(tiler: Tiler): CuteLayout {
		if (!Array.isArray(tiler)) return logicalDivide(this, tiler);
		if (tiler.length > this.rank) {
			throw new Error(
				`Tiler has ${tiler.length} modes, layout ${this} only ${this.rank}`,
			);
		}
		const split = tiler.map((t, i) => logicalDivide(this.mode(i), t));
		const extra = Array.from({ length: this.rank - tiler.length }, (_, i) =>
			this.mode(tiler.length + i),
		);
		return makeLayout([
			makeLayout(split.map((m) => m.mode(0))),
			makeLayout([...split.map((m) => m.mode(1)), ...extra]),
		]);
	}

	/**
	 * Flat coordinate (one 1-D index per top-level mode) of the coordinate this layout maps to
	 * `idx`, as CuTe's get_flat_coord. Thread layout (2,4):(4,1) puts thread 5 at (1,1).
	 */
	flatCoord(idx: number): number[] {
		// idx2crd: each leaf of the coordinate is (idx / stride) % shape.
		const hier = (shape: IntTuple, stride: IntTuple): IntTuple =>
			typeof shape === "number"
				? stride === 0
					? 0 // a broadcast mode, e.g. the K mode (1):(0) of an MMA thread layout
					: Math.floor(idx / (stride as number)) % shape
				: shape.map((m, i) => hier(m, (stride as IntTuple[])[i]));
		return Array.from({ length: this.rank }, (_, i) => {
			const m = this.mode(i);
			return colexIndex(hier(m.shape, m.stride), m.shape);
		});
	}

	/**
	 * local_partition: the elements thread `index` owns when `thrLayout` is tiled over this layout.
	 * Divides by the thread layout's shape, then fixes the tile coordinate to the thread's, so
	 * the thread gets the same position in every tile. Returns the rest-mode layout and the
	 * offset of the thread's first element (CuTe moves the tensor's pointer by it).
	 *   new CuteLayout([8, 8], [1, 8]).localPartition(new CuteLayout([2, 4]), 5)
	 *   // { layout: (4,2):(2,32), offset: 17 }
	 */
	localPartition(
		thrLayout: CuteLayout,
		index: number,
	): { layout: CuteLayout; offset: number } {
		const tiler = (
			typeof thrLayout.shape === "number" ? [thrLayout.shape] : thrLayout.shape
		).map(product);
		const tiled = this.zippedDivide(tiler);
		const tile = tiled.mode(0);
		const offset = thrLayout
			.flatCoord(index)
			.reduce((off, c, i) => off + tile.mode(i).offset(c), 0);
		return { layout: tiled.mode(1), offset };
	}

	toString(): string {
		return `${formatTuple(this.shape)}:${formatTuple(this.stride)}`;
	}
}
