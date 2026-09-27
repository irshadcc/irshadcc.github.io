// A CuTe layout: a (possibly nested) shape paired with a congruent stride.
// It maps a coordinate to an offset, e.g. ((2,2),4):((1,8),2).

export type IntTuple = number | IntTuple[];

function product(t: IntTuple): number {
	return typeof t === "number" ? t : t.reduce<number>((acc, m) => acc * product(m), 1);
}

export function formatTuple(t: IntTuple): string {
	return typeof t === "number" ? `${t}` : `(${t.map(formatTuple).join(",")})`;
}

function congruent(a: IntTuple, b: IntTuple): boolean {
	if (typeof a === "number" || typeof b === "number") return typeof a === typeof b;
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

export class CuteLayout {
	readonly shape: IntTuple;
	readonly stride: IntTuple;

	/** Omit `stride` for the compact column-major layout of `shape`. */
	constructor(shape: IntTuple, stride: IntTuple = compactStride(shape)) {
		if (!congruent(shape, stride)) {
			throw new Error(`Shape ${formatTuple(shape)} and stride ${formatTuple(stride)} are not congruent`);
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

	toString(): string {
		return `${formatTuple(this.shape)}:${formatTuple(this.stride)}`;
	}
}
