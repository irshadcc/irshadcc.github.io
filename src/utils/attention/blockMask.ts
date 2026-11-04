// FlexAttention's BlockMask, computed the way torch/nn/attention/flex_attention.py builds it
// (create_mask -> _convert_mask_to_block_mask -> _dense_to_ordered), plus the order in which the
// Triton template visits blocks (flex_attention.py.jinja: a partial-block loop, then a full-block
// loop, stepping with get_offset_for_next_block from common.py.jinja). Checked against
// create_block_mask from torch 2.13.0 (git cf30153c); see the post's verification notes.

export type MaskPreset = "causal" | "sliding" | "document" | "prefix";

export interface MaskSpec {
	preset: MaskPreset;
	/** sliding: q - kv < window. */
	window?: number;
	/** document: start index of each document after the first. */
	docStarts?: number[];
	/** prefix: keys below this index are visible to every query. */
	prefix?: number;
}

export type BlockKind = "empty" | "partial" | "full";

export interface BlockMaskData {
	qLen: number;
	kvLen: number;
	qBlock: number;
	kvBlock: number;
	/** Number of query blocks (rows) and key/value blocks (columns). */
	rows: number;
	cols: number;
	/** Element mask, [q][kv]. */
	dense: boolean[][];
	/** Block classification, [row][col]. */
	kind: BlockKind[][];
	kvNumBlocks: number[];
	kvIndices: number[][];
	fullKvNumBlocks: number[];
	fullKvIndices: number[][];
	/** Transposed (per key/value block), used by the backward pass. */
	qNumBlocks: number[];
	qIndices: number[][];
	fullQNumBlocks: number[];
	fullQIndices: number[][];
}

export function maskFn(spec: MaskSpec): (q: number, kv: number) => boolean {
	switch (spec.preset) {
		case "causal":
			return (q, kv) => q >= kv;
		case "sliding": {
			const w = spec.window ?? 4;
			return (q, kv) => q >= kv && q - kv < w;
		}
		case "document": {
			const starts = spec.docStarts ?? [];
			const doc = (i: number) => starts.filter((s) => i >= s).length;
			return (q, kv) => q >= kv && doc(q) === doc(kv);
		}
		case "prefix": {
			const p = spec.prefix ?? 4;
			return (q, kv) => kv < p || q >= kv;
		}
	}
}

/** Window, prefix or document starts that suit a sequence of `len` tokens. */
export function defaultSpec(preset: MaskPreset, len: number): MaskSpec {
	switch (preset) {
		case "sliding":
			return { preset, window: Math.round(len / 2) + 1 };
		case "document":
			return {
				preset,
				docStarts: [Math.round(len * 0.3), Math.round(len * 0.7)],
			};
		case "prefix":
			return { preset, prefix: Math.round(len / 4) + 2 };
		default:
			return { preset };
	}
}

export const presetLabel: Record<MaskPreset, string> = {
	causal: "causal",
	sliding: "sliding window",
	document: "document (causal)",
	prefix: "prefix LM",
};

/** The mask_mod a reader would write for this preset, as Python. */
export function maskSource(spec: MaskSpec): string {
	switch (spec.preset) {
		case "causal":
			return "q_idx >= kv_idx";
		case "sliding":
			return `(q_idx >= kv_idx) & (q_idx - kv_idx < ${spec.window ?? 4})`;
		case "document":
			return "(q_idx >= kv_idx) & (doc_id[q_idx] == doc_id[kv_idx])";
		case "prefix":
			return `(kv_idx < ${spec.prefix ?? 4}) | (q_idx >= kv_idx)`;
	}
}

// _dense_to_ordered: argsort(dense, descending=True, stable=True). Set blocks come first in
// ascending order, then the unset ones, also ascending. Only the first num_blocks are read.
function ordered(row: boolean[]): { num: number; idx: number[] } {
	const on: number[] = [];
	const off: number[] = [];
	row.forEach((b, i) => (b ? on : off).push(i));
	return { num: on.length, idx: [...on, ...off] };
}

export function buildBlockMask(
	spec: MaskSpec,
	qLen: number,
	kvLen: number,
	qBlock: number,
	kvBlock: number,
): BlockMaskData {
	const f = maskFn(spec);
	const dense = Array.from({ length: qLen }, (_, q) =>
		Array.from({ length: kvLen }, (_, kv) => f(q, kv)),
	);
	const rows = Math.ceil(qLen / qBlock);
	const cols = Math.ceil(kvLen / kvBlock);
	// The mask is padded with False up to a multiple of the block size, so a block that hangs
	// over the end of the sequence can be partial but never full.
	const kind: BlockKind[][] = [];
	for (let r = 0; r < rows; r++) {
		const row: BlockKind[] = [];
		for (let c = 0; c < cols; c++) {
			let n = 0;
			for (let q = r * qBlock; q < Math.min(qLen, (r + 1) * qBlock); q++)
				for (
					let kv = c * kvBlock;
					kv < Math.min(kvLen, (c + 1) * kvBlock);
					kv++
				)
					if (dense[q][kv]) n++;
			row.push(n === 0 ? "empty" : n === qBlock * kvBlock ? "full" : "partial");
		}
		kind.push(row);
	}
	const pick = (k: BlockKind) =>
		kind.map((row) => ordered(row.map((x) => x === k)));
	const pickT = (k: BlockKind) =>
		Array.from({ length: cols }, (_, c) =>
			ordered(kind.map((row) => row[c] === k)),
		);
	const part = pick("partial");
	const full = pick("full");
	const partT = pickT("partial");
	const fullT = pickT("full");
	return {
		qLen,
		kvLen,
		qBlock,
		kvBlock,
		rows,
		cols,
		dense,
		kind,
		kvNumBlocks: part.map((o) => o.num),
		kvIndices: part.map((o) => o.idx),
		fullKvNumBlocks: full.map((o) => o.num),
		fullKvIndices: full.map((o) => o.idx),
		qNumBlocks: partT.map((o) => o.num),
		qIndices: partT.map((o) => o.idx),
		fullQNumBlocks: fullT.map((o) => o.num),
		fullQIndices: fullT.map((o) => o.idx),
	};
}

/** get_offset_for_next_block from common.py.jinja, with BLOCKS_ARE_CONTIGUOUS = False. */
export function offsetForNextBlock(
	loopIter: number,
	colIndices: number[],
	totalBlocks: number,
	sparseBlock: number,
	multiple: number,
	block: number,
): number {
	const cur = Math.floor(loopIter / multiple);
	const curBlock = colIndices[cur];
	// The kernel's masked load returns 0 past the last block; the loop ends before it is used.
	const nextBlock = cur + 1 < totalBlocks ? colIndices[cur + 1] : 0;
	const needsJump = (loopIter + 1) % multiple === 0;
	const jump = (nextBlock - curBlock) * sparseBlock - (multiple - 1) * block;
	return needsJump ? jump : block;
}

export interface Tile {
	/** First key/value (forward) or query (backward) index of the tile. */
	start: number;
	/** Sparse block the tile lies in. */
	block: number;
	full: boolean;
	/** Offset added after this tile, as get_offset_for_next_block returns it. */
	offset: number;
	/** True when the offset jumps to another sparse block. */
	jump: boolean;
}

/**
 * Tiles one program visits, in order: the partial-block loop, then the full-block loop.
 * Forward: a query block's row, tiles of BLOCK_N along keys. Backward (dK/dV): a key block's
 * column, tiles of BLOCK_M1 along queries, read from the transposed q_num_blocks / q_indices.
 */
export function visitOrder(
	bm: BlockMaskData,
	line: number,
	tile: number,
	dir: "fwd" | "bwd" = "fwd",
): Tile[] {
	const fwd = dir === "fwd";
	const sparse = fwd ? bm.kvBlock : bm.qBlock;
	const len = fwd ? bm.kvLen : bm.qLen;
	const multiple = sparse / tile;
	const loops: [boolean, number, number[]][] = fwd
		? [
				[false, bm.kvNumBlocks[line], bm.kvIndices[line]],
				[true, bm.fullKvNumBlocks[line], bm.fullKvIndices[line]],
			]
		: [
				[false, bm.qNumBlocks[line], bm.qIndices[line]],
				[true, bm.fullQNumBlocks[line], bm.fullQIndices[line]],
			];
	const out: Tile[] = [];
	for (const [full, num, idx] of loops) {
		let start = idx[0] * sparse;
		const end = Math.min(num * multiple, Math.max(Math.ceil(len / tile), 1));
		for (let n = 0; n < end; n++) {
			const offset = offsetForNextBlock(n, idx, num, sparse, multiple, tile);
			out.push({
				start,
				block: Math.floor(start / sparse),
				full,
				offset,
				jump: (n + 1) % multiple === 0,
			});
			start += offset;
		}
	}
	return out;
}
