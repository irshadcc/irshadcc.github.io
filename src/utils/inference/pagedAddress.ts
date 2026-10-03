// Where a token's key and value live in a paged KV cache, for PagedAddress.astro. Token
// position p of a request sits in logical block p / B (integer division) at offset p % B; the
// request's block table maps the logical block to a physical block, so its slot is
//   slot = block_table[p / B] * B + p % B
// which is the formula of vLLM's _compute_slot_mappings_kernel (vllm/v1/worker/gpu/block_table.py)
// and, with B = 1, the req_to_token lookup of SGLang.

export interface PagedConfig {
	blockSize: number;
	/** Number of tokens in the request. */
	tokens: number;
	/** Physical block of each logical block. */
	blockTable: number[];
	/** Physical blocks drawn. */
	numBlocks: number;
}

export interface TokenAddress {
	pos: number;
	logical: number;
	offset: number;
	physical: number;
	slot: number;
}

export function validatePaged(c: PagedConfig): string | undefined {
	if (c.blockSize < 1) return "blockSize must be >= 1";
	const need = Math.ceil(c.tokens / c.blockSize);
	if (c.blockTable.length < need)
		return `block table has ${c.blockTable.length} entries, ${need} needed`;
	if (new Set(c.blockTable).size !== c.blockTable.length)
		return "block table repeats a block";
	for (const b of c.blockTable)
		if (b < 0 || b >= c.numBlocks)
			return `block ${b} outside 0..${c.numBlocks - 1}`;
	return undefined;
}

export function addresses(c: PagedConfig): TokenAddress[] {
	return Array.from({ length: c.tokens }, (_, pos) => {
		const logical = Math.floor(pos / c.blockSize);
		const offset = pos % c.blockSize;
		const physical = c.blockTable[logical];
		return {
			pos,
			logical,
			offset,
			physical,
			slot: physical * c.blockSize + offset,
		};
	});
}

export function drawPaged(c: PagedConfig): { svg: string; width: number } {
	const cell = 24;
	const g = 2;
	const bg = c.blockSize > 1 ? 8 : 2;
	const addr = addresses(c);
	const out: string[] = [];
	const xOfPos = (p: number) =>
		p * (cell + g) + Math.floor(p / c.blockSize) * (bg - g);
	// Logical row.
	let y = 14;
	out.push(`<text class="pa-label" x="0" y="${y - 4}">token position p</text>`);
	for (const a of addr) {
		const x = xOfPos(a.pos);
		out.push(
			`<g class="pa-tok" data-pos="${a.pos}"><rect x="${x}" y="${y}" width="${cell}" height="${cell}" rx="3"/><text x="${x + cell / 2}" y="${y + 16}">${a.pos}</text></g>`,
		);
	}
	const nLogical = Math.ceil(c.tokens / c.blockSize);
	// Block table row.
	y += cell + 30;
	out.push(
		`<text class="pa-label" x="0" y="${y - 4}">${c.blockSize > 1 ? "block table (logical → physical)" : "req_to_token row (position → slot)"}</text>`,
	);
	for (let l = 0; l < nLogical; l++) {
		const x0 = xOfPos(l * c.blockSize);
		const w = c.blockSize > 1 ? c.blockSize * (cell + g) - g : cell;
		out.push(
			`<g class="pa-bt" data-logical="${l}"><rect x="${x0}" y="${y}" width="${w}" height="${cell}" rx="3"/><text x="${x0 + w / 2}" y="${y + 16}">${c.blockTable[l]}</text></g>`,
		);
	}
	// Physical memory.
	y += cell + 34;
	out.push(
		`<text class="pa-label" x="0" y="${y - 4}">${c.blockSize > 1 ? "physical KV blocks (slot ids)" : "KV pool slots"}</text>`,
	);
	const perRow = Math.max(1, Math.floor(16 / c.blockSize));
	// Width of one physical block with its gap; page size 1 packs slots like an array.
	const step =
		c.blockSize > 1 ? c.blockSize * (cell + g) - g + bg + 18 : cell + g;
	const used = new Map(addr.map((a) => [a.slot, a.pos]));
	for (let b = 0; b < c.numBlocks; b++) {
		const bx = (b % perRow) * step;
		const by = y + Math.floor(b / perRow) * (cell + 22);
		if (c.blockSize > 1)
			out.push(`<text class="pa-bid" x="${bx}" y="${by + 16}">${b}</text>`);
		const off = c.blockSize > 1 ? 16 : 0;
		for (let o = 0; o < c.blockSize; o++) {
			const slot = b * c.blockSize + o;
			const x = bx + off + o * (cell + g);
			const p = used.get(slot);
			out.push(
				`<g class="pa-slot${p === undefined ? "" : " mine"}" data-slot="${slot}"><rect x="${x}" y="${by}" width="${cell}" height="${cell}" rx="3"/><text x="${x + cell / 2}" y="${by + 16}">${slot}</text></g>`,
			);
		}
	}
	const rows = Math.ceil(c.numBlocks / perRow);
	const H = y + rows * (cell + 22);
	const physW = Math.min(perRow, c.numBlocks) * step;
	const W = Math.max(xOfPos(c.tokens - 1) + cell, physW, 300);
	return {
		svg: `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Token positions, block table and physical KV slots">${out.join("")}</svg>`,
		width: W,
	};
}
