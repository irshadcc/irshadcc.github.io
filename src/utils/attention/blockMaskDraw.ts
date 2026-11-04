// SVG and HTML for BlockMaskExplorer.astro: the element mask with its block grid, the tiles one
// program visits (numbered in order), and the BlockMask tensors as a table. Shared by the
// server render and the client script, so both draw the same thing.
import type { BlockMaskData, Tile } from "./blockMask";

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

export function cellSize(len: number): number {
	return Math.max(9, Math.min(20, Math.floor(320 / len)));
}

export function drawMask(
	bm: BlockMaskData,
	line: number,
	tiles: Tile[],
	dir: "fwd" | "bwd",
	tile: number,
): string {
	const c = cellSize(Math.max(bm.qLen, bm.kvLen));
	const L = 30; // left gutter for row labels
	const T = 26; // top gutter for column labels
	const W = L + bm.kvLen * c + 4;
	const H = T + bm.qLen * c + 4;
	const parts: string[] = [];
	parts.push(
		`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Attention mask, ${bm.qLen} queries by ${bm.kvLen} keys, in blocks of ${bm.qBlock} by ${bm.kvBlock}">`,
	);
	// Block tints.
	for (let r = 0; r < bm.rows; r++)
		for (let k = 0; k < bm.cols; k++) {
			const kind = bm.kind[r][k];
			const x = L + k * bm.kvBlock * c;
			const y = T + r * bm.qBlock * c;
			const w = Math.min(bm.kvBlock, bm.kvLen - k * bm.kvBlock) * c;
			const h = Math.min(bm.qBlock, bm.qLen - r * bm.qBlock) * c;
			const on = dir === "fwd" ? r === line : k === line;
			parts.push(
				`<rect class="bm-blk ${kind}${on ? " on" : ""}" data-r="${r}" data-k="${k}" x="${x}" y="${y}" width="${w}" height="${h}"/>`,
			);
		}
	// Allowed elements.
	for (let q = 0; q < bm.qLen; q++)
		for (let kv = 0; kv < bm.kvLen; kv++)
			if (bm.dense[q][kv])
				parts.push(
					`<rect class="bm-el" x="${L + kv * c + 1.5}" y="${T + q * c + 1.5}" width="${c - 3}" height="${c - 3}"/>`,
				);
	// Block grid lines.
	for (let k = 0; k <= bm.cols; k++) {
		const x = L + Math.min(k * bm.kvBlock, bm.kvLen) * c;
		parts.push(
			`<line class="bm-grid" x1="${x}" y1="${T}" x2="${x}" y2="${T + bm.qLen * c}"/>`,
		);
	}
	for (let r = 0; r <= bm.rows; r++) {
		const y = T + Math.min(r * bm.qBlock, bm.qLen) * c;
		parts.push(
			`<line class="bm-grid" x1="${L}" y1="${y}" x2="${L + bm.kvLen * c}" y2="${y}"/>`,
		);
	}
	// Block labels.
	for (let k = 0; k < bm.cols; k++)
		parts.push(
			`<text class="bm-lab${dir === "bwd" && k === line ? " on" : ""}" x="${L + (k * bm.kvBlock + Math.min(bm.kvBlock, bm.kvLen - k * bm.kvBlock) / 2) * c}" y="${T - 14}">kv${k}</text>`,
		);
	for (let r = 0; r < bm.rows; r++) {
		const h = Math.min(bm.qBlock, bm.qLen - r * bm.qBlock) * c;
		parts.push(
			`<text class="bm-lab bm-row${dir === "fwd" && r === line ? " on" : ""}" x="${L - 5}" y="${T + r * bm.qBlock * c + h / 2 + 3}">q${r}</text>`,
		);
	}
	// Element index ticks at block boundaries.
	for (let k = 0; k < bm.cols; k++)
		parts.push(
			`<text class="bm-tick" x="${L + k * bm.kvBlock * c + 2}" y="${T - 3}">${k * bm.kvBlock}</text>`,
		);
	// Visited tiles, numbered in visit order.
	tiles.forEach((t, i) => {
		let x: number;
		let y: number;
		let w: number;
		let h: number;
		if (dir === "fwd") {
			x = L + t.start * c;
			y = T + line * bm.qBlock * c;
			w = Math.min(tile, bm.kvLen - t.start) * c;
			h = Math.min(bm.qBlock, bm.qLen - line * bm.qBlock) * c;
		} else {
			x = L + line * bm.kvBlock * c;
			y = T + t.start * c;
			w = Math.min(bm.kvBlock, bm.kvLen - line * bm.kvBlock) * c;
			h = Math.min(tile, bm.qLen - t.start) * c;
		}
		parts.push(
			`<rect class="bm-tile ${t.full ? "full" : "partial"}" x="${x + 1}" y="${y + 1}" width="${Math.max(w - 2, 1)}" height="${Math.max(h - 2, 1)}"/>`,
		);
		const r = Math.min(7, c * 0.42 + 2);
		parts.push(
			`<circle class="bm-num ${t.full ? "full" : "partial"}" cx="${x + w / 2}" cy="${y + h / 2}" r="${r}"/><text class="bm-numt" x="${x + w / 2}" y="${y + h / 2 + r * 0.42}" style="font-size:${(r * 1.25).toFixed(1)}px">${i + 1}</text>`,
		);
	});
	parts.push("</svg>");
	return parts.join("");
}

function idxCells(idx: number[], num: number): string {
	return idx
		.map((v, i) => `<span class="${i < num ? "live" : "dead"}">${v}</span>`)
		.join(" ");
}

export function drawTable(
	bm: BlockMaskData,
	line: number,
	dir: "fwd" | "bwd",
): string {
	const fwd = dir === "fwd";
	const n = fwd ? bm.rows : bm.cols;
	const head = fwd
		? [
				"q block",
				"kv_num_blocks",
				"kv_indices",
				"full_kv_num_blocks",
				"full_kv_indices",
			]
		: [
				"kv block",
				"q_num_blocks",
				"q_indices",
				"full_q_num_blocks",
				"full_q_indices",
			];
	const rows: string[] = [];
	for (let i = 0; i < n; i++) {
		const num = fwd ? bm.kvNumBlocks[i] : bm.qNumBlocks[i];
		const idx = fwd ? bm.kvIndices[i] : bm.qIndices[i];
		const fnum = fwd ? bm.fullKvNumBlocks[i] : bm.fullQNumBlocks[i];
		const fidx = fwd ? bm.fullKvIndices[i] : bm.fullQIndices[i];
		rows.push(
			`<tr data-i="${i}" class="${i === line ? "on" : ""}"><th>${i}</th><td class="num">${num}</td><td class="idx">[${idxCells(idx, num)}]</td><td class="num">${fnum}</td><td class="idx">[${idxCells(fidx, fnum)}]</td></tr>`,
		);
	}
	return `<table><thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

export function describe(
	bm: BlockMaskData,
	line: number,
	tiles: Tile[],
	dir: "fwd" | "bwd",
	tile: number,
): string {
	const fwd = dir === "fwd";
	const sparse = fwd ? bm.kvBlock : bm.qBlock;
	const lo = line * (fwd ? bm.qBlock : bm.kvBlock);
	const hi =
		Math.min(lo + (fwd ? bm.qBlock : bm.kvBlock), fwd ? bm.qLen : bm.kvLen) - 1;
	const who = fwd
		? `The program for query block ${line} (rows ${lo}–${hi})`
		: `The dK/dV program for key block ${line} (keys ${lo}–${hi})`;
	const along = fwd ? "key" : "query";
	const tileName = fwd ? "BLOCK_N" : "BLOCK_M1";
	const part = tiles.filter((t) => !t.full);
	const full = tiles.filter((t) => t.full);
	const blocks = (ts: Tile[]) =>
		[...new Set(ts.map((t) => t.block))].join(", ") || "none";
	const steps = tiles
		.map((t, i) => {
			const range = `${t.start}–${t.start + tile - 1}`;
			const last = i === tiles.length - 1 || tiles[i + 1].full !== t.full;
			const off = last
				? "loop ends"
				: t.jump
					? `jump ${t.offset >= 0 ? "+" : ""}${t.offset} to block ${tiles[i + 1].block}`
					: `+${t.offset}`;
			return `<li class="${t.full ? "full" : "partial"}"><b>${i + 1}</b> ${along}s ${range} <span>(block ${t.block}${t.full ? ", full" : ", partial"}; ${off})</span></li>`;
		})
		.join("");
	return `<p>${who} runs ${tiles.length} tile${tiles.length === 1 ? "" : "s"} of ${tileName} = ${tile} ${along}s. The partial loop applies <code>mask_mod</code> and visits block${part.length === 1 ? "" : "s"} ${blocks(part)}; the full loop skips <code>mask_mod</code> and visits ${blocks(full)}. A block of ${sparse} holds ${sparse / tile} tile${sparse / tile === 1 ? "" : "s"}, so <code>get_offset_for_next_block</code> jumps after every ${sparse / tile === 1 ? "tile" : `${sparse / tile}th tile`}.</p><ol class="bm-steps">${steps}</ol>`;
}
