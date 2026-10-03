// Layout for TokenStrip.astro: a prompt drawn as a row of token chips. A chip is one token, or a
// run of `count` identical tokens drawn once (an image's placeholder tokens). This module gives
// each chip its positions in the sequence and, for a block size, the KV-cache blocks it touches.

export type TokenKind =
	| "special"
	| "role"
	| "text"
	| "space"
	| "vision"
	| "image"
	| "partial";

export interface TokenChip {
	/** What the token decodes to; use "↵" for a newline and "␣" for a leading space. */
	text: string;
	id?: number;
	kind?: TokenKind;
	/** Draw `count` identical tokens as one chip (default 1). */
	count?: number;
	/** Shown under the figure when the chip is selected. HTML allowed. */
	note?: string;
}

export interface TokenView {
	label: string;
	tokens: TokenChip[];
}

export interface PlacedChip extends TokenChip {
	kind: TokenKind;
	count: number;
	/** First and last position in the sequence (inclusive). */
	start: number;
	end: number;
	/** Block of `start` and of `end`, when a block size is given. */
	firstBlock?: number;
	lastBlock?: number;
	/** True when a new block starts at this chip's first position. */
	blockStart: boolean;
}

export function placeTokens(
	tokens: readonly TokenChip[],
	blockSize?: number,
): PlacedChip[] {
	let pos = 0;
	return tokens.map((t) => {
		const count = t.count ?? 1;
		const start = pos;
		const end = pos + count - 1;
		pos += count;
		const placed: PlacedChip = {
			...t,
			kind: t.kind ?? "text",
			count,
			start,
			end,
			blockStart: blockSize !== undefined && start % blockSize === 0,
		};
		if (blockSize !== undefined) {
			placed.firstBlock = Math.floor(start / blockSize);
			placed.lastBlock = Math.floor(end / blockSize);
		}
		return placed;
	});
}

export function sequenceLength(tokens: readonly TokenChip[]): number {
	return tokens.reduce((n, t) => n + (t.count ?? 1), 0);
}

/** One line describing a placed chip, shown under the figure. */
export function describeChip(c: PlacedChip): string {
	const where =
		c.count === 1
			? `position ${c.start}`
			: `positions ${c.start}–${c.end} (${c.count} tokens)`;
	const id = c.id === undefined ? "" : ` · id ${c.id}`;
	let block = "";
	if (c.firstBlock !== undefined && c.lastBlock !== undefined)
		block =
			c.firstBlock === c.lastBlock
				? ` · block ${c.firstBlock}`
				: ` · blocks ${c.firstBlock}–${c.lastBlock}`;
	return `${where}${id}${block}`;
}

export function validateViews(
	views: readonly TokenView[],
	blockSize?: number,
): string | undefined {
	if (views.length === 0) return "needs at least one view";
	if (
		blockSize !== undefined &&
		(!Number.isInteger(blockSize) || blockSize < 1)
	)
		return `blockSize must be a positive integer, got ${blockSize}`;
	for (const v of views) {
		if (v.tokens.length === 0) return `view "${v.label}" has no tokens`;
		for (const t of v.tokens) {
			const n = t.count ?? 1;
			if (!Number.isInteger(n) || n < 1)
				return `token "${t.text}" in "${v.label}" has count ${n}`;
		}
	}
	return undefined;
}
