// Routing keys to MySQL shards, used by ShardRouting.astro and the payment-gateway post.
//
//   range    each shard owns a contiguous range of the key's number. Going from N - 1 to N
//            shards splits the fullest range at its median key; the new shard takes the top half.
//   mod      shard = hash(key) mod N
//   buckets  bucket = hash(key) mod B, then a directory maps each bucket to a shard. Going from
//            N - 1 to N shards moves only the buckets the new shard takes over.
//
// hash() is 32-bit FNV-1a over the key's UTF-8 bytes. Also here: the 64-bit payment ID that
// carries its bucket (time | bucket | sequence) and the sizing estimate from the post.

export type Strategy = "range" | "mod" | "buckets";

export const STRATEGY_NAMES: Record<Strategy, string> = {
	range: "Range of IDs",
	mod: "hash mod N",
	buckets: "hash → bucket → shard",
};

/** 32-bit FNV-1a of the UTF-8 bytes of s, as an unsigned integer. */
export function fnv1a32(s: string): number {
	let h = 0x811c9dc5;
	for (const b of new TextEncoder().encode(s)) {
		h ^= b;
		h = Math.imul(h, 0x01000193) >>> 0;
	}
	return h >>> 0;
}

/**
 * Bucket-to-shard directory for 1..n shards, built one shard at a time. The new shard takes
 * floor(B / n) buckets, each taken from whichever shard currently holds the most (ties: the
 * lower shard; within it, the highest bucket), so every step moves the fewest buckets.
 */
export function bucketDirectory(buckets: number, shards: number): number[] {
	const dir = new Array<number>(buckets).fill(0);
	for (let n = 2; n <= shards; n++) {
		const target = Math.floor(buckets / n);
		for (let k = 0; k < target; k++) {
			const count = new Array<number>(n - 1).fill(0);
			for (const s of dir) if (s < n - 1) count[s]++;
			const donor = count.indexOf(Math.max(...count));
			const b = dir.lastIndexOf(donor);
			dir[b] = n - 1;
		}
	}
	return dir;
}

export interface Placement {
	/** Shard of each key. */
	shard: number[];
	/** Keys per shard. */
	load: number[];
	/** Bucket of each key (buckets strategy only). */
	bucket?: number[];
	/** Bucket -> shard (buckets strategy only). */
	directory?: number[];
}

export interface RouteOptions {
	strategy: Strategy;
	shards: number;
	/** Buckets, for the buckets strategy. */
	buckets?: number;
}

export function keyNumber(key: string): number {
	const m = key.match(/(\d+)$/);
	return m ? Number(m[1]) : 0;
}

export function route(keys: readonly string[], o: RouteOptions): Placement {
	const n = o.shards;
	const load = new Array<number>(n).fill(0);
	let shard: number[];
	let bucket: number[] | undefined;
	let directory: number[] | undefined;
	if (o.strategy === "range") {
		const nums = keys.map(keyNumber);
		shard = nums.map(() => 0);
		for (let m = 1; m < n; m++) {
			const count = new Array<number>(m).fill(0);
			for (const s of shard) count[s]++;
			const full = count.indexOf(Math.max(...count));
			const mine = nums
				.filter((_, i) => shard[i] === full)
				.sort((a, b) => a - b);
			const cut = mine[Math.ceil(mine.length / 2)];
			nums.forEach((v, i) => {
				if (shard[i] === full && v >= cut) shard[i] = m;
			});
		}
	} else if (o.strategy === "mod") {
		shard = keys.map((k) => fnv1a32(k) % n);
	} else {
		const b = o.buckets ?? 16;
		directory = bucketDirectory(b, n);
		bucket = keys.map((k) => fnv1a32(k) % b);
		shard = bucket.map((x) => (directory as number[])[x]);
	}
	for (const s of shard) load[s]++;
	return { shard, load, bucket, directory };
}

/** Indices of keys whose shard differs between two placements. */
export function movedKeys(a: Placement, b: Placement): number[] {
	return a.shard.flatMap((s, i) => (s === b.shard[i] ? [] : [i]));
}

// --- Payment IDs that carry their bucket -------------------------------------------------------

/** 41 bits of milliseconds since ID_EPOCH, 12 bits of bucket, 11 bits of sequence. */
export const ID_BITS = { time: 41, bucket: 12, seq: 11 } as const;
export const ID_EPOCH = Date.UTC(2026, 0, 1);

export function encodeId(ms: number, bucket: number, seq: number): bigint {
	return (
		(BigInt(ms - ID_EPOCH) << BigInt(ID_BITS.bucket + ID_BITS.seq)) |
		(BigInt(bucket) << BigInt(ID_BITS.seq)) |
		BigInt(seq)
	);
}

export function decodeId(id: bigint): {
	ms: number;
	bucket: number;
	seq: number;
} {
	const seq = Number(id & ((1n << BigInt(ID_BITS.seq)) - 1n));
	const bucket = Number(
		(id >> BigInt(ID_BITS.seq)) & ((1n << BigInt(ID_BITS.bucket)) - 1n),
	);
	const ms = Number(id >> BigInt(ID_BITS.bucket + ID_BITS.seq)) + ID_EPOCH;
	return { ms, bucket, seq };
}

// --- Sizing --------------------------------------------------------------------------------------

export interface SizingInput {
	paymentsPerDay: number;
	peakToAverage: number;
	/** MySQL transactions per payment (authorize, capture). */
	txPerPayment: number;
	/** Rows written per payment across those transactions. */
	rowsPerPayment: number;
	/** Bytes kept per payment, indexes included. */
	bytesPerPayment: number;
	/** Days of history kept online. */
	retentionDays: number;
	/** Growth the layout should absorb without resharding. */
	growth: number;
	/** Planning limits for one shard (one primary). */
	shardTxPerSec: number;
	shardBytes: number;
}

export function sizing(x: SizingInput) {
	const avgPayments = x.paymentsPerDay / 86_400;
	const peakPayments = avgPayments * x.peakToAverage;
	const peakTx = peakPayments * x.txPerPayment;
	const peakRows = peakPayments * x.rowsPerPayment;
	const bytesPerDay = x.paymentsPerDay * x.bytesPerPayment;
	const bytesOnline = bytesPerDay * x.retentionDays;
	const shardsForWrites = Math.ceil((peakTx * x.growth) / x.shardTxPerSec);
	const shardsForStorage = Math.ceil((bytesOnline * x.growth) / x.shardBytes);
	return {
		avgPayments,
		peakPayments,
		peakTx,
		peakRows,
		bytesPerDay,
		bytesOnline,
		shardsForWrites,
		shardsForStorage,
		shards: Math.max(shardsForWrites, shardsForStorage),
	};
}

// --- Drawing (ShardRouting.astro) ---------------------------------------------------------------

export interface DrawShardsOptions {
	strategy: Strategy;
	shards: number;
	buckets: number;
	/** Outline keys that moved since shards - 1. */
	showMoves: boolean;
	width?: number;
}

export function drawShards(
	keys: readonly string[],
	o: DrawShardsOptions,
): { svg: string; moved: number; load: number[] } {
	const width = o.width ?? 640;
	const now = route(keys, {
		strategy: o.strategy,
		shards: o.shards,
		buckets: o.buckets,
	});
	const before =
		o.shards > 1
			? route(keys, {
					strategy: o.strategy,
					shards: o.shards - 1,
					buckets: o.buckets,
				})
			: now;
	const moved = new Set(movedKeys(before, now));
	const parts: string[] = [];
	let y = 4;
	if (o.strategy === "buckets" && now.directory && before.directory) {
		const bw = width / o.buckets;
		parts.push(
			`<text class="sh-h" x="0" y="${y + 9}">bucket → shard directory</text>`,
		);
		y += 14;
		now.directory.forEach((s, b) => {
			const was = before.directory?.[b];
			const mv = o.showMoves && was !== s;
			parts.push(
				`<g class="s${s}${mv ? " mv" : ""}"><rect class="bk" x="${b * bw + 1}" y="${y}" width="${bw - 2}" height="22" rx="3"><title>bucket ${b} → shard ${s}${mv ? ` (was shard ${was})` : ""}</title></rect><text class="bt" x="${b * bw + bw / 2}" y="${y + 15}">${b}</text></g>`,
			);
		});
		y += 34;
	}
	const colW = width / o.shards;
	const chipH = 17;
	const cols: string[][] = Array.from({ length: o.shards }, () => []);
	keys.forEach((k, i) => cols[now.shard[i]].push(String(i)));
	for (let s = 0; s < o.shards; s++) {
		const x = s * colW;
		parts.push(
			`<g class="s${s}"><rect class="col" x="${x + 2}" y="${y}" width="${colW - 4}" height="22" rx="4"/><text class="ch" x="${x + colW / 2}" y="${y + 15}">shard ${s} · ${now.load[s]}</text></g>`,
		);
		cols[s].forEach((iStr, r) => {
			const i = Number(iStr);
			const k = keys[i];
			const h = fnv1a32(k);
			const mv = o.showMoves && moved.has(i);
			const cy = y + 28 + r * (chipH + 3);
			const route =
				o.strategy === "range"
					? `number ${keyNumber(k)}`
					: o.strategy === "mod"
						? `hash 0x${h.toString(16).padStart(8, "0")} mod ${o.shards} = ${h % o.shards}`
						: `hash 0x${h.toString(16).padStart(8, "0")} mod ${o.buckets} = bucket ${h % o.buckets}`;
			parts.push(
				`<g class="s${s}${mv ? " mv" : ""}"><rect class="chip" x="${x + 6}" y="${cy}" width="${colW - 12}" height="${chipH}" rx="8"><title>${k}: ${route} → shard ${s}${mv ? ` (was shard ${before.shard[i]})` : ""}</title></rect><text class="kt" x="${x + colW / 2}" y="${cy + 12}">${colW > 90 ? k : keyNumber(k)}${mv ? ` ←${before.shard[i]}` : ""}</text></g>`,
			);
		});
	}
	const tallest = Math.max(...cols.map((c) => c.length));
	const height = y + 28 + tallest * (chipH + 3) + 4;
	return {
		svg: `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img">${parts.join("")}</svg>`,
		moved: moved.size,
		load: now.load,
	};
}
