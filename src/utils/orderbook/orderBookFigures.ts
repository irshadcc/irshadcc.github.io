// Presets for the order book post's figures. The book in the index figure is the worked
// example's book after its first seven messages (exampleFrames()[6]); benchmark numbers come
// from orderBookBench.json, produced by the C++ benchmark described in the post.
import bench from "./orderBookBench.json";
import type { IndexLevel, IndexState } from "./orderBookDraw";

export { bench };

const book7 = (): IndexLevel[] => [
	{ price: 104, side: "sell", orders: [{ id: 7, qty: 3, slot: 6 }] },
	{ price: 102, side: "sell", orders: [{ id: 5, qty: 6, slot: 4 }] },
	{
		price: 101,
		side: "sell",
		orders: [
			{ id: 3, qty: 4, slot: 2 },
			{ id: 4, qty: 2, slot: 3 },
		],
	},
	{
		price: 99,
		side: "buy",
		orders: [
			{ id: 1, qty: 5, slot: 0 },
			{ id: 6, qty: 2, slot: 5 },
		],
	},
	{ price: 98, side: "buy", orders: [{ id: 2, qty: 3, slot: 1 }] },
];

function without(
	levels: IndexLevel[],
	id: number,
	dropEmpty = true,
): IndexLevel[] {
	return levels
		.map((l) => ({ ...l, orders: l.orders.filter((o) => o.id !== id) }))
		.filter((l) => !dropEmpty || l.orders.length > 0);
}

const after6 = without(book7(), 6);
const after2 = without(after6, 2);
const with103: IndexLevel[] = [
	...after2,
	{ price: 103, side: "sell", orders: [{ id: 11, qty: 2, slot: 1 }] },
];

export const INDEX_STATES: IndexState[] = [
	{
		label: "the book",
		levels: book7(),
		freeSlots: [],
		note: "The worked example's book after seven orders. Three structures hold it: a tree of price levels per side (std::map), a FIFO of orders hanging off each level, and an index from order id to the order's pool slot (s0, s1, ...).",
	},
	{
		label: "cancel #6: look up",
		levels: book7(),
		hotId: 6,
		freeSlots: [],
		note: "Cancel #6. The message carries only the id, so the first step is the id index: entry 6 says slot s5. One array read (or one hash probe).",
	},
	{
		label: "neighbours",
		levels: book7(),
		hotId: 6,
		hotLevel: 99,
		hotOrders: [1],
		freeSlots: [],
		note: "The order in s5 stores its price level and its prev / next neighbours in the FIFO: prev = #1, next = none. No search anywhere.",
	},
	{
		label: "unlink",
		levels: after6,
		ghost: { price: 99, order: { id: 6, qty: 2, slot: 5 }, at: 1 },
		hotLevel: 99,
		hotOrders: [1],
		freeSlots: [5],
		note: "Unlink: #1.next = none and the level's tail = #1. The level total drops 7 → 5. The level is not empty, so the tree is untouched. Slot s5 goes on the free list, and index entry 6 is cleared. Cost: O(1).",
	},
	{
		label: "cancel #2",
		levels: after6,
		hotId: 2,
		hotLevel: 98,
		freeSlots: [5],
		note: "Cancel #2: index entry 2 → s1 → level 98. #2 is the only order there.",
	},
	{
		label: "erase level",
		levels: [...after2, { price: 98, side: "buy", orders: [] }],
		ghost: { price: 98, order: { id: 2, qty: 3, slot: 1 }, at: 0 },
		treeChange: { price: 98, kind: "erase" },
		freeSlots: [5, 1],
		note: "The level is now empty, so its node is erased from the bid tree. This is the only part of a cancel that touches the tree; a red-black tree erase is O(log M) for M levels, and frees one node.",
	},
	{
		label: "add: descend",
		levels: after2,
		treePath: [102, 104],
		parents: { 102: null, 101: 102, 104: 102, 99: null },
		freeSlots: [5, 1],
		note: "Add #11: sell 2 at 103. Is there a level at 103? Descend the ask tree: 103 > 102, go right; 103 < 104, go left; there is no left child, so 103 is new. O(log M) comparisons, each one a pointer chase to another heap node.",
	},
	{
		label: "insert",
		levels: with103,
		treeChange: { price: 103, kind: "insert" },
		parents: { 102: null, 101: 102, 104: 102, 103: 104, 99: null },
		hotId: 11,
		freeSlots: [5],
		note: "A new level node is linked in as 104's left child (std::map allocates it), #11 takes the most recently freed slot s1 from the free list, and becomes both head and tail of the new FIFO. An add at an existing price skips the tree altogether: Selph also keeps each Limit in a hash map keyed by price, so only the first order at a price pays O(log M).",
	},
];

/** A bid side with 13 levels, best (99) at the back, as Gross's "reverse" vector stores it. */
export const BID_KEYS = [84, 86, 87, 89, 90, 91, 93, 94, 95, 96, 97, 98, 99];
/** The ask side stores -price so that it too is ascending with the best price at the back. */
export const ASK_KEYS = [-104, -102, -101];

/** 64 ticks of the bid side; the best level (tick 50) has just emptied. */
export const TICKS_OCCUPIED: boolean[] = Array.from({ length: 64 }, (_, i) =>
	[3, 9, 12, 17, 21].includes(i),
);
export const TICKS_FROM = 50;
export const TICKS_LO = 100;
