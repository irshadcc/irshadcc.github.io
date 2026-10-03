export type LsmStage = {
	label: string;
	active: string[];
	note: string;
};

export const writeStages: LsmStage[] = [
	{
		label: "Put",
		active: ["client", "wal", "mem"],
		note: "The write batch receives a sequence number, is appended to the WAL, and is inserted into the active memtable.",
	},
	{
		label: "Switch",
		active: ["mem", "imm"],
		note: "When the write buffer fills, it becomes immutable. A fresh WAL and memtable accept new writes.",
	},
	{
		label: "Flush",
		active: ["imm", "l0"],
		note: "A background job writes the immutable memtable, already sorted by internal key, into a new level-0 SST file.",
	},
	{
		label: "Compact",
		active: ["l0", "l1"],
		note: "Compaction merge-sorts overlapping runs, discards obsolete versions when safe, and installs replacement files through the MANIFEST.",
	},
];

export const readStages: LsmStage[] = [
	{
		label: "Memtables",
		active: ["client", "mem", "imm"],
		note: "A point read checks the mutable memtable and immutable memtables from newest to oldest.",
	},
	{
		label: "L0",
		active: ["l0"],
		note: "Level-0 files may overlap, so every plausible file is checked newest first; filters can reject most misses.",
	},
	{
		label: "Lower levels",
		active: ["l1", "l2"],
		note: "From level 1 downward, files have disjoint key ranges, so at most one file per level is a candidate.",
	},
	{
		label: "Inside SST",
		active: ["cache", "sst"],
		note: "The filter tests membership, the index selects a data block, and the restart array bounds the in-block scan.",
	},
];

export function validateStages(stages: LsmStage[]): string | undefined {
	const nodes = new Set([
		"client",
		"wal",
		"mem",
		"imm",
		"l0",
		"l1",
		"l2",
		"cache",
		"sst",
	]);
	if (stages.length === 0) return "at least one stage is required";
	for (const stage of stages) {
		if (!stage.label || !stage.note)
			return "every stage needs a label and note";
		for (const id of stage.active)
			if (!nodes.has(id)) return `unknown node: ${id}`;
	}
}
