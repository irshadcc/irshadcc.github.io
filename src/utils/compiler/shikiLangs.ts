// Small TextMate grammars for two languages Shiki doesn't ship: LLVM's TableGen (.td) and
// NVIDIA's PTX assembly. They are registered in astro.config.ts, so ```tablegen and ```ptx
// fences highlight in posts, and passed to <Code> by CodeStages.astro.
// They cover what the LLVM post shows (keywords, types, strings, comments, operands, registers),
// not the full languages.

import type { LanguageRegistration } from "shiki";

export const tablegen: LanguageRegistration = {
	name: "tablegen",
	scopeName: "source.tablegen",
	aliases: ["td"],
	patterns: [
		{ include: "#comment" },
		{ include: "#string" },
		{
			match: "\\b(class|def|defm|multiclass)\\s+([A-Za-z_][A-Za-z0-9_]*)",
			captures: {
				1: { name: "keyword.control.tablegen" },
				2: { name: "entity.name.type.tablegen" },
			},
		},
		{
			match:
				"\\b(class|def|defm|defset|defvar|multiclass|let|in|foreach|if|then|else|field|include|assert)\\b",
			name: "keyword.control.tablegen",
		},
		{
			match: "\\b(bit|bits|int|string|list|dag|code)\\b",
			name: "storage.type.tablegen",
		},
		{ match: "![a-z_]+", name: "support.function.tablegen" },
		{ match: "\\$[A-Za-z_][A-Za-z0-9_]*", name: "variable.parameter.tablegen" },
		{
			match: "\\b(0x[0-9a-fA-F]+|0b[01]+|\\d+)\\b",
			name: "constant.numeric.tablegen",
		},
	],
	repository: {
		comment: {
			patterns: [
				{ match: "//.*$", name: "comment.line.double-slash.tablegen" },
				{ begin: "/\\*", end: "\\*/", name: "comment.block.tablegen" },
			],
		},
		string: {
			begin: '"',
			end: '"',
			name: "string.quoted.double.tablegen",
			patterns: [
				{ match: "\\\\.", name: "constant.character.escape.tablegen" },
			],
		},
	},
};

export const ptx: LanguageRegistration = {
	name: "ptx",
	scopeName: "source.ptx",
	patterns: [
		{ match: "//.*$", name: "comment.line.double-slash.ptx" },
		{
			// An instruction: optional guard predicate, then the opcode with its dotted modifiers.
			match:
				"^\\s*(@!?%[A-Za-z0-9_]+\\s+)?([a-z][a-z0-9_]*(?:\\.[a-z0-9_]+)*)(?=\\s|;)",
			captures: {
				1: { name: "variable.other.predicate.ptx" },
				2: { name: "keyword.control.ptx" },
			},
		},
		{ match: "\\.[a-z_][a-z0-9_]*", name: "storage.type.ptx" },
		{
			match: "%[A-Za-z_][A-Za-z0-9_]*(?:\\.[xyzw])?(?:<\\d+>)?",
			name: "variable.other.register.ptx",
		},
		{ match: "\\$?L__[A-Za-z0-9_]+", name: "entity.name.function.ptx" },
		{
			match: "\\b(0[xX][0-9a-fA-F]+|0[fFdD][0-9a-fA-F]+|\\d+)\\b",
			name: "constant.numeric.ptx",
		},
	],
};
