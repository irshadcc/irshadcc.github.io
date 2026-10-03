// Example graphs for DependencyGraph.astro and its demo page.
import type { BadgeIcon, GraphEdge, GraphNode } from "./dependencyGraph";

const ts: BadgeIcon = { text: "TS", bg: "#3178c6", fg: "#ffffff" };
const js: BadgeIcon = { text: "JS", bg: "#f0db4f", fg: "#1c1c1a" };
const go: BadgeIcon = { text: "Go", bg: "#00acd7", fg: "#ffffff" };
const svelte: BadgeIcon = { text: "S", bg: "#ff3e00", fg: "#ffffff" };

/** A code-context graph: files, functions, a folder and a commit, as in a code search tool. */
export const codeNodes: GraphNode[] = [
	{
		id: "tooltip",
		title: "Tooltip.tsx",
		subtitle: "client/wildcard/components",
		icon: ts,
		note: "A React component that imports the shared TypeScript helpers.",
	},
	{
		id: "useExt",
		title: "useExtensionAPI",
		subtitle: "lib/prompt-editor/API.tsx",
		icon: "fn",
		note: "A hook the prompt editor calls to reach the extension host.",
	},
	{ id: "ph-a", kind: "placeholder" },
	{ id: "ph-b", kind: "placeholder" },
	{ id: "ts", kind: "icon", icon: ts, note: "TypeScript sources." },
	{ id: "hex", kind: "icon", icon: "hexagon", note: "The GraphQL schema." },
	{ id: "js", kind: "icon", icon: js, note: "Built JavaScript bundles." },
	{ id: "j1", kind: "junction" },
	{ id: "ph-c", kind: "placeholder" },
	{
		id: "components",
		title: "components",
		subtitle: "92 files · 58 KB total size",
		icon: "folder",
		note: "The shared component folder most of the graph depends on.",
	},
	{
		id: "auth",
		title: "auth.go",
		subtitle: "Go · 58 lines · 1.43 KB",
		icon: go,
		note: "The Go file that defines the auth middleware.",
	},
	{ id: "j2", kind: "junction" },
	{
		id: "sparkle",
		kind: "icon",
		icon: "sparkle",
		note: "An AI-generated summary.",
	},
	{
		id: "compose",
		title: "composeMiddleware",
		subtitle: "cmd/frontend/auth/auth.go",
		icon: "fn",
		note: "Chains the HTTP middleware; the node most edges lead to.",
	},
	{
		id: "avatar",
		title: "Avatar.svelte",
		subtitle: "client/web-sveltekit/src/lib",
		icon: svelte,
	},
	{ id: "go", kind: "icon", icon: go },
	{
		id: "commit",
		title: "timeouts and missing repos (#1505)",
		subtitle: "79afe59 · committed 16h ago",
		icon: "commit",
		note: "The most recent commit to touch composeMiddleware.",
	},
	{ id: "ph-d", kind: "placeholder" },
	{ id: "ph-e", kind: "placeholder" },
	{ id: "ph-f", kind: "placeholder" },
];

export const codeEdges: GraphEdge[] = [
	{ from: "tooltip", to: "ts" },
	{ from: "tooltip", to: "hex" },
	{ from: "useExt", to: "hex" },
	{ from: "useExt", to: "js" },
	{ from: "ph-a", to: "js" },
	{ from: "ph-b", to: "j1" },
	{ from: "ts", to: "ph-c" },
	{ from: "ts", to: "components" },
	{ from: "hex", to: "components" },
	{ from: "js", to: "auth" },
	{ from: "js", to: "sparkle" },
	{ from: "j1", to: "auth" },
	{ from: "ph-a", to: "j2" },
	{ from: "j2", to: "auth" },
	{ from: "ph-c", to: "avatar" },
	{ from: "components", to: "compose" },
	{ from: "sparkle", to: "compose" },
	{ from: "auth", to: "compose" },
	{ from: "auth", to: "go" },
	{ from: "go", to: "ph-f" },
	{ from: "avatar", to: "ph-e" },
	{ from: "compose", to: "commit" },
	{ from: "compose", to: "ph-d" },
	{ from: "go", to: "commit" },
];

/** A small build pipeline, laid out top to bottom. The wires into the junction have no arrow. */
export const buildNodes: GraphNode[] = [
	{ id: "src", title: "src/", subtitle: "214 files", icon: "folder" },
	{ id: "lock", title: "package-lock.json", icon: "file" },
	{ id: "tsc", title: "tsc --noEmit", subtitle: "type check", icon: ts },
	{ id: "bundle", title: "esbuild", subtitle: "bundle + minify", icon: js },
	{ id: "j", kind: "junction" },
	{ id: "dist", title: "dist/", subtitle: "3 files · 412 KB", icon: "folder" },
];

export const buildEdges: GraphEdge[] = [
	{ from: "src", to: "tsc" },
	{ from: "src", to: "bundle" },
	{ from: "lock", to: "bundle" },
	{ from: "tsc", to: "j", directed: false },
	{ from: "bundle", to: "j", directed: false },
	{ from: "j", to: "dist" },
];
