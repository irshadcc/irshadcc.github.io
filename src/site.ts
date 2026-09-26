// Site-wide settings. Edit these to change the header, footer and metadata.
export const SITE = {
	title: "irshadcc",
	author: "Irshad Kandy",
	description: "Irshad Kandy's technical blog on ML systems.",
	socials: [
		{ name: "GitHub", url: "https://github.com/irshadcc" },
		{ name: "LinkedIn", url: "https://linkedin.com/in/irshadcc" },
	],
};

export function formatDate(date: Date): string {
	return date.toLocaleDateString("en-US", { month: "short", day: "2-digit", year: "numeric", timeZone: "UTC" });
}

/** Rough reading time for a Markdown body, in minutes. */
export function readingTime(body = ""): number {
	const words = body.replace(/```[\s\S]*?```/g, " ").split(/\s+/).filter(Boolean).length;
	return Math.max(1, Math.round(words / 230));
}

