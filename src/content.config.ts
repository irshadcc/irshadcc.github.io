import { file, glob } from "astro/loaders";
import { defineCollection, z } from "astro:content";

const posts = defineCollection({
	loader: glob({ base: "src/content/posts", pattern: "**/*.{md,mdx}" }),
	schema: z.object({
		title: z.string(),
		// Shown under the title on the home page.
		description: z.string(),
		date: z.coerce.date(),
		updated: z.coerce.date().optional(),
		draft: z.boolean().optional().default(false),
	}),
});

const workExperience = defineCollection({
	loader: file("src/content/work.json"),
	schema: z.object({
		id: z.number(),
		title: z.string(),
		company: z.string(),
		duration: z.string(),
		description: z.string(),
	}),
});

const publications = defineCollection({
	loader: file("src/content/publications.json"),
	schema: z.object({
		id: z.number(),
		shortTitle: z.string(),
		paperTitle: z.string(),
		link: z.string(),
		conference: z.string(),
	}),
});

export const collections = { posts, workExperience, publications };
