import { getCollection } from "astro:content";

/**
 * Posts newest first. Drafts are never listed; pass `includeDrafts` to also
 * build their pages during `npm run dev` so they can be previewed by URL.
 */
export async function getPosts({ includeDrafts = false } = {}) {
	const posts = await getCollection("posts", (post) => !post.data.draft || (includeDrafts && import.meta.env.DEV));
	return posts.sort((a, b) => b.data.date.getTime() - a.data.date.getTime());
}

export async function getPublishedPosts() {
	const posts = await getPosts();
	return posts.filter((post) => !post.data.tags.includes("Draft"));
}

export async function getDraftPosts() {
	const posts = await getPosts();
	return posts.filter((post) => post.data.tags.includes("Draft"));
}
