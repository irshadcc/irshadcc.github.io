// Browser-side behaviour of a NeuralNetworkGraph figure: hovering or focusing a node highlights
// it and its neighbours, and shows its card (if it has one) beside it.

/** Wires up one `.nn-graph` figure. */
export function attachInteractions(fig: HTMLElement) {
	const svg = fig.querySelector<SVGSVGElement>(".canvas svg");
	if (!svg) return;
	const highlight = highlighter(svg);
	const showCard = cardShower(fig);
	for (const node of svg.querySelectorAll<SVGGElement>(".node")) {
		const on = () => {
			highlight(node.dataset.id ?? null);
			showCard(node);
		};
		const off = () => {
			highlight(null);
			showCard(null);
		};
		node.addEventListener("pointerenter", on);
		node.addEventListener("pointerleave", off);
		node.addEventListener("focus", on);
		node.addEventListener("blur", off);
	}
	addEventListener("keydown", (e) => {
		if (e.key === "Escape") showCard(null);
	});
}

/**
 * Returns a function that highlights a node, its incoming and outgoing edges, and the nodes at
 * their other ends, fading everything else (null clears it). Edges carry data-from/data-to.
 */
function highlighter(svg: SVGSVGElement) {
	const nodes = [...svg.querySelectorAll<SVGGElement>(".node")];
	const edges = [...svg.querySelectorAll<SVGGElement>(".edge")];
	return (id: string | null) => {
		svg.classList.toggle("focusing", id !== null);
		const near = new Set([id]);
		for (const e of edges) {
			const hot = id !== null && (e.dataset.from === id || e.dataset.to === id);
			e.classList.toggle("hot", hot);
			if (hot) near.add(e.dataset.from ?? null).add(e.dataset.to ?? null);
		}
		for (const n of nodes)
			n.classList.toggle("hot", near.has(n.dataset.id ?? null));
	};
}

/** Returns a function that shows a node's card (and hides the previous one; null hides it). */
function cardShower(fig: HTMLElement) {
	const cards = new Map(
		[...fig.querySelectorAll<HTMLElement>(".card")].map((c) => [
			c.dataset.for,
			c,
		]),
	);
	let shown: HTMLElement | undefined;
	return (node: SVGGElement | null) => {
		if (shown) shown.hidden = true;
		shown = node ? cards.get(node.dataset.id) : undefined;
		if (!shown || !node) return;
		shown.hidden = false;
		place(shown, node, fig);
	};
}

/**
 * Puts a card beside its node: to the right if it fits in the viewport, else to the left, else
 * below. The card is positioned against the figure (which is position: relative).
 */
function place(card: HTMLElement, node: Element, fig: HTMLElement) {
	const gap = 10;
	const margin = 8;
	const r = node.getBoundingClientRect();
	const f = fig.getBoundingClientRect();
	const { width: w, height: h } = card.getBoundingClientRect();
	let x = r.right + gap;
	let y = r.top + r.height / 2 - h / 2;
	if (x + w > innerWidth - margin) x = r.left - gap - w;
	if (x < margin) {
		x = Math.min(
			Math.max(margin, r.left + r.width / 2 - w / 2),
			innerWidth - margin - w,
		);
		y = r.bottom + gap;
	} else {
		y = Math.min(Math.max(margin, y), innerHeight - margin - h);
	}
	card.style.left = `${x - f.left}px`;
	card.style.top = `${y - f.top}px`;
}
