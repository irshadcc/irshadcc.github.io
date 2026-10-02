// Client-side behaviour shared by the step-through figures in this folder (CodeStages,
// SelectionDag): the first / previous / play / next buttons and the labelled timeline drawn by
// StepControls.astro, plus the arrow keys while the figure has focus. The figure supplies
// render(i); this keeps the current step, the buttons and the timeline in sync.

export function bindStepper(
	fig: HTMLElement,
	count: number,
	render: (i: number) => void,
	intervalMs = 2500,
): void {
	const segs = [...fig.querySelectorAll<HTMLButtonElement>(".st-seg")];
	const counter = fig.querySelector<HTMLElement>(".st-counter");
	const play = fig.querySelector<HTMLButtonElement>("[data-act=play]");
	let at = Math.max(0, Math.min(count - 1, Number(fig.dataset.step ?? 0)));
	let timer: number | undefined;

	const go = (i: number) => {
		at = Math.max(0, Math.min(count - 1, i));
		segs.forEach((s, k) => {
			s.classList.toggle("done", k <= at);
			s.setAttribute("aria-current", String(k === at));
		});
		if (counter) counter.textContent = `${at + 1} / ${count}`;
		render(at);
	};
	const stop = () => {
		if (timer !== undefined) clearInterval(timer);
		timer = undefined;
		play?.classList.remove("playing");
		play?.setAttribute("aria-label", "Play");
	};
	const start = () => {
		if (at >= count - 1) go(0);
		play?.classList.add("playing");
		play?.setAttribute("aria-label", "Pause");
		timer = window.setInterval(
			() => (at >= count - 1 ? stop() : go(at + 1)),
			intervalMs,
		);
	};

	for (const b of fig.querySelectorAll<HTMLButtonElement>("[data-act]")) {
		b.addEventListener("click", () => {
			const act = b.dataset.act;
			if (act === "play") return timer === undefined ? start() : stop();
			stop();
			if (act === "first") go(0);
			if (act === "prev") go(at - 1);
			if (act === "next") go(at + 1);
		});
	}
	segs.forEach((s, k) =>
		s.addEventListener("click", () => {
			stop();
			go(k);
		}),
	);
	fig.addEventListener("keydown", (e) => {
		if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
		e.preventDefault();
		stop();
		go(at + (e.key === "ArrowRight" ? 1 : -1));
	});
	go(at);
}
