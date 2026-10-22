// Browser helpers for the diffusion figures: resolve a CSS colour to RGB and paint a density
// heat map into an image that an SVG <image> can show.

/** The computed rgb(...) of any CSS colour expression, e.g. "var(--accent)", inside `root`. */
export function resolveColor(
	root: Element,
	css: string,
): [number, number, number] {
	const probe = document.createElement("span");
	probe.style.color = css;
	probe.style.display = "none";
	root.append(probe);
	const m = getComputedStyle(probe).color.match(/[\d.]+/g) ?? ["0", "0", "0"];
	probe.remove();
	return [Number(m[0]), Number(m[1]), Number(m[2])];
}

/**
 * Paint f(column, row) into a cols × rows image and return it as a data URL. Each column is
 * scaled to its own maximum (or, with perColumn = false, to the image maximum); opacity follows
 * (value / max)^0.6.
 */
export function heatImage(
	cols: number,
	rows: number,
	f: (c: number, r: number) => number,
	rgb: [number, number, number],
	perColumn = true,
): string {
	const canvas = document.createElement("canvas");
	canvas.width = cols;
	canvas.height = rows;
	const ctx = canvas.getContext("2d");
	if (!ctx) return "";
	const img = ctx.createImageData(cols, rows);
	const values = Array.from({ length: cols }, (_, c) =>
		Array.from({ length: rows }, (_, r) => f(c, r)),
	);
	const globalMax = Math.max(...values.map((col) => Math.max(...col))) || 1;
	for (let c = 0; c < cols; c++) {
		const col = values[c];
		const max = perColumn ? Math.max(...col) || 1 : globalMax;
		col.forEach((v, r) => {
			const o = (r * cols + c) * 4;
			img.data[o] = rgb[0];
			img.data[o + 1] = rgb[1];
			img.data[o + 2] = rgb[2];
			img.data[o + 3] = Math.round(200 * (v / max) ** 0.6);
		});
	}
	ctx.putImageData(img, 0, 0);
	return canvas.toDataURL();
}
