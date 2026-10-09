// Continuous colour scales for heat-maps: map a number in [min, max] to a colour.
// Named scales are ColorBrewer (sequential single-hue and diverging) and the matplotlib
// perceptual scales (viridis & co.), each given as 9 evenly spaced stops, low -> high.
// Any array of CSS hex colours works as a custom scale.
//
//   const scale = colorScale("viridis", [2.0, 2.5]);
//   scale.color(2.3)   // "#35b779"-ish
//   scale.text(2.3)    // "#1c1c1a" or "#ffffff", whichever reads on that fill

export const COLOR_SCALES = {
	// Sequential, one hue: light = low, dark = high.
	reds: ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"],
	blues: ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"],
	greens: ["#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#006d2c", "#00441b"],
	oranges: ["#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#a63603", "#7f2704"],
	purples: ["#fcfbfd", "#efedf5", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#54278f", "#3f007d"],
	greys: ["#ffffff", "#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252", "#252525", "#000000"],
	// Pale pink to red.
	pinkRed: ["#fbe3e3", "#f7c9c7", "#f4b0ab", "#f1968f", "#ef7c73", "#ee6357", "#ee4a3b", "#e8392a", "#d92b1d"],
	// Perceptual multi-hue (matplotlib): even steps in lightness, readable in greyscale.
	viridis: ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c", "#28ae80", "#5ec962", "#addc30", "#fde725"],
	magma: ["#000004", "#1c1044", "#4f127b", "#812581", "#b5367a", "#e55964", "#fb8761", "#fec287", "#fcfdbf"],
	inferno: ["#000004", "#1f0c48", "#550f6d", "#88226a", "#ba3655", "#e35933", "#f98e09", "#f9cb35", "#fcffa4"],
	plasma: ["#0d0887", "#4c02a1", "#7e03a8", "#a92395", "#cc4778", "#e56b5d", "#f89540", "#fdc527", "#f0f921"],
	cividis: ["#00224e", "#123570", "#3b496c", "#575d6d", "#707173", "#8a8779", "#a69d75", "#c4b56c", "#fee838"],
	turbo: ["#30123b", "#4662d7", "#36aaf9", "#1ae4b6", "#72fe5e", "#c7ef34", "#fbb938", "#f56918", "#7a0403"],
	// Diverging: two hues around a neutral middle; centre the domain on the value that means
	// "nothing unusual" (e.g. the median time).
	rdbu: ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"],
	rdylgn: ["#1a9850", "#66bd63", "#a6d96a", "#d9ef8b", "#ffffbf", "#fee08b", "#fdae61", "#f46d43", "#d73027"],
	piyg: ["#4d9221", "#7fbc41", "#b8e186", "#e6f5d0", "#f7f7f7", "#fde0ef", "#f1b6da", "#de77ae", "#c51b7d"],
	brbg: ["#01665e", "#35978f", "#80cdc1", "#c7eae5", "#f5f5f5", "#f6e8c3", "#dfc27d", "#bf812d", "#8c510a"],
} as const satisfies Record<string, readonly string[]>;

export type ColorScaleName = keyof typeof COLOR_SCALES;

/** A named scale, or your own stops (at least two hex colours, low -> high). */
export type ColorScaleSpec = ColorScaleName | readonly string[];

type RGB = [number, number, number];

const toRgb = (hex: string): RGB => {
	const h = hex.replace("#", "");
	const full = h.length === 3 ? [...h].map((c) => c + c).join("") : h;
	if (!/^[0-9a-f]{6}$/i.test(full)) throw new Error(`colorScale: "${hex}" is not a hex colour`);
	return [0, 2, 4].map((i) => Number.parseInt(full.slice(i, i + 2), 16)) as RGB;
};
const toHex = (c: RGB) => `#${c.map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")}`;

// Interpolate in linear light, not raw sRGB, so midpoints aren't muddy or too dark.
const toLinear = (v: number) => {
	const c = v / 255;
	return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
};
const fromLinear = (v: number) => 255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055);

/** WCAG relative luminance, 0 (black) to 1 (white). */
export function luminance(hex: string): number {
	const [r, g, b] = toRgb(hex).map(toLinear);
	return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export interface ColorScale {
	/** Fill colour for a value; values outside the domain are clamped to its ends. */
	color(v: number): string;
	/** Dark or white text, whichever has more contrast on color(v). */
	text(v: number): string;
	/** A CSS linear-gradient() of the whole scale, for a colour bar. */
	gradient(direction?: string): string;
	domain: readonly [number, number];
}

export function colorScale(spec: ColorScaleSpec, domain: readonly [number, number], reverse = false): ColorScale {
	const hexes = typeof spec === "string" ? COLOR_SCALES[spec as ColorScaleName] : spec;
	if (!hexes) throw new Error(`colorScale: unknown scale "${spec}"`);
	if (hexes.length < 2) throw new Error("colorScale: a custom scale needs at least two colours");
	const stops = (reverse ? [...hexes].reverse() : [...hexes]).map((h) => toRgb(h).map(toLinear) as RGB);
	const [lo, hi] = domain;

	const color = (v: number) => {
		const t = hi === lo ? 0.5 : Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
		const x = t * (stops.length - 1);
		const i = Math.min(Math.floor(x), stops.length - 2);
		const f = x - i;
		return toHex(stops[i].map((c, k) => fromLinear(c + (stops[i + 1][k] - c) * f)) as RGB);
	};
	// Contrast against white vs. near-black text; pick the larger (WCAG contrast ratio).
	const dark = "#1c1c1a";
	const darkL = luminance(dark);
	const text = (v: number) => {
		const l = luminance(color(v));
		return 1.05 / (l + 0.05) > (l + 0.05) / (darkL + 0.05) ? "#ffffff" : dark;
	};
	const gradient = (direction = "to right") =>
		`linear-gradient(${direction}, ${Array.from({ length: 11 }, (_, i) => color(lo + ((hi - lo) * i) / 10)).join(", ")})`;

	return { color, text, gradient, domain };
}
