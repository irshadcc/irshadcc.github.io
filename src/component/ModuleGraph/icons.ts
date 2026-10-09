// Icons for ModuleGraph cards, drawn as SVG markup in a square of side s: thin outlines and small
// solid marks in one accent colour each, set by the classes styled in ModuleGraph.astro.
//   fn        an outlined rounded square with "Fn": an op
//   hexagon   a hexagon with an inner triangle and dots at its corners: a leaf module (Linear, ...)
//   exchange  an outlined rounded square with two opposite arrows: a collective
//   file      a page with a folded corner: an input
//   commit    a line through a solid dot: an output
//   sparkle   a four-pointed star
//   dot       a faint disc
// A badge ({ text, color?, outline? }) is a small square with up to three characters.

export const ICON_NAMES = [
	"fn",
	"hexagon",
	"exchange",
	"file",
	"commit",
	"sparkle",
	"dot",
] as const;
export type IconName = (typeof ICON_NAMES)[number];

export interface BadgeIcon {
	text: string;
	/** Fill (or outline) colour; defaults to the muted text colour. */
	color?: string;
	outline?: boolean;
}

const f = (v: number) => (Math.round(v * 10) / 10).toString();
const esc = (t: string) =>
	t
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;");

/** The icon's markup, with its top-left corner at (x, y) and side s. */
export function iconSvg(
	icon: IconName | BadgeIcon,
	x: number,
	y: number,
	s: number,
): string {
	const cx = x + s / 2;
	const cy = y + s / 2;
	if (typeof icon === "object") {
		const color = icon.color ?? "var(--mg-muted)";
		const size = f((icon.text.length > 2 ? 0.34 : 0.45) * s);
		const square = `<rect x="${f(x + 0.5)}" y="${f(y + 0.5)}" width="${f(s - 1)}" height="${f(s - 1)}" rx="3" style="fill:${icon.outline ? "none" : color};stroke:${color};stroke-width:1.3"/>`;
		const label = `<text x="${f(cx)}" y="${f(cy + s * 0.15)}" text-anchor="middle" font-size="${size}" font-weight="700" style="fill:${icon.outline ? color : "var(--mg-card)"}">${esc(icon.text)}</text>`;
		return square + label;
	}
	switch (icon) {
		case "fn":
			return `<rect class="mg-ic-fn" x="${f(x + 1)}" y="${f(y + 1)}" width="${f(s - 2)}" height="${f(s - 2)}" rx="4"/><text class="mg-ic-fn-t" x="${f(cx)}" y="${f(cy + s * 0.16)}" text-anchor="middle" font-size="${f(s * 0.42)}">Fn</text>`;
		case "hexagon": {
			const r = s * 0.44;
			const corner = (i: number) => {
				const a = Math.PI / 6 + (i * Math.PI) / 3;
				return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
			};
			const corners = [0, 1, 2, 3, 4, 5].map(corner);
			const pts = (ps: number[][]) =>
				ps.map(([px, py]) => `${f(px)},${f(py)}`).join(" ");
			const dots = corners
				.map(
					([px, py]) =>
						`<circle class="mg-ic-hex-d" cx="${f(px)}" cy="${f(py)}" r="${f(s * 0.07)}"/>`,
				)
				.join("");
			return `<polygon class="mg-ic-hex" points="${pts(corners)}"/><polygon class="mg-ic-hex" points="${pts([corners[1], corners[3], corners[5]])}"/>${dots}`;
		}
		case "exchange": {
			const l = x + s * 0.25;
			const r = x + s * 0.75;
			const t = y + s * 0.37;
			const b = y + s * 0.63;
			const h = s * 0.12;
			return `<rect class="mg-ic-x" x="${f(x + 1)}" y="${f(y + 1)}" width="${f(s - 2)}" height="${f(s - 2)}" rx="4"/><path class="mg-ic-x" d="M${f(l)},${f(t)}H${f(r)}M${f(r - h)},${f(t - h)}L${f(r)},${f(t)}L${f(r - h)},${f(t + h)}M${f(r)},${f(b)}H${f(l)}M${f(l + h)},${f(b - h)}L${f(l)},${f(b)}L${f(l + h)},${f(b + h)}"/>`;
		}
		case "file":
			return `<path class="mg-ic-line" d="M${f(x + s * 0.22)},${f(y + 1.5)}H${f(x + s * 0.6)}L${f(x + s * 0.8)},${f(y + s * 0.22)}V${f(y + s - 1.5)}H${f(x + s * 0.22)}ZM${f(x + s * 0.6)},${f(y + 1.5)}V${f(y + s * 0.22)}H${f(x + s * 0.8)}"/>`;
		case "commit":
			return `<path class="mg-ic-line" d="M${f(cx)},${f(y)}V${f(y + s)}"/><circle class="mg-ic-solid" cx="${f(cx)}" cy="${f(cy)}" r="${f(s * 0.17)}"/>`;
		case "sparkle": {
			const r = s / 2;
			const k = r * 0.2;
			return `<path class="mg-ic-spark" d="M${f(cx)},${f(cy - r)}Q${f(cx + k)},${f(cy - k)} ${f(cx + r)},${f(cy)}Q${f(cx + k)},${f(cy + k)} ${f(cx)},${f(cy + r)}Q${f(cx - k)},${f(cy + k)} ${f(cx - r)},${f(cy)}Q${f(cx - k)},${f(cy - k)} ${f(cx)},${f(cy - r)}Z"/>`;
		}
		default:
			return `<circle class="mg-ic-dot" cx="${f(cx)}" cy="${f(cy)}" r="${f(s * 0.36)}"/>`;
	}
}
