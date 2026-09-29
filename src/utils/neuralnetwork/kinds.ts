// Node kinds as NeuralNetworkGraph presents them: legend order and names, and the CSS class
// that gives a node its colour (nnPalette.css maps each class to a colour in --c).
import type { NodeKind } from "./NNGraph";

/** Legend order and names. Outputs share the input entry (see legendKind). */
export const KINDS: { kind: NodeKind; name: string }[] = [
	{ kind: "input", name: "Input / output" },
	{ kind: "embedding", name: "Embedding" },
	{ kind: "linear", name: "Linear / MLP" },
	{ kind: "conv", name: "Convolution" },
	{ kind: "pool", name: "Pooling" },
	{ kind: "attention", name: "Attention" },
	{ kind: "norm", name: "Normalization" },
	{ kind: "activation", name: "Activation" },
	{ kind: "op", name: "Operation" },
	{ kind: "other", name: "Other" },
];

/** The kind a node is listed and coloured as: outputs look like inputs. */
export const legendKind = (k: NodeKind): NodeKind =>
	k === "output" ? "input" : k;

/** The class that sets a node kind's colour. */
export const kindClass = (k: NodeKind) => `k-${legendKind(k)}`;
