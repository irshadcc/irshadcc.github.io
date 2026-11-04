// Figure data for the post vllm-sglang-model-integration: the BoxDiagram of what a model file
// owns and what the serving engine owns. Notes cite vLLM 54beee8 and SGLang 5eb7018.
import type { BoxEdge, BoxNode } from "../payments/boxDiagram";

export const integrationNodes: BoxNode[] = [
	{
		id: "model",
		label: "Model file",
		sub: "LlamaForCausalLM",
		col: 0,
		row: 0,
		group: 0,
		note: "Written by the engine's authors, one file per architecture (<code>vllm/model_executor/models/llama.py</code>, <code>python/sglang/srt/models/llama.py</code>). Built from the engine's parallel layers (<code>QKVParallelLinear</code>, <code>RowParallelLinear</code>, <code>RMSNorm</code>) and loads Hugging Face checkpoints by name. It never allocates or indexes the KV cache.",
	},
	{
		id: "attn",
		label: "Attention layer",
		sub: "Attention, RadixAttention",
		col: 1,
		row: 0,
		group: 1,
		note: "The one layer that touches the cache. The model calls it with <code>q, k, v</code> (vLLM) or <code>q, k, v, forward_batch</code> (SGLang). It finds the cache and the per-step metadata itself: by its layer name in vLLM's forward context, by its <code>layer_id</code> in SGLang's KV pool.",
	},
	{
		id: "backend",
		label: "Attention backend",
		sub: "FlashAttention, ...",
		col: 2,
		row: 0,
		group: 2,
		note: "Chosen at start-up for the GPU and model. Writes the new keys and values (<code>reshape_and_cache_flash</code> in vLLM, <code>store_kvcache_kernel</code> in SGLang) and runs the attention kernel over the cached ones, using the metadata the runner built for this step.",
	},
	{
		id: "cache",
		label: "KV cache",
		sub: "one tensor per layer",
		col: 3,
		row: 0,
		group: 3,
		note: "Allocated after profiling, from the memory left over. vLLM: blocks of 16 tokens, a page per token holding each head's K then V. SGLang: one row per token slot in a <code>k_buffer</code> and a <code>v_buffer</code> per layer.",
	},
	{
		id: "sched",
		label: "Scheduler",
		sub: "block table, radix tree",
		col: 2,
		row: 1,
		group: 4,
		note: "Decides which tokens run this step and where their keys and values live. vLLM's KV cache manager hands out blocks and finds cached prefixes by block hash; SGLang's radix tree finds the longest cached prefix and the allocator hands out free slots. Both run on the CPU.",
	},
	{
		id: "runner",
		label: "Model runner",
		sub: "forward context, batch",
		col: 1,
		row: 1,
		group: 4,
		note: "Turns the scheduler's decision into GPU tensors: vLLM's <code>slot_mapping</code> and block table, SGLang's <code>out_cache_loc</code> and <code>req_to_token</code> rows, each built by a small Triton kernel. Publishes them (vLLM's <code>set_forward_context</code>, SGLang's <code>ForwardBatch</code>) and calls the model.",
	},
	{
		id: "graph",
		label: "Compiler, CUDA graphs",
		sub: "torch.compile, graph runners",
		col: 0,
		row: 1,
		group: 5,
		note: "vLLM traces the model's <code>forward</code> once with Dynamo, splits the graph at attention, compiles the pieces with Inductor and captures them as CUDA graphs. SGLang captures CUDA graphs of the model as written (full for decode, breakable for prefill) and uses <code>torch.compile</code> only when asked.",
	},
];

export const integrationEdges: BoxEdge[] = [
	{ from: "model", to: "attn", label: "q, k, v" },
	{ from: "attn", to: "backend", label: "q, k, v" },
	{ from: "backend", to: "cache", label: "r/w", both: true },
	{ from: "sched", to: "runner", label: "slot ids" },
	{ from: "runner", to: "attn", label: "metadata", dashed: true },
	{ from: "graph", to: "model", label: "wraps", dashed: true },
];
