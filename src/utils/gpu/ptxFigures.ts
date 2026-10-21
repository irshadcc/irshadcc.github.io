// Programs for the PtxTrace figures in the PTX post (src/content/posts/ptx.mdx). saxpy, heavy and
// branchy are LLVM 23.1.2 output (clang -x cuda --cuda-device-only -O3 --cuda-gpu-arch=sm_90)
// with the comment lines removed; the two warp sums are written by hand for an 8-lane segment
// (c = 0x181f), the scaled-down form of the 32-lane reduction clang emits.
import type { BoxEdge, BoxNode } from "../payments/boxDiagram";
import type { PtxSimConfig } from "./ptxSim";

const X = 0x1000;
const Y = 0x2000;

export const saxpyTrace: PtxSimConfig = {
	code: `
ld.param.b32        %r3, [saxpy_param_0];   // n
ld.param.b64        %rd3, [saxpy_param_2];  // x
cvta.to.global.u64  %rd1, %rd3;
ld.param.b64        %rd4, [saxpy_param_3];  // y
cvta.to.global.u64  %rd2, %rd4;
mov.u32             %r4, %ctaid.x;
mov.u32             %r5, %ntid.x;
mov.u32             %r6, %tid.x;
mad.lo.s32          %r1, %r4, %r5, %r6;     // i
setp.ge.u32         %p1, %r1, %r3;          // i >= n ?
@%p1 bra            $L__BB0_2;
ld.param.b32        %r2, [saxpy_param_1];   // a
mul.wide.u32        %rd5, %r1, 4;
add.s64             %rd6, %rd1, %rd5;
ld.global.b32       %r7, [%rd6];            // x[i]
add.s64             %rd7, %rd2, %rd5;
ld.global.b32       %r8, [%rd7];            // y[i]
fma.rn.f32          %r9, %r2, %r7, %r8;     // a*x[i] + y[i]
st.global.b32       [%rd7], %r9;
$L__BB0_2:
ret;`,
	ntid: 8,
	ctaid: 1,
	params: {
		saxpy_param_0: 12,
		saxpy_param_1: 2,
		saxpy_param_2: X,
		saxpy_param_3: Y,
	},
	buffers: {
		x: { base: X, values: Array.from({ length: 16 }, (_, i) => i / 2) },
		y: { base: Y, values: Array.from({ length: 16 }, () => 1) },
	},
	show: ["%r6", "%r1", "%p1", "%rd5", "%r7", "%r8", "%r9"],
	floats: ["%r7", "%r8", "%r9"],
	notes: {
		9: "Each lane computes its global index i = ctaid.x · ntid.x + tid.x = 1 · 8 + tid.x.",
		10: "Only i = 8…11 are below n = 12.",
		11: "Lanes 4–7 jump to the label, lanes 0–3 fall through: the warp has diverged.",
		21: "The four lanes that skipped the body were waiting here; the warp is whole again.",
	},
};

export const heavyTrace: PtxSimConfig = {
	code: `
ld.param.b64        %rd3, [heavy_param_0];
cvta.to.global.u64  %rd4, %rd3;
ld.param.b64        %rd5, [heavy_param_1];
cvta.to.global.u64  %rd1, %rd5;
mov.u32             %r2, %tid.x;
cvt.u64.u32         %rd2, %r2;
mul.wide.u32        %rd6, %r2, 4;
add.s64             %rd7, %rd4, %rd6;
ld.global.b32       %r1, [%rd7];            // x
setp.leu.f32        %p1, %r1, 0f00000000;   // !(x > 0)
@%p1 bra            $L__BB1_2;
sin.approx.f32      %r7, %r1;               // then: sin(x) * 2^x + sqrt(x)
ex2.approx.f32      %r8, %r1;
sqrt.rn.f32         %r9, %r1;
fma.rn.f32          %r10, %r7, %r8, %r9;
bra.uni             $L__BB1_3;
$L__BB1_2:
cos.approx.f32      %r3, %r1;               // else: cos(x) * 2 - x^3
mul.f32             %r4, %r1, %r1;
mul.f32             %r5, %r1, %r4;
neg.f32             %r6, %r5;
fma.rn.f32          %r10, %r3, 0f40000000, %r6;
$L__BB1_3:
shl.b64             %rd8, %rd2, 2;
add.s64             %rd9, %rd1, %rd8;
st.global.b32       [%rd9], %r10;
ret;`,
	params: { heavy_param_0: X, heavy_param_1: Y },
	buffers: {
		in: { base: X, values: [1, -1, 2, -2, 0.5, -0.5, 3, -3] },
		out: { base: Y, values: Array.from({ length: 8 }, () => 0) },
	},
	show: ["%r1", "%p1", "%r7", "%r8", "%r9", "%r3", "%r6", "%r10"],
	floats: ["%r1", "%r7", "%r8", "%r9", "%r3", "%r6", "%r10"],
	notes: {
		11: "The odd lanes hold x ≤ 0 and jump to the else block; the even lanes fall through.",
		12: "The then block issues for the even lanes only.",
		16: "bra.uni tells ptxas the branch is not divergent: every lane that reaches it jumps.",
		18: "Now the else block issues for the odd lanes, while the even lanes wait at the join.",
		24: "Both groups meet at the join label and continue together.",
	},
};

export const branchyTrace: PtxSimConfig = {
	code: `
ld.param.b64        %rd1, [branchy_param_0];
cvta.to.global.u64  %rd2, %rd1;
ld.param.b64        %rd3, [branchy_param_1];
cvta.to.global.u64  %rd4, %rd3;
mov.u32             %r1, %tid.x;
mul.wide.u32        %rd5, %r1, 4;
add.s64             %rd6, %rd2, %rd5;
ld.global.b32       %r2, [%rd6];            // x
and.b32             %r3, %r2, 1;
setp.ne.b32         %p1, %r3, 0;            // x odd ?
mad.lo.s32          %r4, %r2, 3, 1;         // 3x + 1
shr.s32             %r5, %r2, 1;            // x / 2
selp.b32            %r6, %r4, %r5, %p1;     // odd ? 3x+1 : x/2
add.s64             %rd7, %rd4, %rd5;
st.global.b32       [%rd7], %r6;
ret;`,
	params: { branchy_param_0: X, branchy_param_1: Y },
	buffers: {
		in: { base: X, values: [3, 4, 7, 10, 5, 6, 1, 8] },
		out: { base: Y, values: Array.from({ length: 8 }, () => 0) },
	},
	show: ["%r2", "%r3", "%p1", "%r4", "%r5", "%r6"],
	notes: {
		11: "Both sides of the if are computed in every lane, with no branch.",
		13: "selp keeps one of the two results per lane.",
	},
};

const warpSum = (mode: "down" | "bfly") => `
mov.u32             %r1, %laneid;
ld.param.b64        %rd1, [in];
mul.wide.u32        %rd2, %r1, 4;
add.s64             %rd3, %rd1, %rd2;
ld.global.b32       %r2, [%rd3];            // v
shfl.sync.${mode}.b32  %r3, %r2, 4, 0x181f, -1;
add.f32             %r2, %r2, %r3;
shfl.sync.${mode}.b32  %r3, %r2, 2, 0x181f, -1;
add.f32             %r2, %r2, %r3;
shfl.sync.${mode}.b32  %r3, %r2, 1, 0x181f, -1;
add.f32             %r2, %r2, %r3;`;

const warpValues = [3, 1, 4, 1, 5, 9, 2, 6];

export const shflDownTrace: PtxSimConfig = {
	code: warpSum("down"),
	params: { in: X },
	buffers: { in: { base: X, values: warpValues } },
	show: ["%r1", "%r2", "%r3"],
	floats: ["%r2", "%r3"],
	notes: {
		6: "Lane i reads v from lane i + 4. Lanes 4–7 would read past lane 7, the clamp in c, so they get their own v.",
		8: "Lane i reads from lane i + 2.",
		10: "Lane i reads from lane i + 1.",
		11: "Lane 0 now holds the sum of all eight values, 31. The other lanes hold partial sums nobody needs.",
	},
};

export const shflBflyTrace: PtxSimConfig = {
	code: warpSum("bfly"),
	params: { in: X },
	buffers: { in: { base: X, values: warpValues } },
	show: ["%r1", "%r2", "%r3"],
	floats: ["%r2", "%r3"],
	notes: {
		6: "Lane i swaps with lane i XOR 4: 0↔4, 1↔5, 2↔6, 3↔7.",
		8: "Lane i swaps with lane i XOR 2.",
		10: "Lane i swaps with lane i XOR 1.",
		11: "Every lane now holds the full sum, 31.",
	},
};

// BoxDiagram data: where PTX sits in the toolchain, and the state spaces by who can see them.
export const toolchainNodes: BoxNode[] = [
	{
		id: "cuda",
		label: "CUDA C++",
		sub: ".cu",
		col: 0,
		row: 0,
		group: 0,
		note: "Kernels written in CUDA C++. <code>nvcc</code> splits the file into host code (sent to the host C++ compiler) and device code.",
	},
	{
		id: "dsl",
		label: "Triton, CuTe DSL, …",
		sub: "Python",
		col: 0,
		row: 1,
		group: 0,
		note: "Python DSLs lower through MLIR to LLVM IR and then use LLVM's NVPTX backend, so they also produce PTX.",
	},
	{
		id: "fe",
		label: "nvcc / clang",
		sub: "NVVM / LLVM",
		col: 1,
		row: 0,
		group: 1,
		note: "Both compilers are built on LLVM: nvcc's device compiler is NVIDIA's NVVM, clang uses the open-source NVPTX backend. Each optimises the kernel and emits PTX text.",
	},
	{
		id: "llvm",
		label: "LLVM NVPTX",
		sub: "backend",
		col: 1,
		row: 1,
		group: 1,
		note: "The same backend clang uses. See the <a href='/blog/llvm-nvptx-backend/'>LLVM post</a> for how it selects PTX instructions.",
	},
	{
		id: "ptx",
		label: "PTX",
		sub: "virtual ISA, .ptx text",
		col: 2,
		row: 0,
		h: 2,
		group: 2,
		note: "The stable, documented layer. It has unlimited typed virtual registers and no fixed encoding, so the same PTX can be compiled for later GPUs.",
	},
	{
		id: "ptxas",
		label: "ptxas",
		sub: "ahead of time",
		col: 3,
		row: 0,
		group: 3,
		note: "NVIDIA's assembler (really an optimising compiler): register allocation, scheduling and the real instruction encoding for one GPU.",
	},
	{
		id: "jit",
		label: "driver JIT",
		sub: "at load time",
		col: 3,
		row: 1,
		group: 3,
		note: "If the binary has no SASS for the GPU it runs on, the CUDA driver compiles the embedded PTX with the same ptxas logic and caches the result.",
	},
	{
		id: "sass",
		label: "SASS",
		sub: "cubin for sm_90",
		col: 4,
		row: 0,
		group: 4,
		note: "The machine code the SM executes. Its encoding changes between generations and is not formally documented.",
	},
	{
		id: "sass2",
		label: "SASS",
		sub: "for a newer GPU",
		col: 4,
		row: 1,
		group: 4,
		note: "Compiled on the user's machine from PTX that was shipped years earlier. This is why a fat binary should include PTX for its newest target.",
	},
];

export const toolchainEdges: BoxEdge[] = [
	{ from: "cuda", to: "fe" },
	{ from: "dsl", to: "llvm" },
	{ from: "fe", to: "ptx" },
	{ from: "llvm", to: "ptx" },
	{ from: "ptx", to: "ptxas" },
	{ from: "ptx", to: "jit", dashed: true },
	{ from: "ptxas", to: "sass" },
	{ from: "jit", to: "sass2" },
];

export const spaceNodes: BoxNode[] = [
	{
		id: "reg",
		label: ".reg",
		sub: "registers · one thread",
		col: 0,
		row: 0,
		group: 0,
		note: "Not addressable. PTX registers are virtual and unlimited; ptxas maps them to the physical register file and spills the rest to <code>.local</code>.",
	},
	{
		id: "local",
		label: ".local",
		sub: "stack, spills · one thread",
		col: 0,
		row: 1,
		group: 0,
		note: "Private to a thread but stored in device memory (cached in L1/L2), so it is slow. Arrays indexed by a runtime value often end up here.",
	},
	{
		id: "shared",
		label: ".shared",
		sub: "on-chip · one CTA",
		col: 1,
		row: 0,
		group: 1,
		note: "The SM's scratchpad, shared by the threads of a CTA for the CTA's lifetime. Since sm_90, threads in the same cluster can reach other CTAs' shared memory with <code>.shared::cluster</code> addresses.",
	},
	{
		id: "dsmem",
		label: ".shared::cluster",
		sub: "peer CTAs · one cluster",
		col: 1,
		row: 1,
		group: 1,
		note: "Distributed shared memory: <code>mapa</code> turns an address in my CTA's shared memory into the same variable's address in a peer CTA of the cluster.",
	},
	{
		id: "global",
		label: ".global",
		sub: "DRAM · every thread",
		col: 2,
		row: 0,
		group: 2,
		note: "Device memory, visible to all threads and the host, and kept across kernel launches. The slowest space; caching is controlled with operators such as <code>.ca</code> and <code>.cg</code>.",
	},
	{
		id: "const",
		label: ".const / .param",
		sub: "read-only · the grid",
		col: 2,
		row: 1,
		group: 2,
		note: "<code>.const</code> is cached read-only memory. Kernel parameters live in <code>.param</code> and are read with <code>ld.param</code>.",
	},
];

export const spaceEdges: BoxEdge[] = [
	{ from: "reg", to: "local", label: "spill", dashed: true },
	{ from: "shared", to: "dsmem", label: "mapa" },
];
