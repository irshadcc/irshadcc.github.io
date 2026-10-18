// One-paragraph descriptions of the passes in Triton 3.8.0's NVIDIA pipeline (make_ttir, make_ttgir
// and make_llir in third_party/nvidia/backend/compiler.py at git c01b6774), for TritonPasses.astro in
// the Triton JIT post. Each is summarised from the pass's TableGen description and, where that is
// thin, its source; the remarks about the vector add and the matmul come from the dumps in
// tritonJitPasses.ts.

export const JIT_PASS_INFO: Record<string, string> = {
	Inliner:
		"MLIR's inliner. Calls to other @triton.jit functions are separate tt.func ops after parsing; this pass inlines them into the kernel and simplifies each callee with a nested canonicalizer. In the vector add it has no calls to inline, but the nested canonicalizer deletes the dead overflow checks the frontend emitted. In the matmul it inlines three calls to tl.cdiv and one to tl.zeros, which are @triton.jit functions in triton/language/standard.py, each specialised for its constant arguments (cdiv__i32_c128).",
	Canonicalizer:
		"MLIR's canonicalizer: every op's folding and rewrite patterns, applied greedily to a fixed point (constant folding, x + 0 → x, dead code).",
	TritonCombineOps:
		"Triton peepholes: dot(a, b, 0) + c → dot(a, b, c), addptr(addptr(p, i), j) → addptr(p, i + j), a select around a masked load folded into the load's `other`, broadcast of a constant → a constant.",
	TritonReorderBroadcast:
		"Moves broadcast and splat after elementwise ops: elementwise(broadcast(a)) → broadcast(elementwise(a)), so the arithmetic runs on the small tensor.",
	CSE: "MLIR's common subexpression elimination over the whole module.",
	SymbolDCE: "Deletes functions and other symbols nothing references.",
	SCCP: "MLIR's sparse conditional constant propagation: folds values that are constant on every path and removes unreachable blocks.",
	TritonLoopUnroll:
		"Unrolls scf.for loops that carry a tt.loop_unroll_factor attribute (set from Python with tl.range(..., loop_unroll_factor=n)). Neither kernel asks for it.",
	ConvertTritonToTritonGPU:
		"The step from the hardware-neutral tt dialect to ttg: every tensor type gets a layout encoding, which says which thread of which warp holds each element. Each value gets a default #blocked layout built from num_warps and 32 threads per warp, one element per thread per repetition (sizePerThread = [1]).",
	TritonGPUCoalesce:
		"Picks a layout for every load and store from the axis analysis of its pointer (which dimension is contiguous, how long the runs are, how aligned) so the threads of a warp touch consecutive addresses and each thread holds as wide a vector as allowed (128 bits). In the vector add it gives the loads and the store sizePerThread = [4] and wraps them in ttg.convert_layout.",
	TritonGPUF32DotTC:
		"Decomposes fp32 tt.dot into several lower-precision dots (3xTF32 or BF16 splits) when the dot asks for that input precision. The matmul here is fp16.",
	TritonGPUPlanCTAPass:
		"Plans how dots, reductions and stores split across the CTAs of a cluster when num_ctas > 1. Here num_ctas = 1.",
	TritonGPURemoveLayoutConversions:
		"The main layout pass. It propagates a good layout forward and backward through elementwise ops, rematerializes cheap values in the layout their user wants, and deletes ttg.convert_layout ops that become identities. In the vector add it pushes the coalesced layout onto every op, so all the conversions disappear.",
	TritonGPUOptimizeThreadLocality:
		"Reshapes reductions yielded by a loop so each thread keeps a partial result until after the loop, and picks warp-local layouts for gathers. Neither applies here.",
	TritonGPUAccelerateMatmul:
		"Turns tt.dot into the tensor-core form for the target. On sm_120 the result gets an #mma layout of version 2 (mma.sync, instruction shape 16×8); on sm_90 version 3 with shape 16×128×16, and the dot becomes ttng.warp_group_dot (wgmma), which reads A and B from shared memory.",
	TritonGPUOptimizeDotOperands:
		"Moves conversions to the dot-operand layout earlier (past elementwise ops and transposes) so transposition can be done on the way into shared memory or by ldmatrix.",
	TritonNvidiaGPUOptimizeDescriptorEncodingPass:
		"Chooses the shared-memory encoding (swizzle, message size) of TMA tensor descriptors. No descriptors here.",
	TritonLoopAwareCSE:
		"CSE that also merges loop-carried values that are provably the same in every iteration.",
	TritonGPUFuseNestedLoops:
		"Flattens a loop nest into one loop so the pipeliner can overlap across the outer iterations (persistent kernels). The matmul has one loop.",
	TritonLoopInvariantCodeMotion:
		"MLIR's LICM plus hoisting of loads from read-only loops. In the matmul it moves BLOCK_SIZE_K * stride_bk, the pointer step of B, out of the K loop.",
	TritonGPUCombineTensorSelectAndIf:
		"Folds an arith.select into an scf.if with the same condition, returning the two operands from its branches.",
	NVGPUWarpSpecialization:
		"Hopper automatic warp specialization: splits a loop marked with warp_specialize into producer and consumer warp groups that talk through shared memory and barriers. The tutorial loop is not marked.",
	TritonGPUOptimizeAccumulatorInit:
		"For MMAs that have a use-accumulator flag (tcgen05 on sm_100), replaces the zero-initialised accumulator with a flag that is false on the first iteration. Not used by mma.sync.",
	TritonGPUHoistTMEMAlloc:
		"Moves tensor-memory allocations out of loops, so accumulators stay in tensor memory across iterations (sm_100). The matmul on sm_120 has no tensor memory; the pass only touches token plumbing.",
	TritonNvidiaGPUPromoteLHSToTMemPass:
		"Moves the left operand of a tcgen05 MMA into tensor memory when that is cheaper. sm_100 only.",
	TritonGPUAssignLatencies:
		"First pipelining step: tags each op whose latency should be hidden with tt.latency, measured in loop iterations. In the matmul the loads of A and B get tt.latency = 2, which is num_stages − 1.",
	TritonGPUScheduleLoops:
		"Builds the software-pipelining schedule: every op in the loop gets loop.stage (how many iterations ahead it runs) and loop.cluster (its order within an iteration). The loads land in stage 0, the dot in stage 2.",
	TritonGPUAutomaticWarpSpecialization:
		"Tries to split an eligible loop into ttg.warp_specialize partitions (producer warps loading, consumer warps computing). It wraps the passes below; neither kernel qualifies, so nothing changes.",
	TritonGPUPartitionScheduling:
		"Part of warp specialization: decides which partition each load, MMA and other op of the loop belongs to.",
	NVWSHoistTmemStore:
		"Part of warp specialization: hoists a tensor-memory allocation and its initialising store out of a loop nest.",
	NVWSInsertAref:
		"Part of warp specialization: inserts arefs (asynchronous references) between producer and consumer partitions, so the barrier handshakes can be generated automatically.",
	NVWSInsertTmemAref:
		"Part of warp specialization: inserts arefs where ownership of a tensor-memory buffer passes between two partitions.",
	NVWSLowerAref:
		"Part of warp specialization: lowers arefs to mbarrier wait and arrive ops.",
	NVWSAssignStagePhase:
		"Part of warp specialization: gives each aref op its buffer stage and barrier phase.",
	TritonGPUPartitionLoops:
		"Part of warp specialization: splits a scheduled loop into ttg.warp_specialize regions, one per partition.",
	NVWSLowerWarpGroup:
		"Part of warp specialization: converts nvws.warp_group to ttg.warp_specialize.",
	TritonGPUPipeline:
		"Applies the schedule. Loads with a latency become ttg.async_copy_global_to_local (cp.async) into a ring of shared-memory buffers; a prologue issues the first copies; each iteration waits for the oldest copy (ttg.async_wait), runs the dot and issues the copy for a later iteration. On sm_90 the wgmma also becomes asynchronous. The ring has 2 buffers on sm_120 and 3 on sm_90 in this compile.",
	TritonGPUOptimizePartitionWarps:
		"Part of warp specialization: shrinks the number of warps and registers given to each partition.",
	TritonNvidiaGPURemoveTMEMTokensPass:
		"Removes the tensor-memory dependency tokens once they are no longer needed. It leaves a ub.poison token behind, which the next canonicalizer deletes.",
	TritonGPUPrefetch:
		"For sm_80 mma: splits each dot along K into slices whose operands are loaded from shared memory one slice ahead, overlapping ldmatrix with mma. Only added when capability // 10 == 8.",
	TritonGPUCoalesceAsyncCopy:
		"Narrows the per-thread vector of an async copy when the destination's shared layout cannot take the full width, keeping cp.async transactions valid.",
	TritonNvidiaGPUOptimizeTMemLayoutsPass:
		"sm_100 only: chooses tensor-memory layouts for tcgen05 accumulators.",
	TritonNvidiaGPUFuseTMEMLoadReducePass:
		"sm_100 only: fuses a tensor-memory load followed by a row reduction into one tcgen05.ld.red.",
	TritonNvidiaGPUTMALoweringPass:
		"Lowers loads and stores through tensor descriptors to TMA copies (sm_90 and later). No descriptors here.",
	TritonNvidiaGPUInterleaveTMemPass:
		"sm_100 only: reorders tensor-memory loads and stores to cut register pressure.",
	TritonGPUReduceDataDuplication:
		"Routes conversions from a distributed layout to a dot-operand layout through shared memory, so several users share one copy instead of duplicating data in registers.",
	TritonGPUReorderInstructions:
		"Moves ops to cut register pressure (for example, shared-memory loads next to their first use) and into an order ptxas schedules well.",
	TritonGPUFenceInsertion:
		"Inserts fence.proxy.async where a normal write to shared memory is later read by the async proxy (wgmma, TMA).",
	TritonNvidiaGPUMMALoweringPass:
		"Prepares MMA ops for LLVM conversion where needed (for example, scaled MMAs).",
	TritonGPUAllocateWarpGroups:
		"Counts the warps the kernel needs, including any warp-specialized partitions, and records ttg.total-num-warps (4 for both kernels).",
	SCFToControlFlowPass:
		"Lowers structured control flow (scf.for, scf.if) to basic blocks and branches (cf dialect).",
	GluonInline:
		"Inlines any remaining Gluon helper functions. Nothing to do here.",
	AllocateSharedMemoryNv:
		"Plans shared memory: computes the lifetime of every shared buffer and of the scratch space each layout conversion needs, assigns byte offsets that reuse memory between non-overlapping lifetimes, and records the total as ttg.shared: 0 bytes for the vector add, 32768 for the matmul on sm_120, 49152 on sm_90.",
	TritonTensorMemoryAllocationPass:
		"sm_100 only: allocates tensor memory; records ttg.tensor_memory_size = 0 here.",
	TritonNvidiaGPUCheckMatmulTwoCTAPass:
		"Checks that all matmuls agree on two-CTA mode and records the choice on the module.",
	TritonGPUProxyFenceInsertion:
		"Adds any fence.proxy.async still required between normal writes and async-proxy reads of shared memory. On sm_90 it adds one fence in the matmul, before the wgmma reads the buffers.",
	TritonNvidiaGPUTMemBarrierInsertionPass:
		"Inserts CTA barriers between tensor-memory accesses that reuse the same physical storage (sm_100).",
	ConvertTritonGPUToLLVM:
		"The big lowering. Each ttg op becomes LLVM-dialect code for one thread: a tensor becomes the registers that thread holds under its layout, tt.get_program_id becomes a read of ctaid.x, loads and stores become inline-PTX ld.global / st.global guarded by predicates, dots become mma.sync or wgmma. Many hundred lines appear, most of them redundant.",
	InitializeWSClusterBarriers:
		"Initialises the mbarriers of warp-specialized cluster kernels. Nothing here.",
	CanonicalizeLLVMIR:
		"Simplifies the LLVM dialect produced by the conversion. For the vector add it shrinks the module from about 510 lines to 120.",
	ConvertWarpSpecializeToLLVM:
		"Lowers ttg.warp_specialize regions into one function where warps branch on their id. Nothing here.",
	ReconcileUnrealizedCastsPass:
		"Removes the casts left between type systems during the dialect conversions.",
	ConvertNVGPUToLLVM:
		"Lowers Triton's nvgpu helper ops (warp id, cluster id, wgmma waits, barriers) to NVVM or inline PTX.",
	ConvertNVVMToLLVMPass:
		"Lowers the remaining NVVM ops to LLVM intrinsics or inline PTX.",
	LLVMDIScope:
		"Attaches debug scopes so line information survives into LLVM IR and PTX.",
};
