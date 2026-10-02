// One-paragraph descriptions of the passes in Triton's NVIDIA pipeline (make_ttir, make_ttgir and
// make_llir in third_party/nvidia/backend/compiler.py at Triton 5d6048aa), for TritonPasses.astro.
// Each is summarised from the pass's TableGen description and, where that is thin, its source.

export const PASS_INFO: Record<string, string> = {
	Inliner:
		"MLIR's inliner. Calls to @triton.jit helpers (triton_helpers.maximum, libdevice.exp, tl.zeros) are separate tt.func ops after parsing; this pass inlines them into the kernel, simplifying each callee with a nested canonicalizer, and deletes the dead helpers.",
	TritonRewriteTensorPointer:
		"Rewrites loads and stores through block pointers (tt.make_tensor_ptr, tt.advance) into plain pointer tensors with explicit masks. Inductor's kernels use plain pointers, so it has nothing to do.",
	Canonicalizer:
		"MLIR's canonicalizer: every op's folding and rewrite patterns, applied greedily to a fixed point (constant folding, x + 0 → x, dead code).",
	TritonCombineOps:
		"Triton peepholes: dot(a, b, 0) + c → dot(a, b, c), addptr(addptr(p, i), j) → addptr(p, i + j), a select around a masked load folded into the load's `other`, broadcast of a constant → a constant.",
	TritonReorderBroadcast:
		"Moves broadcast and splat after elementwise ops: elementwise(broadcast(a)) → broadcast(elementwise(a)), so the arithmetic runs on the small tensor.",
	CSE: "MLIR's common subexpression elimination over the whole module.",
	SymbolDCE: "Deletes functions and other symbols nothing references.",
	TritonLoopUnroll:
		"Unrolls scf.for loops that carry a tt.loop_unroll_factor attribute (set from Python with tl.range(..., loop_unroll_factor=n)). Neither kernel asks for it.",
	ConvertTritonToTritonGPU:
		"The step from the hardware-neutral tt dialect to ttg: every tensor type gets a layout encoding, which says which thread of which warp holds each element. This pass gives each value a default #blocked layout from num_warps and threads_per_warp, and inserts ttg.convert_layout wherever two values must agree.",
	TritonGPUCoalesce:
		"Picks a layout for each load and store from the axis analysis of its pointer (which dimension is contiguous, how many elements, how aligned) so that a warp's threads touch consecutive addresses and each thread loads as wide a vector as alignment allows. Conversions are inserted around the memory op.",
	TritonGPUF32DotTC:
		"Decomposes fp32 tt.dot into several lower-precision dots (3xTF32 or BF16 splits) when the dot asks for that input precision, so tensor cores can be used. The matmul here is fp16.",
	TritonGPUPlanCTAPass:
		"Plans how dots, reductions and stores split across the CTAs of a cluster when num_ctas > 1. Here num_ctas = 1.",
	TritonGPURemoveLayoutConversions:
		"The workhorse layout pass. It propagates a good layout forward and backward through elementwise ops (blocked for expensive loads and stores, mma for dot results), rematerializes cheap values in the layout their user wants, and deletes ttg.convert_layout ops that become identities.",
	TritonGPUOptimizeThreadLocality:
		"Reshapes reductions yielded by a loop so each thread keeps a partial result until after the loop, and picks warp-local layouts for gathers. Neither applies here.",
	TritonGPUAccelerateMatmul:
		"Turns tt.dot into the tensor-core op for the target: an #mma layout for the result, a shared-memory layout for the operands, and on sm_90 ttng.warp_group_dot (wgmma), which reads A and B from shared memory. It picks the instruction shape and warp arrangement.",
	TritonGPUOptimizeDotOperands:
		"Moves conversions to the dot-operand layout earlier (past elementwise ops and transposes) so transposition can be done by the hardware path into shared memory or ldmatrix.",
	TritonNvidiaGPUOptimizeDescriptorEncodingPass:
		"Chooses the shared-memory encoding (swizzle, message size) of TMA tensor descriptors. No descriptors here.",
	TritonLoopAwareCSE:
		"CSE that also merges loop-carried values that are provably the same in every iteration.",
	TritonGPUFuseNestedLoops:
		"Flattens a loop nest into one loop so the pipeliner can overlap across the outer iterations (persistent kernels). Only one loop here.",
	TritonLoopInvariantCodeMotion:
		"MLIR's LICM plus hoisting of loads from loops whose body is read-only, guarded by a trip-count check. Here it moves the loop-invariant address arithmetic (the row offsets of A, the column offsets of B, the base pointers) out of the K loop.",
	TritonGPUCombineTensorSelectAndIf:
		"Folds an arith.select into an scf.if with the same condition, returning the two operands from its branches.",
	NVGPUWarpSpecialization:
		"Hopper automatic warp specialization: splits a loop marked tt.warp_specialize into producer and consumer warp groups that communicate through shared memory and barriers. Runs only on such loops and only with 4 warps; Inductor's template loop is not marked.",
	TritonGPUAssignLatencies:
		"First pipelining step: tags each op whose latency should be hidden with tt.latency, in loop iterations. Here the A and B loads that feed the dot get tt.latency = 2, num_stages - 1.",
	TritonGPUScheduleLoops:
		"Builds the software-pipelining schedule: every op in the loop gets loop.stage (which iteration ahead it runs in) and loop.cluster (its order within an iteration). The loads land in stage 0, the shared-memory store and the dot in stage 2.",
	TritonGPUPipeline:
		"Applies the schedule. Loads with latency become ttg.async_copy_global_to_local into a ring of num_stages shared-memory buffers; a prologue issues the first stages; the loop waits on the oldest copy (ttg.async_wait), runs the dot, and issues the copy for a later iteration; the wgmma becomes asynchronous with a bounded number of groups in flight.",
	TritonGPUPrefetch:
		"For sm_80-style mma: splits each dot along K into slices whose operands are loaded from shared memory into registers one slice ahead, overlapping ldmatrix with mma. On sm_90 the wgmma reads shared memory directly, so it does nothing here.",
	TritonGPUCoalesceAsyncCopy:
		"Narrows the per-thread vector of an async copy when the destination's shared layout cannot take the full width, keeping cp.async transactions valid.",
	TritonNvidiaGPUOptimizeTMemLayoutsPass:
		"Blackwell (sm_100) only: chooses tensor-memory layouts for tcgen05 accumulators.",
	TritonNvidiaGPUTMALoweringPass:
		"Lowers loads and stores through tensor descriptors to TMA copy operations (sm_90 and later). No descriptors here.",
	TritonNvidiaGPUInterleaveTMemPass:
		"Blackwell only: reorders tensor-memory loads and stores to cut register pressure.",
	TritonGPUReduceDataDuplication:
		"Routes conversions from a distributed layout to a dot-operand layout through shared memory, so several users can share one copy instead of duplicating data in registers.",
	TritonGPUReorderInstructions:
		"Moves ops to cut register pressure (for example, shared-memory loads next to their first use) and into an order ptxas schedules well.",
	TritonGPUFenceInsertion:
		"Inserts fence.proxy.async where a generic-proxy write to shared memory (a register store) is later read by the async proxy (wgmma, TMA), at optimized positions.",
	TritonNvidiaGPUMMALoweringPass:
		"Prepares MMA ops for LLVM conversion where needed (for example, scaled MMAs).",
	SCCP: "MLIR's sparse conditional constant propagation; here it mostly hoists and renames constants.",
	TritonGPUAllocateWarpGroups:
		"Counts the warps the kernel needs, including any warp-specialized partitions, and records ttg.total-num-warps.",
	SCFToControlFlowPass:
		"Lowers structured control flow (scf.for, scf.if) to basic blocks and branches (cf dialect).",
	GluonInline:
		"Inlines any remaining Gluon helper functions. Nothing to do here.",
	AllocateSharedMemoryNv:
		"Plans shared memory: computes the lifetime of every shared buffer and of the scratch space each layout conversion or reduction needs, assigns byte offsets that reuse memory between non-overlapping lifetimes, and records the total as ttg.shared.",
	TritonTensorMemoryAllocationPass:
		"Blackwell only: allocates tensor memory; records ttg.tensor_memory_size = 0 here.",
	TritonNvidiaGPUCheckMatmulTwoCTAPass:
		"Checks that all matmuls agree on two-CTA mode and records the choice on the module.",
	TritonGPUGlobalScratchAllocationPass:
		"Allocates global-memory scratch some ops need; records a size of 0 here.",
	TritonGPUProxyFenceInsertion:
		"The functional counterpart of FenceInsertion: adds any fence.proxy.async still required between generic writes and async-proxy reads of shared memory. Here it adds one inside the K loop, right before the wgmma: cp.async fills the shared buffers through the generic proxy, and wgmma reads them through the async proxy.",
	ConvertTritonGPUToLLVM:
		"The big lowering. Each ttg op becomes LLVM-dialect code for one thread: a tensor becomes the registers that thread holds under its layout, reductions become warp shuffles plus shared memory when a reduction crosses warps, layout conversions go through shared memory, wgmma and cp.async become NVVM ops or inline PTX.",
	ConvertNVGPUToLLVM:
		"Lowers Triton's nvgpu helper ops (cluster ids, wgmma waits, barriers) to inline PTX.",
	ConvertWarpSpecializeToLLVM:
		"Lowers ttg.warp_specialize regions into one function where warps branch on their id and talk through shared memory and barriers. Nothing here.",
	ReconcileUnrealizedCastsPass:
		"Removes the casts left between type systems during the dialect conversions.",
	ConvertNVVMToLLVMPass:
		"Lowers the remaining NVVM ops to LLVM intrinsics or inline PTX.",
	LLVMDIScope:
		"Attaches debug scopes so line information survives into LLVM IR and PTX.",
};
