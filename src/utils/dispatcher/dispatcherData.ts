// Generated from PyTorch 2.9.1 (git 5811a8d7da, macOS arm64 wheel) with
// torch._C._dispatch_dump / _dispatch_dump_table. Do not edit by hand.

export interface Registration {
	key: string;
	site: string;
	fallthrough?: boolean;
}

/** Backend fallbacks: the computed table of an operator with no kernels of its own. */
export const BACKEND_FALLBACKS: Registration[] = [
	{
		key: "MPS",
		site: "aten/src/ATen/mps/MPSFallback.mm:79",
	},
	{
		key: "Meta",
		site: "aten/src/ATen/core/MetaFallbackKernel.cpp:23",
	},
	{
		key: "BackendSelect",
		site: "aten/src/ATen/core/BackendSelectFallbackKernel.cpp:3",
		fallthrough: true,
	},
	{
		key: "Python",
		site: "aten/src/ATen/core/PythonFallbackKernel.cpp:194",
	},
	{
		key: "FuncTorchDynamicLayerBackMode",
		site: "aten/src/ATen/functorch/DynamicLayer.cpp:479",
	},
	{
		key: "Functionalize",
		site: "aten/src/ATen/FunctionalizeFallbackKernel.cpp:387",
	},
	{
		key: "Named",
		site: "aten/src/ATen/core/NamedRegistrations.cpp:7",
	},
	{
		key: "Conjugate",
		site: "aten/src/ATen/ConjugateFallback.cpp:17",
	},
	{
		key: "Negative",
		site: "aten/src/ATen/native/NegateFallback.cpp:18",
	},
	{
		key: "ZeroTensor",
		site: "aten/src/ATen/ZeroTensorFallback.cpp:115",
	},
	{
		key: "ADInplaceOrView",
		site: "aten/src/ATen/core/VariableFallbackKernel.cpp:104",
		fallthrough: true,
	},
	{
		key: "AutogradOther",
		site: "aten/src/ATen/core/VariableFallbackKernel.cpp:63",
	},
	{
		key: "AutogradCPU",
		site: "aten/src/ATen/core/VariableFallbackKernel.cpp:67",
	},
	{
		key: "AutogradCUDA",
		site: "aten/src/ATen/core/VariableFallbackKernel.cpp:75",
	},
	{
		key: "AutogradXLA",
		site: "aten/src/ATen/core/VariableFallbackKernel.cpp:87",
	},
	{
		key: "AutogradMPS",
		site: "aten/src/ATen/core/VariableFallbackKernel.cpp:95",
	},
	{
		key: "AutogradXPU",
		site: "aten/src/ATen/core/VariableFallbackKernel.cpp:71",
	},
	{
		key: "AutogradHPU",
		site: "aten/src/ATen/core/VariableFallbackKernel.cpp:108",
	},
	{
		key: "AutogradLazy",
		site: "aten/src/ATen/core/VariableFallbackKernel.cpp:91",
	},
	{
		key: "AutogradMTIA",
		site: "aten/src/ATen/core/VariableFallbackKernel.cpp:79",
	},
	{
		key: "AutogradMAIA",
		site: "aten/src/ATen/core/VariableFallbackKernel.cpp:83",
	},
	{
		key: "AutogradMeta",
		site: "aten/src/ATen/core/VariableFallbackKernel.cpp:99",
	},
	{
		key: "Tracer",
		site: "torch/csrc/autograd/TraceTypeManual.cpp:294",
	},
	{
		key: "AutocastCPU",
		site: "aten/src/ATen/autocast_mode.cpp:324",
		fallthrough: true,
	},
	{
		key: "AutocastMTIA",
		site: "aten/src/ATen/autocast_mode.cpp:468",
		fallthrough: true,
	},
	{
		key: "AutocastMAIA",
		site: "aten/src/ATen/autocast_mode.cpp:506",
		fallthrough: true,
	},
	{
		key: "AutocastXPU",
		site: "aten/src/ATen/autocast_mode.cpp:544",
		fallthrough: true,
	},
	{
		key: "AutocastMPS",
		site: "aten/src/ATen/autocast_mode.cpp:209",
		fallthrough: true,
	},
	{
		key: "AutocastCUDA",
		site: "aten/src/ATen/autocast_mode.cpp:165",
		fallthrough: true,
	},
	{
		key: "FuncTorchBatched",
		site: "aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:731",
	},
	{
		key: "BatchedNestedTensor",
		site: "aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:758",
	},
	{
		key: "FuncTorchVmapMode",
		site: "aten/src/ATen/functorch/VmapModeRegistrations.cpp:27",
		fallthrough: true,
	},
	{
		key: "Batched",
		site: "aten/src/ATen/LegacyBatchingRegistrations.cpp:1075",
	},
	{
		key: "VmapMode",
		site: "aten/src/ATen/VmapModeRegistrations.cpp:33",
		fallthrough: true,
	},
	{
		key: "FuncTorchGradWrapper",
		site: "aten/src/ATen/functorch/TensorWrapper.cpp:210",
	},
	{
		key: "PythonTLSSnapshot",
		site: "aten/src/ATen/core/PythonFallbackKernel.cpp:202",
	},
	{
		key: "FuncTorchDynamicLayerFrontMode",
		site: "aten/src/ATen/functorch/DynamicLayer.cpp:475",
	},
	{
		key: "PreDispatch",
		site: "aten/src/ATen/core/PythonFallbackKernel.cpp:206",
	},
	{
		key: "PythonDispatcher",
		site: "aten/src/ATen/core/PythonFallbackKernel.cpp:198",
	},
];

/** Kernels registered for aten::add.Tensor, oldest first per key. */
export const ADD_TENSOR_REGISTRATIONS: Registration[] = [
	{
		key: "CompositeExplicitAutogradNonFunctional",
		site: "build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1374",
	},
	{
		key: "Autograd",
		site: "torch/csrc/autograd/generated/VariableType_2.cpp:20142",
	},
	{
		key: "NestedTensorHPU",
		site: "build/aten/src/ATen/RegisterNestedTensorHPU_0.cpp:297",
	},
	{
		key: "NestedTensorCPU",
		site: "build/aten/src/ATen/RegisterNestedTensorCPU_0.cpp:309",
	},
	{
		key: "SparseCsrMeta",
		site: "build/aten/src/ATen/RegisterSparseCsrMeta_0.cpp:384",
	},
	{
		key: "SparseCsrCPU",
		site: "build/aten/src/ATen/RegisterSparseCsrCPU_0.cpp:393",
	},
	{
		key: "SparseMeta",
		site: "build/aten/src/ATen/RegisterSparseMeta_0.cpp:142",
	},
	{
		key: "SparseMPS",
		site: "build/aten/src/ATen/RegisterSparseMPS_0.cpp:350",
	},
	{
		key: "SparseCPU",
		site: "build/aten/src/ATen/RegisterSparseCPU_0.cpp:341",
	},
	{
		key: "Meta",
		site: "build/aten/src/ATen/RegisterMeta_0.cpp:1158",
	},
	{
		key: "Meta",
		site: "torch/_meta_registrations.py:50",
	},
	{
		key: "MPS",
		site: "build/aten/src/ATen/RegisterMPS_0.cpp:1628",
	},
	{
		key: "CPU",
		site: "build/aten/src/ATen/RegisterCPU_0.cpp:1309",
	},
	{
		key: "Batched",
		site: "aten/src/ATen/LegacyBatchingRegistrations.cpp:1079",
	},
	{
		key: "FuncTorchBatched",
		site: "aten/src/ATen/functorch/BatchRulesBinaryOps.cpp:352",
	},
	{
		key: "Tracer",
		site: "torch/csrc/autograd/generated/TraceType_2.cpp:17917",
	},
	{
		key: "ZeroTensor",
		site: "build/aten/src/ATen/RegisterZeroTensor_0.cpp:114",
	},
	{
		key: "Named",
		site: "aten/src/ATen/core/NamedRegistrations.cpp:11",
		fallthrough: true,
	},
	{
		key: "MkldnnCPU",
		site: "build/aten/src/ATen/RegisterMkldnnCPU_0.cpp:162",
	},
];

/** Kernels registered for aten::mm, oldest first per key. */
export const MM_REGISTRATIONS: Registration[] = [
	{
		key: "CompositeExplicitAutogradNonFunctional",
		site: "build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:7805",
	},
	{
		key: "Autograd",
		site: "torch/csrc/autograd/generated/VariableType_3.cpp:19687",
	},
	{
		key: "SparseCsrMeta",
		site: "build/aten/src/ATen/RegisterSparseCsrMeta_0.cpp:1080",
	},
	{
		key: "SparseCsrCPU",
		site: "build/aten/src/ATen/RegisterSparseCsrCPU_0.cpp:1115",
	},
	{
		key: "SparseCPU",
		site: "build/aten/src/ATen/RegisterSparseCPU_0.cpp:1272",
	},
	{
		key: "Meta",
		site: "build/aten/src/ATen/RegisterMeta_0.cpp:9425",
	},
	{
		key: "Meta",
		site: "torch/_meta_registrations.py:50",
	},
	{
		key: "MPS",
		site: "build/aten/src/ATen/RegisterMPS_0.cpp:12807",
	},
	{
		key: "CPU",
		site: "build/aten/src/ATen/RegisterCPU_0.cpp:3456",
	},
	{
		key: "Batched",
		site: "aten/src/ATen/LegacyBatchingRegistrations.cpp:1079",
	},
	{
		key: "FuncTorchBatched",
		site: "aten/src/ATen/functorch/BatchRulesLinearAlgebra.cpp:749",
	},
	{
		key: "AutocastCUDA",
		site: "aten/src/ATen/autocast_mode.cpp:169",
	},
	{
		key: "AutocastMPS",
		site: "aten/src/ATen/autocast_mode.cpp:213",
	},
	{
		key: "AutocastXPU",
		site: "aten/src/ATen/autocast_mode.cpp:548",
	},
	{
		key: "AutocastMAIA",
		site: "aten/src/ATen/autocast_mode.cpp:510",
	},
	{
		key: "AutocastMTIA",
		site: "aten/src/ATen/autocast_mode.cpp:472",
	},
	{
		key: "AutocastCPU",
		site: "aten/src/ATen/autocast_mode.cpp:329",
	},
	{
		key: "Tracer",
		site: "torch/csrc/autograd/generated/TraceType_3.cpp:15107",
	},
	{
		key: "Conjugate",
		site: "aten/src/ATen/ConjugateFallback.cpp:21",
		fallthrough: true,
	},
	{
		key: "Named",
		site: "aten/src/ATen/core/NamedRegistrations.cpp:11",
		fallthrough: true,
	},
];
