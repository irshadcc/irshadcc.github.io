---
title: 'Pytorch Dispatcher'
description: 'Pytorch Dispatcher'
pubDate: 'Jul 01 2025'
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

One of the core components of Pytorch is its Dispatcher module. Its main responsibility is to find the correct Operator and call the Kernel whenever users does any operation on Tensors.


For the illustration purpose, I have written a simple script to add two tensors 

```python
import torch

# Define two tensors
x = torch.tensor([1,2], requires_grad=False, dtype=torch.float32)
w = torch.tensor([3,4], requires_grad=True, dtype=torch.float32)

# Add the tensors
result_tensor = torch.add(x, w)
sum_tensor = result_tensor.sum()
sum_tensor.backward()
print(x.grad)
print(w.grad)

```


### High level - Code walkthrough

1. Create the operator handle for and call the kernel

```c++
static C10_NOINLINE c10::TypedOperatorHandle<add_Tensor::schema> create_add_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(add_Tensor::name, add_Tensor::overload_name)
      .typed<add_Tensor::schema>();
}

// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor add_Tensor::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {

    // Find the operator using the mapping stored in the dispatcher
    // Note that 'op' is static
    static auto op = create_add_Tensor_typed_handle();
    return op.call(self, other, alpha);
}
```

2. Extract dispatch key set for the operator based upon its input and lookup the kernel

```c++
// I have deleted many lines in this function to show the main function calls
// If the pytorch is build with USE_DEBUG or HAS_TORCH_SHOW_DISPATCH_TRACE, 
// you can set env variable TORCH_SHOW_DISPATCH_TRACE=1 to show dispatch traces
template <class Return, class... Args>
C10_ALWAYS_INLINE_UNLESS_MOBILE Return Dispatcher::call(
    const TypedOperatorHandle<Return(Args...)>& op,
    Args... args) const {
  auto dispatchKeySet =op.operatorDef_->op.dispatchKeyExtractor().template getDispatchKeySetUnboxed<Args...>(args...);
  const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);
  return kernel.template call<Return, Args...>(
      op, dispatchKeySet, std::forward<Args>(args)...);
}
```




### OperatorEntry
During the loading of pytorch library, it will register the Kernels to the dispatcher through TORCH_LIBRARY_IMPL

1. Registering the kernels to Dispatcher using macro TORCH_LIBRARY_IMPL
```c++
TORCH_LIBRARY_IMPL(aten, Batched, m) {
    m.impl("add", native::add);
}

// Macro expands to 
static const torch::detail::TorchLibraryInit TORCH_LIBRARY_IMPL_static_init_aten_Batched_1(
    torch::Library::IMPL,
    &TORCH_LIBRARY_IMPL_static_init_aten_Batched_1, // Function to call to register the kernel
    "aten", // namespace 
    c10::make_optional(c10::DispatchKey::Batched), // Dispatch key
    "pytorch/aten/src/ATen/LegacyBatchingRegistrations.cpp"
    1079 
);

```

2. The function Library::impl() will call Dispatcher::registerImpl() function


```c++
RegistrationHandleRAII Dispatcher::registerImpl(
  OperatorName op_name,
  std::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  std::optional<impl::CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  std::lock_guard<std::mutex> lock(guard_->mutex);

  // Create OperatorHandle
  auto op = findOrRegisterName_(op_name);

  // Register kernel to the OperatorHandle
  auto handle = op.operatorDef_->op.registerKernel(
    *this,
    dispatch_key,
    std::move(kernel),
    std::move(cpp_signature),
    std::move(inferred_function_schema),
    std::move(debug)
  );

  ++op.operatorDef_->def_and_impl_count;

  cond_var_.notify_all();

  return RegistrationHandleRAII([guard = this->guard_, this, op, op_name, dispatch_key, handle] {
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    deregisterImpl_(op, op_name, dispatch_key, handle);
  });
}
```