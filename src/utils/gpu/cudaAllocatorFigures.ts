import type { BoxNode, BoxEdge } from "../payments/boxDiagram";

export const allocatorNodes: BoxNode[] = [
  { id: "api", x: 2, y: 0, w: 2, label: "PyTorch API", group: 1, note: "User calls torch.empty or allocates a tensor, hitting the C10 allocator." },
  { id: "malloc", x: 2, y: 1, w: 2, label: "malloc(size, stream)", group: 2, note: "The entry point in DeviceCachingAllocator. Acquires a mutex and checks pools." },
  
  { id: "size_check", x: 2, y: 2, w: 2, label: "Size Check", group: 0, note: "Determines if the request is small (<= 1MB) or large (> 1MB)." },
  
  { id: "small_pool", x: 0, y: 3, w: 2, label: "Small BlockPool", group: 3, note: "Holds unallocated cached blocks <= 1MB." },
  { id: "large_pool", x: 4, y: 3, w: 2, label: "Large BlockPool", group: 3, note: "Holds unallocated cached blocks > 1MB." },
  
  { id: "split", x: 2, y: 4, w: 2, label: "Split Block", group: 4, note: "If the found block is larger than requested, it is split. The remainder is put back in the pool." },
  
  { id: "cuda_malloc", x: 2, y: 5, w: 2, label: "cudaMalloc", group: 5, note: "If no suitable cached block is found, asks the OS for new memory via cudaMalloc." },
  
  { id: "active", x: 2, y: 6, w: 2, label: "active_blocks", group: 1, note: "The chosen block is marked allocated and added to the active_blocks set." }
];

export const allocatorEdges: BoxEdge[] = [
  { from: "api", to: "malloc" },
  { from: "malloc", to: "size_check" },
  { from: "size_check", to: "small_pool", label: "<= 1MB" },
  { from: "size_check", to: "large_pool", label: "> 1MB" },
  { from: "small_pool", to: "split", label: "Found block" },
  { from: "large_pool", to: "split", label: "Found block" },
  { from: "small_pool", to: "cuda_malloc", label: "No block", dashed: true },
  { from: "large_pool", to: "cuda_malloc", label: "No block", dashed: true },
  { from: "split", to: "active" },
  { from: "cuda_malloc", to: "active" }
];
