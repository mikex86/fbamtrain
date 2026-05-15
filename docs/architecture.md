# Architecture Overview

This repository is split into a few large layers that build on each other to turn high level model code into GPU kernels:

- `tensorlib/`: minimal tensor runtime with graph tracing, memory management, kernel descriptors, GPU streams, and the execution backend.
- `src/passes/`: compiler passes that lower graph ops to concrete kernels (mostly Triton- or CUTLASS-generated) and fold patterns such as fused matmul + bias/activation.
- `kernel_gen/`: ahead-of-time kernel build system (Triton + CUTLASS) that emits binary blobs and metadata which are embedded into the binary.
- `src/*.cpp`: application code that builds graphs for the frame head and action model, wires dataset loading, and drives execution.
- `tests/` and `benchmarks/`: validation and perf coverage for the tensor runtime and kernels.

## End-to-end flow
1. Application code builds modules (see `src/framehead_model.cpp`, `src/main_action_model.cpp`) using tensorlib functional APIs. Each operation records into an `OpGraph`.
2. The graph is frozen via `OpGraph::finalize()` and turned into an `ExecutionPlan` (traces -> real tensors) when inputs/parameters are supplied.
3. Compiler passes in `src/passes/` rewrite plan entries: fusing patterns, validating layout, and attaching `ComputeKernelDescriptor` objects that point at embedded kernel binaries.
4. `Executor` runs the plan via `ExecutionBackend`, which manages streams, events, memory allocation, and kernel launches. `KernelCache` loads modules on demand from the embedded blobs provided by `kernel_gen`.
5. Results are surfaced back to the application; repeated iterations reuse the plan and caches but rebind new input tensors.

## Component relationships
- The application layer only depends on tensorlib headers and the pass interfaces; it is ignorant of kernel_gen details.
- Passes depend on `kernel_gen/include` for the predeclared `kernel_bin_t` structs and metadata that describe block sizes, shared memory, etc. They use this to compute grid/block shapes and to populate `ComputeKernelDescriptor` on `ExecutionEntry`.
- `ExecutionBackend` interprets `ComputeKernelDescriptor` by loading the proper module (CUDA or HIP) and launching through device streams tracked in `ctx_management`/`gpu_stream`.
- `kernel_gen` outputs are compiled into static libs `kernels_nv` / `kernels_amd` that are linked into `fbamtrain_lib`. `src/passes/kernel_binaries.cpp` selects the right architecture bucket at compile time.

## Files to know
- Tensor runtime: `tensorlib/include/*.h`, `tensorlib/src/*.cpp`
- Passes: `include/passes.h`, `include/passes/pass_utils.h`, `src/passes/*.cpp`
- Kernel embeddings: `kernel_gen/include/nv/kernels_nv.h`, `kernel_gen/src/embed_kernels_nv.asm`, `kernel_gen/src/embed_kernels_amd.asm`
- Application entrypoint: `src/main.cpp`
- Models: `src/framehead_model.cpp`, `src/main_action_model.cpp`
