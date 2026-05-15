# Tensorlib Primer

## Core abstractions
- **Device/DataType**: simple enums in `tensorlib/include/tensorlib.h` used throughout the runtime; GPU ordinals are integers.
- **Shape/Strides**: lightweight containers that keep dimensions and layout; stride helpers ensure row-major contiguity checks in passes.
- **TraceTensor vs RealTensor**: `TraceTensor` records intent inside `OpGraph` (ids, layout, device) without owning memory. `RealTensor` wraps actual `Storage` with allocator hooks, tracks views via `storage_offset`, and exposes device/shape metadata.
- **Storage and allocators**: `Storage` owns raw buffers via `allocator::Allocator` implementations (see `tensorlib/include/allocator.h`). Pinned buffers are supported for H2D/D2H overlap. `allocator::AllocatorRegistry` is passed into execution to resolve per-device allocators.

## Graph recording
- `OpGraph` captures an ordered list of `OperationEntry` structs. Each entry carries an `OpType`, inputs/outputs (as `TraceTensor`), and arbitrary attributes (`std::any`) for kernels or custom ops.
- The graph surface mirrors common DL ops: matmul, conv, pooling, normalization, attention, casting, device copies, contiguous/view transforms, and an escape hatch `CUSTOM_OP` (used by the frame-head embed builder).
- Graph inputs/parameters are described up front via `GraphInputDescriptor` and are validated before execution to ensure matching shapes/dtypes/devices.

## Execution plan materialization
- `ExecutionPlan::FromGraph` binds runtime tensors to traces. It checks provided inputs/params, creates missing outputs, and emits an ordered vector of `ExecutionEntry` records.
- `ExecutionEntry` optionally carries an `op_type` (for fallback execution) or a `ComputeKernelDescriptor` (for direct kernel launches after passes run). It also tracks input/output tensors, a `flop_estimate`, an `is_useful` flag for profiling, and a free-form attributes map.
- `ExecutionPlan::updateInputDescriptors` lets callers rebind inputs between iterations without rebuilding the plan structure.

## Execution runtime
- `Executor` owns a plan and drives execution through `ExecutionBackend`, reusing the `real_tensors` map from the plan. It also tracks which GPU ordinals were touched to coordinate waits.
- `ExecutionBackend` is responsible for:
  - Picking GPU streams from `gpustream::GpuStreamBundle` (main, h2d, d2h) via `ctx_management`.
  - Loading kernels with `KernelCache` and launching functions based on `ComputeKernelDescriptor` (CUDA/HIP).
  - Synchronization primitives: per-device `Event`, stream waits, and `await` to drain outstanding work.
- CPU-only ops (e.g., host side copies) run without GPU stream bundles; async copy wait ops reuse the last seen GPU ordinal to select the right stream.

## Functional layer and modules
- Functional helpers (see `tensorlib/include/functional.h` and friends) build `OpGraph` ops for math, reductions, and tensor layout changes; higher-level modules in `src/framehead_model.cpp` and `src/main_action_model.cpp` compose these to form trainable models.
- Modules hold parameters as `TraceTensor` instances and expose a `buildForward` method that records graph nodes. Parameters are later bound to `RealTensor` inputs when constructing the `ExecutionPlan`.

## Testing hooks
- `tensorlib/src/testing.cpp` provides utilities used by `tests/` to create deterministic tensors and validate correctness/performance of kernels and graph execution without invoking the full training application.
