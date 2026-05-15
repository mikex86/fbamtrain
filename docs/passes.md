# Pass Pipeline

Passes live in `src/passes/` and implement `pi::tensorlib::passes::CompilerPass` (see `include/passes.h`). They take an `ExecutionPlan` and mutate `ExecutionEntry` records to attach concrete kernels, adjust shapes/metadata, or fuse patterns. The default sequence is assembled in `src/main.cpp` inside `ApplyDefaultPasses()`.

## Major passes and responsibilities
- **MatmulFusePass** (`src/passes/matmul_pass.cpp`): folds bias and activation into GEMMs where possible and rewrites entries to use fused kernels (Triton or CUTLASS). Selects kernel variants based on dtype, accumulation mode, and output type.
- **Fill passes** (`src/passes/fill_passes.cpp`): lower `FILL_ZEROS`, `FILL_UNIFORM`, `FILL_NORMAL` to elementwise kernels and set `flop_estimate`.
- **Norm passes** (`src/passes/norm_passes.cpp`): map layer norm and RMS norm to specialized Triton kernels; validate contiguous layout and feature dimension sizes.
- **Pool passes** (`src/passes/pool_passes.cpp`): pick 1D/2D pooling kernels and derive grid/block sizes from tensor shapes.
- **Conv2dImplPass** (`src/passes/conv2d_pass.cpp`): choose either Triton conv kernels or CUTLASS conv packages depending on parameters; encode stride/padding/dilation metadata for runtime launch.
- **MeanImplPass** (`src/passes/mean_pass.cpp`): lower reductions to elementwise kernels and compute grid based on reduce dimension.
- **MhaAttentionImplPass** (`src/passes/mha_attention_pass.cpp`): map full attention to fused flash-attention Triton kernels, validating head dims and layout.
- **CastImplPass** (`src/passes/cast_pass.cpp`): use elementwise cast kernels to move between fp16/fp32/bf16.
- **trailing broadcast passes** (`src/passes/trailing_broadcast_passes.cpp`): provide add/mul kernels that broadcast along trailing dimensions while keeping row-major expectations.
- **ActImplPass** (`src/passes/act_pass.cpp`): select GELU or ReLU elementwise kernels.
- **BuildCellEmbedPass** (`src/passes/build_cell_embed_pass.cpp`): custom lowering for the frame-head embedding builder (aggregates multiple embedding tables into NHWC output).
- **ContiguousImplPass** (`src/passes/contiguous_pass.cpp`): materialize views into contiguous buffers using the appropriate reshape kernels.
- **LstmCellImplPass** (`src/passes/lstm_pass.cpp`): wire the streaming LSTM cell forward kernel and derive launch parameters from sequence shapes.

## Kernel binding
- Passes rely on `include/passes/kernel_binaries.h` and `kernel_gen/include/*/kernels_*.h`, which expose `kernel_bin_t<Meta>` structs for every architecture-specific kernel. `src/passes/kernel_binaries.cpp` selects the right architecture variant at compile time using `NV_KERNEL_ARCH`/`AMD_KERNEL_ARCH`.
- `pass_utils.h` provides helpers for dtype/device validation, contiguity checks, and lightweight math helpers (e.g., `CEIL_DIV`) to compute grid dimensions.
- Each lowering builds a `ComputeKernelDescriptor` with:
  - `kernel_name`/`function_name` from the embedded metadata,
  - backend (`CUDA` or `HIP`),
  - an argument provider lambda that validates tensor shapes/layout, computes grid/block sizes from kernel meta fields, and returns `KernelLaunchArguments`.

## Execution interplay
- After passes run, `ExecutionEntry::op_type` is typically cleared and `kernel_descriptor` populated. The executor therefore skips generic op dispatch and directly launches the chosen kernel.
- If a pass cannot lower (unsupported dtype/layout), it throws early, preventing runtime failure.
- `ExecutionEntry::is_useful` and `flop_estimate` are set by passes to enable downstream profiling (`ExecutionPlan::totalUsefulFlops()`).
