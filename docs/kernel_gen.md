# Kernel Generation Pipeline

`kernel_gen/` houses the toolchain that produces the GPU kernels consumed by passes and the executor. The outputs are small binary blobs plus metadata describing launch parameters.

## Source inputs
- Triton kernels under `kernel_gen/kernel_gen/kernels/` and helper compiler in `kernel_gen/kernel_gen/pysrc/triton_aot_compiler.py`.
- CUTLASS codegen configs in `kernel_gen/kernel_gen/kernels/cutlass_codegen/` (JSON autotune constants and Python wrapper).
- Shell wrappers `kernel_gen/kernel_gen/compile_*.sh` define the signatures, meta-parameters (block sizes, warps, stages), and target architectures.

## Build chain
1. Run `kernel_gen/kernel_gen/compile_all_kernels.sh` (or individual `compile_*.sh`). Scripts expect a Triton-enabled Python (`PYTHON_BIN`) and CUDA/HIP toolchains (`CUDA_HOME`, ROCm for HIP).
2. Each script calls the Triton AOT compiler to emit `.cubin` (CUDA) or `.hsaco` (HIP) files into `kernel_gen/kernel_gen/output/` with architecture suffixes (e.g., `*_sm89`, `*_sm100`, `*_gfx942`).
3. CUTLASS kernels are packaged via `kernels/cutlass_codegen/compile.py`, which emits pre-compiled cubins alongside JSON metadata.
4. The `kernel_gen/src/embed_kernels_{nv,amd}.asm` files use the `EMBED_ASSET` macro from `declare_asset.inc.asm` to embed those binaries as linker-visible symbols (`<name>_start`, `<name>_end`, function names, and launch meta constants).

## Headers and metadata
- `kernel_gen/include/declare_kernel.inc.h` defines `kernel_bin_t<Meta>` plus macros like `DECLARE_GEMM_KERNEL_BINARY` that wire together the embedded symbols, shared memory requirements, warp counts, and operation-specific meta (tile sizes, block pixels, head dim, etc.).
- Architecture-specific headers (`kernel_gen/include/nv/kernels_nv.h`, `kernel_gen/include/amd/kernels_amd.h`) enumerate all kernels and provide aliases without the architecture suffix for ease of use in passes.
- `src/passes/kernel_binaries.cpp` includes the proper header via `NV_KERNEL_ARCH`/`AMD_KERNEL_ARCH` and exposes the `k<name>` instances used by passes.

## CMake integration
- `kernel_gen/CMakeLists.txt` builds two static libraries: `kernels_nv` and `kernels_amd`, setting `NV_KERNEL_ARCH`/`AMD_KERNEL_ARCH` defaults (89 and gfx942). The assembler sources include the generated assets directory and use `-x assembler-with-cpp` to honor `#if` guards.
- The top-level `CMakeLists.txt` links `kernels_nv` into `fbamtrain_lib` when CUDA is enabled and `kernels_amd` when HIP is enabled. Kernel binaries are therefore always available to the runtime without dynamic loading from disk.

## How passes use it
- Each pass chooses a `kernel_bin_t` from the predeclared symbols, inspects its meta fields (block sizes, swizzle, max kernel size, etc.), and produces a `ComputeKernelDescriptor` that points to the embedded module bytes.
- The executor loads modules lazily through `KernelCache::loadKernel()` using the `module_data`/`module_size` supplied by the descriptor, then launches the `function_name` with the grid/block sizes computed from meta.

## Adding or updating kernels
- Modify or add a Triton/CUTLASS source under `kernel_gen/kernel_gen/kernels/` and extend the appropriate `compile_*.sh` script with a signature line.
- Run the compile script to regenerate artifacts in `kernel_gen/kernel_gen/output/`, then rebuild C++ to re-embed the new blobs (the assembler files consume whatever is present in `output/`).
- Keep architecture suffixes consistent; the headers assume naming like `<op>_<dtype>_<arch>.cubin/hsaco` for aliasing.

## Conv2d bin-tuned CUTLASS kernels
Conv2d forward kernels are tuned against a fixed set of input shapes ("bins") derived from run-config JSON files. This keeps kernel selection stable and lets the autotuner specialize per bin.

Workflow:
1. Regenerate bins when configs change:
   `python3 kernel_gen/kernel_gen/pysrc/generate_conv2d_bins.py --config working_dir/run-configs/debug.json \
   --manifest-out kernel_gen/kernel_gen/kernels/cutlass_codegen/configs/conv2d_bin_manifest.json \
   --config-out kernel_gen/kernel_gen/kernels/cutlass_codegen/configs/conv2d_bin_configs.json \
   --kernel-header-out kernel_gen/include/nv/conv2d_bin_kernels_nv.h \
   --map-header-out include/passes/conv2d_bin_map.h \
   --embed-sm89-out kernel_gen/src/conv2d_bin_kernels_sm89.inc.asm \
   --embed-sm90-out kernel_gen/src/conv2d_bin_kernels_sm90.inc.asm \
   --embed-sm100-out kernel_gen/src/conv2d_bin_kernels_sm100.inc.asm \
   --kinfo-sm89-out kernel_gen/src/conv2d_bin_kinfo_sm89.inc.asm \
   --kinfo-sm90-out kernel_gen/src/conv2d_bin_kinfo_sm90.inc.asm \
   --kinfo-sm100-out kernel_gen/src/conv2d_bin_kinfo_sm100.inc.asm \
   --base-config kernel_gen/kernel_gen/kernels/cutlass_codegen/configs/input_configs/base_autotune_configs.json`
2. Autotune CUTLASS conv2d using the bin config file to produce `autotune_constants_sm89_conv2d_bins.json` (or the target SM equivalent).
3. Re-run `kernel_gen/kernel_gen/compile_conv2d.sh` to compile the bin-specific kernels and embed them.

`src/passes/conv2d_pass.cpp` reads `include/passes/conv2d_bin_map.h` to match batch/height/width bins, falling back to the default CUTLASS list when a bin is not present.

For the full workflow and generated artifacts, see `docs/conv2d_bins.md`.
