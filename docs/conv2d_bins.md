# Conv2d Bin Tuning

This project uses a bin-based tuning system for CUTLASS conv2d forward kernels. A "bin" is a concrete
input shape (batch, height, width) paired with the conv2d attributes (channels, kernel, stride,
padding, dilation). Each bin is tuned and compiled into its own kernel variant so that runtime
dispatch can choose the best kernel for the exact shape we expect to see in training.

## What Gets Generated
- `kernel_gen/kernel_gen/kernels/cutlass_codegen/configs/conv2d_bin_manifest.json`
  - The list of input shapes collected from run-config JSON files.
- `kernel_gen/kernel_gen/kernels/cutlass_codegen/configs/conv2d_bin_configs.json`
  - CUTLASS config entries for each bin (bf16, fp16, fp16_acc_fp16).
- `kernel_gen/kernel_gen/kernels/cutlass_codegen/configs/autotune_constants_sm89_conv2d_bins.json`
  - Autotuned constants for the bin configs on sm89 (sm100 left untuned).
- `include/passes/conv2d_bin_map.h`
  - Bin to kernel mapping consumed by the conv2d pass.
- `kernel_gen/include/nv/conv2d_bin_kernels_nv.h`
  - Kernel declarations and aliases.
- `kernel_gen/src/conv2d_bin_kernels_sm89.inc.asm`
- `kernel_gen/src/conv2d_bin_kernels_sm100.inc.asm`
- `kernel_gen/src/conv2d_bin_kinfo_sm89.inc.asm`
- `kernel_gen/src/conv2d_bin_kinfo_sm100.inc.asm`
  - Embedded asset/kinfo includes consumed by `kernel_gen/src/embed_kernels_nv.asm`.

## Dispatch Behavior
`src/passes/conv2d_pass.cpp` first checks the bin map using:
- input shape (batch, height, width)
- in/out channels
- kernel size, stride, dilation
- padding

If a bin match exists, its CUTLASS kernel is selected. If not, the pass falls back to the default
CUTLASS list; if that is missing it falls back to Triton. Backend preference can be forced via
`FBAMTRAIN_PREFER_CONV2D_BACKEND`.

## Update Workflow
1. Generate bins from run-configs:
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
   --kinfo-sm100-out kernel_gen/src/conv2d_bin_kinfo_sm100.inc.asm`
2. Autotune the bin configs on the target SM (example: sm89):
   `python3 -m kernel_gen.kernel_gen.kernels.cutlass_codegen.autotune \
   --sm sm_89 \
   --base-config kernel_gen/kernel_gen/kernels/cutlass_codegen/configs/input_configs/base_autotune_configs.json \
   --conv2d-configs-file kernel_gen/kernel_gen/kernels/cutlass_codegen/configs/conv2d_bin_configs.json \
   --output kernel_gen/kernel_gen/kernels/cutlass_codegen/configs/autotune_constants_sm89_conv2d_bins.json \
   --config <each bin config name>`
3. Recompile the kernels:
   `kernel_gen/kernel_gen/compile_conv2d.sh`
4. Rebuild the C++ targets so the new embedded assets are linked in.

The compile script automatically includes any configs listed in `conv2d_bin_configs.json`, so you
do not need to edit the list by hand.

## Troubleshooting
- If autotune writes output to a nested path, re-run it from the repo root using
  `python3 -m kernel_gen.kernel_gen.kernels.cutlass_codegen.autotune` (module mode) to keep paths correct.
- If a bin is missing at runtime, the pass will fall back to the non-bin CUTLASS kernels or Triton.
