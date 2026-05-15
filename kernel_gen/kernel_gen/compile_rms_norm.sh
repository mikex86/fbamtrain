#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1
# Tuned on RTX 4090: BLOCK_SIZE=256, NUM_WARPS=2, NUM_STAGES=2
# signature: rms_norm_fwd(input_ptr, weight_ptr, output_ptr,
#             num_rows, num_cols,
#             eps,
#             BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE=256
RMS_NORM_SIGNATURE_BF16="*bf16:16, *bf16:16, *bf16:16, i32, i32, fp32, $BLOCK_SIZE"
RMS_NORM_BWD_SIGNATURE_BF16="*bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, i32, i32, fp32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_fwd" \
  --signature "$RMS_NORM_SIGNATURE_BF16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_fwd_bf16_sm80 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_fwd" \
  --signature "$RMS_NORM_SIGNATURE_BF16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_fwd_bf16_sm89 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_fwd" \
  --signature "$RMS_NORM_SIGNATURE_BF16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_fwd_bf16_sm90 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_bwd" \
  --signature "$RMS_NORM_BWD_SIGNATURE_BF16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_bwd_bf16_sm80 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_bwd" \
  --signature "$RMS_NORM_BWD_SIGNATURE_BF16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_bwd_bf16_sm89 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_bwd" \
  --signature "$RMS_NORM_BWD_SIGNATURE_BF16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_bwd_bf16_sm90 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_fwd" \
  --signature "$RMS_NORM_SIGNATURE_BF16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_fwd_bf16_sm100 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_bwd" \
  --signature "$RMS_NORM_BWD_SIGNATURE_BF16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_bwd_bf16_sm100 \
  kernels/rms_norm.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_fwd" \
  --signature "$RMS_NORM_SIGNATURE_BF16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_fwd_bf16_gfx942 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_fwd" \
  --signature "$RMS_NORM_SIGNATURE_BF16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_fwd_bf16_gfx1101 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_bwd" \
  --signature "$RMS_NORM_BWD_SIGNATURE_BF16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_bwd_bf16_gfx942 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_bwd" \
  --signature "$RMS_NORM_BWD_SIGNATURE_BF16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_bwd_bf16_gfx1101 \
  kernels/rms_norm.py

RMS_NORM_SIGNATURE_FP16="*fp16:16, *fp16:16, *fp16:16, i32, i32, fp32, $BLOCK_SIZE"
RMS_NORM_BWD_SIGNATURE_FP16="*fp16:16, *fp16:16, *fp16:16, *fp16:16, *fp16:16, i32, i32, fp32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_fwd" \
  --signature "$RMS_NORM_SIGNATURE_FP16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_fwd_fp16_sm80 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_fwd" \
  --signature "$RMS_NORM_SIGNATURE_FP16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_fwd_fp16_sm89 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_fwd" \
  --signature "$RMS_NORM_SIGNATURE_FP16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_fwd_fp16_sm90 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_bwd" \
  --signature "$RMS_NORM_BWD_SIGNATURE_FP16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_bwd_fp16_sm80 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_bwd" \
  --signature "$RMS_NORM_BWD_SIGNATURE_FP16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_bwd_fp16_sm89 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_bwd" \
  --signature "$RMS_NORM_BWD_SIGNATURE_FP16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_bwd_fp16_sm90 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_fwd" \
  --signature "$RMS_NORM_SIGNATURE_FP16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_fwd_fp16_sm100 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_bwd" \
  --signature "$RMS_NORM_BWD_SIGNATURE_FP16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_bwd_fp16_sm100 \
  kernels/rms_norm.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_fwd" \
  --signature "$RMS_NORM_SIGNATURE_FP16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_fwd_fp16_gfx942 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_fwd" \
  --signature "$RMS_NORM_SIGNATURE_FP16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_fwd_fp16_gfx1101 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_bwd" \
  --signature "$RMS_NORM_BWD_SIGNATURE_FP16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_bwd_fp16_gfx942 \
  kernels/rms_norm.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "rms_norm_bwd" \
  --signature "$RMS_NORM_BWD_SIGNATURE_FP16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/rms_norm_bwd_fp16_gfx1101 \
  kernels/rms_norm.py
