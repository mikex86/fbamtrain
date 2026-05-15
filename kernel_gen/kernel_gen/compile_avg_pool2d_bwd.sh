#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1
# Tuned on RTX 4090: BLOCK_SIZE_H=16, BLOCK_SIZE_W=8, MAX_KERNEL_H=7, MAX_KERNEL_W=7, NUM_WARPS=4, NUM_STAGES=2
# signature: avg_pool2d_bwd_kernel(upstream_ptr, grad_input_ptr,
#             outer_stride_upstream0, outer_stride_upstream1,
#             pool_stride_upstream_h, pool_stride_upstream_w,
#             outer_stride_grad0, outer_stride_grad1,
#             pool_stride_grad_h, pool_stride_grad_w,
#             outer_size0, outer_size1,
#             pool_h_in, pool_w_in,
#             pool_h_out, pool_w_out,
#             kernel_h, kernel_w,
#             stride_h, stride_w,
#             padding_h, padding_w,
#             inv_kernel_size,
#             BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
#             MAX_KERNEL_H: tl.constexpr, MAX_KERNEL_W: tl.constexpr)
BLOCK_SIZE_H=16
BLOCK_SIZE_W=8
MAX_KERNEL_H=7
MAX_KERNEL_W=7
AVG_POOL2D_BWD_SIGNATURE_BF16="*bf16:16, *bf16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, fp32, $BLOCK_SIZE_H, $BLOCK_SIZE_W, $MAX_KERNEL_H, $MAX_KERNEL_W"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_BF16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_bf16_sm80 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_BF16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_bf16_sm89 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_BF16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_bf16_sm90 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_BF16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_bf16_sm80 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_BF16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_bf16_sm89 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_BF16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_bf16_sm90 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_BF16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_bf16_sm100 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_BF16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_bf16_sm100 \
  kernels/avg_pool2d_bwd.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_BF16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_bf16_gfx942 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_BF16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_bf16_gfx1101 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_BF16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_bf16_gfx942 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_BF16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_bf16_gfx1101 \
  kernels/avg_pool2d_bwd.py

AVG_POOL2D_BWD_SIGNATURE_FP16="*fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, fp32, $BLOCK_SIZE_H, $BLOCK_SIZE_W, $MAX_KERNEL_H, $MAX_KERNEL_W"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_FP16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_fp16_sm80 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_FP16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_fp16_sm89 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_FP16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_fp16_sm90 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_FP16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_fp16_sm80 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_FP16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_fp16_sm89 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_FP16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_fp16_sm90 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_FP16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_fp16_sm100 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_FP16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_fp16_sm100 \
  kernels/avg_pool2d_bwd.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_FP16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_fp16_gfx942 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_FP16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_fp16_gfx1101 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_FP16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_fp16_gfx942 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_kernel" \
  --signature "$AVG_POOL2D_BWD_SIGNATURE_FP16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_fp16_gfx1101 \
  kernels/avg_pool2d_bwd.py

FAST_BLOCK_SIZE_H=16
FAST_BLOCK_SIZE_W=128
FAST_MAX_KERNEL_H=2
FAST_MAX_KERNEL_W=2
AVG_POOL2D_BWD_FAST_SIGNATURE_BF16="*bf16:16, *bf16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, fp32, $FAST_BLOCK_SIZE_H, $FAST_BLOCK_SIZE_W, $FAST_MAX_KERNEL_H, $FAST_MAX_KERNEL_W"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_nhwc_2x2_s2_kernel" \
  --signature "$AVG_POOL2D_BWD_FAST_SIGNATURE_BF16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_nhwc_2x2_s2_bf16_sm80 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_nhwc_2x2_s2_kernel" \
  --signature "$AVG_POOL2D_BWD_FAST_SIGNATURE_BF16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_nhwc_2x2_s2_bf16_sm89 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_nhwc_2x2_s2_kernel" \
  --signature "$AVG_POOL2D_BWD_FAST_SIGNATURE_BF16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_nhwc_2x2_s2_bf16_sm90 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_nhwc_2x2_s2_kernel" \
  --signature "$AVG_POOL2D_BWD_FAST_SIGNATURE_BF16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_nhwc_2x2_s2_bf16_sm100 \
  kernels/avg_pool2d_bwd.py

AVG_POOL2D_BWD_FAST_SIGNATURE_FP16="*fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, fp32, $FAST_BLOCK_SIZE_H, $FAST_BLOCK_SIZE_W, $FAST_MAX_KERNEL_H, $FAST_MAX_KERNEL_W"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_nhwc_2x2_s2_kernel" \
  --signature "$AVG_POOL2D_BWD_FAST_SIGNATURE_FP16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_nhwc_2x2_s2_fp16_sm80 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_nhwc_2x2_s2_kernel" \
  --signature "$AVG_POOL2D_BWD_FAST_SIGNATURE_FP16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_nhwc_2x2_s2_fp16_sm89 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_nhwc_2x2_s2_kernel" \
  --signature "$AVG_POOL2D_BWD_FAST_SIGNATURE_FP16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_nhwc_2x2_s2_fp16_sm90 \
  kernels/avg_pool2d_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool2d_bwd_noaccum_nhwc_2x2_s2_kernel" \
  --signature "$AVG_POOL2D_BWD_FAST_SIGNATURE_FP16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/avg_pool2d_bwd_noaccum_nhwc_2x2_s2_fp16_sm100 \
  kernels/avg_pool2d_bwd.py
