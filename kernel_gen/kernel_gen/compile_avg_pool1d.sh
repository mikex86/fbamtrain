#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1
# Tuned on RTX 4090: BLOCK_SIZE=64, MAX_KERNEL_SIZE=32, NUM_WARPS=2, NUM_STAGES=2
# signature: avg_pool1d_kernel(input_ptr, output_ptr,
#             outer_stride_in0, outer_stride_in1, pool_stride_in,
#             outer_stride_out0, outer_stride_out1, pool_stride_out,
#             outer_size0, outer_size1,
#             pool_in, pool_out,
#             kernel_size, stride, inv_kernel_size,
#             BLOCK_SIZE: tl.constexpr, MAX_KERNEL_SIZE: tl.constexpr)
BLOCK_SIZE=64
MAX_KERNEL_SIZE=32
AVG_POOL1D_SIGNATURE_BF16="*bf16:16, *bf16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, fp32, $BLOCK_SIZE, $MAX_KERNEL_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool1d_kernel" \
  --signature "$AVG_POOL1D_SIGNATURE_BF16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/avg_pool1d_bf16_sm80 \
  kernels/avg_pool1d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool1d_kernel" \
  --signature "$AVG_POOL1D_SIGNATURE_BF16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/avg_pool1d_bf16_sm89 \
  kernels/avg_pool1d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool1d_kernel" \
  --signature "$AVG_POOL1D_SIGNATURE_BF16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/avg_pool1d_bf16_sm90 \
  kernels/avg_pool1d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool1d_kernel" \
  --signature "$AVG_POOL1D_SIGNATURE_BF16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/avg_pool1d_bf16_sm100 \
  kernels/avg_pool1d.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool1d_kernel" \
  --signature "$AVG_POOL1D_SIGNATURE_BF16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/avg_pool1d_bf16_gfx942 \
  kernels/avg_pool1d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool1d_kernel" \
  --signature "$AVG_POOL1D_SIGNATURE_BF16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/avg_pool1d_bf16_gfx1101 \
  kernels/avg_pool1d.py

AVG_POOL1D_SIGNATURE_FP16="*fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, fp32, $BLOCK_SIZE, $MAX_KERNEL_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool1d_kernel" \
  --signature "$AVG_POOL1D_SIGNATURE_FP16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/avg_pool1d_fp16_sm80 \
  kernels/avg_pool1d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool1d_kernel" \
  --signature "$AVG_POOL1D_SIGNATURE_FP16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/avg_pool1d_fp16_sm89 \
  kernels/avg_pool1d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool1d_kernel" \
  --signature "$AVG_POOL1D_SIGNATURE_FP16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/avg_pool1d_fp16_sm90 \
  kernels/avg_pool1d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool1d_kernel" \
  --signature "$AVG_POOL1D_SIGNATURE_FP16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/avg_pool1d_fp16_sm100 \
  kernels/avg_pool1d.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool1d_kernel" \
  --signature "$AVG_POOL1D_SIGNATURE_FP16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/avg_pool1d_fp16_gfx942 \
  kernels/avg_pool1d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "avg_pool1d_kernel" \
  --signature "$AVG_POOL1D_SIGNATURE_FP16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 2 \
  --out-name output/avg_pool1d_fp16_gfx1101 \
  kernels/avg_pool1d.py
