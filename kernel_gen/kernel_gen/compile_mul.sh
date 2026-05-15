#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1
# Tuned on RTX 4090 with TRITON_PRINT_AUTOTUNING=1:
# mul_trailing_broadcast -> BLOCK_SIZE=256, NUM_WARPS=4, NUM_STAGES=2
# signature: mul_trailing_broadcast(out_ptr, in_ptr, scale_ptr,
#             rows, cols,
#             stride_row, stride_col, stride_scale,
#             BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE=256
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
SIGNATURE_BF16="*bf16:16, *bf16:16, *bf16:16, i32, i32, i32, i32, i32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_BF16" --target cuda:80:32 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_bf16_sm80 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_BF16" --target cuda:89:32 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_bf16_sm89 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_BF16" --target cuda:90:32 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_bf16_sm90 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_BF16" --target cuda:100:32 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_bf16_sm100 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_bf16_gfx942 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_bf16_gfx1101 kernels/mul.py

SIGNATURE_FP16="*fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_FP16" --target cuda:80:32 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_fp16_sm80 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_FP16" --target cuda:89:32 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_fp16_sm89 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_FP16" --target cuda:90:32 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_fp16_sm90 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_FP16" --target cuda:100:32 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_fp16_sm100 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_fp16_gfx942 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_fp16_gfx1101 kernels/mul.py

SIGNATURE_FP32="*fp32:16, *fp32:16, *fp32:16, i32, i32, i32, i32, i32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_FP32" --target cuda:80:32 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_fp32_sm80 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_FP32" --target cuda:89:32 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_fp32_sm89 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_FP32" --target cuda:90:32 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_fp32_sm90 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_FP32" --target cuda:100:32 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_fp32_sm100 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_FP32" --target hip:gfx942:64 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_fp32_gfx942 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_trailing_broadcast --signature "$SIGNATURE_FP32" --target hip:gfx1101:32 --num-stages 2 --num-warps 4 --out-name output/mul_trailing_broadcast_fp32_gfx1101 kernels/mul.py

# Tuned on RTX 4090 with TRITON_PRINT_AUTOTUNING=1:
# mul_elementwise -> BLOCK_SIZE=1024, NUM_WARPS=8, NUM_STAGES=2
# signature: mul_elementwise(out_ptr, lhs_ptr, rhs_ptr,
#             n_elements,
#             BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE_ELEMWISE=1024
SIGNATURE_ELEMWISE_BF16="*bf16:16, *bf16:16, *bf16:16, i32, $BLOCK_SIZE_ELEMWISE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_bf16_sm80 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_bf16_sm89 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_bf16_sm90 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_bf16_sm100 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_bf16_gfx942 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_bf16_gfx1101 kernels/mul.py

SIGNATURE_ELEMWISE_FP16="*fp16:16, *fp16:16, *fp16:16, i32, $BLOCK_SIZE_ELEMWISE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_fp16_sm80 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_fp16_sm89 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_fp16_sm90 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_fp16_sm100 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_fp16_gfx942 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_fp16_gfx1101 kernels/mul.py

SIGNATURE_ELEMWISE_FP32="*fp32:16, *fp32:16, *fp32:16, i32, $BLOCK_SIZE_ELEMWISE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_fp32_sm80 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_fp32_sm89 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_fp32_sm90 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_fp32_sm100 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_fp32_gfx942 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/mul_elementwise_fp32_gfx1101 kernels/mul.py

# mul_scalar -> BLOCK_SIZE=1024, NUM_WARPS=8, NUM_STAGES=2
# signature: mul_scalar(out_ptr, lhs_ptr, rhs_ptr,
#             n_elements,
#             BLOCK_SIZE: tl.constexpr)
SIGNATURE_SCALAR_BF16="*bf16:16, *bf16:16, *bf16:16, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_BF16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_bf16_sm80 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_BF16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_bf16_sm89 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_BF16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_bf16_sm90 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_BF16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_bf16_sm100 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_bf16_gfx942 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_bf16_gfx1101 kernels/mul.py

SIGNATURE_SCALAR_FP16="*fp16:16, *fp16:16, *fp16:16, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_FP16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_fp16_sm80 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_FP16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_fp16_sm89 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_FP16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_fp16_sm90 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_FP16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_fp16_sm100 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_fp16_gfx942 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_fp16_gfx1101 kernels/mul.py

SIGNATURE_SCALAR_FP32="*fp32:16, *fp32:16, *fp32:16, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_FP32" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_fp32_sm80 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_FP32" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_fp32_sm89 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_FP32" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_fp32_sm90 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_FP32" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_fp32_sm100 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_FP32" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_fp32_gfx942 kernels/mul.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_scalar --signature "$SIGNATURE_SCALAR_FP32" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/mul_scalar_fp32_gfx1101 kernels/mul.py
