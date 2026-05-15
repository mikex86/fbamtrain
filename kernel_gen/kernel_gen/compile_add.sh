#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1
# Tuned on RTX 4090: BLOCK_SIZE=256, NUM_WARPS=4, NUM_STAGES=2
# signature: add_trailing_broadcast(out_ptr, in_ptr, bias_ptr,
#             rows, cols,
#             stride_row, stride_col, stride_bias,
#             BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE=256
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}

SIGNATURE_BF16="*bf16:16, *bf16:16, *bf16:16, i32, i32, i32, i32, i32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_BF16" --target cuda:80:32 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_bf16_sm80 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_BF16" --target cuda:89:32 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_bf16_sm89 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_BF16" --target cuda:90:32 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_bf16_sm90 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_BF16" --target cuda:100:32 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_bf16_sm100 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_bf16_gfx942 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_bf16_gfx1101 kernels/add.py

SIGNATURE_FP16="*fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_FP16" --target cuda:80:32 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_fp16_sm80 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_FP16" --target cuda:89:32 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_fp16_sm89 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_FP16" --target cuda:90:32 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_fp16_sm90 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_FP16" --target cuda:100:32 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_fp16_sm100 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_fp16_gfx942 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_fp16_gfx1101 kernels/add.py

SIGNATURE_FP32="*fp32:16, *fp32:16, *fp32:16, i32, i32, i32, i32, i32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_FP32" --target cuda:80:32 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_fp32_sm80 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_FP32" --target cuda:89:32 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_fp32_sm89 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_FP32" --target cuda:90:32 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_fp32_sm90 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_FP32" --target cuda:100:32 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_fp32_sm100 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_FP32" --target hip:gfx942:64 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_fp32_gfx942 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_trailing_broadcast --signature "$SIGNATURE_FP32" --target hip:gfx1101:32 --num-stages 2 --num-warps 4 --out-name output/add_trailing_broadcast_fp32_gfx1101 kernels/add.py

# Tuned on RTX 4090: BLOCK_SIZE=1024, NUM_WARPS=8, NUM_STAGES=2
# signature: add_elementwise(out_ptr, lhs_ptr, rhs_ptr,
#             n_elements,
#             BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE_ELEMWISE=1024
SIGNATURE_ELEMWISE_BF16="*bf16:16, *bf16:16, *bf16:16, i32, $BLOCK_SIZE_ELEMWISE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_bf16_sm80 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_bf16_sm89 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_bf16_sm90 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_bf16_sm100 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_bf16_gfx942 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_bf16_gfx1101 kernels/add.py

SIGNATURE_ELEMWISE_FP16="*fp16:16, *fp16:16, *fp16:16, i32, $BLOCK_SIZE_ELEMWISE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp16_sm80 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp16_sm89 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp16_sm90 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp16_sm100 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp16_gfx942 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp16_gfx1101 kernels/add.py

SIGNATURE_ELEMWISE_FP32="*fp32:16, *fp32:16, *fp32:16, i32, $BLOCK_SIZE_ELEMWISE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp32_sm80 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp32_sm89 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp32_sm90 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp32_sm100 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp32_gfx942 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp32_gfx1101 kernels/add.py

SIGNATURE_ELEMWISE_FP32_OUT_FP16="*fp16:16, *fp32:16, *fp32:16, i32, $BLOCK_SIZE_ELEMWISE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise_fp32_to_fp16 --signature "$SIGNATURE_ELEMWISE_FP32_OUT_FP16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp32_out_fp16_sm80 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise_fp32_to_fp16 --signature "$SIGNATURE_ELEMWISE_FP32_OUT_FP16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp32_out_fp16_sm89 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise_fp32_to_fp16 --signature "$SIGNATURE_ELEMWISE_FP32_OUT_FP16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp32_out_fp16_sm90 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise_fp32_to_fp16 --signature "$SIGNATURE_ELEMWISE_FP32_OUT_FP16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp32_out_fp16_sm100 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise_fp32_to_fp16 --signature "$SIGNATURE_ELEMWISE_FP32_OUT_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp32_out_fp16_gfx942 kernels/add.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name add_elementwise_fp32_to_fp16 --signature "$SIGNATURE_ELEMWISE_FP32_OUT_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/add_elementwise_fp32_out_fp16_gfx1101 kernels/add.py
