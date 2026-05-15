#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1

# Tuned on RTX 4090 with TRITON_PRINT_AUTOTUNING=1:
# div_elementwise -> BLOCK_SIZE=1024, NUM_WARPS=8, NUM_STAGES=2
# signature: div_elementwise(out_ptr, lhs_ptr, rhs_ptr,
#             n_elements,
#             BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE_ELEMWISE=1024

SIGNATURE_ELEMWISE_BF16="*bf16:16, *bf16:16, *bf16:16, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_bf16_sm80 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_bf16_sm89 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_bf16_sm90 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_bf16_sm100 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_bf16_gfx942 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_bf16_gfx1101 kernels/div.py

SIGNATURE_ELEMWISE_FP16="*fp16:16, *fp16:16, *fp16:16, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_fp16_sm80 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_fp16_sm89 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_fp16_sm90 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_fp16_sm100 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_fp16_gfx942 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_fp16_gfx1101 kernels/div.py

SIGNATURE_ELEMWISE_FP32="*fp32:16, *fp32:16, *fp32:16, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_fp32_sm80 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_fp32_sm89 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_fp32_sm90 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_fp32_sm100 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_fp32_gfx942 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_elementwise --signature "$SIGNATURE_ELEMWISE_FP32" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/div_elementwise_fp32_gfx1101 kernels/div.py

# div_scalar -> BLOCK_SIZE=1024, NUM_WARPS=8, NUM_STAGES=2
# signature: div_scalar(out_ptr, lhs_ptr, rhs_ptr,
#             n_elements,
#             BLOCK_SIZE: tl.constexpr)
SIGNATURE_SCALAR_BF16="*bf16:16, *bf16:16, *bf16:16, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_BF16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_bf16_sm80 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_BF16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_bf16_sm89 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_BF16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_bf16_sm90 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_BF16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_bf16_sm100 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/div_scalar_bf16_gfx942 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_bf16_gfx1101 kernels/div.py

SIGNATURE_SCALAR_FP16="*fp16:16, *fp16:16, *fp16:16, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_FP16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_fp16_sm80 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_FP16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_fp16_sm89 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_FP16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_fp16_sm90 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_FP16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_fp16_sm100 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/div_scalar_fp16_gfx942 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_fp16_gfx1101 kernels/div.py

SIGNATURE_SCALAR_FP32="*fp32:16, *fp32:16, *fp32:16, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_FP32" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_fp32_sm80 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_FP32" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_fp32_sm89 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_FP32" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_fp32_sm90 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_FP32" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_fp32_sm100 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_FP32" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/div_scalar_fp32_gfx942 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar --signature "$SIGNATURE_SCALAR_FP32" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_fp32_gfx1101 kernels/div.py

# div_add -> BLOCK_SIZE=1024, NUM_WARPS=8, NUM_STAGES=2
# signature: div_add(out_ptr, lhs_ptr, rhs_ptr, denom_ptr,
#             n_elements,
#             BLOCK_SIZE: tl.constexpr)
SIGNATURE_DIV_ADD_BF16="*bf16:16, *bf16:16, *bf16:16, *bf16:16, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_BF16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/div_add_bf16_sm80 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_BF16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/div_add_bf16_sm89 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_BF16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/div_add_bf16_sm90 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_BF16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/div_add_bf16_sm100 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/div_add_bf16_gfx942 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/div_add_bf16_gfx1101 kernels/div.py

SIGNATURE_DIV_ADD_FP16="*fp16:16, *fp16:16, *fp16:16, *fp16:16, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_FP16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/div_add_fp16_sm80 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_FP16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/div_add_fp16_sm89 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_FP16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/div_add_fp16_sm90 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_FP16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/div_add_fp16_sm100 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/div_add_fp16_gfx942 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/div_add_fp16_gfx1101 kernels/div.py

SIGNATURE_DIV_ADD_FP32="*fp32:16, *fp32:16, *fp32:16, *fp32:16, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_FP32" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/div_add_fp32_sm80 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_FP32" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/div_add_fp32_sm89 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_FP32" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/div_add_fp32_sm90 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_FP32" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/div_add_fp32_sm100 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_FP32" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/div_add_fp32_gfx942 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_add --signature "$SIGNATURE_DIV_ADD_FP32" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/div_add_fp32_gfx1101 kernels/div.py

# div_scalar_add -> BLOCK_SIZE=1024, NUM_WARPS=8, NUM_STAGES=2
# signature: div_scalar_add(out_ptr, lhs_ptr, rhs_ptr, denom_ptr,
#             n_elements,
#             BLOCK_SIZE: tl.constexpr)
SIGNATURE_DIV_SCALAR_ADD_BF16="*bf16:16, *bf16:16, *bf16:16, *bf16:16, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_BF16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_bf16_sm80 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_BF16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_bf16_sm89 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_BF16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_bf16_sm90 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_BF16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_bf16_sm100 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_bf16_gfx942 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_bf16_gfx1101 kernels/div.py

SIGNATURE_DIV_SCALAR_ADD_FP16="*fp16:16, *fp16:16, *fp16:16, *fp16:16, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_FP16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_fp16_sm80 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_FP16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_fp16_sm89 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_FP16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_fp16_sm90 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_FP16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_fp16_sm100 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_fp16_gfx942 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_fp16_gfx1101 kernels/div.py

SIGNATURE_DIV_SCALAR_ADD_FP32="*fp32:16, *fp32:16, *fp32:16, *fp32:16, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_FP32" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_fp32_sm80 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_FP32" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_fp32_sm89 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_FP32" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_fp32_sm90 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_FP32" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_fp32_sm100 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_FP32" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_fp32_gfx942 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add --signature "$SIGNATURE_DIV_SCALAR_ADD_FP32" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_fp32_gfx1101 kernels/div.py

# div_scalar_add_broadcast -> BLOCK_SIZE=1024, NUM_WARPS=8, NUM_STAGES=2
# signature: div_scalar_add_broadcast(out_ptr, lhs_ptr, rhs_ptr, denom_ptr,
#             n_elements, inner_size, cols,
#             BLOCK_SIZE: tl.constexpr)
SIGNATURE_DIV_SCALAR_ADD_BCAST_BF16="*bf16:16, *bf16:16, *bf16:16, *bf16:16, i32, i32, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_BF16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_bf16_sm80 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_BF16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_bf16_sm89 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_BF16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_bf16_sm90 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_BF16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_bf16_sm100 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_bf16_gfx942 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_bf16_gfx1101 kernels/div.py

SIGNATURE_DIV_SCALAR_ADD_BCAST_FP16="*fp16:16, *fp16:16, *fp16:16, *fp16:16, i32, i32, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_FP16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_fp16_sm80 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_FP16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_fp16_sm89 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_FP16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_fp16_sm90 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_FP16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_fp16_sm100 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_fp16_gfx942 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_fp16_gfx1101 kernels/div.py

SIGNATURE_DIV_SCALAR_ADD_BCAST_FP32="*fp32:16, *fp32:16, *fp32:16, *fp32:16, i32, i32, i32, $BLOCK_SIZE_ELEMWISE"
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_FP32" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_fp32_sm80 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_FP32" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_fp32_sm89 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_FP32" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_fp32_sm90 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_FP32" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_fp32_sm100 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_FP32" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_fp32_gfx942 kernels/div.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name div_scalar_add_broadcast --signature "$SIGNATURE_DIV_SCALAR_ADD_BCAST_FP32" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/div_scalar_add_broadcast_fp32_gfx1101 kernels/div.py
