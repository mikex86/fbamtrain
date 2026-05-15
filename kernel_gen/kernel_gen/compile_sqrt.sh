#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1

# Tuned on RTX 4090: BLOCK_SIZE=1024, NUM_WARPS=8, NUM_STAGES=2
# signature: sqrt_elementwise(out_ptr, in_ptr,
#             n_elements,
#             BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE=1024
NUM_WARPS=8
NUM_STAGES=2

SIGNATURE_BF16="*bf16, *bf16, i32, $BLOCK_SIZE"
SIGNATURE_FP16="*fp16, *fp16, i32, $BLOCK_SIZE"
SIGNATURE_FP32="*fp32, *fp32, i32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_BF16" --target cuda:80:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_bf16_sm80 kernels/sqrt.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_BF16" --target cuda:89:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_bf16_sm89 kernels/sqrt.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_BF16" --target cuda:90:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_bf16_sm90 kernels/sqrt.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_BF16" --target cuda:100:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_bf16_sm100 kernels/sqrt.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_BF16" --target hip:gfx942:64 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_bf16_gfx942 kernels/sqrt.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_BF16" --target hip:gfx1101:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_bf16_gfx1101 kernels/sqrt.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_FP16" --target cuda:80:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_fp16_sm80 kernels/sqrt.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_FP16" --target cuda:89:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_fp16_sm89 kernels/sqrt.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_FP16" --target cuda:90:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_fp16_sm90 kernels/sqrt.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_FP16" --target cuda:100:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_fp16_sm100 kernels/sqrt.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_FP16" --target hip:gfx942:64 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_fp16_gfx942 kernels/sqrt.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_FP16" --target hip:gfx1101:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_fp16_gfx1101 kernels/sqrt.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_FP32" --target cuda:80:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_fp32_sm80 kernels/sqrt.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_FP32" --target cuda:89:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_fp32_sm89 kernels/sqrt.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_FP32" --target cuda:90:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_fp32_sm90 kernels/sqrt.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_FP32" --target cuda:100:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_fp32_sm100 kernels/sqrt.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_FP32" --target hip:gfx942:64 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_fp32_gfx942 kernels/sqrt.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sqrt_elementwise --signature "$SIGNATURE_FP32" --target hip:gfx1101:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/sqrt_elementwise_fp32_gfx1101 kernels/sqrt.py
