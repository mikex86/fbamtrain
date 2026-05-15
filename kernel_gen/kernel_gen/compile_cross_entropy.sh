#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1

# Tuned on RTX 4090: BLOCK_SIZE=1024, NUM_WARPS=16, NUM_STAGES=3
# signature: cross_entropy_on_targets(out_ptr, logits_ptr, targets_ptr,
#             rows, cols, _unused,
#             BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE=1024
NUM_WARPS=16
NUM_STAGES=3

PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/.venv/bin/python"}

SIGNATURE_BF16_MEAN="*bf16:16, *bf16:16, *i32, i32, i32, $BLOCK_SIZE, $NUM_WARPS, $NUM_STAGES"
SIGNATURE_FP16_MEAN="*fp16:16, *fp16:16, *i32, i32, i32, $BLOCK_SIZE, $NUM_WARPS, $NUM_STAGES"
SIGNATURE_BF16_ADD_FP32="*fp32, *bf16:16, *i32, i32, i32, $BLOCK_SIZE, $NUM_WARPS, $NUM_STAGES"
SIGNATURE_FP16_ADD_FP32="*fp32, *fp16:16, *i32, i32, i32, $BLOCK_SIZE, $NUM_WARPS, $NUM_STAGES"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_BF16_MEAN" --target cuda:80:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_bf16_sm80 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_BF16_MEAN" --target cuda:89:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_bf16_sm89 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_BF16_MEAN" --target cuda:90:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_bf16_sm90 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_BF16_MEAN" --target cuda:100:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_bf16_sm100 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_BF16_MEAN" --target hip:gfx942:64 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_bf16_gfx942 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_BF16_MEAN" --target hip:gfx1101:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_bf16_gfx1101 kernels/cross_entropy.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_FP16_MEAN" --target cuda:80:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_fp16_sm80 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_FP16_MEAN" --target cuda:89:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_fp16_sm89 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_FP16_MEAN" --target cuda:90:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_fp16_sm90 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_FP16_MEAN" --target cuda:100:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_fp16_sm100 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_FP16_MEAN" --target hip:gfx942:64 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_fp16_gfx942 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_FP16_MEAN" --target hip:gfx1101:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_fp16_gfx1101 kernels/cross_entropy.py

# add reduction uses FP32 output
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_BF16_ADD_FP32" --target cuda:80:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_add_fp32_bf16_sm80 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_BF16_ADD_FP32" --target cuda:89:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_add_fp32_bf16_sm89 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_BF16_ADD_FP32" --target cuda:90:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_add_fp32_bf16_sm90 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_BF16_ADD_FP32" --target cuda:100:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_add_fp32_bf16_sm100 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_BF16_ADD_FP32" --target hip:gfx942:64 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_add_fp32_bf16_gfx942 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_BF16_ADD_FP32" --target hip:gfx1101:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_add_fp32_bf16_gfx1101 kernels/cross_entropy.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_FP16_ADD_FP32" --target cuda:80:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_add_fp32_fp16_sm80 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_FP16_ADD_FP32" --target cuda:89:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_add_fp32_fp16_sm89 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_FP16_ADD_FP32" --target cuda:90:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_add_fp32_fp16_sm90 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_FP16_ADD_FP32" --target cuda:100:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_add_fp32_fp16_sm100 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_FP16_ADD_FP32" --target hip:gfx942:64 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_add_fp32_fp16_gfx942 kernels/cross_entropy.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name cross_entropy_on_targets --signature "$SIGNATURE_FP16_ADD_FP32" --target hip:gfx1101:32 --num-stages $NUM_STAGES --num-warps $NUM_WARPS --out-name output/cross_entropy_on_targets_add_fp32_fp16_gfx1101 kernels/cross_entropy.py
