#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
set -euo pipefail
export TRITON_ALWAYS_COMPILE=1

# signature: fill_constant(out_ptr, n_elements, value, BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE=1024

FILL_CONSTANT_SIGNATURE_BF16="*bf16:16, i32, fp32, $BLOCK_SIZE"
FILL_CONSTANT_SIGNATURE_FP16="*fp16:16, i32, fp32, $BLOCK_SIZE"
FILL_CONSTANT_SIGNATURE_FP32="*fp32:16, i32, fp32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_BF16" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/fill_constant_bf16_sm80 kernels/fill_constant.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_BF16" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/fill_constant_bf16_sm89 kernels/fill_constant.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_BF16" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/fill_constant_bf16_sm90 kernels/fill_constant.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_BF16" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/fill_constant_bf16_sm100 kernels/fill_constant.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/fill_constant_bf16_gfx942 kernels/fill_constant.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/fill_constant_bf16_gfx1101 kernels/fill_constant.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_FP16" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/fill_constant_fp16_sm80 kernels/fill_constant.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_FP16" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/fill_constant_fp16_sm89 kernels/fill_constant.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_FP16" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/fill_constant_fp16_sm90 kernels/fill_constant.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_FP16" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/fill_constant_fp16_sm100 kernels/fill_constant.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/fill_constant_fp16_gfx942 kernels/fill_constant.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/fill_constant_fp16_gfx1101 kernels/fill_constant.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_FP32" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/fill_constant_fp32_sm80 kernels/fill_constant.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_FP32" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/fill_constant_fp32_sm89 kernels/fill_constant.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_FP32" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/fill_constant_fp32_sm90 kernels/fill_constant.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_FP32" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/fill_constant_fp32_sm100 kernels/fill_constant.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_FP32" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/fill_constant_fp32_gfx942 kernels/fill_constant.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_constant --signature "$FILL_CONSTANT_SIGNATURE_FP32" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/fill_constant_fp32_gfx1101 kernels/fill_constant.py
