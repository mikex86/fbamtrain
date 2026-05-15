#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1
# BLOCK_SIZE: 1024, num_stages: 2, num_warps: 2 ; best on RTX 4090
# signature: fill_normal(out_ptr, n_elements, mean, std, seed, BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE=1024
FILL_NORMAL_SIGNATURE_BF16="*bf16:16, i32, fp32, fp32, i32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_normal --signature "$FILL_NORMAL_SIGNATURE_BF16" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/fill_normal_bf16_sm80 kernels/fill_normal.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_normal --signature "$FILL_NORMAL_SIGNATURE_BF16" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/fill_normal_bf16_sm89 kernels/fill_normal.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_normal --signature "$FILL_NORMAL_SIGNATURE_BF16" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/fill_normal_bf16_sm90 kernels/fill_normal.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_normal --signature "$FILL_NORMAL_SIGNATURE_BF16" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/fill_normal_bf16_sm100 kernels/fill_normal.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_normal --signature "$FILL_NORMAL_SIGNATURE_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/fill_normal_bf16_gfx942 kernels/fill_normal.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_normal --signature "$FILL_NORMAL_SIGNATURE_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/fill_normal_bf16_gfx1101 kernels/fill_normal.py

FILL_NORMAL_SIGNATURE_FP16="*fp16:16, i32, fp32, fp32, i32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_normal --signature "$FILL_NORMAL_SIGNATURE_FP16" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/fill_normal_fp16_sm80 kernels/fill_normal.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_normal --signature "$FILL_NORMAL_SIGNATURE_FP16" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/fill_normal_fp16_sm89 kernels/fill_normal.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_normal --signature "$FILL_NORMAL_SIGNATURE_FP16" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/fill_normal_fp16_sm90 kernels/fill_normal.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_normal --signature "$FILL_NORMAL_SIGNATURE_FP16" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/fill_normal_fp16_sm100 kernels/fill_normal.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_normal --signature "$FILL_NORMAL_SIGNATURE_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/fill_normal_fp16_gfx942 kernels/fill_normal.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_normal --signature "$FILL_NORMAL_SIGNATURE_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/fill_normal_fp16_gfx1101 kernels/fill_normal.py
