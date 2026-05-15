#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1

BLOCK_SIZE=1024

SIGNATURE_ADAMW_BF16="*bf16:16, *bf16:16, *fp32:16, *fp32:16, *fp32:16, *fp32:16, i32, fp32, fp32, fp32, fp32, fp32, $BLOCK_SIZE"
SIGNATURE_ADAMW_FP16="*fp16:16, *fp16:16, *fp32:16, *fp32:16, *fp32:16, *fp32:16, i32, fp32, fp32, fp32, fp32, fp32, $BLOCK_SIZE"
SIGNATURE_ADAMW_FP32="*fp32:16, *fp32:16, *fp32:16, *fp32:16, *fp32:16, *fp32:16, i32, fp32, fp32, fp32, fp32, fp32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_BF16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/adamw_step_bf16_sm80 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_BF16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/adamw_step_bf16_sm89 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_BF16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/adamw_step_bf16_sm90 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_BF16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/adamw_step_bf16_sm100 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/adamw_step_bf16_gfx942 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/adamw_step_bf16_gfx1101 kernels/optimizer.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_FP16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/adamw_step_fp16_sm80 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_FP16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/adamw_step_fp16_sm89 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_FP16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/adamw_step_fp16_sm90 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_FP16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/adamw_step_fp16_sm100 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/adamw_step_fp16_gfx942 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/adamw_step_fp16_gfx1101 kernels/optimizer.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_FP32" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/adamw_step_fp32_sm80 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_FP32" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/adamw_step_fp32_sm89 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_FP32" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/adamw_step_fp32_sm90 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_FP32" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/adamw_step_fp32_sm100 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_FP32" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/adamw_step_fp32_gfx942 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name adamw_step --signature "$SIGNATURE_ADAMW_FP32" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/adamw_step_fp32_gfx1101 kernels/optimizer.py

SIGNATURE_SGD_BF16="*bf16:16, *bf16:16, *fp32:16, i32, fp32, fp32, fp32, i32, $BLOCK_SIZE"
SIGNATURE_SGD_FP16="*fp16:16, *fp16:16, *fp32:16, i32, fp32, fp32, fp32, i32, $BLOCK_SIZE"
SIGNATURE_SGD_FP32="*fp32:16, *fp32:16, *fp32:16, i32, fp32, fp32, fp32, i32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_BF16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/sgd_step_bf16_sm80 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_BF16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/sgd_step_bf16_sm89 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_BF16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/sgd_step_bf16_sm90 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_BF16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/sgd_step_bf16_sm100 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/sgd_step_bf16_gfx942 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/sgd_step_bf16_gfx1101 kernels/optimizer.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_FP16" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/sgd_step_fp16_sm80 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_FP16" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/sgd_step_fp16_sm89 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_FP16" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/sgd_step_fp16_sm90 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_FP16" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/sgd_step_fp16_sm100 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/sgd_step_fp16_gfx942 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/sgd_step_fp16_gfx1101 kernels/optimizer.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_FP32" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/sgd_step_fp32_sm80 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_FP32" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/sgd_step_fp32_sm89 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_FP32" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/sgd_step_fp32_sm90 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_FP32" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/sgd_step_fp32_sm100 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_FP32" --target hip:gfx942:64 --num-stages 2 --num-warps 8 --out-name output/sgd_step_fp32_gfx942 kernels/optimizer.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name sgd_step --signature "$SIGNATURE_SGD_FP32" --target hip:gfx1101:32 --num-stages 2 --num-warps 8 --out-name output/sgd_step_fp32_gfx1101 kernels/optimizer.py
