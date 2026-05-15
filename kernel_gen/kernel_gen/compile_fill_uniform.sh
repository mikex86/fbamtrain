#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1
# Tuned on RTX 4090: BLOCK_SIZE=1024, NUM_WARPS=2, NUM_STAGES=2
# signature: fill_uniform(out_ptr, n_elements, low, high, seed, BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE=1024

# Compile for BF16 on different architectures
FILL_UNIFORM_SIGNATURE_BF16="*bf16:16, i32, fp32, fp32, i32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_BF16" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_bf16_sm80 kernels/fill_uniform.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_BF16" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_bf16_sm89 kernels/fill_uniform.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_BF16" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_bf16_sm90 kernels/fill_uniform.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_BF16" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_bf16_sm100 kernels/fill_uniform.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_bf16_gfx942 kernels/fill_uniform.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_bf16_gfx1101 kernels/fill_uniform.py

# Compile for FP16 on different architectures
FILL_UNIFORM_SIGNATURE_FP16="*fp16:16, i32, fp32, fp32, i32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_FP16" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_fp16_sm80 kernels/fill_uniform.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_FP16" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_fp16_sm89 kernels/fill_uniform.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_FP16" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_fp16_sm90 kernels/fill_uniform.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_FP16" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_fp16_sm100 kernels/fill_uniform.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_fp16_gfx942 kernels/fill_uniform.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_fp16_gfx1101 kernels/fill_uniform.py

# Compile for FP32 on different architectures
FILL_UNIFORM_SIGNATURE_FP32="*fp32:16, i32, fp32, fp32, i32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_FP32" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_fp32_sm80 kernels/fill_uniform.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_FP32" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_fp32_sm89 kernels/fill_uniform.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_FP32" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_fp32_sm90 kernels/fill_uniform.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_FP32" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_fp32_sm100 kernels/fill_uniform.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_FP32" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_fp32_gfx942 kernels/fill_uniform.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_uniform --signature "$FILL_UNIFORM_SIGNATURE_FP32" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/fill_uniform_fp32_gfx1101 kernels/fill_uniform.py
