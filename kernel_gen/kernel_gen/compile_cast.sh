#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1
# BF16->FP32: BLOCK_SIZE=256, NUM_WARPS=4, NUM_STAGES=2 ; best on RTX 4090
# signature: cast_bf16_to_fp32(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE_BF16_TO_FP32=256
CAST_BF16_TO_FP32_SIGNATURE="*bf16:16, *fp32:16, i32, $BLOCK_SIZE_BF16_TO_FP32"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_bf16_to_fp32_kernel" \
  --signature "$CAST_BF16_TO_FP32_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_bf16_to_fp32_sm80 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_bf16_to_fp32_kernel" \
  --signature "$CAST_BF16_TO_FP32_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_bf16_to_fp32_sm89 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_bf16_to_fp32_kernel" \
  --signature "$CAST_BF16_TO_FP32_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_bf16_to_fp32_sm90 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_bf16_to_fp32_kernel" \
  --signature "$CAST_BF16_TO_FP32_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_bf16_to_fp32_sm100 \
  kernels/cast.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_bf16_to_fp32_kernel" \
  --signature "$CAST_BF16_TO_FP32_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_bf16_to_fp32_gfx942 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_bf16_to_fp32_kernel" \
  --signature "$CAST_BF16_TO_FP32_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_bf16_to_fp32_gfx1101 \
  kernels/cast.py

# FP32->BF16: BLOCK_SIZE=512, NUM_WARPS=4, NUM_STAGES=2 ; best on RTX 4090
# signature: cast_fp32_to_bf16(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE_FP32_TO_BF16=512
CAST_FP32_TO_BF16_SIGNATURE="*fp32:16, *bf16:16, i32, $BLOCK_SIZE_FP32_TO_BF16"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp32_to_bf16_kernel" \
  --signature "$CAST_FP32_TO_BF16_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp32_to_bf16_sm80 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp32_to_bf16_kernel" \
  --signature "$CAST_FP32_TO_BF16_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp32_to_bf16_sm89 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp32_to_bf16_kernel" \
  --signature "$CAST_FP32_TO_BF16_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp32_to_bf16_sm90 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp32_to_bf16_kernel" \
  --signature "$CAST_FP32_TO_BF16_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp32_to_bf16_sm100 \
  kernels/cast.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp32_to_bf16_kernel" \
  --signature "$CAST_FP32_TO_BF16_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp32_to_bf16_gfx942 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp32_to_bf16_kernel" \
  --signature "$CAST_FP32_TO_BF16_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp32_to_bf16_gfx1101 \
  kernels/cast.py

# FP16<->FP32 casts reuse similar tuning
BLOCK_SIZE_FP16_TO_FP32=256
CAST_FP16_TO_FP32_SIGNATURE="*fp16:16, *fp32:16, i32, $BLOCK_SIZE_FP16_TO_FP32"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp16_to_fp32_kernel" \
  --signature "$CAST_FP16_TO_FP32_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp16_to_fp32_sm80 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp16_to_fp32_kernel" \
  --signature "$CAST_FP16_TO_FP32_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp16_to_fp32_sm89 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp16_to_fp32_kernel" \
  --signature "$CAST_FP16_TO_FP32_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp16_to_fp32_sm90 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp16_to_fp32_kernel" \
  --signature "$CAST_FP16_TO_FP32_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp16_to_fp32_sm100 \
  kernels/cast.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp16_to_fp32_kernel" \
  --signature "$CAST_FP16_TO_FP32_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp16_to_fp32_gfx942 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp16_to_fp32_kernel" \
  --signature "$CAST_FP16_TO_FP32_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp16_to_fp32_gfx1101 \
  kernels/cast.py

BLOCK_SIZE_FP32_TO_FP16=1024
CAST_FP32_TO_FP16_SIGNATURE="*fp32:16, *fp16:16, i32, $BLOCK_SIZE_FP32_TO_FP16"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp32_to_fp16_kernel" \
  --signature "$CAST_FP32_TO_FP16_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp32_to_fp16_sm80 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp32_to_fp16_kernel" \
  --signature "$CAST_FP32_TO_FP16_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp32_to_fp16_sm89 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp32_to_fp16_kernel" \
  --signature "$CAST_FP32_TO_FP16_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp32_to_fp16_sm90 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp32_to_fp16_kernel" \
  --signature "$CAST_FP32_TO_FP16_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp32_to_fp16_sm100 \
  kernels/cast.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp32_to_fp16_kernel" \
  --signature "$CAST_FP32_TO_FP16_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp32_to_fp16_gfx942 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp32_to_fp16_kernel" \
  --signature "$CAST_FP32_TO_FP16_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp32_to_fp16_gfx1101 \
  kernels/cast.py

# BF16<->FP16 casts
BLOCK_SIZE_BF16_TO_FP16=1024
CAST_BF16_TO_FP16_SIGNATURE="*bf16:16, *fp16:16, i32, $BLOCK_SIZE_BF16_TO_FP16"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_bf16_to_fp16_kernel" \
  --signature "$CAST_BF16_TO_FP16_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_bf16_to_fp16_sm80 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_bf16_to_fp16_kernel" \
  --signature "$CAST_BF16_TO_FP16_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_bf16_to_fp16_sm89 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_bf16_to_fp16_kernel" \
  --signature "$CAST_BF16_TO_FP16_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_bf16_to_fp16_sm90 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_bf16_to_fp16_kernel" \
  --signature "$CAST_BF16_TO_FP16_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_bf16_to_fp16_sm100 \
  kernels/cast.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_bf16_to_fp16_kernel" \
  --signature "$CAST_BF16_TO_FP16_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_bf16_to_fp16_gfx942 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_bf16_to_fp16_kernel" \
  --signature "$CAST_BF16_TO_FP16_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_bf16_to_fp16_gfx1101 \
  kernels/cast.py

BLOCK_SIZE_FP16_TO_BF16=1024
CAST_FP16_TO_BF16_SIGNATURE="*fp16:16, *bf16:16, i32, $BLOCK_SIZE_FP16_TO_BF16"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp16_to_bf16_kernel" \
  --signature "$CAST_FP16_TO_BF16_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp16_to_bf16_sm80 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp16_to_bf16_kernel" \
  --signature "$CAST_FP16_TO_BF16_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp16_to_bf16_sm89 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp16_to_bf16_kernel" \
  --signature "$CAST_FP16_TO_BF16_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp16_to_bf16_sm90 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp16_to_bf16_kernel" \
  --signature "$CAST_FP16_TO_BF16_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp16_to_bf16_sm100 \
  kernels/cast.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp16_to_bf16_kernel" \
  --signature "$CAST_FP16_TO_BF16_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp16_to_bf16_gfx942 \
  kernels/cast.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "cast_fp16_to_bf16_kernel" \
  --signature "$CAST_FP16_TO_BF16_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/cast_fp16_to_bf16_gfx1101 \
  kernels/cast.py
