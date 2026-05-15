#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1

BLOCK_SIZE=128
SIGNATURE="*fp32, *fp32, i32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name reduce_sum_partial \
  --signature "$SIGNATURE" \
  --target cuda:80:32 \
  --num-warps 4 \
  --num-stages 2 \
  --out-name output/reduce_sum_partial_fp32_sm80 \
  kernels/reduce_sum_partial.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name reduce_sum_partial \
  --signature "$SIGNATURE" \
  --target cuda:89:32 \
  --num-warps 4 \
  --num-stages 2 \
  --out-name output/reduce_sum_partial_fp32_sm89 \
  kernels/reduce_sum_partial.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name reduce_sum_partial \
  --signature "$SIGNATURE" \
  --target cuda:90:32 \
  --num-warps 4 \
  --num-stages 2 \
  --out-name output/reduce_sum_partial_fp32_sm90 \
  kernels/reduce_sum_partial.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name reduce_sum_partial \
  --signature "$SIGNATURE" \
  --target cuda:100:32 \
  --num-warps 4 \
  --num-stages 2 \
  --out-name output/reduce_sum_partial_fp32_sm100 \
  kernels/reduce_sum_partial.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name reduce_sum_partial \
  --signature "$SIGNATURE" \
  --target hip:gfx942:64 \
  --num-warps 4 \
  --num-stages 2 \
  --out-name output/reduce_sum_partial_fp32_gfx942 \
  kernels/reduce_sum_partial.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name reduce_sum_partial \
  --signature "$SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-warps 4 \
  --num-stages 2 \
  --out-name output/reduce_sum_partial_fp32_gfx1101 \
  kernels/reduce_sum_partial.py
