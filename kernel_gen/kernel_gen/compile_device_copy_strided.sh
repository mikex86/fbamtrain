#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}

set -euo pipefail
export TRITON_ALWAYS_COMPILE=1

# Tuned on RTX 4090: BLOCK_SIZE_X=256, BLOCK_SIZE_Y=8, NUM_WARPS=4, NUM_STAGES=2
# signature: device_copy_strided_2d(dst_ptr, src_ptr, src_stride_elems, dst_stride_elems, width_elems, height,
#             BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr)
BLOCK_SIZE_X=256
BLOCK_SIZE_Y=8
DEVICE_COPY_SIGNATURE="*u16, *u16, i32, i32, i32, i32, $BLOCK_SIZE_X, $BLOCK_SIZE_Y"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "device_copy_strided_2d" \
  --signature "$DEVICE_COPY_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/device_copy_strided_2d_sm80 \
  kernels/device_copy_strided_2d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "device_copy_strided_2d" \
  --signature "$DEVICE_COPY_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/device_copy_strided_2d_sm89 \
  kernels/device_copy_strided_2d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "device_copy_strided_2d" \
  --signature "$DEVICE_COPY_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/device_copy_strided_2d_sm90 \
  kernels/device_copy_strided_2d.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "device_copy_strided_2d" \
  --signature "$DEVICE_COPY_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/device_copy_strided_2d_sm100 \
  kernels/device_copy_strided_2d.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "device_copy_strided_2d" \
  --signature "$DEVICE_COPY_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/device_copy_strided_2d_gfx942 \
  kernels/device_copy_strided_2d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "device_copy_strided_2d" \
  --signature "$DEVICE_COPY_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/device_copy_strided_2d_gfx1101 \
  kernels/device_copy_strided_2d.py
