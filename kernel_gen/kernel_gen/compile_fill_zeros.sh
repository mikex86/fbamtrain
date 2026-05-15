#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1
# Tuned on RTX 4090: BLOCK_SIZE=1024, NUM_WARPS=2, NUM_STAGES=2
# signature: fill_zeros(out_ptr, n_bytes, BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE=1024
FILL_ZEROS_SIGNATURE="*i8:16, i32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_zeros --signature "$FILL_ZEROS_SIGNATURE" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/fill_zeros_sm80 kernels/fill_zeros.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_zeros --signature "$FILL_ZEROS_SIGNATURE" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/fill_zeros_sm89 kernels/fill_zeros.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_zeros --signature "$FILL_ZEROS_SIGNATURE" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/fill_zeros_sm90 kernels/fill_zeros.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_zeros --signature "$FILL_ZEROS_SIGNATURE" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/fill_zeros_sm100 kernels/fill_zeros.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_zeros --signature "$FILL_ZEROS_SIGNATURE" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/fill_zeros_gfx942 kernels/fill_zeros.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name fill_zeros --signature "$FILL_ZEROS_SIGNATURE" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/fill_zeros_gfx1101 kernels/fill_zeros.py
