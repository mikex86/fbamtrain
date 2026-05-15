#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1
# Tuned on RTX 4090: BLOCK_SIZE=64, NUM_WARPS=2, NUM_STAGES=2
# signature: mean_reduce_contiguous(out_ptr, in_ptr,
#             outer, inner, cols,
#             BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE_CONTIG=64
BLOCK_SIZE_COL=32
SIGNATURE_BF16_CONTIG="*bf16:16, *bf16:16, i32, i32, i32, $BLOCK_SIZE_CONTIG"
SIGNATURE_BF16_COL="*bf16:16, *bf16:16, i32, i32, i32, $BLOCK_SIZE_COL"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_contiguous --signature "$SIGNATURE_BF16_CONTIG" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_contiguous_bf16_sm80 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_contiguous --signature "$SIGNATURE_BF16_CONTIG" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_contiguous_bf16_sm89 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_contiguous --signature "$SIGNATURE_BF16_CONTIG" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_contiguous_bf16_sm90 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_contiguous --signature "$SIGNATURE_BF16_CONTIG" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_contiguous_bf16_sm100 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_contiguous --signature "$SIGNATURE_BF16_CONTIG" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_contiguous_bf16_gfx942 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_contiguous --signature "$SIGNATURE_BF16_CONTIG" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_contiguous_bf16_gfx1101 kernels/mean_reduce.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_column_tiled --signature "$SIGNATURE_BF16_COL" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_column_tiled_bf16_sm80 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_column_tiled --signature "$SIGNATURE_BF16_COL" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_column_tiled_bf16_sm89 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_column_tiled --signature "$SIGNATURE_BF16_COL" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_column_tiled_bf16_sm90 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_column_tiled --signature "$SIGNATURE_BF16_COL" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_column_tiled_bf16_sm100 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_column_tiled --signature "$SIGNATURE_BF16_COL" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_column_tiled_bf16_gfx942 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_column_tiled --signature "$SIGNATURE_BF16_COL" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_column_tiled_bf16_gfx1101 kernels/mean_reduce.py

SIGNATURE_FP16_CONTIG="*fp16:16, *fp16:16, i32, i32, i32, $BLOCK_SIZE_CONTIG"
SIGNATURE_FP16_COL="*fp16:16, *fp16:16, i32, i32, i32, $BLOCK_SIZE_COL"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_contiguous --signature "$SIGNATURE_FP16_CONTIG" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_contiguous_fp16_sm80 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_contiguous --signature "$SIGNATURE_FP16_CONTIG" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_contiguous_fp16_sm89 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_contiguous --signature "$SIGNATURE_FP16_CONTIG" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_contiguous_fp16_sm90 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_contiguous --signature "$SIGNATURE_FP16_CONTIG" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_contiguous_fp16_sm100 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_contiguous --signature "$SIGNATURE_FP16_CONTIG" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_contiguous_fp16_gfx942 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_contiguous --signature "$SIGNATURE_FP16_CONTIG" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_contiguous_fp16_gfx1101 kernels/mean_reduce.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_column_tiled --signature "$SIGNATURE_FP16_COL" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_column_tiled_fp16_sm80 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_column_tiled --signature "$SIGNATURE_FP16_COL" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_column_tiled_fp16_sm89 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_column_tiled --signature "$SIGNATURE_FP16_COL" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_column_tiled_fp16_sm90 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_column_tiled --signature "$SIGNATURE_FP16_COL" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_column_tiled_fp16_sm100 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_column_tiled --signature "$SIGNATURE_FP16_COL" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_column_tiled_fp16_gfx942 kernels/mean_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mean_reduce_column_tiled --signature "$SIGNATURE_FP16_COL" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/mean_reduce_column_tiled_fp16_gfx1101 kernels/mean_reduce.py
