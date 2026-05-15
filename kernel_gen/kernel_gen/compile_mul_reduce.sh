#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1
# signature: mul_reduce_contiguous(out_ptr, lhs_ptr, rhs_ptr,
#             outer, inner, cols,
#             BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE_CONTIG=64
BLOCK_SIZE_COL=32
BLOCK_SIZE_SPLIT_X=64
BLOCK_SIZE_SPLIT_Y=128
SIGNATURE_BF16_CONTIG="*fp32:16, *bf16:16, *bf16:16, i32, i32, i32, $BLOCK_SIZE_CONTIG"
SIGNATURE_BF16_COL="*fp32:16, *bf16:16, *bf16:16, i32, i32, i32, $BLOCK_SIZE_COL"
SIGNATURE_BF16_SPLIT="*fp32:16, *bf16:16, *bf16:16, i32, i32, i32, i32, $BLOCK_SIZE_SPLIT_X, $BLOCK_SIZE_SPLIT_Y"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_contiguous --signature "$SIGNATURE_BF16_CONTIG" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_contiguous_bf16_out_fp32_sm80 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_contiguous --signature "$SIGNATURE_BF16_CONTIG" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_contiguous_bf16_out_fp32_sm89 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_contiguous --signature "$SIGNATURE_BF16_CONTIG" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_contiguous_bf16_out_fp32_sm90 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_contiguous --signature "$SIGNATURE_BF16_CONTIG" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_contiguous_bf16_out_fp32_sm100 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_contiguous --signature "$SIGNATURE_BF16_CONTIG" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_contiguous_bf16_out_fp32_gfx942 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_contiguous --signature "$SIGNATURE_BF16_CONTIG" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_contiguous_bf16_out_fp32_gfx1101 kernels/mul_reduce.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_tiled --signature "$SIGNATURE_BF16_COL" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_column_tiled_bf16_out_fp32_sm80 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_tiled --signature "$SIGNATURE_BF16_COL" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_column_tiled_bf16_out_fp32_sm89 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_tiled --signature "$SIGNATURE_BF16_COL" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_column_tiled_bf16_out_fp32_sm90 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_tiled --signature "$SIGNATURE_BF16_COL" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_column_tiled_bf16_out_fp32_sm100 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_tiled --signature "$SIGNATURE_BF16_COL" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_column_tiled_bf16_out_fp32_gfx942 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_tiled --signature "$SIGNATURE_BF16_COL" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_column_tiled_bf16_out_fp32_gfx1101 kernels/mul_reduce.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_split_partials --signature "$SIGNATURE_BF16_SPLIT" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/mul_reduce_column_split_partials_bf16_out_fp32_sm80 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_split_partials --signature "$SIGNATURE_BF16_SPLIT" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/mul_reduce_column_split_partials_bf16_out_fp32_sm89 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_split_partials --signature "$SIGNATURE_BF16_SPLIT" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/mul_reduce_column_split_partials_bf16_out_fp32_sm90 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_split_partials --signature "$SIGNATURE_BF16_SPLIT" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/mul_reduce_column_split_partials_bf16_out_fp32_sm100 kernels/mul_reduce.py

SIGNATURE_FP16_CONTIG="*fp32:16, *fp16:16, *fp16:16, i32, i32, i32, $BLOCK_SIZE_CONTIG"
SIGNATURE_FP16_COL="*fp32:16, *fp16:16, *fp16:16, i32, i32, i32, $BLOCK_SIZE_COL"
SIGNATURE_FP16_SPLIT="*fp32:16, *fp16:16, *fp16:16, i32, i32, i32, i32, $BLOCK_SIZE_SPLIT_X, $BLOCK_SIZE_SPLIT_Y"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_contiguous --signature "$SIGNATURE_FP16_CONTIG" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_contiguous_fp16_out_fp32_sm80 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_contiguous --signature "$SIGNATURE_FP16_CONTIG" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_contiguous_fp16_out_fp32_sm89 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_contiguous --signature "$SIGNATURE_FP16_CONTIG" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_contiguous_fp16_out_fp32_sm90 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_contiguous --signature "$SIGNATURE_FP16_CONTIG" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_contiguous_fp16_out_fp32_sm100 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_contiguous --signature "$SIGNATURE_FP16_CONTIG" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_contiguous_fp16_out_fp32_gfx942 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_contiguous --signature "$SIGNATURE_FP16_CONTIG" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_contiguous_fp16_out_fp32_gfx1101 kernels/mul_reduce.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_tiled --signature "$SIGNATURE_FP16_COL" --target cuda:80:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_column_tiled_fp16_out_fp32_sm80 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_tiled --signature "$SIGNATURE_FP16_COL" --target cuda:89:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_column_tiled_fp16_out_fp32_sm89 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_tiled --signature "$SIGNATURE_FP16_COL" --target cuda:90:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_column_tiled_fp16_out_fp32_sm90 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_tiled --signature "$SIGNATURE_FP16_COL" --target cuda:100:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_column_tiled_fp16_out_fp32_sm100 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_tiled --signature "$SIGNATURE_FP16_COL" --target hip:gfx942:64 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_column_tiled_fp16_out_fp32_gfx942 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_tiled --signature "$SIGNATURE_FP16_COL" --target hip:gfx1101:32 --num-stages 2 --num-warps 2 --out-name output/mul_reduce_column_tiled_fp16_out_fp32_gfx1101 kernels/mul_reduce.py

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_split_partials --signature "$SIGNATURE_FP16_SPLIT" --target cuda:80:32 --num-stages 2 --num-warps 8 --out-name output/mul_reduce_column_split_partials_fp16_out_fp32_sm80 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_split_partials --signature "$SIGNATURE_FP16_SPLIT" --target cuda:89:32 --num-stages 2 --num-warps 8 --out-name output/mul_reduce_column_split_partials_fp16_out_fp32_sm89 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_split_partials --signature "$SIGNATURE_FP16_SPLIT" --target cuda:90:32 --num-stages 2 --num-warps 8 --out-name output/mul_reduce_column_split_partials_fp16_out_fp32_sm90 kernels/mul_reduce.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name mul_reduce_column_split_partials --signature "$SIGNATURE_FP16_SPLIT" --target cuda:100:32 --num-stages 2 --num-warps 8 --out-name output/mul_reduce_column_split_partials_fp16_out_fp32_sm100 kernels/mul_reduce.py
