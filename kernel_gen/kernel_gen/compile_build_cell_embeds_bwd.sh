#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1

# build_cell_embeds_bwd_cp signature:
#   grad_out_ptr, cell_states_ptr, grad_cp_ptr,
#   n_cells, embed_dim, vocab_size,
#   stride_out_n, stride_out_d, stride_cp_v, stride_cp_d,
#   BLOCK_SIZE (constexpr)
BLOCK_SIZE_CP=256
BLOCK_SIZE_CP_OUT_FP32=512
NUM_WARPS_CP=8
NUM_WARPS_CP_OUT_FP32=8
# Tuned on RTX 4090: 256x8 improves half-output atomic throughput, while
# 512x8 is consistently faster for fp32-output variants.
BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_BF16="*bf16:16, *u32:16, *bf16:16, u32, u32, u32, u32, u32, u32, u32, $BLOCK_SIZE_CP"
BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_BF16_OUT_FP32="*bf16:16, *u32:16, *fp32:16, u32, u32, u32, u32, u32, u32, u32, $BLOCK_SIZE_CP_OUT_FP32"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_BF16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP \
  --out-name output/build_cell_embeds_bwd_cp_bf16_sm80 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_BF16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP \
  --out-name output/build_cell_embeds_bwd_cp_bf16_sm89 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_BF16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP \
  --out-name output/build_cell_embeds_bwd_cp_bf16_sm90 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_BF16_OUT_FP32" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP_OUT_FP32 \
  --out-name output/build_cell_embeds_bwd_cp_bf16_out_fp32_sm80 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_BF16_OUT_FP32" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP_OUT_FP32 \
  --out-name output/build_cell_embeds_bwd_cp_bf16_out_fp32_sm89 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_BF16_OUT_FP32" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP_OUT_FP32 \
  --out-name output/build_cell_embeds_bwd_cp_bf16_out_fp32_sm90 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_BF16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP \
  --out-name output/build_cell_embeds_bwd_cp_bf16_sm100 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_BF16_OUT_FP32" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP_OUT_FP32 \
  --out-name output/build_cell_embeds_bwd_cp_bf16_out_fp32_sm100 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_BF16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP \
  --out-name output/build_cell_embeds_bwd_cp_bf16_gfx942 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_BF16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP \
  --out-name output/build_cell_embeds_bwd_cp_bf16_gfx1101 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_BF16_OUT_FP32" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP_OUT_FP32 \
  --out-name output/build_cell_embeds_bwd_cp_bf16_out_fp32_gfx942 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_BF16_OUT_FP32" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP_OUT_FP32 \
  --out-name output/build_cell_embeds_bwd_cp_bf16_out_fp32_gfx1101 \
  kernels/build_cell_embeds_bwd.py

BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_FP16="*fp16:16, *u32:16, *fp16:16, u32, u32, u32, u32, u32, u32, u32, $BLOCK_SIZE_CP"
BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_FP16_OUT_FP32="*fp16:16, *u32:16, *fp32:16, u32, u32, u32, u32, u32, u32, u32, $BLOCK_SIZE_CP_OUT_FP32"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_FP16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP \
  --out-name output/build_cell_embeds_bwd_cp_fp16_sm80 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_FP16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP \
  --out-name output/build_cell_embeds_bwd_cp_fp16_sm89 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_FP16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP \
  --out-name output/build_cell_embeds_bwd_cp_fp16_sm90 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_FP16_OUT_FP32" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP_OUT_FP32 \
  --out-name output/build_cell_embeds_bwd_cp_fp16_out_fp32_sm80 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_FP16_OUT_FP32" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP_OUT_FP32 \
  --out-name output/build_cell_embeds_bwd_cp_fp16_out_fp32_sm89 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_FP16_OUT_FP32" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP_OUT_FP32 \
  --out-name output/build_cell_embeds_bwd_cp_fp16_out_fp32_sm90 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_FP16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP \
  --out-name output/build_cell_embeds_bwd_cp_fp16_sm100 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_FP16_OUT_FP32" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP_OUT_FP32 \
  --out-name output/build_cell_embeds_bwd_cp_fp16_out_fp32_sm100 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_FP16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP \
  --out-name output/build_cell_embeds_bwd_cp_fp16_gfx942 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_FP16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP \
  --out-name output/build_cell_embeds_bwd_cp_fp16_gfx1101 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_FP16_OUT_FP32" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP_OUT_FP32 \
  --out-name output/build_cell_embeds_bwd_cp_fp16_out_fp32_gfx942 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_cp_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_CP_SIGNATURE_FP16_OUT_FP32" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps $NUM_WARPS_CP_OUT_FP32 \
  --out-name output/build_cell_embeds_bwd_cp_fp16_out_fp32_gfx1101 \
  kernels/build_cell_embeds_bwd.py

# build_cell_embeds_bwd_pos signature:
#   grad_out_ptr, grad_pos_ptr,
#   batch, max_positions, embed_dim,
#   stride_out_n, stride_out_d, stride_pos_p, stride_pos_d,
#   BLOCK_SIZE_X, BLOCK_SIZE_Y
BLOCK_SIZE_POS_X=128
BLOCK_SIZE_POS_Y=32
BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_BF16="*bf16:16, *bf16:16, u32, u32, u32, u32, u32, u32, u32, $BLOCK_SIZE_POS_X, $BLOCK_SIZE_POS_Y"
BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_BF16_OUT_FP32="*bf16:16, *fp32:16, u32, u32, u32, u32, u32, u32, u32, $BLOCK_SIZE_POS_X, $BLOCK_SIZE_POS_Y"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_BF16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_bf16_sm80 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_BF16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_bf16_sm89 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_BF16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_bf16_sm90 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_BF16_OUT_FP32" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_bf16_out_fp32_sm80 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_BF16_OUT_FP32" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_bf16_out_fp32_sm89 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_BF16_OUT_FP32" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_bf16_out_fp32_sm90 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_BF16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_bf16_sm100 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_BF16_OUT_FP32" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_bf16_out_fp32_sm100 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_BF16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_bf16_gfx942 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_BF16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_bf16_gfx1101 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_BF16_OUT_FP32" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_bf16_out_fp32_gfx942 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_BF16_OUT_FP32" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_bf16_out_fp32_gfx1101 \
  kernels/build_cell_embeds_bwd.py

BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_FP16="*fp16:16, *fp16:16, u32, u32, u32, u32, u32, u32, u32, $BLOCK_SIZE_POS_X, $BLOCK_SIZE_POS_Y"
BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_FP16_OUT_FP32="*fp16:16, *fp32:16, u32, u32, u32, u32, u32, u32, u32, $BLOCK_SIZE_POS_X, $BLOCK_SIZE_POS_Y"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_FP16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_fp16_sm80 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_FP16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_fp16_sm89 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_FP16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_fp16_sm90 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_FP16_OUT_FP32" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_fp16_out_fp32_sm80 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_FP16_OUT_FP32" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_fp16_out_fp32_sm89 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_FP16_OUT_FP32" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_fp16_out_fp32_sm90 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_FP16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_fp16_sm100 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_FP16_OUT_FP32" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_fp16_out_fp32_sm100 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_FP16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_fp16_gfx942 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_FP16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_fp16_gfx1101 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_FP16_OUT_FP32" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_fp16_out_fp32_gfx942 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_pos_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_POS_SIGNATURE_FP16_OUT_FP32" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_pos_fp16_out_fp32_gfx1101 \
  kernels/build_cell_embeds_bwd.py

# build_cell_embeds_bwd_color signature:
#   grad_out_ptr, cell_states_ptr,
#   grad_fg_r_ptr, grad_fg_g_ptr, grad_fg_b_ptr,
#   grad_bg_r_ptr, grad_bg_g_ptr, grad_bg_b_ptr,
#   n_cells, embed_dim, stride_out_n, stride_out_d,
#   BLOCK_SIZE_X, BLOCK_SIZE_Y
BLOCK_SIZE_X=128
BLOCK_SIZE_Y=8
BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_BF16="*bf16:16, *u32:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, u32, u32, u32, u32, $BLOCK_SIZE_X, $BLOCK_SIZE_Y"
BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_BF16_OUT_FP32="*bf16:16, *u32:16, *fp32:16, *fp32:16, *fp32:16, *fp32:16, *fp32:16, *fp32:16, u32, u32, u32, u32, $BLOCK_SIZE_X, $BLOCK_SIZE_Y"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_BF16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_bf16_sm80 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_BF16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_bf16_sm89 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_BF16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_bf16_sm90 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_BF16_OUT_FP32" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_bf16_out_fp32_sm80 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_BF16_OUT_FP32" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_bf16_out_fp32_sm89 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_BF16_OUT_FP32" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_bf16_out_fp32_sm90 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_BF16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_bf16_sm100 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_BF16_OUT_FP32" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_bf16_out_fp32_sm100 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_BF16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_bf16_gfx942 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_BF16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_bf16_gfx1101 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_BF16_OUT_FP32" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_bf16_out_fp32_gfx942 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_BF16_OUT_FP32" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_bf16_out_fp32_gfx1101 \
  kernels/build_cell_embeds_bwd.py

BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_FP16="*fp16:16, *u32:16, *fp16:16, *fp16:16, *fp16:16, *fp16:16, *fp16:16, *fp16:16, u32, u32, u32, u32, $BLOCK_SIZE_X, $BLOCK_SIZE_Y"
BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_FP16_OUT_FP32="*fp16:16, *u32:16, *fp32:16, *fp32:16, *fp32:16, *fp32:16, *fp32:16, *fp32:16, u32, u32, u32, u32, $BLOCK_SIZE_X, $BLOCK_SIZE_Y"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_FP16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_fp16_sm80 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_FP16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_fp16_sm89 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_FP16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_fp16_sm90 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_FP16_OUT_FP32" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_fp16_out_fp32_sm80 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_FP16_OUT_FP32" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_fp16_out_fp32_sm89 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_FP16_OUT_FP32" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_fp16_out_fp32_sm90 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_FP16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_fp16_sm100 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_FP16_OUT_FP32" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_fp16_out_fp32_sm100 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_FP16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_fp16_gfx942 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_FP16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_fp16_gfx1101 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_FP16_OUT_FP32" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_fp16_out_fp32_gfx942 \
  kernels/build_cell_embeds_bwd.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name build_cell_embeds_bwd_color_out_fp32 \
  --signature "$BUILD_CELL_EMBEDS_BWD_COLOR_SIGNATURE_FP16_OUT_FP32" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/build_cell_embeds_bwd_color_fp16_out_fp32_gfx1101 \
  kernels/build_cell_embeds_bwd.py
