#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1
# BLOCK_SIZE_X: 128, BLOCK_SIZE_Y: 4, num_stages: 2, num_warps: 4 ; best on RTX 4090
# signature: build_cell_embeds(out_ptr,      # *T, shape (N, D)
#        input_cell_states,                  # *u32, shape (N, 3) laid out contiguously
#        cp_embed,                           # *T, shape (V, D)
#        position_embed,                     # *T, shape (P, D)
#        fg_r_embed, fg_g_embed, fg_b_embed, # *T, shape (D,)
#        bg_r_embed, bg_g_embed, bg_b_embed, # *T, shape (D,)
#        n_cells: tl.uint32,                  # N
#        embed_dim: tl.uint32,                # D
#        vocab_size: tl.uint32,               # V
#        max_positions: tl.uint32,            # P
#        stride_out_n: tl.uint32,             # usually D
#        stride_out_d: tl.uint32,             # usually 1
#        stride_cp_v: tl.uint32,              # usually D
#        stride_cp_d: tl.uint32,              # usually 1
#        stride_pos_p: tl.uint32,             # usually D
#        stride_pos_d: tl.uint32,             # usually 1
#        BLOCK_SIZE_X: tl.constexpr,
#        BLOCK_SIZE_Y: tl.constexpr)
BLOCK_SIZE_X=128
BLOCK_SIZE_Y=4
BUILD_CELL_EMBEDS_SIGNATURE_BF16="*bf16:16, *u32:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, $BLOCK_SIZE_X, $BLOCK_SIZE_Y"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name build_cell_embeds_2d --signature "$BUILD_CELL_EMBEDS_SIGNATURE_BF16" --target cuda:80:32 --num-stages 2 --num-warps 4 --out-name output/build_cell_embeds_bf16_sm80 kernels/build_cell_embeds.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name build_cell_embeds_2d --signature "$BUILD_CELL_EMBEDS_SIGNATURE_BF16" --target cuda:89:32 --num-stages 2 --num-warps 4 --out-name output/build_cell_embeds_bf16_sm89 kernels/build_cell_embeds.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name build_cell_embeds_2d --signature "$BUILD_CELL_EMBEDS_SIGNATURE_BF16" --target cuda:90:32 --num-stages 2 --num-warps 4 --out-name output/build_cell_embeds_bf16_sm90 kernels/build_cell_embeds.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name build_cell_embeds_2d --signature "$BUILD_CELL_EMBEDS_SIGNATURE_BF16" --target cuda:100:32 --num-stages 2 --num-warps 4 --out-name output/build_cell_embeds_bf16_sm100 kernels/build_cell_embeds.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name build_cell_embeds_2d --signature "$BUILD_CELL_EMBEDS_SIGNATURE_BF16" --target hip:gfx942:64 --num-stages 2 --num-warps 4 --out-name output/build_cell_embeds_bf16_gfx942 kernels/build_cell_embeds.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name build_cell_embeds_2d --signature "$BUILD_CELL_EMBEDS_SIGNATURE_BF16" --target hip:gfx1101:32 --num-stages 2 --num-warps 4 --out-name output/build_cell_embeds_bf16_gfx1101 kernels/build_cell_embeds.py

BUILD_CELL_EMBEDS_SIGNATURE_FP16="*fp16:16, *u32:16, *fp16:16, *fp16:16, *fp16:16, *fp16:16, *fp16:16, *fp16:16, *fp16:16, *fp16:16, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, $BLOCK_SIZE_X, $BLOCK_SIZE_Y"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name build_cell_embeds_2d --signature "$BUILD_CELL_EMBEDS_SIGNATURE_FP16" --target cuda:80:32 --num-stages 2 --num-warps 4 --out-name output/build_cell_embeds_fp16_sm80 kernels/build_cell_embeds.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name build_cell_embeds_2d --signature "$BUILD_CELL_EMBEDS_SIGNATURE_FP16" --target cuda:89:32 --num-stages 2 --num-warps 4 --out-name output/build_cell_embeds_fp16_sm89 kernels/build_cell_embeds.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name build_cell_embeds_2d --signature "$BUILD_CELL_EMBEDS_SIGNATURE_FP16" --target cuda:90:32 --num-stages 2 --num-warps 4 --out-name output/build_cell_embeds_fp16_sm90 kernels/build_cell_embeds.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name build_cell_embeds_2d --signature "$BUILD_CELL_EMBEDS_SIGNATURE_FP16" --target cuda:100:32 --num-stages 2 --num-warps 4 --out-name output/build_cell_embeds_fp16_sm100 kernels/build_cell_embeds.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name build_cell_embeds_2d --signature "$BUILD_CELL_EMBEDS_SIGNATURE_FP16" --target hip:gfx942:64 --num-stages 2 --num-warps 4 --out-name output/build_cell_embeds_fp16_gfx942 kernels/build_cell_embeds.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name build_cell_embeds_2d --signature "$BUILD_CELL_EMBEDS_SIGNATURE_FP16" --target hip:gfx1101:32 --num-stages 2 --num-warps 4 --out-name output/build_cell_embeds_fp16_gfx1101 kernels/build_cell_embeds.py
