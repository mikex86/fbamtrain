#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
CUTLASS_RUNNER="${SCRIPT_DIR}/kernels/cutlass_codegen/run_cutlass_py.sh"
export TRITON_ALWAYS_COMPILE=1
# Autotuned on sm89 for B=8,H=8,T=1024,HD=128 (tmp/mha.py):
# BLOCK_M=128, BLOCK_N=32, num_warps=8, num_stages=3
BLOCK_SIZE_X=128
BLOCK_SIZE_Y=32
HEAD_DIM=128
FP8_OUTPUT=0
STAGE=1 # full attention (non causal)
WARP_SPECIALIZE=1
IS_HOPPER=0
ACCUM_FP16_FALSE=0
ACCUM_FP16_TRUE=1
WRITE_M_FALSE=0
WRITE_M_TRUE=1
EVEN_N_CTX_FALSE=0
EVEN_N_CTX_TRUE=1

KERNEL_NAME="mha_attn_fwd"
KERNEL_SRC="kernels/flash_attention_fwd.py"

# signature: mha_attn_fwd(
#   sm_scale, M,
#   Z, H,
#   q, k, v, o,
#   stride_q,
#   N_CTX,
#   # meta: HEAD_DIM, BLOCK_SIZE_X, BLOCK_SIZE_Y, FP8_OUTPUT, STAGE, warp_specialize, IS_HOPPER, ACCUMULATE_IN_FP16,
#   #       WRITE_M, EVEN_N_CTX
# )
ATTN_FWD_BF16_SIGNATURE="fp32, *fp32:16, i32, i32, \
*bf16:16, *bf16:16, *bf16:16, *bf16:16, \
i32, i32, \
$HEAD_DIM, $BLOCK_SIZE_X, $BLOCK_SIZE_Y, $FP8_OUTPUT, $STAGE, $WARP_SPECIALIZE, $IS_HOPPER, $ACCUM_FP16_FALSE, $WRITE_M_TRUE, $EVEN_N_CTX_TRUE"

ATTN_FWD_BF16_NOLSE_SIGNATURE="fp32, *fp32:16, i32, i32, \
*bf16:16, *bf16:16, *bf16:16, *bf16:16, \
i32, i32, \
$HEAD_DIM, $BLOCK_SIZE_X, $BLOCK_SIZE_Y, $FP8_OUTPUT, $STAGE, $WARP_SPECIALIZE, $IS_HOPPER, $ACCUM_FP16_FALSE, $WRITE_M_FALSE, $EVEN_N_CTX_TRUE"

ATTN_FWD_BF16_UNEVEN_SIGNATURE="fp32, *fp32:16, i32, i32, \
*bf16:16, *bf16:16, *bf16:16, *bf16:16, \
i32, i32, \
$HEAD_DIM, $BLOCK_SIZE_X, $BLOCK_SIZE_Y, $FP8_OUTPUT, $STAGE, $WARP_SPECIALIZE, $IS_HOPPER, $ACCUM_FP16_FALSE, $WRITE_M_TRUE, $EVEN_N_CTX_FALSE"

ATTN_FWD_BF16_UNEVEN_NOLSE_SIGNATURE="fp32, *fp32:16, i32, i32, \
*bf16:16, *bf16:16, *bf16:16, *bf16:16, \
i32, i32, \
$HEAD_DIM, $BLOCK_SIZE_X, $BLOCK_SIZE_Y, $FP8_OUTPUT, $STAGE, $WARP_SPECIALIZE, $IS_HOPPER, $ACCUM_FP16_FALSE, $WRITE_M_FALSE, $EVEN_N_CTX_FALSE"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_UNEVEN_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_uneven_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_UNEVEN_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_uneven_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_UNEVEN_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_uneven_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_NOLSE_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_nolse_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_NOLSE_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_nolse_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_NOLSE_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_nolse_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_UNEVEN_NOLSE_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_uneven_nolse_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_UNEVEN_NOLSE_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_uneven_nolse_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_UNEVEN_NOLSE_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_uneven_nolse_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_UNEVEN_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_uneven_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_NOLSE_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_nolse_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_UNEVEN_NOLSE_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_uneven_nolse_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_gfx942 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_gfx1101 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_UNEVEN_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_uneven_gfx942 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_UNEVEN_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_uneven_gfx1101 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_NOLSE_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_nolse_gfx942 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_NOLSE_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_nolse_gfx1101 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_UNEVEN_NOLSE_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_uneven_nolse_gfx942 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_BF16_UNEVEN_NOLSE_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_bf16_uneven_nolse_gfx1101 \
  "$KERNEL_SRC"

ATTN_FWD_FP16_SIGNATURE="fp32, *fp32:16, i32, i32, \
*fp16:16, *fp16:16, *fp16:16, *fp16:16, \
i32, i32, \
$HEAD_DIM, $BLOCK_SIZE_X, $BLOCK_SIZE_Y, $FP8_OUTPUT, $STAGE, $WARP_SPECIALIZE, $IS_HOPPER, $ACCUM_FP16_FALSE, $WRITE_M_TRUE, $EVEN_N_CTX_TRUE"

ATTN_FWD_FP16_NOLSE_SIGNATURE="fp32, *fp32:16, i32, i32, \
*fp16:16, *fp16:16, *fp16:16, *fp16:16, \
i32, i32, \
$HEAD_DIM, $BLOCK_SIZE_X, $BLOCK_SIZE_Y, $FP8_OUTPUT, $STAGE, $WARP_SPECIALIZE, $IS_HOPPER, $ACCUM_FP16_FALSE, $WRITE_M_FALSE, $EVEN_N_CTX_TRUE"

ATTN_FWD_FP16_ACC_SIGNATURE="fp32, *fp32:16, i32, i32, \
*fp16:16, *fp16:16, *fp16:16, *fp16:16, \
i32, i32, \
$HEAD_DIM, $BLOCK_SIZE_X, $BLOCK_SIZE_Y, $FP8_OUTPUT, $STAGE, $WARP_SPECIALIZE, $IS_HOPPER, $ACCUM_FP16_TRUE, $WRITE_M_TRUE, $EVEN_N_CTX_TRUE"

ATTN_FWD_FP16_ACC_NOLSE_SIGNATURE="fp32, *fp32:16, i32, i32, \
*fp16:16, *fp16:16, *fp16:16, *fp16:16, \
i32, i32, \
$HEAD_DIM, $BLOCK_SIZE_X, $BLOCK_SIZE_Y, $FP8_OUTPUT, $STAGE, $WARP_SPECIALIZE, $IS_HOPPER, $ACCUM_FP16_TRUE, $WRITE_M_FALSE, $EVEN_N_CTX_TRUE"

ATTN_FWD_FP16_UNEVEN_SIGNATURE="fp32, *fp32:16, i32, i32, \
*fp16:16, *fp16:16, *fp16:16, *fp16:16, \
i32, i32, \
$HEAD_DIM, $BLOCK_SIZE_X, $BLOCK_SIZE_Y, $FP8_OUTPUT, $STAGE, $WARP_SPECIALIZE, $IS_HOPPER, $ACCUM_FP16_FALSE, $WRITE_M_TRUE, $EVEN_N_CTX_FALSE"

ATTN_FWD_FP16_UNEVEN_NOLSE_SIGNATURE="fp32, *fp32:16, i32, i32, \
*fp16:16, *fp16:16, *fp16:16, *fp16:16, \
i32, i32, \
$HEAD_DIM, $BLOCK_SIZE_X, $BLOCK_SIZE_Y, $FP8_OUTPUT, $STAGE, $WARP_SPECIALIZE, $IS_HOPPER, $ACCUM_FP16_FALSE, $WRITE_M_FALSE, $EVEN_N_CTX_FALSE"

ATTN_FWD_FP16_ACC_UNEVEN_SIGNATURE="fp32, *fp32:16, i32, i32, \
*fp16:16, *fp16:16, *fp16:16, *fp16:16, \
i32, i32, \
$HEAD_DIM, $BLOCK_SIZE_X, $BLOCK_SIZE_Y, $FP8_OUTPUT, $STAGE, $WARP_SPECIALIZE, $IS_HOPPER, $ACCUM_FP16_TRUE, $WRITE_M_TRUE, $EVEN_N_CTX_FALSE"

ATTN_FWD_FP16_ACC_UNEVEN_NOLSE_SIGNATURE="fp32, *fp32:16, i32, i32, \
*fp16:16, *fp16:16, *fp16:16, *fp16:16, \
i32, i32, \
$HEAD_DIM, $BLOCK_SIZE_X, $BLOCK_SIZE_Y, $FP8_OUTPUT, $STAGE, $WARP_SPECIALIZE, $IS_HOPPER, $ACCUM_FP16_TRUE, $WRITE_M_FALSE, $EVEN_N_CTX_FALSE"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_UNEVEN_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_uneven_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_UNEVEN_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_uneven_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_UNEVEN_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_uneven_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_NOLSE_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_nolse_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_NOLSE_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_nolse_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_NOLSE_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_nolse_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_UNEVEN_NOLSE_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_uneven_nolse_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_UNEVEN_NOLSE_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_uneven_nolse_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_UNEVEN_NOLSE_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_uneven_nolse_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_UNEVEN_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_uneven_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_NOLSE_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_nolse_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_UNEVEN_NOLSE_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_uneven_nolse_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_gfx942 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_gfx1101 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_UNEVEN_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_uneven_gfx942 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_UNEVEN_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_uneven_gfx1101 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_NOLSE_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_nolse_gfx942 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_NOLSE_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_nolse_gfx1101 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_UNEVEN_NOLSE_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_uneven_nolse_gfx942 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_UNEVEN_NOLSE_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_uneven_nolse_gfx1101 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_UNEVEN_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_UNEVEN_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_UNEVEN_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_NOLSE_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_nolse_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_NOLSE_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_nolse_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_NOLSE_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_nolse_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_UNEVEN_NOLSE_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven_nolse_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_UNEVEN_NOLSE_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven_nolse_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_UNEVEN_NOLSE_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven_nolse_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_UNEVEN_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_NOLSE_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_nolse_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_NAME" \
  --signature "$ATTN_FWD_FP16_ACC_UNEVEN_NOLSE_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven_nolse_sm100 \
  "$KERNEL_SRC"

${CUTLASS_RUNNER} fetch_cutlass_sources.py

CUDA_SMS=(sm_80 sm_89 sm_90 sm_100)
for sm in "${CUDA_SMS[@]}"; do
  cfg="kernels/cutlass_codegen/configs/output_configs/autotune_constants_${sm//_/}.json"
  ${CUTLASS_RUNNER} compile.py \
    --mha-config mha_full_attn_cutlass_bf16 \
    --mha-config mha_full_attn_cutlass_bf16_lse \
    --mha-config mha_full_attn_cutlass_fp16 \
    --mha-config mha_full_attn_cutlass_fp16_lse \
    --autotune-config "${cfg}" \
    --package \
    --sm "${sm}"
done

$PYTHON_BIN -m kernels.flash_attention_codegen.fetch_flash_attention

sm_csv="$(IFS=,; echo "${CUDA_SMS[*]}")"
autotune_args=()
for sm in "${CUDA_SMS[@]}"; do
  autotune_args+=("--autotune-config" "kernels/flash_attention_codegen/configs/autotune_constants_${sm//_/}.json")
done

$PYTHON_BIN -m kernels.flash_attention_codegen.compile \
  --sm "${sm_csv}" \
  "${autotune_args[@]}"
