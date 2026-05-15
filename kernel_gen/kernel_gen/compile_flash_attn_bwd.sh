#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
CUTLASS_RUNNER="${SCRIPT_DIR}/kernels/cutlass_codegen/run_cutlass_py.sh"
export TRITON_ALWAYS_COMPILE=1

BLOCK_M1=32
BLOCK_N1=128
BLOCK_M2=128
BLOCK_N2=32
BLK_SLICE_FACTOR=2
HEAD_DIM=128
CAUSAL=0
EVEN_N_CTX=1
EVEN_N_CTX_UNEVEN=0

PRE_BLOCK=128

KERNEL_SRC="kernels/flash_attention_bwd.py"

KERNEL_PRE_NAME="mha_attn_bwd_pre"
KERNEL_BWD_NAME="mha_attn_bwd"

ATTN_BWD_PRE_BF16_SIGNATURE="*bf16:16, *bf16:16, *fp32:16, i32, i32, i32, \
i32, i32, i32, i32, \
i32, i32, i32, \
$PRE_BLOCK, $HEAD_DIM, $PRE_BLOCK, $HEAD_DIM"

ATTN_BWD_PRE_FP16_SIGNATURE="*fp16:16, *fp16:16, *fp32:16, i32, i32, i32, \
i32, i32, i32, i32, \
i32, i32, i32, \
$PRE_BLOCK, $HEAD_DIM, $PRE_BLOCK, $HEAD_DIM"

ATTN_BWD_BF16_SIGNATURE="*bf16:16, *bf16:16, *bf16:16, *bf16:16, fp32, \
*bf16:16, *bf16:16, *bf16:16, *bf16:16, \
*fp32:16, *fp32:16, \
i32, i32, i32, i32, i32, \
$BLOCK_M1, $BLOCK_N1, $BLOCK_M2, $BLOCK_N2, $BLK_SLICE_FACTOR, $HEAD_DIM, $CAUSAL, $BLOCK_N1, $BLOCK_M2, $EVEN_N_CTX"

ATTN_BWD_FP16_SIGNATURE="*fp16:16, *fp16:16, *fp16:16, *fp16:16, fp32, \
*fp16:16, *fp16:16, *fp16:16, *fp16:16, \
*fp32:16, *fp32:16, \
i32, i32, i32, i32, i32, \
$BLOCK_M1, $BLOCK_N1, $BLOCK_M2, $BLOCK_N2, $BLK_SLICE_FACTOR, $HEAD_DIM, $CAUSAL, $BLOCK_N1, $BLOCK_M2, $EVEN_N_CTX"

ATTN_BWD_BF16_UNEVEN_SIGNATURE="*bf16:16, *bf16:16, *bf16:16, *bf16:16, fp32, \
*bf16:16, *bf16:16, *bf16:16, *bf16:16, \
*fp32:16, *fp32:16, \
i32, i32, i32, i32, i32, \
$BLOCK_M1, $BLOCK_N1, $BLOCK_M2, $BLOCK_N2, $BLK_SLICE_FACTOR, $HEAD_DIM, $CAUSAL, $BLOCK_N1, $BLOCK_M2, $EVEN_N_CTX_UNEVEN"

ATTN_BWD_FP16_UNEVEN_SIGNATURE="*fp16:16, *fp16:16, *fp16:16, *fp16:16, fp32, \
*fp16:16, *fp16:16, *fp16:16, *fp16:16, \
*fp32:16, *fp32:16, \
i32, i32, i32, i32, i32, \
$BLOCK_M1, $BLOCK_N1, $BLOCK_M2, $BLOCK_N2, $BLK_SLICE_FACTOR, $HEAD_DIM, $CAUSAL, $BLOCK_N1, $BLOCK_M2, $EVEN_N_CTX_UNEVEN"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_PRE_NAME" \
  --signature "$ATTN_BWD_PRE_BF16_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_pre_hs128_bf16_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_PRE_NAME" \
  --signature "$ATTN_BWD_PRE_BF16_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_pre_hs128_bf16_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_PRE_NAME" \
  --signature "$ATTN_BWD_PRE_BF16_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_pre_hs128_bf16_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_PRE_NAME" \
  --signature "$ATTN_BWD_PRE_BF16_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_pre_hs128_bf16_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_PRE_NAME" \
  --signature "$ATTN_BWD_PRE_BF16_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_pre_hs128_bf16_gfx942 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_PRE_NAME" \
  --signature "$ATTN_BWD_PRE_BF16_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_pre_hs128_bf16_gfx1101 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_PRE_NAME" \
  --signature "$ATTN_BWD_PRE_FP16_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_pre_hs128_fp16_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_PRE_NAME" \
  --signature "$ATTN_BWD_PRE_FP16_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_pre_hs128_fp16_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_PRE_NAME" \
  --signature "$ATTN_BWD_PRE_FP16_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_pre_hs128_fp16_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_PRE_NAME" \
  --signature "$ATTN_BWD_PRE_FP16_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_pre_hs128_fp16_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_PRE_NAME" \
  --signature "$ATTN_BWD_PRE_FP16_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_pre_hs128_fp16_gfx942 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_PRE_NAME" \
  --signature "$ATTN_BWD_PRE_FP16_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_pre_hs128_fp16_gfx1101 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_BF16_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_bwd_hs128_bf16_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_BF16_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_bwd_hs128_bf16_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_BF16_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_bwd_hs128_bf16_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_BF16_UNEVEN_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_bwd_hs128_bf16_uneven_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_BF16_UNEVEN_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_bwd_hs128_bf16_uneven_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_BF16_UNEVEN_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_bwd_hs128_bf16_uneven_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_BF16_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 5 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_hs128_bf16_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_BF16_UNEVEN_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 5 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_hs128_bf16_uneven_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_BF16_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 5 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_hs128_bf16_gfx942 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_BF16_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 5 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_hs128_bf16_gfx1101 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_BF16_UNEVEN_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 5 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_hs128_bf16_uneven_gfx942 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_BF16_UNEVEN_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 5 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_hs128_bf16_uneven_gfx1101 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_FP16_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_bwd_hs128_fp16_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_FP16_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_bwd_hs128_fp16_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_FP16_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_bwd_hs128_fp16_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_FP16_UNEVEN_SIGNATURE" \
  --target cuda:80:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_bwd_hs128_fp16_uneven_sm80 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_FP16_UNEVEN_SIGNATURE" \
  --target cuda:89:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_bwd_hs128_fp16_uneven_sm89 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_FP16_UNEVEN_SIGNATURE" \
  --target cuda:90:32 \
  --num-stages 3 \
  --num-warps 8 \
  --out-name output/mha_full_attn_bwd_hs128_fp16_uneven_sm90 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_FP16_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 5 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_hs128_fp16_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_FP16_UNEVEN_SIGNATURE" \
  --target cuda:100:32 \
  --num-stages 5 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_hs128_fp16_uneven_sm100 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_FP16_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 5 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_hs128_fp16_gfx942 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_FP16_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 5 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_hs128_fp16_gfx1101 \
  "$KERNEL_SRC"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_FP16_UNEVEN_SIGNATURE" \
  --target hip:gfx942:64 \
  --num-stages 5 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_hs128_fp16_uneven_gfx942 \
  "$KERNEL_SRC"
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "$KERNEL_BWD_NAME" \
  --signature "$ATTN_BWD_FP16_UNEVEN_SIGNATURE" \
  --target hip:gfx1101:32 \
  --num-stages 5 \
  --num-warps 4 \
  --out-name output/mha_full_attn_bwd_hs128_fp16_uneven_gfx1101 \
  "$KERNEL_SRC"

${CUTLASS_RUNNER} fetch_cutlass_sources.py

CUDA_SMS=(sm_80 sm_89 sm_90 sm_100)
for sm in "${CUDA_SMS[@]}"; do
  cfg="kernels/cutlass_codegen/configs/output_configs/autotune_constants_${sm//_/}.json"
  ${CUTLASS_RUNNER} compile.py \
    --mha-bwd-config mha_full_attn_bwd_cutlass_bf16 \
    --mha-bwd-config mha_full_attn_bwd_cutlass_fp16 \
    --autotune-config "${cfg}" \
    --package \
    --sm "${sm}"
done
