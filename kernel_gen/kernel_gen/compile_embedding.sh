#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1
# Tuned on RTX 4090: BLOCK_SIZE=128, NUM_WARPS=4, NUM_STAGES=2
# signature: embedding_lookup(out_ptr, table_ptr, indices_ptr,
#             num_indices, embedding_dim,
#             stride_out_row, stride_out_col,
#             stride_table_row, stride_table_col,
#             BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE=128
EMBEDDING_SIGNATURE_BF16="*bf16:16, *bf16:16, *u32:16, u32, u32, u32, u32, u32, u32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "embedding_lookup" \
  --signature "$EMBEDDING_SIGNATURE_BF16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/embedding_lookup_bf16_sm80 \
  kernels/embedding.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "embedding_lookup" \
  --signature "$EMBEDDING_SIGNATURE_BF16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/embedding_lookup_bf16_sm89 \
  kernels/embedding.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "embedding_lookup" \
  --signature "$EMBEDDING_SIGNATURE_BF16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/embedding_lookup_bf16_sm90 \
  kernels/embedding.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "embedding_lookup" \
  --signature "$EMBEDDING_SIGNATURE_BF16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/embedding_lookup_bf16_sm100 \
  kernels/embedding.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "embedding_lookup" \
  --signature "$EMBEDDING_SIGNATURE_BF16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/embedding_lookup_bf16_gfx942 \
  kernels/embedding.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "embedding_lookup" \
  --signature "$EMBEDDING_SIGNATURE_BF16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/embedding_lookup_bf16_gfx1101 \
  kernels/embedding.py

EMBEDDING_SIGNATURE_FP16="*fp16:16, *fp16:16, *u32:16, u32, u32, u32, u32, u32, u32, $BLOCK_SIZE"

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "embedding_lookup" \
  --signature "$EMBEDDING_SIGNATURE_FP16" \
  --target cuda:80:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/embedding_lookup_fp16_sm80 \
  kernels/embedding.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "embedding_lookup" \
  --signature "$EMBEDDING_SIGNATURE_FP16" \
  --target cuda:89:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/embedding_lookup_fp16_sm89 \
  kernels/embedding.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "embedding_lookup" \
  --signature "$EMBEDDING_SIGNATURE_FP16" \
  --target cuda:90:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/embedding_lookup_fp16_sm90 \
  kernels/embedding.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "embedding_lookup" \
  --signature "$EMBEDDING_SIGNATURE_FP16" \
  --target cuda:100:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/embedding_lookup_fp16_sm100 \
  kernels/embedding.py

$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "embedding_lookup" \
  --signature "$EMBEDDING_SIGNATURE_FP16" \
  --target hip:gfx942:64 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/embedding_lookup_fp16_gfx942 \
  kernels/embedding.py
$PYTHON_BIN -m pysrc.triton_aot_compiler \
  --kernel-name "embedding_lookup" \
  --signature "$EMBEDDING_SIGNATURE_FP16" \
  --target hip:gfx1101:32 \
  --num-stages 2 \
  --num-warps 4 \
  --out-name output/embedding_lookup_fp16_gfx1101 \
  kernels/embedding.py
