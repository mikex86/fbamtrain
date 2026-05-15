#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}

set -euo pipefail
export TRITON_ALWAYS_COMPILE=1

CUDA_SMS=(sm_80 sm_89 sm_90 sm_100)
sm_csv="$(IFS=,; echo "${CUDA_SMS[*]}")"
autotune_args=()
for sm in "${CUDA_SMS[@]}"; do
  autotune_args+=("--autotune-config" "${SCRIPT_DIR}/kernels/flash_attention_codegen/configs/autotune_constants_${sm//_/}.json")
done

$PYTHON_BIN -m kernels.flash_attention_codegen.fetch_flash_attention

$PYTHON_BIN -m kernels.flash_attention_codegen.compile \
  --sm "${sm_csv}" \
  --head-dim 128 \
  "${autotune_args[@]}"
