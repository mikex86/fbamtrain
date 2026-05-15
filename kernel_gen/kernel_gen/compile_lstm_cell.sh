#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
set -euo pipefail
export TRITON_ALWAYS_COMPILE=1

# Tuned on RTX 4090: BLOCK_SIZE=128, NUM_WARPS=4, NUM_STAGES=2
# signature: lstm_cell_fwd(gates, c_prev, h_out, c_out, y_out, batch, hidden, BLOCK_SIZE: tl.constexpr)
CUDA_ARCHES=(sm80 sm89 sm90 sm100)
HIP_ARCHES=(gfx942 gfx1101)

for arch in "${CUDA_ARCHES[@]}"; do
  TARGET="cuda:${arch:2}:32"
  OUT_BASE="${SCRIPT_DIR}/output/lstm_cell_fwd_out_fp16_${arch}"
  ${PYTHON_BIN} -m pysrc.triton_aot_compiler \
    ${SCRIPT_DIR}/kernels/lstm_cell.py \
    --kernel-name lstm_cell_fwd_out_fp16 \
    --signature "*fp16, *fp16, *fp16, *fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, 128" \
    --target "${TARGET}" \
    --num-warps 4 \
    --num-stages 2 \
    --out-name "${OUT_BASE}.cubin" \
    --out-path "${OUT_BASE}"

  OUT_BASE_FP32_STATE="${SCRIPT_DIR}/output/lstm_cell_fwd_fp32_state_out_fp16_${arch}"
  ${PYTHON_BIN} -m pysrc.triton_aot_compiler \
    ${SCRIPT_DIR}/kernels/lstm_cell.py \
    --kernel-name lstm_cell_fwd_fp32_state_out_fp16 \
    --signature "*fp32, *fp32, *fp16, *fp32, *fp16, i32, i32, i32, i32, i32, i32, i32, 128" \
    --target "${TARGET}" \
    --num-warps 4 \
    --num-stages 2 \
    --out-name "${OUT_BASE_FP32_STATE}.cubin" \
    --out-path "${OUT_BASE_FP32_STATE}"

  OUT_BASE_FP32_STATE_BF16="${SCRIPT_DIR}/output/lstm_cell_fwd_fp32_state_out_bf16_${arch}"
  ${PYTHON_BIN} -m pysrc.triton_aot_compiler \
    ${SCRIPT_DIR}/kernels/lstm_cell.py \
    --kernel-name lstm_cell_fwd_fp32_state_out_bf16 \
    --signature "*fp32, *fp32, *bf16, *fp32, *bf16, i32, i32, i32, i32, i32, i32, i32, 128" \
    --target "${TARGET}" \
    --num-warps 4 \
    --num-stages 2 \
    --out-name "${OUT_BASE_FP32_STATE_BF16}.cubin" \
    --out-path "${OUT_BASE_FP32_STATE_BF16}"
done

for arch in "${HIP_ARCHES[@]}"; do
  num="${arch#gfx}"
  if [[ "${num}" =~ ^[0-9]+$ ]]; then
    major=$((num / 100))
    if (( major < 10 )); then
      warp=64
    else
      warp=32
    fi
  else
    echo "Warning: unexpected HIP arch '${arch}'; defaulting warp size to 32" >&2
    warp=32
  fi
  TARGET="hip:${arch}:${warp}"
  OUT_BASE="${SCRIPT_DIR}/output/lstm_cell_fwd_out_fp16_${arch}"
  ${PYTHON_BIN} -m pysrc.triton_aot_compiler \
    ${SCRIPT_DIR}/kernels/lstm_cell.py \
    --kernel-name lstm_cell_fwd_out_fp16 \
    --signature "*fp16, *fp16, *fp16, *fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, 128" \
    --target "${TARGET}" \
    --num-warps 4 \
    --num-stages 2 \
    --out-name "${OUT_BASE}" \
    --out-path "${OUT_BASE}"

  OUT_BASE_FP32_STATE="${SCRIPT_DIR}/output/lstm_cell_fwd_fp32_state_out_fp16_${arch}"
  ${PYTHON_BIN} -m pysrc.triton_aot_compiler \
    ${SCRIPT_DIR}/kernels/lstm_cell.py \
    --kernel-name lstm_cell_fwd_fp32_state_out_fp16 \
    --signature "*fp32, *fp32, *fp16, *fp32, *fp16, i32, i32, i32, i32, i32, i32, i32, 128" \
    --target "${TARGET}" \
    --num-warps 4 \
    --num-stages 2 \
    --out-name "${OUT_BASE_FP32_STATE}" \
    --out-path "${OUT_BASE_FP32_STATE}"

  OUT_BASE_FP32_STATE_BF16="${SCRIPT_DIR}/output/lstm_cell_fwd_fp32_state_out_bf16_${arch}"
  ${PYTHON_BIN} -m pysrc.triton_aot_compiler \
    ${SCRIPT_DIR}/kernels/lstm_cell.py \
    --kernel-name lstm_cell_fwd_fp32_state_out_bf16 \
    --signature "*fp32, *fp32, *bf16, *fp32, *bf16, i32, i32, i32, i32, i32, i32, i32, 128" \
    --target "${TARGET}" \
    --num-warps 4 \
    --num-stages 2 \
    --out-name "${OUT_BASE_FP32_STATE_BF16}" \
    --out-path "${OUT_BASE_FP32_STATE_BF16}"
done
