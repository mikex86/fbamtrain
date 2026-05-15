#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
CUTLASS_RUNNER="${SCRIPT_DIR}/kernels/cutlass_codegen/run_cutlass_py.sh"

export TRITON_ALWAYS_COMPILE=1

cd "${SCRIPT_DIR}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  ptxas_path="$(command -v ptxas || true)"
  if [[ -n "${ptxas_path}" ]]; then
    export CUDA_HOME="$(dirname "$(dirname "${ptxas_path}")")"
  elif [[ -d "/usr/local/cuda" ]]; then
    export CUDA_HOME="/usr/local/cuda"
  fi
fi
if [[ -z "${CUDA_HOME:-}" ]]; then
  echo "CUDA_HOME not set and ptxas not found; cannot compile CUDA Triton kernels." >&2
  exit 1
fi

\
  FBAMTRAIN_TRITON_TARGET_BACKENDS=cuda \
  $PYTHON_BIN -m pysrc.compile_matmul \
    --autotune-config configs/triton_matmul_autotune_sm80.json \
    --autotune-config configs/triton_matmul_autotune_sm89.json \
    --autotune-config configs/triton_matmul_autotune_sm90.json \
    --autotune-config configs/triton_matmul_autotune_sm100.json \
    --variant-set base

ROCM_PYTHON_BIN=${ROCM_PYTHON_BIN:-"${SCRIPT_DIR}/.venv-rocm/bin/python"}
if [[ -x "${ROCM_PYTHON_BIN}" ]]; then
  FBAMTRAIN_TRITON_TARGET_BACKENDS=hip \
    $ROCM_PYTHON_BIN -m pysrc.compile_matmul \
      --autotune-config configs/triton_matmul_autotune_gfx942.json \
      --autotune-config configs/triton_matmul_autotune_gfx1101.json \
      --variant-set base
else
  echo "ROCm venv not found at ${ROCM_PYTHON_BIN}; skipping HIP matmul compilation." >&2
fi

${CUTLASS_RUNNER} fetch_cutlass_sources.py

CUTLASS_SMS=("sm_80" "sm_89" "sm_90" "sm_100")

run_cutlass_pack() {
  local sm="$1"
  local autotune_cfg="kernels/cutlass_codegen/configs/output_configs/autotune_constants_${sm//_/}.json"
  ${CUTLASS_RUNNER} compile.py \
      --gemm-config addmm_cutlass_bf16 \
      --gemm-config addmm_cutlass_fp16 \
      --gemm-config addmm_cutlass_fp16_acc_fp16 \
      --gemm-config addmm_cutlass_fp16_out_fp32 \
      --gemm-config addmm_cutlass_fp16_acc_fp32 \
      --autotune-config "${autotune_cfg}" \
      --package --sm "${sm}"

  ${CUTLASS_RUNNER} compile.py \
      --gemm-config matmul_cutlass_bf16 \
      --gemm-config matmul_cutlass_bf16_ta \
      --gemm-config matmul_cutlass_bf16_tb \
      --gemm-config matmul_cutlass_bf16_tab \
      --gemm-config matmul_cutlass_bf16_out_fp32 \
      --gemm-config matmul_cutlass_bf16_out_fp32_ta \
      --gemm-config matmul_cutlass_bf16_out_fp32_tb \
      --gemm-config matmul_cutlass_bf16_out_fp32_tab \
      --gemm-config matmul_cutlass_fp16 \
      --gemm-config matmul_cutlass_fp16_acc_fp16 \
      --gemm-config matmul_cutlass_fp16_out_fp32 \
      --gemm-config matmul_cutlass_fp16_acc_fp32 \
      --gemm-config matmul_cutlass_fp16_ta \
      --gemm-config matmul_cutlass_fp16_tb \
      --gemm-config matmul_cutlass_fp16_tab \
      --gemm-config matmul_cutlass_fp16_acc_fp16_ta \
      --gemm-config matmul_cutlass_fp16_acc_fp16_tb \
      --gemm-config matmul_cutlass_fp16_acc_fp16_tab \
      --gemm-config matmul_cutlass_fp16_out_fp32_ta \
      --gemm-config matmul_cutlass_fp16_out_fp32_tb \
      --gemm-config matmul_cutlass_fp16_out_fp32_tab \
      --gemm-config matmul_cutlass_fp16_acc_fp32_ta \
      --gemm-config matmul_cutlass_fp16_acc_fp32_tb \
      --gemm-config matmul_cutlass_fp16_acc_fp32_tab \
      --autotune-config "${autotune_cfg}" \
      --package --sm "${sm}"
}

for sm in "${CUTLASS_SMS[@]}"; do
  run_cutlass_pack "${sm}"
done
