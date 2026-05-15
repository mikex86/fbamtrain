#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
ROCM_PYTHON_BIN=${ROCM_PYTHON_BIN:-"${SCRIPT_DIR}/.venv-rocm/bin/python"}
TRITON_JOBS=${TRITON_JOBS:-"$(nproc)"}
CUTLASS_RUNNER="${SCRIPT_DIR}/kernels/cutlass_codegen/run_cutlass_py.sh"

set -euo pipefail
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
  $PYTHON_BIN -m pysrc.compile_conv2d \
    --autotune-config configs/triton_conv2d_autotune_sm80.json \
    --autotune-config configs/triton_conv2d_autotune_sm89.json \
    --autotune-config configs/triton_conv2d_autotune_sm90.json \
    --autotune-config configs/triton_conv2d_autotune_sm100.json &
cuda_pid=$!

if [[ -x "${ROCM_PYTHON_BIN}" ]]; then
  FBAMTRAIN_TRITON_TARGET_BACKENDS=hip \
    $ROCM_PYTHON_BIN -m pysrc.compile_conv2d \
      --autotune-config configs/triton_conv2d_autotune_gfx942.json \
      --autotune-config configs/triton_conv2d_autotune_gfx1101.json &
  hip_pid=$!
else
  echo "ROCm venv not found at ${ROCM_PYTHON_BIN}; skipping HIP conv2d compilation." >&2
fi
wait "${cuda_pid}" ${hip_pid:+${hip_pid}}

${CUTLASS_RUNNER} fetch_cutlass_sources.py

CUTLASS_SMS=(sm_80 sm_89 sm_90 sm_100)
cutlass_pids=()
for sm in "${CUTLASS_SMS[@]}"; do
  cfg="kernels/cutlass_codegen/configs/output_configs/autotune_constants_${sm//_/}.json"
  ${CUTLASS_RUNNER} compile.py \
    --all-configs \
    --autotune-config "${cfg}" \
    --conv2d-configs-file kernels/cutlass_codegen/configs/conv2d_bin_configs.json \
    --package \
    --sm "${sm}" &
  cutlass_pids+=("$!")
done

for pid in "${cutlass_pids[@]}"; do
  wait "${pid}"
done
