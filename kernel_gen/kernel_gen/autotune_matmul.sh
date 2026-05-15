#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
ROCM_PYTHON_BIN=${ROCM_PYTHON_BIN:-"${SCRIPT_DIR}/.venv-rocm/bin/python"}

set -euo pipefail
export TRITON_ALWAYS_COMPILE=1

cd "${SCRIPT_DIR}"

# TODO: run both amd and nvidia when available
DETECT_PYTHON="${PYTHON_BIN}"
if [[ -x "${ROCM_PYTHON_BIN}" ]]; then
  DETECT_PYTHON="${ROCM_PYTHON_BIN}"
fi

TARGET_SUFFIX=$($DETECT_PYTHON - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA/HIP device required for autotuning")
props = torch.cuda.get_device_properties(0)
if torch.version.hip:
    gcn = getattr(props, "gcnArchName", None)
    if not gcn:
        raise SystemExit("Unable to determine GCN architecture for ROCm device")
    print(gcn.split(":")[0])
else:
    print(f"sm{props.major}{props.minor}")
PY
)

OUT="configs/triton_matmul_autotune_${TARGET_SUFFIX}.json"
if [[ "${TARGET_SUFFIX}" == sm* ]]; then
  if [[ -z "${CUDA_HOME:-}" ]]; then
    ptxas_path="$(command -v ptxas || true)"
    if [[ -n "${ptxas_path}" ]]; then
      export CUDA_HOME="$(dirname "$(dirname "${ptxas_path}")")"
    elif [[ -d "/usr/local/cuda" ]]; then
      export CUDA_HOME="/usr/local/cuda"
    fi
  fi
  if [[ -z "${CUDA_HOME:-}" ]]; then
    echo "CUDA_HOME not set and ptxas not found; cannot autotune CUDA Triton kernels." >&2
    exit 1
  fi
  TRITON_PTXAS_PATH=$CUDA_HOME/bin/ptxas \
    $PYTHON_BIN -m pysrc.autotune_matmul --output "${OUT}" --target "${TARGET_SUFFIX}"
else
  if [[ -x "${ROCM_PYTHON_BIN}" ]]; then
    PYTHON_BIN="${ROCM_PYTHON_BIN}"
  fi
  if [[ "${TARGET_SUFFIX}" == "gfx1101" && -z "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
  fi
  $PYTHON_BIN -m pysrc.autotune_matmul --output "${OUT}" --target "${TARGET_SUFFIX}"
fi
