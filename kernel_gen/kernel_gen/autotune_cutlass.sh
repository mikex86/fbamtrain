#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
CUTLASS_AUTOTUNE_SH="${SCRIPT_DIR}/kernels/cutlass_codegen/autotune_cutlass_kernels.sh"

set -euo pipefail

cd "${SCRIPT_DIR}"

TARGET_SUFFIX=$($PYTHON_BIN - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA device required for CUTLASS autotuning")
if torch.version.hip:
    raise SystemExit("CUTLASS autotuning is CUDA-only")
props = torch.cuda.get_device_properties(0)
print(f"sm{props.major}{props.minor}")
PY
)

if [[ "${TARGET_SUFFIX}" != sm* ]]; then
  echo "Unsupported target suffix for CUTLASS autotune: ${TARGET_SUFFIX}" >&2
  exit 1
fi

SM="sm_${TARGET_SUFFIX#sm}"
OUT="${SCRIPT_DIR}/kernels/cutlass_codegen/configs/output_configs/autotune_constants_${TARGET_SUFFIX}.json"

SM="${SM}" OUTPUT="${OUT}" "${CUTLASS_AUTOTUNE_SH}"
