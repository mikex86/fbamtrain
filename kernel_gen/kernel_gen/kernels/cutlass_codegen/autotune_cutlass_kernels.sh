#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
CUTLASS_RUNNER="${SCRIPT_DIR}/run_cutlass_py.sh"
BASE_CONFIG="${BASE_CONFIG:-"${SCRIPT_DIR}/configs/input_configs/base_autotune_configs.json"}"
MHA_ONLY=0

FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mha_only)
      MHA_ONLY=1
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${SM:-}" ]]; then
  SM="$("${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA device required to auto-detect SM.")
major, minor = torch.cuda.get_device_capability()
print(f"sm_{major}{minor}")
PY
)"
fi
SM_TAG="${SM//_/}"
OUTPUT=${OUTPUT:-"${SCRIPT_DIR}/configs/output_configs/autotune_constants_${SM_TAG}.json"}
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Base config file not found at ${BASE_CONFIG}. Set BASE_CONFIG to a valid base config JSON." >&2
  exit 1
fi

if [[ "${MHA_ONLY}" -eq 1 ]]; then
  exec "${CUTLASS_RUNNER}" autotune.py \
    --sm "${SM}" \
    --output "${OUTPUT}" \
    --base-config "${BASE_CONFIG}" \
    --mha-only \
    "${FORWARD_ARGS[@]}"
fi

exec "${CUTLASS_RUNNER}" autotune.py \
  --sm "${SM}" \
  --output "${OUTPUT}" \
  --base-config "${BASE_CONFIG}" \
  --base-config "${SCRIPT_DIR}/configs/conv2d_bin_configs.json" \
  --all-configs \
  --all-conv2d-dgrad-configs \
  --all-conv2d-wgrad-configs \
  --all-gemm-configs \
  --all-mha-configs \
  --all-mha-bwd-configs \
  "${FORWARD_ARGS[@]}"
