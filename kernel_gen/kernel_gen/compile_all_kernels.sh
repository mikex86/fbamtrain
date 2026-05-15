#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}

set -euo pipefail
export TRITON_ALWAYS_COMPILE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the virtualenv interpreter by default but allow callers to override.
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if ! PYTHON_BIN="$(command -v "${PYTHON_BIN}" 2>/dev/null)"; then
    echo "Error: Python interpreter '${PYTHON_BIN}' is not executable or not found." >&2
    echo "       Set PYTHON_BIN to a Python with Triton installed." >&2
    exit 1
  fi
fi

PYTHON_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)/$(basename "${PYTHON_BIN}")"

if ! "${PYTHON_BIN}" -c "import triton" >/dev/null 2>&1; then
  echo "Error: '${PYTHON_BIN}' does not have the 'triton' package available." >&2
  exit 1
fi

# Ensure both PYTHON_BIN and bare `python` resolve to the same interpreter.
export PYTHON_BIN
PYTHON_DIR="$(dirname "${PYTHON_BIN}")"
export PATH="${PYTHON_DIR}:${PATH}"

echo "Using Python interpreter: ${PYTHON_BIN}"

shopt -s nullglob
compile_scripts=("${SCRIPT_DIR}"/compile_*.sh)
shopt -u nullglob

if [[ ${#compile_scripts[@]} -eq 0 ]]; then
  echo "No compile scripts found in ${SCRIPT_DIR}." >&2
  exit 1
fi

# Create ./output directory if it doesn't exist
OUTPUT_DIR="${SCRIPT_DIR}/output"
mkdir -p "${OUTPUT_DIR}"

pids=()
names=()

for script_path in "${compile_scripts[@]}"; do
  script_name="$(basename "${script_path}")"
  [[ "${script_name}" == "compile_all_kernels.sh" ]] && continue
  echo
  echo "==> Running ${script_name}"
  (cd "${SCRIPT_DIR}" && bash "${script_name}") &
  pids+=("$!")
  names+=("${script_name}")
done

failures=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "Error: ${names[$i]} failed." >&2
    failures=1
  fi
done

if [[ "${failures}" -ne 0 ]]; then
  exit 1
fi

echo
echo "All kernel compile scripts completed successfully."
