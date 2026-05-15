#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Cutlass venv not found at ${PYTHON_BIN}. Create it in ${SCRIPT_DIR}." >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <script.py> [args...]" >&2
  exit 2
fi

SCRIPT_PATH="$1"
shift

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  if [[ -f "${SCRIPT_DIR}/${SCRIPT_PATH}" ]]; then
    SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_PATH}"
  else
    echo "Cutlass script not found: ${SCRIPT_PATH}" >&2
    exit 3
  fi
fi

PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH

exec "${PYTHON_BIN}" "${SCRIPT_PATH}" "$@"
