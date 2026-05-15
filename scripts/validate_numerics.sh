#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: validate_numerics.sh [options]

Options:
  --steps N         Number of validation steps to dump (default: 10)
  --threshold T     Tolerance for tenscmp (default: 0.3)
  --build-dir DIR   CMake build dir (default: cmake-build-debug)
  --work-dir DIR    Working dir with run-configs/ and data (default: working_dir)
  --config PATH     Run config path (default: working_dir/run-configs/debug.json)
  -h, --help        Show this help
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STEPS=10
THRESHOLD="0.3"
BUILD_DIR="${ROOT}/cmake-build-release"
WORK_DIR="${ROOT}/working_dir"
CONFIG="${WORK_DIR}/run-configs/debug.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -x /usr/bin/cmake ]]; then
  CMAKE_BIN="/usr/bin/cmake"
elif command -v cmake >/dev/null 2>&1; then
  CMAKE_BIN="$(command -v cmake)"
else
  echo "cmake not found in PATH and /usr/bin/cmake missing." >&2
  exit 1
fi

if [[ "${BUILD_DIR}" != /* ]]; then
  BUILD_DIR="${ROOT}/${BUILD_DIR}"
fi
if [[ "${WORK_DIR}" != /* ]]; then
  WORK_DIR="${ROOT}/${WORK_DIR}"
fi
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${ROOT}/${CONFIG}"
fi

FBAMTRAIN_BIN="${BUILD_DIR}/fbamtrain"
TENSCMP_BIN="${BUILD_DIR}/tenscmp"
PYTHON_BIN="${ROOT}/toyimpl/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python venv not found at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found at ${CONFIG}" >&2
  exit 1
fi

echo "[INFO] Building fbamtrain and tenscmp in ${BUILD_DIR}"
${CMAKE_BIN} --build "${BUILD_DIR}" --target fbamtrain tenscmp

CPP_BASE="validation_dump_cpp.safetensors"
PY_BASE="validation_dump_py.safetensors"

rm -f "${WORK_DIR}/validation_dump_cpp.step"*.safetensors \
      "${WORK_DIR}/validation_dump_py.step"*.safetensors

pushd "${WORK_DIR}" >/dev/null
echo "[INFO] Running C++ validation dump (${STEPS} steps)"
"${FBAMTRAIN_BIN}" --config "${CONFIG}" \
  --validation-dump \
  --validation-dump-steps "${STEPS}" \
  --validation-dump-file "${CPP_BASE}"

echo "[INFO] Running Python validation dump (${STEPS} steps)"
"${PYTHON_BIN}" "${ROOT}/toyimpl/src/main.py" -c "${CONFIG}" \
  --validation-dump \
  --validation-dump-steps "${STEPS}" \
  --validation-dump-file "${PY_BASE}"
popd >/dev/null

echo "[INFO] Comparing dumps with tenscmp threshold ${THRESHOLD}"
for step in $(seq 0 $((STEPS - 1))); do
  "${TENSCMP_BIN}" -t "${THRESHOLD}" \
    "${WORK_DIR}/validation_dump_py.step${step}.safetensors" \
    "${WORK_DIR}/validation_dump_cpp.step${step}.safetensors"
done

echo "[INFO] PASS: all steps within threshold ${THRESHOLD}"
