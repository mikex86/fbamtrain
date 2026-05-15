#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN_PATH=${BIN_PATH:-"${ROOT_DIR}/cmake-build-release/fbamtrain"}
CONFIG_PATH=${CONFIG_PATH:-"${ROOT_DIR}/working_dir/run-configs/debug.json"}
WORKDIR=${WORKDIR:-"${ROOT_DIR}/working_dir"}

FRAME_HEAD_PARALLEL_WORLD_SIZE=${FRAME_HEAD_PARALLEL_WORLD_SIZE:-${FRAME_HEAD_PARALLEL_GPUS:-2}}
if ! [[ "${FRAME_HEAD_PARALLEL_WORLD_SIZE}" =~ ^[0-9]+$ ]]; then
  echo "FRAME_HEAD_PARALLEL_WORLD_SIZE must be an integer from 2 to 8 (got '${FRAME_HEAD_PARALLEL_WORLD_SIZE}')." >&2
  exit 1
fi
if (( FRAME_HEAD_PARALLEL_WORLD_SIZE < 2 || FRAME_HEAD_PARALLEL_WORLD_SIZE > 8 )); then
  echo "Unsupported FRAME_HEAD_PARALLEL_WORLD_SIZE=${FRAME_HEAD_PARALLEL_WORLD_SIZE}. Supported values: 2..8." >&2
  exit 1
fi

DEFAULT_PARALLEL_CONFIG_PATH="${ROOT_DIR}/working_dir/run-configs/frame_head_parallel/parallel_config_fheadparallel.json"

PARALLEL_CONFIG_PATH=${PARALLEL_CONFIG_PATH:-"${DEFAULT_PARALLEL_CONFIG_PATH}"}
if [[ ! -f "${PARALLEL_CONFIG_PATH}" ]]; then
  echo "Parallel config not found: ${PARALLEL_CONFIG_PATH}" >&2
  exit 1
fi
COMMON_ARGS=(--config "${CONFIG_PATH}" --parallel-config "${PARALLEL_CONFIG_PATH}" --world-size "${FRAME_HEAD_PARALLEL_WORLD_SIZE}")
if [[ -n "${DDP_WORLD_SIZE:-}" ]]; then
  if ! [[ "${DDP_WORLD_SIZE}" =~ ^[0-9]+$ ]] || (( DDP_WORLD_SIZE < 2 )); then
    echo "DDP_WORLD_SIZE must be an integer >= 2 when set (got '${DDP_WORLD_SIZE}')." >&2
    exit 1
  fi
  COMMON_ARGS+=(--ddp-world-size "${DDP_WORLD_SIZE}")
fi

if [[ -n "${FRAME_HEAD_PARALLEL_WORKER_IDS:-}" ]]; then
  read -r -a WORKER_IDS <<< "${FRAME_HEAD_PARALLEL_WORKER_IDS}"
else
  mapfile -t WORKER_IDS < <(seq 1 $((FRAME_HEAD_PARALLEL_WORLD_SIZE - 1)))
fi

if [[ ${#WORKER_IDS[@]} -eq 0 ]]; then
  echo "No worker IDs resolved. Set FRAME_HEAD_PARALLEL_WORLD_SIZE >= 2 or FRAME_HEAD_PARALLEL_WORKER_IDS." >&2
  exit 1
fi
if [[ ${#WORKER_IDS[@]} -ne $((FRAME_HEAD_PARALLEL_WORLD_SIZE - 1)) ]]; then
  echo "Expected $((FRAME_HEAD_PARALLEL_WORLD_SIZE - 1)) worker ids for world size ${FRAME_HEAD_PARALLEL_WORLD_SIZE}, got ${#WORKER_IDS[@]}." >&2
  exit 1
fi
for worker_id in "${WORKER_IDS[@]}"; do
  if ! [[ "${worker_id}" =~ ^[0-9]+$ ]] || (( worker_id == 0 || worker_id >= FRAME_HEAD_PARALLEL_WORLD_SIZE )); then
    echo "Worker id must be in [1, world_size): ${worker_id}" >&2
    exit 1
  fi
done

WORKER_PIDS=()
cleanup() {
  for pid in "${WORKER_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

for worker_id in "${WORKER_IDS[@]}"; do
  (cd "${WORKDIR}" && "${BIN_PATH}" "${COMMON_ARGS[@]}" --worker_id "${worker_id}" "$@") &
  WORKER_PIDS+=($!)
done

if [[ -n "${USE_WANDB_LOG:-}" ]]; then
  WANDB_PROJECT=${WANDB_PROJECT:-"fbamtrain"}
  WANDB_ENTITY=${WANDB_ENTITY:-""}
  WANDB_NAME=${WANDB_NAME:-""}
  WANDB_GROUP=${WANDB_GROUP:-""}
  WANDB_TAGS=${WANDB_TAGS:-""}

  ARGS=(
    "${SCRIPT_DIR}/wandb_log_wrapper.py"
    --binary "${BIN_PATH}"
    --workdir "${WORKDIR}"
    --wandb-project "${WANDB_PROJECT}"
    --fbamtrainargs "$(printf '%q ' "${COMMON_ARGS[@]}") --master"
  )

  if [[ -n "${WANDB_ENTITY}" ]]; then
    ARGS+=(--wandb-entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_NAME}" ]]; then
    ARGS+=(--wandb-name "${WANDB_NAME}")
  fi
  if [[ -n "${WANDB_GROUP}" ]]; then
    ARGS+=(--wandb-group "${WANDB_GROUP}")
  fi
  if [[ -n "${WANDB_TAGS}" ]]; then
    ARGS+=(--wandb-tags "${WANDB_TAGS}")
  fi

  if [[ $# -gt 0 ]]; then
    (cd "${WORKDIR}" && FBAMTRAIN_LOG_LEVEL=DEBUG python3 "${ARGS[@]}" -- "$@")
  else
    (cd "${WORKDIR}" && FBAMTRAIN_LOG_LEVEL=DEBUG python3 "${ARGS[@]}")
  fi
else
  (cd "${WORKDIR}" && FBAMTRAIN_LOG_LEVEL=DEBUG "${BIN_PATH}" "${COMMON_ARGS[@]}" --master "$@")
fi

for pid in "${WORKER_PIDS[@]}"; do
  wait "${pid}"
done
