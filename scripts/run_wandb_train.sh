#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN_PATH=${BIN_PATH:-"${ROOT_DIR}/cmake-build-release/fbamtrain"}
CONFIG_PATH=${CONFIG_PATH:-"${ROOT_DIR}/working_dir/run-configs/debug.json"}
WORKDIR=${WORKDIR:-"${ROOT_DIR}/working_dir"}

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
  --fbamtrainargs "--config ${CONFIG_PATH}"
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

python3 "${ARGS[@]}" "$@"
