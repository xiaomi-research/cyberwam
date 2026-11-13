#!/usr/bin/env bash
# Copyright (c) 2026 Xiaomi Corporation
# Portions derived from FastWAM (MIT License, https://github.com/yuantianyuan01/FastWAM); see NOTICE.md.

set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_zero1.sh <nproc_per_node> [hydra_overrides...]}"
shift

EXTRA_ARGS=("$@")
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if ! is_integer "${NPROC_PER_NODE}" || ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}" || ! is_integer "${MAIN_PROCESS_PORT}"; then
  echo "Error: NPROC_PER_NODE, NNODES, NODE_RANK, and MASTER_PORT must be integers." >&2
  exit 1
fi
if (( NPROC_PER_NODE < 1 || NUM_MACHINES < 1 )); then
  echo "Error: NPROC_PER_NODE (${NPROC_PER_NODE}) and NNODES (${NUM_MACHINES}) must be greater than zero." >&2
  exit 1
fi
if (( MACHINE_RANK < 0 || MACHINE_RANK >= NUM_MACHINES )); then
  echo "Error: NODE_RANK (${MACHINE_RANK}) must be in [0, NNODES=${NUM_MACHINES})." >&2
  exit 1
fi
if (( MAIN_PROCESS_PORT < 1 || MAIN_PROCESS_PORT > 65535 )); then
  echo "Error: MASTER_PORT (${MAIN_PROCESS_PORT}) must be in [1, 65535]." >&2
  exit 1
fi
if [[ -z "${MAIN_PROCESS_IP}" ]]; then
  echo "Error: MASTER_ADDR must not be empty." >&2
  exit 1
fi
if (( NUM_MACHINES > 1 )) && [[ "${MAIN_PROCESS_IP}" == "127.0.0.1" || "${MAIN_PROCESS_IP}" == "localhost" || "${MAIN_PROCESS_IP}" == "::1" ]]; then
  echo "Error: MASTER_ADDR (${MAIN_PROCESS_IP}) must be reachable from every node for multi-node training." >&2
  exit 1
fi
TOTAL_PROCESSES="$((NPROC_PER_NODE * NUM_MACHINES))"

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

TASK_BASENAME="train"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
  esac
done

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-180}"
    RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"

    if ! is_integer "${RUN_ID_SYNC_TIMEOUT}" || (( RUN_ID_SYNC_TIMEOUT < 1 )); then
      echo "Error: RUN_ID_SYNC_TIMEOUT (${RUN_ID_SYNC_TIMEOUT}) must be a positive integer." >&2
      exit 1
    fi
    if ! is_integer "${RUN_ID_SYNC_PORT}" || (( RUN_ID_SYNC_PORT < 1 || RUN_ID_SYNC_PORT > 65535 )); then
      echo "Error: RUN_ID_SYNC_PORT (${RUN_ID_SYNC_PORT}) must be an integer in [1, 65535]." >&2
      exit 1
    fi

    export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
    export RUN_ID_SYNC_PORT
    export RUN_ID_SYNC_TIMEOUT
    export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
    export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
    export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"

    RUN_ID="$(
      python - <<'PY'
import datetime
import os
from datetime import timedelta

import torch.distributed as dist

host = os.environ["RUN_ID_SYNC_HOST"]
port = int(os.environ["RUN_ID_SYNC_PORT"])
timeout_s = int(os.environ["RUN_ID_SYNC_TIMEOUT"])
machine_rank = int(os.environ["RUN_ID_SYNC_MACHINE_RANK"])
num_machines = int(os.environ["RUN_ID_SYNC_NUM_MACHINES"])
task_basename = os.environ.get("RUN_ID_SYNC_TASK_BASENAME", "train")

store = dist.TCPStore(
    host_name=host,
    port=port,
    world_size=num_machines,
    is_master=(machine_rank == 0),
    timeout=timedelta(seconds=timeout_s),
)
key = f"run_id::{task_basename}"
if machine_rank == 0:
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store.set(key, run_id)
run_id = store.get(key).decode("utf-8")
print(run_id)
PY
    )"

    echo "[run_id_sync] mode=tcpstore host=${RUN_ID_SYNC_HOST} port=${RUN_ID_SYNC_PORT} timeout_s=${RUN_ID_SYNC_TIMEOUT} run_id=${RUN_ID}"
  fi
fi

echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} total_processes=${TOTAL_PROCESSES} machine_rank=${MACHINE_RANK} main_process=${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT} run_id=${RUN_ID}"

DEEPSPEED_MULTINODE_ARGS=()
if (( NUM_MACHINES > 1 )); then
  DEEPSPEED_MULTINODE_ARGS=(
    --deepspeed_multinode_launcher "${DEEPSPEED_MULTINODE_LAUNCHER:-standard}"
    --same_network
  )
fi

ACCELERATE_CONFIG_FILE="${FASTWAM_ACCELERATE_CONFIG_FILE:-scripts/accelerate_configs/accelerate_zero1_ds.yaml}"
# The B1K launch contract is ZeRO-1.  Do not allow an inherited environment
# variable to silently switch this launcher to a different DeepSpeed stage.
EXPECTED_ACCELERATE_CONFIG="scripts/accelerate_configs/accelerate_zero1_ds.yaml"
if [[ "${ACCELERATE_CONFIG_FILE}" != "${EXPECTED_ACCELERATE_CONFIG}" ]]; then
  echo "Error: train_zero1.sh requires ${EXPECTED_ACCELERATE_CONFIG}; refusing FASTWAM_ACCELERATE_CONFIG_FILE=${ACCELERATE_CONFIG_FILE}" >&2
  exit 1
fi
DEEPSPEED_CONFIG_FILE="scripts/ds_configs/ds_zero1_config.json"
if [[ ! -f "${ACCELERATE_CONFIG_FILE}" ]] || [[ ! -f "${DEEPSPEED_CONFIG_FILE}" ]] || ! grep -Eq '"stage"[[:space:]]*:[[:space:]]*1' "${DEEPSPEED_CONFIG_FILE}"; then
  echo "Error: accelerate config is not the verified ZeRO-1 configuration: ${ACCELERATE_CONFIG_FILE}" >&2
  exit 1
fi

accelerate launch \
  --config_file "${ACCELERATE_CONFIG_FILE}" \
  --num_processes "${TOTAL_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${DEEPSPEED_MULTINODE_ARGS[@]}" \
  scripts/train.py \
  "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"
