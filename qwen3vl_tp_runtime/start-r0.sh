#!/usr/bin/env bash
# Launch rank 0 for the minimal two-stage PP debug path on a single machine.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/ssd/code/Qwen3_vl}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/qwen3vl_tp_runtime}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ssd/miniconda3/envs/vlm/bin/python}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Distributed settings for the minimal two-rank PP test.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29601}"
export WORLD_SIZE="${WORLD_SIZE:-2}"
export RANK="${RANK:-0}"

# Set this when you need a specific interface, for example:
# export GLOO_SOCKET_IFNAME=tun0
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}"

# Local device settings.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DEVICE="${DEVICE:-cuda}"

# Runtime inputs.
export STAGE0_BUNDLE_PATH="${STAGE0_BUNDLE_PATH:-${REPO_ROOT}/qwen3vl_text_stage0_case.pt}"
export STAGE1_BUNDLE_PATH="${STAGE1_BUNDLE_PATH:-${REPO_ROOT}/qwen3vl_text_stage1_case.pt}"
export COMPUTE_DTYPE="${COMPUTE_DTYPE:-auto}"
export COMM_DTYPE="${COMM_DTYPE:-float32}"

echo "[launch] repo_root=${REPO_ROOT}"
echo "[launch] runtime_root=${RUNTIME_ROOT}"
echo "[launch] python_bin=${PYTHON_BIN}"
echo "[launch] master=${MASTER_ADDR}:${MASTER_PORT} world_size=${WORLD_SIZE} rank=${RANK}"
echo "[launch] device=${DEVICE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[launch] stage0_bundle=${STAGE0_BUNDLE_PATH}"
echo "[launch] stage1_bundle=${STAGE1_BUNDLE_PATH}"
echo "[launch] compute_dtype=${COMPUTE_DTYPE} comm_dtype=${COMM_DTYPE}"

exec "${PYTHON_BIN}" \
  "${RUNTIME_ROOT}/scripts/two_stage_text.py" pp \
  --stage0-bundle-path "${STAGE0_BUNDLE_PATH}" \
  --stage1-bundle-path "${STAGE1_BUNDLE_PATH}" \
  --device "${DEVICE}" \
  --compute-dtype "${COMPUTE_DTYPE}" \
  --comm-dtype "${COMM_DTYPE}"
