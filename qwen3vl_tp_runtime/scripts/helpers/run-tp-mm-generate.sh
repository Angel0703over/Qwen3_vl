#!/usr/bin/env bash
# Multi-node TP multimodal generate smoke wrapper.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${RUNTIME_ROOT}/.." && pwd)}"
TORCHRUN="${TORCHRUN:-/mnt/ssd/miniconda3/envs/vlm/bin/torchrun}"
CASE_ID="${CASE_ID:-tp-mm-generate}"

usage() {
  cat <<EOF
Usage:
  NODE_RANK=<0..NNODES-1> [NNODES=2] [MASTER_ADDR=10.126.126.2] bash ${0##*/} [extra runtime args]

Environment:
  NNODES           Default: 2
  MASTER_ADDR      Default: 10.126.126.2
  MASTER_PORT      Default: 29536
  MODEL_PATH       Default: /mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct
  FRAME_DIR        Default: /mnt/ssd/code/Qwen3_vl/frames
  MM_PROMPT/PROMPT Default: 请用中文简要介绍一下人工智能。
  MAX_NEW_TOKENS   Default: 4
  NUM_FRAMES       Default: 8
  TP               Default: \${NNODES}
  OUT              Default: \${REPO_ROOT}/baseline_runs/\$(date -u +%Y%m%d)
  LOG_PATH         Default: \${OUT}/${CASE_ID}-rank\${NODE_RANK}.log
  DRY_RUN          Set to 1 to print the command without running it.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

is_uint() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

NNODES="${NNODES:-2}"
MASTER_ADDR="${MASTER_ADDR:-10.126.126.2}"
MASTER_PORT="${MASTER_PORT:-29536}"
NODE_RANK="${NODE_RANK:-}"
MODEL_PATH="${MODEL_PATH:-/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct}"
FRAME_DIR="${FRAME_DIR:-/mnt/ssd/code/Qwen3_vl/frames}"
PROMPT="${PROMPT:-${MM_PROMPT:-请用中文简要介绍一下人工智能。}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
NUM_FRAMES="${NUM_FRAMES:-8}"
TP="${TP:-${NNODES}}"
OUT="${OUT:-${BASELINE_OUT:-${REPO_ROOT}/baseline_runs/$(date -u +%Y%m%d)}}"
LOG_PATH="${LOG_PATH:-${OUT}/${CASE_ID}-rank${NODE_RANK}.log}"
HEXGEN_STARTUP_LOG="${HEXGEN_STARTUP_LOG:-1}"
USE_TIME="${USE_TIME:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${NODE_RANK}" ]]; then
  echo "NODE_RANK is required." >&2
  usage >&2
  exit 2
fi
if ! is_uint "${NNODES}" || (( 10#${NNODES} < 2 )); then
  echo "Invalid NNODES=${NNODES}; pure TP smoke requires at least 2 nodes." >&2
  exit 2
fi
if ! is_uint "${NODE_RANK}" || (( 10#${NODE_RANK} >= 10#${NNODES} )); then
  echo "Invalid NODE_RANK=${NODE_RANK} for NNODES=${NNODES}." >&2
  exit 2
fi
if ! is_uint "${TP}" || (( 10#${TP} < 2 )); then
  echo "Invalid TP=${TP}; pure TP smoke requires TP >= 2." >&2
  exit 2
fi
if (( 10#${TP} != 10#${NNODES} )); then
  echo "Invalid TP=${TP} for NNODES=${NNODES}; pure TP smoke currently requires TP == NNODES." >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUT}"

cmd=(
  "${TORCHRUN}"
  --nnodes "${NNODES}"
  --nproc-per-node 1
  --node-rank "${NODE_RANK}"
  --master-addr "${MASTER_ADDR}"
  --master-port "${MASTER_PORT}"
  "${RUNTIME_ROOT}/scripts/runtime.py"
  --backend tp
  --modality multimodal
  --mode generate
  --model-path "${MODEL_PATH}"
  --frame-dir "${FRAME_DIR}"
  --num-frames "${NUM_FRAMES}"
  --tp "${TP}"
  --prompt "${PROMPT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  "$@"
)

if [[ "${USE_TIME}" == "1" && -x /usr/bin/time ]]; then
  cmd=(/usr/bin/time -p "${cmd[@]}")
fi

echo "[smoke] case=${CASE_ID} rank=${NODE_RANK}/${NNODES} tp=${TP} log=${LOG_PATH}"
printf '[smoke] command: HEXGEN_STARTUP_LOG=%q' "${HEXGEN_STARTUP_LOG}"
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

HEXGEN_STARTUP_LOG="${HEXGEN_STARTUP_LOG}" "${cmd[@]}" 2>&1 | tee "${LOG_PATH}"
