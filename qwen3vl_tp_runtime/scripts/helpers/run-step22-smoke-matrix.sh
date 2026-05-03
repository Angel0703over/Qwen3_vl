#!/usr/bin/env bash
# One-click Step 22 smoke matrix runner.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${RUNTIME_ROOT}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ssd/miniconda3/envs/vlm/bin/python}"
TORCHRUN="${TORCHRUN:-/mnt/ssd/miniconda3/envs/vlm/bin/torchrun}"

MODEL_PATH="${MODEL_PATH:-/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct}"
FRAME_DIR="${FRAME_DIR:-/mnt/ssd/code/Qwen3_vl/frames}"
VIDEO_PATH="${VIDEO_PATH:-}"
VIDEO_URL="${VIDEO_URL:-}"
VIDEO_FPS="${VIDEO_FPS:-}"
VIDEO_NFRAMES="${VIDEO_NFRAMES:-}"
VIDEO_START="${VIDEO_START:-}"
VIDEO_END="${VIDEO_END:-}"
VIDEO_MIN_FRAMES="${VIDEO_MIN_FRAMES:-}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-}"
FULL_VIDEO_NFRAMES="${FULL_VIDEO_NFRAMES:-${VIDEO_NFRAMES:-4}}"

PROMPT="${PROMPT:-请用中文简要介绍一下人工智能。}"
FULL_VIDEO_PROMPT="${FULL_VIDEO_PROMPT:-请用中文简要描述这个视频的主要内容。}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
LONG_MAX_NEW_TOKENS="${LONG_MAX_NEW_TOKENS:-16}"
NUM_FRAMES="${NUM_FRAMES:-8}"

MASTER_ADDR="${MASTER_ADDR:-10.126.126.3}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29670}"
TP_HOSTS="${TP_HOSTS:-local 10.126.126.4}"
PP_HOSTS="${PP_HOSTS:-${TP_HOSTS}}"
PP3_HOSTS="${PP3_HOSTS:-${HYBRID_HOSTS:-local 10.126.126.4 10.126.126.5}}"
HYBRID_HOSTS="${HYBRID_HOSTS:-local 10.126.126.4 10.126.126.5}"
PP_VIDEO_HOSTS="${PP_VIDEO_HOSTS:-${PP_HOSTS}}"
TP_VIDEO_HOSTS="${TP_VIDEO_HOSTS:-${TP_HOSTS}}"
HYBRID_VIDEO_HOSTS="${HYBRID_VIDEO_HOSTS:-${TP_HOSTS}}"

SSH_USER="${SSH_USER:-nvidia}"
SSH_PORT="${SSH_PORT:-22}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519_jetson2}"

OUT="${OUT:-${BASELINE_OUT:-${REPO_ROOT}/baseline_runs/$(date -u +%Y%m%d-step22-smoke-matrix)}}"
INCLUDE_OPTIONAL="${INCLUDE_OPTIONAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_RUN="${SKIP_RUN:-0}"
SKIP_CHECKS="${SKIP_CHECKS:-0}"
SKIP_PERF="${SKIP_PERF:-0}"
USE_TIME="${USE_TIME:-1}"
HEXGEN_STARTUP_LOG="${HEXGEN_STARTUP_LOG:-1}"
SELECTED_CASE_IDS=()
if [[ -n "${CASE_IDS:-}" ]]; then
  read -r -a SELECTED_CASE_IDS <<< "${CASE_IDS}"
fi

usage() {
  cat <<EOF
Usage:
  bash ${0##*/} [options]

Options:
  --out PATH               Baseline output directory. Default: ${OUT}
  --include-optional       Also run full-video optional cases.
  --case-id ID             Run only one matrix case; can be repeated.
                           This is useful when the current lab has only two CUDA hosts.
  --dry-run                Print commands without running them.
  --skip-run               Only run checker/perf on an existing OUT directory.
  --skip-checks            Do not run check_baseline_logs.py after smoke.
  --skip-perf              Do not generate runtime-perf-records/table.
  -h, --help               Show this help.

Host environment:
  TP_HOSTS                 Space-separated hosts for TP ranks. Default: "${TP_HOSTS}"
  PP_HOSTS                 Space-separated hosts for PP ranks. Default: "${PP_HOSTS}"
  PP3_HOSTS                Space-separated hosts for PP=3 smoke. Default: "${PP3_HOSTS}"
  HYBRID_HOSTS             Space-separated hosts for HYBRID ranks. Default: "${HYBRID_HOSTS}"
  MASTER_ADDR              torchrun master addr. Default: ${MASTER_ADDR}
  MASTER_PORT_BASE         First port for distributed cases. Default: ${MASTER_PORT_BASE}

Use "local" as a host to run that rank on this machine. Other hosts are run by ssh.
The repository, model path, and frame/video inputs must already exist on every host.

Input environment:
  MODEL_PATH               Default: ${MODEL_PATH}
  FRAME_DIR                Default: ${FRAME_DIR}
  VIDEO_PATH/VIDEO_URL     Required only with --include-optional full-video cases.
  FULL_VIDEO_NFRAMES       Default: ${FULL_VIDEO_NFRAMES}
  PROMPT                   Default: ${PROMPT}
  FULL_VIDEO_PROMPT        Default: ${FULL_VIDEO_PROMPT}

Examples:
  DRY_RUN=1 bash ${0##*/}
  TP_HOSTS="local 10.126.126.4" HYBRID_HOSTS="local 10.126.126.4 10.126.126.5" bash ${0##*/}
  VIDEO_PATH=/mnt/ssd/code/Qwen3_vl/test/demo.mp4 bash ${0##*/} --include-optional
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT="$2"
      shift 2
      ;;
    --include-optional)
      INCLUDE_OPTIONAL=1
      shift
      ;;
    --case-id)
      SELECTED_CASE_IDS+=("$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-run)
      SKIP_RUN=1
      shift
      ;;
    --skip-checks)
      SKIP_CHECKS=1
      shift
      ;;
    --skip-perf)
      SKIP_PERF=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUT}"

has_selected_cases() {
  (( ${#SELECTED_CASE_IDS[@]} > 0 ))
}

should_run_case() {
  local case_id="$1"
  local selected
  if ! has_selected_cases; then
    return 0
  fi
  for selected in "${SELECTED_CASE_IDS[@]}"; do
    if [[ "${selected}" == "${case_id}" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ -n "${VIDEO_PATH}" && -n "${VIDEO_URL}" ]]; then
  echo "VIDEO_PATH and VIDEO_URL are mutually exclusive." >&2
  exit 2
fi
needs_full_video=0
if [[ "${INCLUDE_OPTIONAL}" == "1" ]]; then
  needs_full_video=1
fi
for selected in "${SELECTED_CASE_IDS[@]}"; do
  if [[ "${selected}" == *"video"* ]]; then
    needs_full_video=1
  fi
done
if [[ "${needs_full_video}" == "1" && -z "${VIDEO_PATH}" && -z "${VIDEO_URL}" ]]; then
  echo "--include-optional requires VIDEO_PATH or VIDEO_URL for full-video smoke." >&2
  exit 2
fi

ssh_base=(ssh -p "${SSH_PORT}")
if [[ -n "${SSH_KEY_PATH}" && -f "${SSH_KEY_PATH}" ]]; then
  ssh_base+=(-i "${SSH_KEY_PATH}" -o IdentitiesOnly=yes)
fi

is_local_host() {
  [[ "$1" == "local" || "$1" == "localhost" || "$1" == "127.0.0.1" ]]
}

shell_join() {
  local out="" part
  for part in "$@"; do
    printf -v part "%q" "${part}"
    out+="${part} "
  done
  printf '%s' "${out% }"
}

run_logged() {
  local label="$1"
  local log_path="$2"
  shift 2
  local cmd=("$@")
  echo "[step22] ${label} -> ${log_path}"
  printf '[step22] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  (
    cd "${REPO_ROOT}"
    "${cmd[@]}"
  ) >"${log_path}" 2>&1
}

base_env=(
  "REPO_ROOT=${REPO_ROOT}"
  "RUNTIME_ROOT=${RUNTIME_ROOT}"
  "PYTHON_BIN=${PYTHON_BIN}"
  "TORCHRUN=${TORCHRUN}"
  "MODEL_PATH=${MODEL_PATH}"
  "FRAME_DIR=${FRAME_DIR}"
  "NUM_FRAMES=${NUM_FRAMES}"
  "VIDEO_PATH="
  "VIDEO_URL="
  "VIDEO_FPS="
  "VIDEO_NFRAMES="
  "VIDEO_START="
  "VIDEO_END="
  "VIDEO_MIN_FRAMES="
  "VIDEO_MAX_FRAMES="
  "PROMPT=${PROMPT}"
  "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
  "OUT=${OUT}"
  "HEXGEN_STARTUP_LOG=${HEXGEN_STARTUP_LOG}"
  "USE_TIME=${USE_TIME}"
)

full_video_env=()
if [[ -n "${VIDEO_PATH}" ]]; then
  full_video_env+=("VIDEO_PATH=${VIDEO_PATH}")
elif [[ -n "${VIDEO_URL}" ]]; then
  full_video_env+=("VIDEO_URL=${VIDEO_URL}")
fi
full_video_env+=(
  "VIDEO_NFRAMES=${FULL_VIDEO_NFRAMES}"
  "PROMPT=${FULL_VIDEO_PROMPT}"
)

runtime_frame_args=(--frame-dir "${FRAME_DIR}" --num-frames "${NUM_FRAMES}")
runtime_full_video_args=()
if [[ -n "${VIDEO_PATH}" ]]; then
  runtime_full_video_args+=(--video-path "${VIDEO_PATH}")
else
  runtime_full_video_args+=(--video-url "${VIDEO_URL}")
fi
runtime_full_video_args+=(--video-nframes "${FULL_VIDEO_NFRAMES}")

run_hf_case() {
  local case_id="$1"
  local modality="$2"
  local prompt="$3"
  shift 3
  local input_args=("$@")
  local log_path="${OUT}/${case_id}.log"
  local cmd=(
    env
    "PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}"
    "HEXGEN_STARTUP_LOG=${HEXGEN_STARTUP_LOG}"
    "${PYTHON_BIN}"
    "${RUNTIME_ROOT}/scripts/runtime.py"
    --backend hf
    --modality "${modality}"
    --mode generate
    --model-path "${MODEL_PATH}"
    "${input_args[@]}"
    --prompt "${prompt}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
  )
  if [[ "${USE_TIME}" == "1" && -x /usr/bin/time ]]; then
    cmd=(/usr/bin/time -p "${cmd[@]}")
  fi
  run_logged "${case_id}" "${log_path}" "${cmd[@]}"
}

run_remote_rank() {
  local host="$1"
  local log_path="$2"
  local stderr_path="$3"
  shift 3
  local remote_cmd=("$@")
  local joined
  joined="$(shell_join cd "${REPO_ROOT}") && $(shell_join "${remote_cmd[@]}")"
  if is_local_host "${host}"; then
    echo "[step22] local rank -> ${log_path}"
    printf '[step22] command: %s\n' "${joined}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      return 0
    fi
    bash -lc "${joined}" >"${log_path}" 2>"${stderr_path}" &
  else
    echo "[step22] ssh ${host} -> ${log_path}"
    printf '[step22] command:'
    printf ' %q' "${ssh_base[@]}" "${SSH_USER}@${host}" "${joined}"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then
      return 0
    fi
    "${ssh_base[@]}" "${SSH_USER}@${host}" "${joined}" >"${log_path}" 2>"${stderr_path}" &
  fi
}

run_distributed_case() {
  local case_id="$1"
  local helper="$2"
  local hosts_string="$3"
  local master_port="$4"
  shift 4
  local extra_env=()
  local extra_args=()
  local parse_args=0
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
      parse_args=1
      shift
      continue
    fi
    if [[ "${parse_args}" == "0" ]]; then
      extra_env+=("$1")
    else
      extra_args+=("$1")
    fi
    shift
  done

  read -r -a hosts <<< "${hosts_string}"
  local nnodes="${#hosts[@]}"
  if (( nnodes < 2 )); then
    echo "${case_id}: at least 2 hosts are required, got: ${hosts_string}" >&2
    exit 2
  fi

  echo "[step22] run distributed case=${case_id} nnodes=${nnodes} master=${MASTER_ADDR}:${master_port}"
  local pids=()
  local rank host log_path stderr_path
  for rank in "${!hosts[@]}"; do
    host="${hosts[$rank]}"
    log_path="${OUT}/${case_id}-rank${rank}.log"
    stderr_path="${OUT}/${case_id}-rank${rank}.ssh.stderr"
    local cmd=(
      env
      "${base_env[@]}"
      "${extra_env[@]}"
      "CASE_ID=${case_id}"
      "NODE_RANK=${rank}"
      "NNODES=${nnodes}"
      "MASTER_ADDR=${MASTER_ADDR}"
      "MASTER_PORT=${master_port}"
      "LOG_PATH=/dev/null"
      bash
      "${helper}"
      "${extra_args[@]}"
    )
    run_remote_rank "${host}" "${log_path}" "${stderr_path}" "${cmd[@]}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      pids+=("$!")
    fi
  done

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    echo "[step22] case failed: ${case_id}" >&2
    exit "${status}"
  fi
}

run_post_checks() {
  if [[ "${SKIP_CHECKS}" != "1" ]]; then
    if has_selected_cases; then
      : >"${OUT}/check-smoke-matrix.txt"
      local case_id
      for case_id in "${SELECTED_CASE_IDS[@]}"; do
        local paths=()
        if [[ -f "${OUT}/${case_id}.log" ]]; then
          paths+=("${OUT}/${case_id}.log")
        fi
        while IFS= read -r path; do
          paths+=("${path}")
        done < <(find "${OUT}" -maxdepth 1 -type f -name "${case_id}-rank*.log" | sort)
        if (( ${#paths[@]} == 0 )); then
          echo "missing log files for ${case_id}" >>"${OUT}/check-smoke-matrix.txt"
          return 1
        fi
        echo "[step22] check ${case_id}" | tee -a "${OUT}/check-smoke-matrix.txt"
        env "PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}" \
          "${PYTHON_BIN}" \
          "${RUNTIME_ROOT}/scripts/check_baseline_logs.py" \
          --case-id "${case_id}" \
          --require-transport-metrics \
          "${paths[@]}" >>"${OUT}/check-smoke-matrix.txt" 2>&1
      done
    else
      local check_cmd=(
        "${PYTHON_BIN}"
        "${RUNTIME_ROOT}/scripts/check_baseline_logs.py"
        --matrix step22
        --baseline-dir "${OUT}"
        --require-transport-metrics
      )
      if [[ "${INCLUDE_OPTIONAL}" == "1" ]]; then
        check_cmd+=(--include-optional)
      fi
      run_logged "check-smoke-matrix" "${OUT}/check-smoke-matrix.txt" env "PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}" "${check_cmd[@]}"
    fi
  fi

  if [[ "${SKIP_PERF}" != "1" ]]; then
    local perf_cmd=(
      "${PYTHON_BIN}"
      "${RUNTIME_ROOT}/scripts/collect_runtime_perf.py"
      --baseline-dir "${OUT}"
      --output-json "${OUT}/runtime-perf-records.json"
      --output-md "${OUT}/runtime-perf-table.md"
    )
    if has_selected_cases; then
      local case_id
      for case_id in "${SELECTED_CASE_IDS[@]}"; do
        perf_cmd+=(--case-id "${case_id}")
      done
    else
      perf_cmd+=(--matrix step22)
      if [[ "${INCLUDE_OPTIONAL}" == "1" ]]; then
        perf_cmd+=(--include-optional)
      fi
    fi
    run_logged "collect-runtime-perf" "${OUT}/collect-runtime-perf.txt" env "PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}" "${perf_cmd[@]}"
  fi
}

write_readme() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  {
    echo "# Step 22 Smoke Matrix Baseline"
    echo
    echo "| item | value |"
    echo "| --- | --- |"
    echo "| created_utc | $(date -u '+%Y-%m-%d %H:%M:%S') |"
    echo "| include_optional | ${INCLUDE_OPTIONAL} |"
    if has_selected_cases; then
      echo "| selected_case_ids | \`${SELECTED_CASE_IDS[*]}\` |"
    fi
    echo "| model | \`${MODEL_PATH}\` |"
    echo "| frame_dir | \`${FRAME_DIR}\` |"
    if [[ -n "${VIDEO_PATH}" ]]; then
      echo "| video_path | \`${VIDEO_PATH}\` |"
    fi
    if [[ -n "${VIDEO_URL}" ]]; then
      echo "| video_url | \`${VIDEO_URL}\` |"
    fi
    echo "| tp_hosts | \`${TP_HOSTS}\` |"
    echo "| pp_hosts | \`${PP_HOSTS}\` |"
    echo "| pp3_hosts | \`${PP3_HOSTS}\` |"
    echo "| hybrid_hosts | \`${HYBRID_HOSTS}\` |"
    echo
    if [[ -f "${OUT}/runtime-perf-table.md" ]]; then
      echo "## Perf Table"
      echo
      cat "${OUT}/runtime-perf-table.md"
      echo
    fi
    echo "## Files"
    echo
    echo "- \`check-smoke-matrix.txt\`"
    echo "- \`collect-runtime-perf.txt\`"
    echo "- \`runtime-perf-records.json\`"
    echo "- \`runtime-perf-table.md\`"
  } >"${OUT}/README.md"
}

echo "[step22] out=${OUT}"
echo "[step22] repo=${REPO_ROOT}"
echo "[step22] include_optional=${INCLUDE_OPTIONAL} dry_run=${DRY_RUN} skip_run=${SKIP_RUN}"
if has_selected_cases; then
  echo "[step22] selected_case_ids=${SELECTED_CASE_IDS[*]}"
fi

if [[ "${SKIP_RUN}" != "1" ]]; then
  if should_run_case "hf-text-generate"; then
    run_hf_case "hf-text-generate" "text" "${PROMPT}"
  fi
  if should_run_case "hf-mm-generate"; then
    run_hf_case "hf-mm-generate" "multimodal" "${PROMPT}" "${runtime_frame_args[@]}"
  fi

  if should_run_case "pp-mm-generate"; then
    run_distributed_case \
      "pp-mm-generate" \
      "${RUNTIME_ROOT}/scripts/helpers/run-pp-mm-generate.sh" \
      "${PP_HOSTS}" \
      "$((MASTER_PORT_BASE + 1))" \
      --
  fi

  if should_run_case "pp3-mm-generate"; then
    run_distributed_case \
      "pp3-mm-generate" \
      "${RUNTIME_ROOT}/scripts/helpers/run-pp-mm-generate.sh" \
      "${PP3_HOSTS}" \
      "$((MASTER_PORT_BASE + 6))" \
      "PP=3" \
      --
  fi

  if should_run_case "tp-mm-generate"; then
    run_distributed_case \
      "tp-mm-generate" \
      "${RUNTIME_ROOT}/scripts/helpers/run-tp-mm-generate.sh" \
      "${TP_HOSTS}" \
      "$((MASTER_PORT_BASE + 2))" \
      --
  fi

  if should_run_case "hybrid-mm-generate"; then
    run_distributed_case \
      "hybrid-mm-generate" \
      "${RUNTIME_ROOT}/scripts/helpers/run-hybrid-mm-generate.sh" \
      "${HYBRID_HOSTS}" \
      "$((MASTER_PORT_BASE + 3))" \
      "PP=2" \
      "TP_DEGREES=2 1" \
      --
  fi

  if should_run_case "tp-mm-generate-long"; then
    run_distributed_case \
      "tp-mm-generate-long" \
      "${RUNTIME_ROOT}/scripts/helpers/run-tp-mm-generate.sh" \
      "${TP_HOSTS}" \
      "$((MASTER_PORT_BASE + 4))" \
      "MAX_NEW_TOKENS=${LONG_MAX_NEW_TOKENS}" \
      --
  fi

  if should_run_case "tp-mm-generate-frame-regression"; then
    run_distributed_case \
      "tp-mm-generate-frame-regression" \
      "${RUNTIME_ROOT}/scripts/helpers/run-tp-mm-generate.sh" \
      "${TP_HOSTS}" \
      "$((MASTER_PORT_BASE + 5))" \
      --
  fi

  if [[ "${INCLUDE_OPTIONAL}" == "1" ]] || has_selected_cases; then
    if should_run_case "hf-mm-generate-video-builder-prompt"; then
      run_hf_case \
        "hf-mm-generate-video-builder-prompt" \
        "multimodal" \
        "${FULL_VIDEO_PROMPT}" \
        "${runtime_full_video_args[@]}"
    fi

    if should_run_case "pp-mm-generate-video"; then
      run_distributed_case \
        "pp-mm-generate-video" \
        "${RUNTIME_ROOT}/scripts/helpers/run-pp-mm-generate.sh" \
        "${PP_VIDEO_HOSTS}" \
        "$((MASTER_PORT_BASE + 7))" \
        "${full_video_env[@]}" \
        --
    fi

    if should_run_case "tp-mm-generate-video"; then
      run_distributed_case \
        "tp-mm-generate-video" \
        "${RUNTIME_ROOT}/scripts/helpers/run-tp-mm-generate.sh" \
        "${TP_VIDEO_HOSTS}" \
        "$((MASTER_PORT_BASE + 8))" \
        "${full_video_env[@]}" \
        --
    fi

    if should_run_case "hybrid-mm-generate-video-pp2tp1"; then
      run_distributed_case \
        "hybrid-mm-generate-video-pp2tp1" \
        "${RUNTIME_ROOT}/scripts/helpers/run-hybrid-mm-generate.sh" \
        "${HYBRID_VIDEO_HOSTS}" \
        "$((MASTER_PORT_BASE + 9))" \
        "${full_video_env[@]}" \
        "PP=2" \
        "TP_DEGREES=1 1" \
        --
    fi
  fi
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  run_post_checks
  write_readme
  echo "[step22] PASS out=${OUT}"
else
  echo "[step22] dry-run complete"
fi
