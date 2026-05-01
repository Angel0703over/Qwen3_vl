#!/usr/bin/env bash
# Minimal regression matrix for runtime-core changes.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${RUNTIME_ROOT}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ssd/miniconda3/envs/vlm/bin/python}"
BASELINE_DIR="${BASELINE_DIR:-${REPO_ROOT}/baseline_runs/20260428}"
INCLUDE_WEIGHT_LOADER=0
SKIP_BASELINE_CHECKS=0

usage() {
  cat <<EOF
Usage:
  bash ${0##*/} [options]

Options:
  --baseline-dir PATH       Frozen baseline log directory.
                            Default: ${BASELINE_DIR}
  --include-weight-loader   Also run test/test_model_weight_loader.py.
  --skip-baseline-checks    Only run local unit tests.
  -h, --help                Show this help.

Environment:
  PYTHON_BIN                Default: ${PYTHON_BIN}
  REPO_ROOT                 Default: ${REPO_ROOT}
  RUNTIME_ROOT              Default: ${RUNTIME_ROOT}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline-dir)
      BASELINE_DIR="$2"
      shift 2
      ;;
    --include-weight-loader)
      INCLUDE_WEIGHT_LOADER=1
      shift
      ;;
    --skip-baseline-checks)
      SKIP_BASELINE_CHECKS=1
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

run() {
  echo "[regression] $*"
  "$@"
}

run_python() {
  run env PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" "$@"
}

check_logs() {
  local case_id="$1"
  shift
  run_python "${RUNTIME_ROOT}/scripts/check_baseline_logs.py" --case-id "${case_id}" "$@"
}

cd "${REPO_ROOT}"

echo "[regression] repo=${REPO_ROOT}"
echo "[regression] python=${PYTHON_BIN}"

run_python test/test_check_baseline_logs.py
run_python test/test_collect_runtime_perf.py
run_python test/test_distributed_single_rank_bypass.py
run_python test/test_runtime_builder_handoffs.py
run_python test/test_generate_buffers.py
run_python test/test_kv_cache.py
run_python test/test_hybrid_runtime_input_schema.py
run_python test/test_tensor_parallel_direct.py
run_python test/test_pipeline_direct_loader.py
run_python test/test_hybrid_direct_loader.py
run_python test/test_runtime_cli_modes.py
run_python test/test_runtime_summary.py
run_python test/test_compat_package_exports.py

if [[ "${INCLUDE_WEIGHT_LOADER}" == "1" ]]; then
  run_python test/test_model_weight_loader.py
fi

if [[ "${SKIP_BASELINE_CHECKS}" != "1" ]]; then
  if [[ ! -d "${BASELINE_DIR}" ]]; then
    echo "Baseline directory does not exist: ${BASELINE_DIR}" >&2
    exit 2
  fi

  check_logs pp-text-generate \
    "${BASELINE_DIR}/pp-text-generate-rank0.log" \
    "${BASELINE_DIR}/pp-text-generate-rank1.log"
  check_logs pp-mm-generate \
    "${BASELINE_DIR}/pp-mm-generate-rank0.log" \
    "${BASELINE_DIR}/pp-mm-generate-rank1.log"
  check_logs tp-text-generate \
    "${BASELINE_DIR}/tp-text-generate-rank0.log" \
    "${BASELINE_DIR}/tp-text-generate-rank1.log"
  check_logs tp-mm-generate \
    "${BASELINE_DIR}/tp-mm-generate-rank0.log" \
    "${BASELINE_DIR}/tp-mm-generate-rank1.log"
  check_logs hybrid-text-generate \
    "${BASELINE_DIR}/hybrid-text-generate-rank0.log" \
    "${BASELINE_DIR}/hybrid-text-generate-rank1.log" \
    "${BASELINE_DIR}/hybrid-text-generate-rank2.log"
  check_logs hybrid-mm-generate \
    "${BASELINE_DIR}/hybrid-mm-generate-rank0.log" \
    "${BASELINE_DIR}/hybrid-mm-generate-rank1.log" \
    "${BASELINE_DIR}/hybrid-mm-generate-rank2.log"
fi

echo "[regression] PASS runtime core minimal matrix"
