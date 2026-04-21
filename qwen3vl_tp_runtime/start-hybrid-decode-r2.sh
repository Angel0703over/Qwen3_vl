#!/usr/bin/env bash
# Launch text-only decode hybrid smoke-test rank 2.

set -euo pipefail

export RANK=2
exec /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/run-hybrid-decode-rank.sh "$@"
