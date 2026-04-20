#!/usr/bin/env bash
# Launch hybrid smoke-test rank 3.

set -euo pipefail

export RANK=3
exec /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/run-hybrid-rank.sh "$@"
