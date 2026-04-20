#!/usr/bin/env bash
# Launch hybrid smoke-test rank 2.

set -euo pipefail

export RANK=2
exec /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/run-hybrid-rank.sh "$@"
