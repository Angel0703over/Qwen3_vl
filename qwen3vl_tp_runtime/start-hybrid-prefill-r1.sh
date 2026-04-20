#!/usr/bin/env bash
# Launch text-only prefill hybrid smoke-test rank 1.

set -euo pipefail

export RANK=1
exec /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/run-hybrid-prefill-rank.sh "$@"
