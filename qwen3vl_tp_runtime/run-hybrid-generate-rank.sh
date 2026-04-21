#!/usr/bin/env bash
# Launch one rank for the minimal 2-stage x TP=2 text-only generate hybrid smoke test.

set -euo pipefail

export MANIFEST_PATH="${MANIFEST_PATH:-/mnt/ssd/code/Qwen3_vl/qwen3vl_text_generate_hybrid_manifest.pt}"
exec bash /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/run-hybrid-rank.sh "$@"
