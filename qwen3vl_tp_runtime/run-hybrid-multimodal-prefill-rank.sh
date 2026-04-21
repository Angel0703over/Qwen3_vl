#!/usr/bin/env bash
# Launch one rank for the minimal 2-stage x TP=2 multimodal prefill hybrid smoke test.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MANIFEST_PATH="${MANIFEST_PATH:-/mnt/ssd/code/Qwen3_vl/qwen3vl_multimodal_prefill_hybrid_manifest.pt}"
exec bash "${SCRIPT_DIR}/run-hybrid-rank.sh" "$@"
