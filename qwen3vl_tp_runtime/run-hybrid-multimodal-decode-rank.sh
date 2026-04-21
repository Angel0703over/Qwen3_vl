#!/usr/bin/env bash
# Launch one rank for the minimal 2-stage x TP=2 multimodal decode hybrid smoke test.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MANIFEST_PATH="${MANIFEST_PATH:-/mnt/ssd/code/Qwen3_vl/qwen3vl_multimodal_decode_hybrid_manifest.pt}"
export COMM_DTYPE="${COMM_DTYPE:-float32}"
export TP_ATTN_MATH="${TP_ATTN_MATH:-float32}"
export TP_MLP_MATH="${TP_MLP_MATH:-float32}"
export HYBRID_DEBUG="${HYBRID_DEBUG:-0}"
case "${HYBRID_DEBUG,,}" in
  1|true|yes|on)
    export DUMP_LAYER="${DUMP_LAYER:--1}"
    ;;
esac
exec bash "${SCRIPT_DIR}/run-hybrid-rank.sh" "$@"
