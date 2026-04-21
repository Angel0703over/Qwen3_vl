#!/usr/bin/env bash
# Prepare a minimal 2-stage x TP=2 multimodal decode hybrid manifest and stage bundles.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PIPELINE_TYPE="${PIPELINE_TYPE:-multimodal_decode}"
export STAGE_RANGES="${STAGE_RANGES:-0:17 18:35}"
export TP_DEGREES="${TP_DEGREES:-2 2}"
export NUM_FRAMES="${NUM_FRAMES:-8}"
export BUNDLE_DIR="${BUNDLE_DIR:-/mnt/ssd/code/Qwen3_vl/qwen3vl_multimodal_decode_hybrid}"
export MANIFEST_PATH="${MANIFEST_PATH:-/mnt/ssd/code/Qwen3_vl/qwen3vl_multimodal_decode_hybrid_manifest.pt}"

exec bash "${SCRIPT_DIR}/prepare-hybrid.sh" "$@"
