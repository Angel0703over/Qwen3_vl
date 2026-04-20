#!/usr/bin/env bash
# Prepare a minimal 2-stage x TP=2 text-only prefill hybrid manifest and stage bundles.

set -euo pipefail

export PIPELINE_TYPE="${PIPELINE_TYPE:-text_prefill}"
export STAGE_RANGES="${STAGE_RANGES:-0:17 18:35}"
export TP_DEGREES="${TP_DEGREES:-2 2}"
export PROMPT="${PROMPT:-请用中文简要介绍一下人工智能。}"
export BUNDLE_DIR="${BUNDLE_DIR:-/mnt/ssd/code/Qwen3_vl/qwen3vl_text_prefill_hybrid}"
export MANIFEST_PATH="${MANIFEST_PATH:-/mnt/ssd/code/Qwen3_vl/qwen3vl_text_prefill_hybrid_manifest.pt}"

exec /mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/prepare-hybrid.sh "$@"
