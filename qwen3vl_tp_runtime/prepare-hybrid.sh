#!/usr/bin/env bash
# Prepare a minimal 2-stage x TP=2 hybrid manifest and its captured stage bundles.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ssd/miniconda3/envs/vlm/bin/python}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export PIPELINE_TYPE="${PIPELINE_TYPE:-text}"
export STAGE_RANGES="${STAGE_RANGES:-0:5 6:11}"
export TP_DEGREES="${TP_DEGREES:-2 2}"
export NUM_FRAMES="${NUM_FRAMES:-8}"
export FRAME_DIR="${FRAME_DIR:-}"
export SAVE_DTYPE="${SAVE_DTYPE:-auto}"
export BUNDLE_DIR="${BUNDLE_DIR:-${REPO_ROOT}/qwen3vl_text_hybrid}"
export MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/qwen3vl_text_hybrid_manifest.pt}"
export PROMPT="${PROMPT:-请用中文简要介绍一下人工智能。}"
export DECODE_TOKEN_ID="${DECODE_TOKEN_ID:-}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"

echo "[prepare] repo_root=${REPO_ROOT}"
echo "[prepare] runtime_root=${RUNTIME_ROOT}"
echo "[prepare] python_bin=${PYTHON_BIN}"
echo "[prepare] pipeline_type=${PIPELINE_TYPE}"
echo "[prepare] stage_ranges=${STAGE_RANGES}"
echo "[prepare] tp_degrees=${TP_DEGREES}"
echo "[prepare] bundle_dir=${BUNDLE_DIR}"
echo "[prepare] manifest_path=${MANIFEST_PATH}"
echo "[prepare] num_frames=${NUM_FRAMES} save_dtype=${SAVE_DTYPE}"
if [[ -n "${FRAME_DIR}" ]]; then
  echo "[prepare] frame_dir=${FRAME_DIR}"
fi
if [[ "${PIPELINE_TYPE}" == "text_prefill" || "${PIPELINE_TYPE}" == "text_decode" || "${PIPELINE_TYPE}" == "text_generate" ]]; then
  echo "[prepare] prompt=${PROMPT}"
fi
if [[ "${PIPELINE_TYPE}" == "text_decode" || "${PIPELINE_TYPE}" == "multimodal_decode" ]] && [[ -n "${DECODE_TOKEN_ID}" ]]; then
  echo "[prepare] decode_token_id=${DECODE_TOKEN_ID}"
fi
if [[ "${PIPELINE_TYPE}" == "text_generate" || "${PIPELINE_TYPE}" == "multimodal_generate" ]]; then
  echo "[prepare] max_new_tokens=${MAX_NEW_TOKENS}"
fi

"${PYTHON_BIN}" - <<'PY'
import json
import os

from qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel import (
    prepare_multimodal_decode_hybrid,
    prepare_multimodal_generate_hybrid,
    prepare_multimodal_prefill_hybrid,
    prepare_text_decode_hybrid,
    prepare_text_generate_hybrid,
    prepare_text_hybrid,
    prepare_text_prefill_hybrid,
)


def parse_stage_ranges(spec: str) -> list[tuple[int, int]]:
    ranges = []
    for item in spec.split():
        start_str, end_str = item.split(":")
        ranges.append((int(start_str), int(end_str)))
    return ranges


def parse_tp_degrees(spec: str) -> list[int]:
    return [int(value) for value in spec.split()]


pipeline_type = os.environ["PIPELINE_TYPE"]
common_kwargs = {
    "stage_ranges": parse_stage_ranges(os.environ["STAGE_RANGES"]),
    "tp_degrees": parse_tp_degrees(os.environ["TP_DEGREES"]),
    "bundle_dir": os.environ["BUNDLE_DIR"],
    "manifest_path": os.environ["MANIFEST_PATH"],
    "save_dtype": os.environ["SAVE_DTYPE"],
}

if pipeline_type == "text":
    manifest = prepare_text_hybrid(
        num_frames=int(os.environ["NUM_FRAMES"]),
        **common_kwargs,
    )
elif pipeline_type == "text_prefill":
    manifest = prepare_text_prefill_hybrid(
        prompt=os.environ["PROMPT"],
        **common_kwargs,
    )
elif pipeline_type == "multimodal_prefill":
    manifest = prepare_multimodal_prefill_hybrid(
        num_frames=int(os.environ["NUM_FRAMES"]),
        **({"frame_dir": os.environ["FRAME_DIR"]} if os.environ.get("FRAME_DIR", "") else {}),
        **common_kwargs,
    )
elif pipeline_type == "multimodal_decode":
    decode_token_id_env = os.environ.get("DECODE_TOKEN_ID", "")
    manifest = prepare_multimodal_decode_hybrid(
        num_frames=int(os.environ["NUM_FRAMES"]),
        decode_token_id=(None if decode_token_id_env == "" else int(decode_token_id_env)),
        **({"frame_dir": os.environ["FRAME_DIR"]} if os.environ.get("FRAME_DIR", "") else {}),
        **common_kwargs,
    )
elif pipeline_type == "multimodal_generate":
    manifest = prepare_multimodal_generate_hybrid(
        num_frames=int(os.environ["NUM_FRAMES"]),
        max_new_tokens=int(os.environ["MAX_NEW_TOKENS"]),
        **({"frame_dir": os.environ["FRAME_DIR"]} if os.environ.get("FRAME_DIR", "") else {}),
        **common_kwargs,
    )
elif pipeline_type == "text_decode":
    decode_token_id_env = os.environ.get("DECODE_TOKEN_ID", "")
    manifest = prepare_text_decode_hybrid(
        prompt=os.environ["PROMPT"],
        decode_token_id=(None if decode_token_id_env == "" else int(decode_token_id_env)),
        **common_kwargs,
    )
elif pipeline_type == "text_generate":
    manifest = prepare_text_generate_hybrid(
        prompt=os.environ["PROMPT"],
        max_new_tokens=int(os.environ["MAX_NEW_TOKENS"]),
        **common_kwargs,
    )
else:
    raise ValueError(f"unsupported PIPELINE_TYPE={pipeline_type!r}")

summary = {
    "runtime": manifest.runtime,
    "pipeline_type": manifest.pipeline_type,
    "stage_ranges": manifest.stage_ranges,
    "tp_degrees": manifest.tp_degrees,
    "stage_rank_groups": manifest.stage_rank_groups,
    "pp_rank_groups": manifest.pp_rank_groups,
    "world_size": manifest.world_size,
    "bundle_dir": manifest.bundle_dir,
    "manifest_path": os.environ["MANIFEST_PATH"],
}

print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
