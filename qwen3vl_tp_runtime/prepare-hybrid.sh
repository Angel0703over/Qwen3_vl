#!/usr/bin/env bash
# Prepare a minimal 2-stage x TP=2 hybrid manifest and its captured stage bundles.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/ssd/code/Qwen3_vl}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/qwen3vl_tp_runtime}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ssd/miniconda3/envs/vlm/bin/python}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export STAGE_RANGES="${STAGE_RANGES:-0:5 6:11}"
export TP_DEGREES="${TP_DEGREES:-2 2}"
export NUM_FRAMES="${NUM_FRAMES:-8}"
export SAVE_DTYPE="${SAVE_DTYPE:-auto}"
export BUNDLE_DIR="${BUNDLE_DIR:-${REPO_ROOT}/qwen3vl_text_hybrid}"
export MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/qwen3vl_text_hybrid_manifest.pt}"

echo "[prepare] repo_root=${REPO_ROOT}"
echo "[prepare] runtime_root=${RUNTIME_ROOT}"
echo "[prepare] python_bin=${PYTHON_BIN}"
echo "[prepare] stage_ranges=${STAGE_RANGES}"
echo "[prepare] tp_degrees=${TP_DEGREES}"
echo "[prepare] bundle_dir=${BUNDLE_DIR}"
echo "[prepare] manifest_path=${MANIFEST_PATH}"
echo "[prepare] num_frames=${NUM_FRAMES} save_dtype=${SAVE_DTYPE}"

"${PYTHON_BIN}" - <<'PY'
import json
import os

from qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel import prepare_text_hybrid


def parse_stage_ranges(spec: str) -> list[tuple[int, int]]:
    ranges = []
    for item in spec.split():
        start_str, end_str = item.split(":")
        ranges.append((int(start_str), int(end_str)))
    return ranges


def parse_tp_degrees(spec: str) -> list[int]:
    return [int(value) for value in spec.split()]


manifest = prepare_text_hybrid(
    stage_ranges=parse_stage_ranges(os.environ["STAGE_RANGES"]),
    tp_degrees=parse_tp_degrees(os.environ["TP_DEGREES"]),
    bundle_dir=os.environ["BUNDLE_DIR"],
    manifest_path=os.environ["MANIFEST_PATH"],
    num_frames=int(os.environ["NUM_FRAMES"]),
    save_dtype=os.environ["SAVE_DTYPE"],
)

summary = {
    "runtime": manifest.runtime,
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
