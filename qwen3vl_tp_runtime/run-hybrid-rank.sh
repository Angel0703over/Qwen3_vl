#!/usr/bin/env bash
# Launch one rank for the minimal 2-stage x TP=2 hybrid PP+TP smoke test.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/ssd/code/Qwen3_vl}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/qwen3vl_tp_runtime}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ssd/miniconda3/envs/vlm/bin/python}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Distributed settings.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29621}"
export WORLD_SIZE="${WORLD_SIZE:-4}"
export RANK="${RANK:-0}"

# Set this when you need a specific interface, for example:
# export GLOO_SOCKET_IFNAME=tun0
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}"

# Local device settings.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DEVICE="${DEVICE:-cuda}"

# Runtime inputs.
export MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/qwen3vl_text_hybrid_manifest.pt}"
export COMPUTE_DTYPE="${COMPUTE_DTYPE:-auto}"
export COMM_DTYPE="${COMM_DTYPE:-float32}"
# For captured replay parity, TP attention / MLP default to original dtype math.
export TP_ATTN_MATH="${TP_ATTN_MATH:-orig}"
export TP_MLP_MATH="${TP_MLP_MATH:-orig}"
export COMPARE_DIRECT="${COMPARE_DIRECT:-1}"
export TRACE_LAYERS="${TRACE_LAYERS:-1}"
export DUMP_LAYER="${DUMP_LAYER:-0}"
export DUMP_TOPK="${DUMP_TOPK:-10}"

echo "[launch] repo_root=${REPO_ROOT}"
echo "[launch] runtime_root=${RUNTIME_ROOT}"
echo "[launch] python_bin=${PYTHON_BIN}"
echo "[launch] master=${MASTER_ADDR}:${MASTER_PORT} world_size=${WORLD_SIZE} rank=${RANK}"
echo "[launch] manifest_path=${MANIFEST_PATH}"
echo "[launch] device=${DEVICE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[launch] compute_dtype=${COMPUTE_DTYPE} comm_dtype=${COMM_DTYPE}"
echo "[launch] tp_attn_math=${TP_ATTN_MATH} tp_mlp_math=${TP_MLP_MATH}"

"${PYTHON_BIN}" - <<'PY'
import json
import os

import torch.distributed as dist

from qwen3vl_tp_runtime.hexgen_core.distributed import get_device, init_dist
from qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel import (
    TextHybridRunner,
    load_hybrid_manifest,
)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_optional_int(name: str) -> int | None:
    value = os.environ.get(name, "")
    if value in {"", "none", "None", "-1"}:
        return None
    return int(value)


manifest = load_hybrid_manifest(os.environ["MANIFEST_PATH"])
rank, world_size = init_dist()
device = get_device(os.environ["DEVICE"])

runner = TextHybridRunner(
    manifest=manifest,
    device=device,
    compute_dtype_arg=os.environ["COMPUTE_DTYPE"],
    comm_dtype_arg=os.environ["COMM_DTYPE"],
    tp_attn_math_mode=os.environ["TP_ATTN_MATH"],
    tp_mlp_math_mode=os.environ["TP_MLP_MATH"],
    compare_direct=env_flag("COMPARE_DIRECT", default=True),
    trace_layers=env_flag("TRACE_LAYERS", default=False),
    dump_layer=env_optional_int("DUMP_LAYER"),
    dump_topk=int(os.environ["DUMP_TOPK"]),
)

stats = runner.run_rank(rank, world_size)
traces = stats["traces"] or []

summary = {
    "rank": stats["rank"],
    "stage_idx": stats["stage_idx"],
    "stage_ranks": stats["stage_ranks"],
    "local_rank": stats["local_rank"],
    "tp_degree": stats["tp_degree"],
    "leader_rank": stats["leader_rank"],
    "current_pp_group": stats["current_pp_group"],
    "input_shape": list(stats["input_shape"]),
    "output_shape": list(stats["output_shape"]),
    "received_payload_keys": stats["received_payload_keys"],
    "sent_payload_keys": stats["sent_payload_keys"],
    "sent_tensor_shapes": {
        key: (None if value is None else list(value))
        for key, value in stats["sent_tensor_shapes"].items()
    },
    "boundary_max_diff": stats["boundary_max_diff"],
    "boundary_mean_diff": stats["boundary_mean_diff"],
    "direct_max_diff": stats["direct_max_diff"],
    "direct_mean_diff": stats["direct_mean_diff"],
    "stage_max_diff": stats["stage_max_diff"],
    "stage_mean_diff": stats["stage_mean_diff"],
    "tp_direct_max_diff": stats["tp_direct_max_diff"],
    "tp_direct_mean_diff": stats["tp_direct_mean_diff"],
    "num_traces": len(traces),
    "outlier_dump": stats["outlier_dump"],
}

print(json.dumps(summary, ensure_ascii=False, indent=2))

dist.barrier()
dist.destroy_process_group()
PY
