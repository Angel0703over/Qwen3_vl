#!/usr/bin/env bash
# Launch one rank for the minimal 2-stage x TP=2 hybrid PP+TP smoke test.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ssd/miniconda3/envs/vlm/bin/python}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29621}"
export WORLD_SIZE="${WORLD_SIZE:-4}"
export RANK="${RANK:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DEVICE="${DEVICE:-cuda}"

export MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/qwen3vl_text_hybrid_manifest.pt}"
export COMPUTE_DTYPE="${COMPUTE_DTYPE:-auto}"
export COMM_DTYPE="${COMM_DTYPE:-float32}"
export HYBRID_DEBUG="${HYBRID_DEBUG:-0}"
export TP_ATTN_MATH="${TP_ATTN_MATH:-orig}"
export TP_MLP_MATH="${TP_MLP_MATH:-orig}"
case "${HYBRID_DEBUG,,}" in
  1|true|yes|on)
    export COMPARE_DIRECT="${COMPARE_DIRECT:-1}"
    export TRACE_LAYERS="${TRACE_LAYERS:-1}"
    export DUMP_LAYER="${DUMP_LAYER:-0}"
    export DUMP_TOPK="${DUMP_TOPK:-10}"
    ;;
  *)
    export COMPARE_DIRECT="${COMPARE_DIRECT:-0}"
    export TRACE_LAYERS="${TRACE_LAYERS:-0}"
    export DUMP_LAYER="${DUMP_LAYER:-}"
    export DUMP_TOPK="${DUMP_TOPK:-10}"
    ;;
esac

echo "[launch] repo_root=${REPO_ROOT}"
echo "[launch] runtime_root=${RUNTIME_ROOT}"
echo "[launch] python_bin=${PYTHON_BIN}"
echo "[launch] master=${MASTER_ADDR}:${MASTER_PORT} world_size=${WORLD_SIZE} rank=${RANK}"
echo "[launch] manifest_path=${MANIFEST_PATH}"
echo "[launch] device=${DEVICE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[launch] compute_dtype=${COMPUTE_DTYPE} comm_dtype=${COMM_DTYPE}"
echo "[launch] tp_attn_math=${TP_ATTN_MATH} tp_mlp_math=${TP_MLP_MATH}"
echo "[launch] hybrid_debug=${HYBRID_DEBUG} compare_direct=${COMPARE_DIRECT} trace_layers=${TRACE_LAYERS} dump_layer=${DUMP_LAYER} dump_topk=${DUMP_TOPK}"

"${PYTHON_BIN}" - <<'PY'
import json
import os

import torch
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
    if value in {"", "none", "None"}:
        return None
    return int(value)


def summarize_last_token_topk(logits, topk: int) -> list[dict]:
    last_token_logits = logits[0, -1].to(torch.float32)
    k = min(topk, last_token_logits.numel())
    values, indices = torch.topk(last_token_logits, k=k)
    return [
        {
            "token_id": int(token_id),
            "logit": float(value),
        }
        for value, token_id in zip(values.tolist(), indices.tolist())
    ]


manifest = load_hybrid_manifest(os.environ["MANIFEST_PATH"])
rank, world_size = init_dist()
device = get_device(os.environ["DEVICE"])
compare_direct = env_flag("COMPARE_DIRECT", default=False)
trace_layers = env_flag("TRACE_LAYERS", default=False)
dump_layer = env_optional_int("DUMP_LAYER")
dump_topk = int(os.environ["DUMP_TOPK"])

runner = TextHybridRunner(
    manifest=manifest,
    device=device,
    compute_dtype_arg=os.environ["COMPUTE_DTYPE"],
    comm_dtype_arg=os.environ["COMM_DTYPE"],
    tp_attn_math_mode=os.environ["TP_ATTN_MATH"],
    tp_mlp_math_mode=os.environ["TP_MLP_MATH"],
    compare_direct=compare_direct,
    trace_layers=trace_layers,
    dump_layer=dump_layer,
    dump_topk=dump_topk,
    return_tensors=(
        manifest.pipeline_type
        in {"text_prefill", "multimodal_prefill", "text_decode", "multimodal_decode", "text_generate"}
    ),
)

stats = runner.run_rank(rank, world_size)
if manifest.pipeline_type == "text_generate":
    prefill_stats = stats["prefill"]
    steps = stats["steps"]
    summary = {
        "rank": stats["rank"],
        "pipeline_type": manifest.pipeline_type,
        "stage_idx": stats["stage_idx"],
        "stage_ranks": stats["stage_ranks"],
        "local_rank": stats["local_rank"],
        "tp_degree": stats["tp_degree"],
        "leader_rank": stats["leader_rank"],
        "current_pp_group": stats["current_pp_group"],
        "debug_mode": compare_direct or trace_layers or dump_layer is not None,
        "compare_direct": compare_direct,
        "trace_layers": trace_layers,
        "dump_layer": dump_layer,
        "dump_topk": dump_topk,
        "prefill_seq_len": stats["prefill_seq_len"],
        "max_new_tokens": stats["max_new_tokens"],
        "prefill": {
            "input_shape": list(prefill_stats["input_shape"]),
            "output_shape": list(prefill_stats["output_shape"]),
            "received_payload_keys": prefill_stats["received_payload_keys"],
            "sent_payload_keys": prefill_stats["sent_payload_keys"],
            "sent_tensor_shapes": {
                key: (None if value is None else list(value))
                for key, value in prefill_stats["sent_tensor_shapes"].items()
            },
            "boundary_max_diff": prefill_stats["boundary_max_diff"],
            "boundary_mean_diff": prefill_stats["boundary_mean_diff"],
            "embedding_max_diff": prefill_stats["embedding_max_diff"],
            "embedding_mean_diff": prefill_stats["embedding_mean_diff"],
            "hidden_stage_max_diff": prefill_stats["hidden_stage_max_diff"],
            "hidden_stage_mean_diff": prefill_stats["hidden_stage_mean_diff"],
            "norm_max_diff": prefill_stats["norm_max_diff"],
            "norm_mean_diff": prefill_stats["norm_mean_diff"],
            "stage_max_diff": prefill_stats["stage_max_diff"],
            "stage_mean_diff": prefill_stats["stage_mean_diff"],
            "predicted_token_id": prefill_stats["predicted_token_id"],
            "reference_token_id": prefill_stats["reference_token_id"],
        },
        "steps": [
            {
                "input_shape": list(step["input_shape"]),
                "output_shape": list(step["output_shape"]),
                "received_payload_keys": step["received_payload_keys"],
                "sent_payload_keys": step["sent_payload_keys"],
                "sent_tensor_shapes": {
                    key: (None if value is None else list(value))
                    for key, value in step["sent_tensor_shapes"].items()
                },
                "boundary_max_diff": step["boundary_max_diff"],
                "boundary_mean_diff": step["boundary_mean_diff"],
                "embedding_max_diff": step["embedding_max_diff"],
                "embedding_mean_diff": step["embedding_mean_diff"],
                "hidden_stage_max_diff": step["hidden_stage_max_diff"],
                "hidden_stage_mean_diff": step["hidden_stage_mean_diff"],
                "norm_max_diff": step["norm_max_diff"],
                "norm_mean_diff": step["norm_mean_diff"],
                "stage_max_diff": step["stage_max_diff"],
                "stage_mean_diff": step["stage_mean_diff"],
                "predicted_token_id": step["predicted_token_id"],
                "reference_token_id": step["reference_token_id"],
            }
            for step in steps
        ],
        "generated_token_ids": stats["generated_token_ids"],
        "reference_generated_token_ids": stats["reference_generated_token_ids"],
        "token_match": stats["generated_token_ids"] == stats["reference_generated_token_ids"],
    }
    prefill_output_tensor = stats.pop("prefill_output_tensor", None)
    step_output_tensors = stats.pop("step_output_tensors", [])
    if (
        stats["stage_idx"] == stats["num_stages"] - 1
        and stats["local_rank"] == 0
        and prefill_output_tensor is not None
    ):
        summary["prefill_topk"] = summarize_last_token_topk(prefill_output_tensor, dump_topk)
        for step_idx, step_output in enumerate(step_output_tensors):
            summary.setdefault("step_topks", []).append(
                {
                    "step_idx": step_idx,
                    "topk": summarize_last_token_topk(step_output, dump_topk),
                }
            )
else:
    traces = stats["traces"] or []
    stage_output = stats.pop("stage_output", None)
    reference_output = stats.pop("reference_output", None)

    summary = {
        "rank": stats["rank"],
        "pipeline_type": manifest.pipeline_type,
        "stage_idx": stats["stage_idx"],
        "stage_ranks": stats["stage_ranks"],
        "local_rank": stats["local_rank"],
        "tp_degree": stats["tp_degree"],
        "leader_rank": stats["leader_rank"],
        "current_pp_group": stats["current_pp_group"],
        "input_shape": list(stats["input_shape"]),
        "output_shape": list(stats["output_shape"]),
        "debug_mode": compare_direct or trace_layers or dump_layer is not None,
        "compare_direct": compare_direct,
        "trace_layers": trace_layers,
        "dump_layer": dump_layer,
        "dump_topk": dump_topk,
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
        "trace_summary": stats["trace_summary"],
        "num_traces": len(traces),
        "outlier_dump": stats["outlier_dump"],
    }
    if (
        manifest.pipeline_type in {"text_prefill", "multimodal_prefill", "text_decode", "multimodal_decode"}
        and stage_output is not None
        and reference_output is not None
        and stats["stage_idx"] == stats["num_stages"] - 1
    ):
        summary["last_stage_topk"] = summarize_last_token_topk(stage_output, dump_topk)
        summary["reference_topk"] = summarize_last_token_topk(reference_output, dump_topk)

print(json.dumps(summary, ensure_ascii=False, indent=2))

dist.barrier()
dist.destroy_process_group()
PY
