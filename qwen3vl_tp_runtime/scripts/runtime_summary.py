"""Summary and JSON formatting helpers for the unified runtime."""

from __future__ import annotations

import sys
from functools import lru_cache

import torch

from qwen3vl_tp_runtime.models.qwen3vl import load_processor
from qwen3vl_tp_runtime.scripts.common import summarize_last_token_topk

GENERATE_PIPELINE_TYPES = {"text_generate", "multimodal_generate"}


def _runtime_dep(name: str, fallback=None):
    runtime_mod = sys.modules.get("qwen3vl_tp_runtime.scripts.runtime")
    if runtime_mod is not None and hasattr(runtime_mod, name):
        return getattr(runtime_mod, name)
    if fallback is not None:
        return fallback
    raise AttributeError(name)


def _tensor_shape_map_to_json(payload: dict[str, tuple[int, ...] | None]) -> dict[str, list[int] | None]:
    return {
        key: (None if value is None else list(value))
        for key, value in payload.items()
    }


@lru_cache(maxsize=4)
def _load_processor_cached(model_path: str):
    return _runtime_dep("load_processor", load_processor)(model_path)


def _decode_generated_token_ids(
    token_ids: list[int],
    *,
    model_path: str,
    keep_special_tokens: bool,
    clean_up_tokenization_spaces: bool,
) -> tuple[str | None, str | None]:
    if not token_ids:
        return "", None

    try:
        processor = _load_processor_cached(model_path)
        decoded_texts = processor.post_process_image_text_to_text(
            [torch.tensor(token_ids, dtype=torch.long)],
            skip_special_tokens=not keep_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        return (decoded_texts[0] if decoded_texts else ""), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _attach_generated_texts(summary: dict, args: argparse.Namespace) -> dict:
    if args.mode != "generate":
        return summary
    if "generated_token_ids" not in summary:
        return summary

    generated_text, generated_error = _decode_generated_token_ids(
        summary["generated_token_ids"],
        model_path=args.model_path,
        keep_special_tokens=args.keep_special_tokens,
        clean_up_tokenization_spaces=args.clean_up_tokenization_spaces,
    )
    reference_generated_token_ids = summary.get("reference_generated_token_ids")
    reference_text = None
    reference_error = None
    if reference_generated_token_ids is not None:
        reference_text, reference_error = _decode_generated_token_ids(
            reference_generated_token_ids,
            model_path=args.model_path,
            keep_special_tokens=args.keep_special_tokens,
            clean_up_tokenization_spaces=args.clean_up_tokenization_spaces,
        )

    if generated_text is not None:
        summary["generated_text"] = generated_text
    if reference_text is not None:
        summary["reference_generated_text"] = reference_text
    if generated_error is not None or reference_error is not None:
        summary["generated_text_decode_error"] = {
            "runtime": generated_error,
            "reference": reference_error,
        }
    return summary


def _summarize_pipeline_run(stats: dict, manifest, topk: int) -> dict:
    summary = {
        "rank": stats["rank"],
        "backend": "pp",
        "pipeline_type": manifest.pipeline_type,
        "stage_idx": stats["stage_idx"],
        "num_stages": stats["num_stages"],
        "start_idx": stats["start_idx"],
        "end_idx": stats["end_idx"],
        "num_layers": stats["num_layers"],
        "device": stats["device"],
        "comm_dtype": stats["comm_dtype"],
        "input_shape": list(stats["input_shape"]),
        "output_shape": list(stats["output_shape"]),
        "received_payload_keys": stats["received_payload_keys"],
        "sent_payload_keys": stats["sent_payload_keys"],
        "sent_tensor_shapes": _tensor_shape_map_to_json(stats["sent_tensor_shapes"]),
        "boundary_max_diff": stats["boundary_max_diff"],
        "boundary_mean_diff": stats["boundary_mean_diff"],
        "stage_max_diff": stats["stage_max_diff"],
        "stage_mean_diff": stats["stage_mean_diff"],
    }

    if "stage_output" in stats and "reference_output" in stats:
        stage_output = stats["stage_output"]
        reference_output = stats["reference_output"]
        topk_fn = _runtime_dep("summarize_last_token_topk", summarize_last_token_topk)
        summary["last_stage_topk"] = topk_fn(stage_output, topk)
        summary["reference_topk"] = topk_fn(reference_output, topk)
    return summary


def _summarize_generate_phase_stats(phase_stats: dict) -> dict:
    return {
        "input_shape": list(phase_stats["input_shape"]),
        "output_shape": list(phase_stats["output_shape"]),
        "received_payload_keys": phase_stats["received_payload_keys"],
        "sent_payload_keys": phase_stats["sent_payload_keys"],
        "sent_tensor_shapes": _tensor_shape_map_to_json(phase_stats["sent_tensor_shapes"]),
        "boundary_max_diff": phase_stats["boundary_max_diff"],
        "boundary_mean_diff": phase_stats["boundary_mean_diff"],
        "embedding_max_diff": phase_stats["embedding_max_diff"],
        "embedding_mean_diff": phase_stats["embedding_mean_diff"],
        "hidden_stage_max_diff": phase_stats["hidden_stage_max_diff"],
        "hidden_stage_mean_diff": phase_stats["hidden_stage_mean_diff"],
        "norm_max_diff": phase_stats["norm_max_diff"],
        "norm_mean_diff": phase_stats["norm_mean_diff"],
        "stage_max_diff": phase_stats["stage_max_diff"],
        "stage_mean_diff": phase_stats["stage_mean_diff"],
        "predicted_token_id": phase_stats["predicted_token_id"],
        "reference_token_id": phase_stats["reference_token_id"],
    }


def _summarize_pipeline_generate_run(stats: dict, manifest, topk: int) -> dict:
    topk_fn = _runtime_dep("summarize_last_token_topk", summarize_last_token_topk)
    summary = {
        "rank": stats["rank"],
        "backend": "pp",
        "pipeline_type": manifest.pipeline_type,
        "stage_idx": stats["stage_idx"],
        "num_stages": stats["num_stages"],
        "start_idx": stats["start_idx"],
        "end_idx": stats["end_idx"],
        "num_layers": stats["num_layers"],
        "device": stats["device"],
        "comm_dtype": stats["comm_dtype"],
        "prefill_seq_len": stats["prefill_seq_len"],
        "max_new_tokens": stats["max_new_tokens"],
        "prefill": _summarize_generate_phase_stats(stats["prefill"]),
        "steps": [_summarize_generate_phase_stats(step) for step in stats["steps"]],
        "generated_token_ids": stats["generated_token_ids"],
    }
    if stats.get("reference_generated_token_ids") is not None:
        summary["reference_generated_token_ids"] = stats["reference_generated_token_ids"]
        summary["token_match"] = stats["generated_token_ids"] == stats["reference_generated_token_ids"]

    if "prefill_output_tensor" in stats:
        summary["prefill_topk"] = topk_fn(stats["prefill_output_tensor"], topk)
        summary["step_topks"] = [
            {
                "step_idx": step_idx,
                "topk": topk_fn(runtime_logits, topk),
            }
            for step_idx, runtime_logits in enumerate(stats["step_output_tensors"])
        ]
    return summary


def _summarize_hybrid_run(
    stats: dict,
    manifest,
    *,
    backend: str,
    topk: int,
    compare_direct: bool,
    trace_layers: bool,
    dump_layer: int | None,
) -> dict:
    topk_fn = _runtime_dep("summarize_last_token_topk", summarize_last_token_topk)
    generate_pipeline_types = _runtime_dep("GENERATE_PIPELINE_TYPES", GENERATE_PIPELINE_TYPES)
    debug_mode = compare_direct or trace_layers or dump_layer is not None
    if manifest.pipeline_type in generate_pipeline_types:
        summary = {
            "rank": stats["rank"],
            "backend": backend,
            "pipeline_type": manifest.pipeline_type,
            "stage_idx": stats["stage_idx"],
            "stage_ranks": stats["stage_ranks"],
            "local_rank": stats["local_rank"],
            "tp_degree": stats["tp_degree"],
            "leader_rank": stats["leader_rank"],
            "current_pp_group": stats["current_pp_group"],
            "debug_mode": debug_mode,
            "compare_direct": compare_direct,
            "trace_layers": trace_layers,
            "dump_layer": dump_layer,
            "dump_topk": topk,
            "prefill_seq_len": stats["prefill_seq_len"],
            "max_new_tokens": stats["max_new_tokens"],
            "prefill": _summarize_generate_phase_stats(stats["prefill"]),
            "steps": [_summarize_generate_phase_stats(step) for step in stats["steps"]],
            "generated_token_ids": stats["generated_token_ids"],
        }
        if stats.get("reference_generated_token_ids") is not None:
            summary["reference_generated_token_ids"] = stats["reference_generated_token_ids"]
            summary["token_match"] = stats["generated_token_ids"] == stats["reference_generated_token_ids"]
        if "prefill_output_tensor" in stats and stats["stage_idx"] == stats["num_stages"] - 1 and stats["local_rank"] == 0:
            summary["prefill_topk"] = topk_fn(stats["prefill_output_tensor"], topk)
            summary["step_topks"] = [
                {
                    "step_idx": step_idx,
                    "topk": topk_fn(runtime_logits, topk),
                }
                for step_idx, runtime_logits in enumerate(stats["step_output_tensors"])
            ]
        return summary

    summary = {
        "rank": stats["rank"],
        "backend": backend,
        "pipeline_type": manifest.pipeline_type,
        "stage_idx": stats["stage_idx"],
        "stage_ranks": stats["stage_ranks"],
        "local_rank": stats["local_rank"],
        "tp_degree": stats["tp_degree"],
        "leader_rank": stats["leader_rank"],
        "current_pp_group": stats["current_pp_group"],
        "input_shape": list(stats["input_shape"]),
        "output_shape": list(stats["output_shape"]),
        "debug_mode": debug_mode,
        "compare_direct": compare_direct,
        "trace_layers": trace_layers,
        "dump_layer": dump_layer,
        "dump_topk": topk,
        "received_payload_keys": stats["received_payload_keys"],
        "sent_payload_keys": stats["sent_payload_keys"],
        "sent_tensor_shapes": _tensor_shape_map_to_json(stats["sent_tensor_shapes"]),
        "boundary_max_diff": stats["boundary_max_diff"],
        "boundary_mean_diff": stats["boundary_mean_diff"],
        "direct_max_diff": stats["direct_max_diff"],
        "direct_mean_diff": stats["direct_mean_diff"],
        "stage_max_diff": stats["stage_max_diff"],
        "stage_mean_diff": stats["stage_mean_diff"],
        "tp_direct_max_diff": stats["tp_direct_max_diff"],
        "tp_direct_mean_diff": stats["tp_direct_mean_diff"],
        "trace_summary": stats["trace_summary"],
        "num_traces": len(stats["traces"] or []),
        "outlier_dump": stats["outlier_dump"],
    }
    if (
        "stage_output" in stats
        and "reference_output" in stats
        and stats["stage_idx"] == stats["num_stages"] - 1
        and stats["local_rank"] == 0
    ):
        summary["last_stage_topk"] = topk_fn(stats["stage_output"], topk)
        summary["reference_topk"] = topk_fn(stats["reference_output"], topk)
    return summary


__all__ = [
    "_attach_generated_texts",
    "_summarize_hybrid_run",
    "_summarize_pipeline_generate_run",
    "_summarize_pipeline_run",
]
