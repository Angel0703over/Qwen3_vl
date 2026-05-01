"""Summary and JSON formatting helpers for the unified runtime."""

from __future__ import annotations

import resource
import sys
import time
from collections import defaultdict
from functools import lru_cache
from typing import Any

import torch

from qwen3vl_tp_runtime.debug.tp_debug import TpDebugConfig
from qwen3vl_tp_runtime.hexgen_core.distributed import (
    get_startup_timing_events,
    get_transport_profile_events,
    reset_startup_timing_events,
    reset_transport_profile_events,
)
from qwen3vl_tp_runtime.models.qwen3vl.processing import load_processor
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


def reset_runtime_metrics() -> None:
    reset_startup_timing_events()
    reset_transport_profile_events()
    if not torch.cuda.is_available():
        return
    try:
        for device_idx in range(torch.cuda.device_count()):
            with torch.cuda.device(device_idx):
                torch.cuda.reset_peak_memory_stats(device_idx)
    except Exception:
        return


def _round_seconds(value: float) -> float:
    return round(float(value), 6)


def _classify_startup_event(event: dict[str, Any]) -> str:
    component = str(event.get("component", ""))
    message = str(event.get("message", "")).lower()
    if "post-load barrier" in message:
        return "post_load_barrier_seconds"
    if "startup_contract" in message:
        if component in {"object-send", "object-recv", "tensor-send", "tensor-recv"}:
            return "startup_contract_transport_seconds"
        return "startup_contract_prepare_seconds"
    if "prepare " in message and " session" in message:
        return "prepare_session_seconds"
    if message.startswith("materialize ") or "materialize local direct shard" in message:
        return "materialize_stage_seconds"
    if "stage_scaffold" in message or "text_scaffold" in message or "stage_state_" in message:
        return "scaffold_transport_seconds"
    return "other_seconds"


def _summarize_startup_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    totals_by_component: defaultdict[str, float] = defaultdict(float)
    totals_by_kind: defaultdict[str, float] = defaultdict(float)
    normalized_events: list[dict[str, Any]] = []
    for event in events:
        elapsed = float(event.get("elapsed_seconds") or 0.0)
        component = str(event.get("component", "unknown"))
        kind = _classify_startup_event(event)
        totals_by_component[component] += elapsed
        totals_by_kind[kind] += elapsed
        normalized = dict(event)
        normalized["kind"] = kind
        normalized["elapsed_seconds"] = _round_seconds(elapsed)
        normalized_events.append(normalized)

    startup_contract_seconds = (
        totals_by_kind["startup_contract_prepare_seconds"]
        + totals_by_kind["startup_contract_transport_seconds"]
    )
    for key in (
        "prepare_session_seconds",
        "startup_contract_prepare_seconds",
        "startup_contract_transport_seconds",
        "materialize_stage_seconds",
        "post_load_barrier_seconds",
        "scaffold_transport_seconds",
        "other_seconds",
    ):
        totals_by_kind.setdefault(key, 0.0)
    totals_by_kind["startup_contract_seconds"] = startup_contract_seconds

    return {
        "event_count": len(normalized_events),
        "events": normalized_events,
        "totals_by_component": {
            key: _round_seconds(value)
            for key, value in sorted(totals_by_component.items())
        },
        "totals_by_kind": {
            key: _round_seconds(value)
            for key, value in sorted(totals_by_kind.items())
        },
    }


def _summarize_transport_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    totals_by_kind: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {
            "event_count": 0,
            "elapsed_seconds": 0.0,
            "object_bytes": 0.0,
            "tensor_bytes": 0.0,
        }
    )
    totals_by_channel: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {
            "event_count": 0,
            "elapsed_seconds": 0.0,
            "object_bytes": 0.0,
            "tensor_bytes": 0.0,
        }
    )
    normalized_events: list[dict[str, Any]] = []
    pin_memory_requested_event_count = 0
    pin_memory_used_event_count = 0
    pin_memory_tensor_count = 0
    for event in events:
        normalized = dict(event)
        kind = str(normalized.get("kind") or "other")
        channel = str(normalized.get("channel") or "unknown")
        elapsed = float(normalized.get("elapsed_seconds") or 0.0)
        object_bytes = float(normalized.get("object_bytes") or 0.0)
        tensor_bytes = float(normalized.get("total_tensor_bytes") or 0.0)
        if bool(normalized.get("transport_pin_memory_requested")):
            pin_memory_requested_event_count += 1
        if bool(normalized.get("transport_pin_memory_used")):
            pin_memory_used_event_count += 1
            pin_memory_tensor_count += int(normalized.get("transport_pin_memory_tensor_count") or 0)
        normalized["elapsed_seconds"] = _round_seconds(elapsed)
        normalized_events.append(normalized)

        for bucket in (totals_by_kind[kind], totals_by_channel[channel]):
            bucket["event_count"] += 1
            bucket["elapsed_seconds"] += elapsed
            bucket["object_bytes"] += object_bytes
            bucket["tensor_bytes"] += tensor_bytes

    def _finalize(payload: dict[str, float]) -> dict[str, int | float]:
        return {
            "event_count": int(payload["event_count"]),
            "elapsed_seconds": _round_seconds(payload["elapsed_seconds"]),
            "object_bytes": int(payload["object_bytes"]),
            "tensor_bytes": int(payload["tensor_bytes"]),
            "total_bytes": int(payload["object_bytes"] + payload["tensor_bytes"]),
        }

    for key in ("startup_contract", "scaffold", "stage_handoff", "tp_collective", "other"):
        totals_by_kind.setdefault(key, totals_by_kind.default_factory())

    return {
        "event_count": len(normalized_events),
        "events": normalized_events,
        "totals_by_kind": {
            key: _finalize(value)
            for key, value in sorted(totals_by_kind.items())
        },
        "totals_by_channel": {
            key: _finalize(value)
            for key, value in sorted(totals_by_channel.items())
        },
        "pin_memory": {
            "requested_event_count": pin_memory_requested_event_count,
            "used_event_count": pin_memory_used_event_count,
            "used_tensor_count": pin_memory_tensor_count,
        },
    }


def _maxrss_bytes() -> int | None:
    try:
        maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if sys.platform == "darwin":
        return maxrss
    return maxrss * 1024


def _collect_cuda_memory() -> dict[str, Any]:
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:
        return {
            "cuda_available": False,
            "cuda_error": f"{type(exc).__name__}: {exc}",
            "devices": [],
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    if not cuda_available:
        return {
            "cuda_available": False,
            "devices": [],
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }

    devices: list[dict[str, Any]] = []
    peak_allocated = 0
    peak_reserved = 0
    try:
        device_count = torch.cuda.device_count()
        for device_idx in range(device_count):
            allocated = int(torch.cuda.memory_allocated(device_idx))
            reserved = int(torch.cuda.memory_reserved(device_idx))
            max_allocated = int(torch.cuda.max_memory_allocated(device_idx))
            max_reserved = int(torch.cuda.max_memory_reserved(device_idx))
            peak_allocated = max(peak_allocated, max_allocated)
            peak_reserved = max(peak_reserved, max_reserved)
            devices.append(
                {
                    "device": f"cuda:{device_idx}",
                    "memory_allocated_bytes": allocated,
                    "memory_reserved_bytes": reserved,
                    "max_memory_allocated_bytes": max_allocated,
                    "max_memory_reserved_bytes": max_reserved,
                }
            )
    except Exception as exc:
        return {
            "cuda_available": True,
            "cuda_error": f"{type(exc).__name__}: {exc}",
            "devices": devices,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }

    return {
        "cuda_available": True,
        "devices": devices,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def _attach_runtime_metrics(
    summary: dict[str, Any],
    *,
    started_at: float,
    extra_timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    timing = {
        "runtime_total_seconds": _round_seconds(time.perf_counter() - started_at),
    }
    for key, value in (extra_timings or {}).items():
        timing[key] = _round_seconds(value)

    summary["runtime_metrics"] = {
        "timing": timing,
        "startup": _summarize_startup_events(get_startup_timing_events()),
        "transport": _summarize_transport_events(get_transport_profile_events()),
        "memory": {
            "cpu_max_rss_bytes": _maxrss_bytes(),
            **_collect_cuda_memory(),
        },
    }
    return summary


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
        "weight_load": stats.get("weight_load"),
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
    summary = {
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
    for key in (
        "runtime_input_source",
        "runtime_input_broadcast_skipped",
        "stage_kv_cache",
        "video_window_cache",
        "video_kv_compression_plan",
    ):
        if key in phase_stats:
            summary[key] = phase_stats[key]
    return summary


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
        "weight_load": stats.get("weight_load"),
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
    debug_config: TpDebugConfig | None = None,
) -> dict:
    topk_fn = _runtime_dep("summarize_last_token_topk", summarize_last_token_topk)
    generate_pipeline_types = _runtime_dep("GENERATE_PIPELINE_TYPES", GENERATE_PIPELINE_TYPES)
    debug_config = debug_config or TpDebugConfig()
    debug_fields = debug_config.to_summary_fields()
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
            "weight_load": stats.get("weight_load"),
            **debug_fields,
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
        "weight_load": stats.get("weight_load"),
        "input_shape": list(stats["input_shape"]),
        "output_shape": list(stats["output_shape"]),
        **debug_fields,
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
    "_attach_runtime_metrics",
    "_summarize_hybrid_run",
    "_summarize_pipeline_generate_run",
    "_summarize_pipeline_run",
    "reset_runtime_metrics",
]
