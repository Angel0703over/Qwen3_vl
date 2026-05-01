"""Planner-only video KV compression metadata.

This module does not mutate StageKVCache. It only records the window-local
budget and estimated KV bytes for future opt-in compression.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


VIDEO_KV_COMPRESSION_PLAN_SCHEMA = "video_kv_compression_plan_v1"
VIDEO_KV_COMPRESSION_METHODS = frozenset({"none", "uniform", "swa", "infinipot-v"})
VIDEO_KV_SELECTOR_METHODS = frozenset({"uniform", "swa"})


def build_video_kv_compression_plan(
    *,
    video_window_cache: Mapping[str, Any] | None,
    stage_kv_cache_summary: Mapping[str, Any] | None,
    method: str = "none",
    keep_ratio: float | None = None,
    keep_tokens_per_window: int | None = None,
) -> dict[str, Any] | None:
    """Build a JSON-friendly no-op compression plan for local video KV windows."""

    if not video_window_cache:
        return None
    windows = video_window_cache.get("windows")
    if not isinstance(windows, list) or not windows:
        return None

    method = str(method or "none")
    if method not in VIDEO_KV_COMPRESSION_METHODS:
        raise ValueError(
            f"unknown video KV compression method {method!r}; "
            f"expected one of {sorted(VIDEO_KV_COMPRESSION_METHODS)}"
        )

    bytes_per_token = _estimate_local_kv_bytes_per_token(stage_kv_cache_summary)
    window_plans: list[dict[str, Any]] = []
    total_original_tokens = 0
    total_keep_tokens = 0
    total_original_bytes = 0
    total_keep_bytes = 0

    for raw_window in windows:
        if not isinstance(raw_window, Mapping):
            continue
        token_start = _safe_int(raw_window.get("token_start"))
        token_end = _safe_int(raw_window.get("token_end"))
        token_count = _safe_int(raw_window.get("token_count"))
        if token_count is None and token_start is not None and token_end is not None:
            token_count = token_end - token_start
        if token_start is None or token_end is None or token_count is None or token_count <= 0:
            continue

        keep_count = _resolve_keep_count(
            token_count=token_count,
            method=method,
            keep_ratio=keep_ratio,
            keep_tokens_per_window=keep_tokens_per_window,
        )
        selected_tokens, selector_status = _candidate_tokens(
            method=method,
            token_start=token_start,
            token_end=token_end,
            keep_count=keep_count,
        )
        original_bytes = _estimate_bytes(token_count, bytes_per_token)
        keep_bytes = _estimate_bytes(keep_count, bytes_per_token)
        selected_ranges = _ranges_from_sorted_tokens(selected_tokens)
        total_original_tokens += token_count
        total_keep_tokens += keep_count
        total_original_bytes += original_bytes or 0
        total_keep_bytes += keep_bytes or 0

        window_plans.append(
            {
                "window_id": raw_window.get("window_id"),
                "token_range": [token_start, token_end],
                "original_token_count": token_count,
                "keep_token_count": keep_count,
                "drop_token_count": token_count - keep_count,
                "expected_keep_ratio": _safe_ratio(keep_count, token_count),
                "selector": method,
                "selector_status": selector_status,
                "selected_token_sample": selected_tokens[:16],
                "selected_token_ranges": selected_ranges,
                "selected_token_count": len(selected_tokens),
                "candidate_token_sample": selected_tokens[:16],
                "candidate_token_ranges": selected_ranges,
                "candidate_token_count": len(selected_tokens),
                "estimated_original_kv_bytes": original_bytes,
                "estimated_keep_kv_bytes": keep_bytes,
                "estimated_savable_kv_bytes": None
                if original_bytes is None or keep_bytes is None
                else max(0, original_bytes - keep_bytes),
                "kv_location": raw_window.get("kv_location"),
            }
        )

    if not window_plans:
        return None

    plan = {
        "schema": VIDEO_KV_COMPRESSION_PLAN_SCHEMA,
        "planner_only": True,
        "mutates_kv": False,
        "compression_enabled": False,
        "selector_enabled": method in VIDEO_KV_SELECTOR_METHODS,
        "method": method,
        "requested_keep_ratio": keep_ratio,
        "requested_keep_tokens_per_window": keep_tokens_per_window,
        "budget_source": _budget_source(method, keep_ratio, keep_tokens_per_window),
        "window_count": len(window_plans),
        "total_original_tokens": total_original_tokens,
        "total_keep_tokens": total_keep_tokens,
        "total_drop_tokens": total_original_tokens - total_keep_tokens,
        "expected_keep_ratio": _safe_ratio(total_keep_tokens, total_original_tokens),
        "estimated_local_kv_bytes_per_token": bytes_per_token,
        "estimated_original_kv_bytes": total_original_bytes if bytes_per_token is not None else None,
        "estimated_keep_kv_bytes": total_keep_bytes if bytes_per_token is not None else None,
        "estimated_savable_kv_bytes": None
        if bytes_per_token is None
        else max(0, total_original_bytes - total_keep_bytes),
        "allocated_kv_tensor_bytes_after_plan": _safe_int(
            None if stage_kv_cache_summary is None else stage_kv_cache_summary.get("tensor_bytes")
        ),
        "windows": window_plans,
    }
    plan["metadata_bytes"] = _json_size_bytes(plan)
    return plan


def _resolve_keep_count(
    *,
    token_count: int,
    method: str,
    keep_ratio: float | None,
    keep_tokens_per_window: int | None,
) -> int:
    if method == "none":
        return token_count
    if keep_tokens_per_window is not None:
        return max(1, min(token_count, int(keep_tokens_per_window)))
    if keep_ratio is not None:
        ratio = max(0.0, min(1.0, float(keep_ratio)))
        return max(1, min(token_count, int(round(token_count * ratio))))
    return token_count


def _budget_source(
    method: str,
    keep_ratio: float | None,
    keep_tokens_per_window: int | None,
) -> str:
    if method == "none":
        return "none"
    if keep_tokens_per_window is not None:
        return "keep_tokens_per_window"
    if keep_ratio is not None:
        return "keep_ratio"
    return "no_budget_all_tokens"


def _candidate_tokens(
    *,
    method: str,
    token_start: int,
    token_end: int,
    keep_count: int,
) -> tuple[list[int], str]:
    token_count = token_end - token_start
    if method == "none" or keep_count >= token_count:
        return list(range(token_start, token_end)), "all_tokens"
    if method == "swa":
        return list(range(token_end - keep_count, token_end)), "planned_recent_window"
    if method == "uniform":
        step = token_count / keep_count
        tokens = [token_start + min(token_count - 1, int(index * step)) for index in range(keep_count)]
        return sorted(set(tokens)), "planned_uniform"
    return [], "requires_layer_kv_scores"


def _estimate_local_kv_bytes_per_token(stage_kv_cache_summary: Mapping[str, Any] | None) -> int | None:
    if not stage_kv_cache_summary:
        return None
    tensor_bytes = _safe_int(stage_kv_cache_summary.get("tensor_bytes"))
    max_seq_len = _safe_int(stage_kv_cache_summary.get("max_seq_len"))
    if tensor_bytes is None or max_seq_len is None or max_seq_len <= 0:
        return None
    return int(round(tensor_bytes / max_seq_len))


def _estimate_bytes(token_count: int, bytes_per_token: int | None) -> int | None:
    if bytes_per_token is None:
        return None
    return int(token_count * bytes_per_token)


def _ranges_from_sorted_tokens(tokens: list[int]) -> list[list[int]]:
    if not tokens:
        return []
    ranges: list[list[int]] = []
    start = tokens[0]
    previous = tokens[0]
    for token in tokens[1:]:
        if token == previous + 1:
            previous = token
            continue
        ranges.append([start, previous + 1])
        start = token
        previous = token
    ranges.append([start, previous + 1])
    return ranges


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_size_bytes(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))


__all__ = [
    "VIDEO_KV_COMPRESSION_METHODS",
    "VIDEO_KV_COMPRESSION_PLAN_SCHEMA",
    "VIDEO_KV_SELECTOR_METHODS",
    "build_video_kv_compression_plan",
]
