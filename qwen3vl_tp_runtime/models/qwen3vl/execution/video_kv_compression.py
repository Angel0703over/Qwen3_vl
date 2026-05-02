"""Video KV compression planning and opt-in compaction helpers.

The default path is metadata-only. Physical StageKVCache compaction is exposed
as an explicit opt-in helper for runtime-only multimodal generate.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import torch

from .kv_cache import StageKVCache

VIDEO_KV_COMPACTION_SCHEMA = "video_kv_compaction_v1"
VIDEO_KV_COMPRESSION_CONTRACT_SCHEMA = "video_kv_compression_contract_v1"
VIDEO_KV_COMPRESSION_PLAN_SCHEMA = "video_kv_compression_plan_v1"
VIDEO_KV_COMPRESSION_METHODS = frozenset({"none", "uniform", "swa", "infinipot-v"})
VIDEO_KV_SELECTOR_METHODS = frozenset({"uniform", "swa", "infinipot-v"})
INFINIPOT_V_DEFAULT_TAR_RATIO = 0.5
INFINIPOT_V_DEFAULT_QUERY_RATIO = 0.25


def build_video_kv_compression_plan(
    *,
    video_window_cache: Mapping[str, Any] | None,
    stage_kv_cache_summary: Mapping[str, Any] | None,
    stage_kv_cache: StageKVCache | None = None,
    method: str = "none",
    keep_ratio: float | None = None,
    keep_tokens_per_window: int | None = None,
    prefill_seq_len: int | None = None,
) -> dict[str, Any] | None:
    """Build a JSON-friendly compression plan for local video KV windows."""

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
    if method == "infinipot-v" and stage_kv_cache is not None:
        plan = materialize_video_kv_compression_plan(
            compression_plan=plan,
            stage_kv_cache=stage_kv_cache,
            prefill_seq_len=prefill_seq_len,
        )
    return plan


def materialize_video_kv_compression_plan(
    *,
    compression_plan: Mapping[str, Any],
    stage_kv_cache: StageKVCache,
    prefill_seq_len: int | None = None,
    tar_ratio: float = INFINIPOT_V_DEFAULT_TAR_RATIO,
    query_ratio: float = INFINIPOT_V_DEFAULT_QUERY_RATIO,
) -> dict[str, Any]:
    """Materialize an InfiniPot-V plan using only local StageKVCache K/V.

    InfiniPot-V selects per-layer/per-head indices. Our runtime compaction
    contract requires one shared token list so attention masks and positions
    stay stage-wide. This helper aggregates local layer/head scores into one
    rank-local selection.
    """

    method = str(compression_plan.get("method", "none"))
    if method != "infinipot-v":
        return dict(compression_plan)

    windows = compression_plan.get("windows")
    if not isinstance(windows, list) or not windows:
        return dict(compression_plan)
    if stage_kv_cache is None:
        raise ValueError("infinipot-v selector requires local StageKVCache K/V scores")

    materialized_windows: list[dict[str, Any]] = []
    total_original_tokens = 0
    total_keep_tokens = 0
    total_original_bytes = 0
    total_keep_bytes = 0

    for raw_window in windows:
        if not isinstance(raw_window, Mapping):
            continue
        window = dict(raw_window)
        token_range = window.get("token_range")
        if not isinstance(token_range, (list, tuple)) or len(token_range) != 2:
            raise ValueError(f"infinipot-v plan window has invalid token_range: {token_range!r}")
        token_start = int(token_range[0])
        token_end = int(token_range[1])
        token_count = int(window.get("original_token_count", token_end - token_start))
        keep_count = int(window.get("keep_token_count", token_count))
        if token_count <= 0 or token_end <= token_start:
            continue
        if token_count != token_end - token_start:
            raise ValueError(
                "infinipot-v plan token_count does not match token_range: "
                f"token_count={token_count} token_range={token_range!r}"
            )

        selected_tokens, score_info = _select_infinipot_v_tokens(
            stage_kv_cache=stage_kv_cache,
            token_start=token_start,
            token_end=token_end,
            keep_count=keep_count,
            prefill_seq_len=prefill_seq_len,
            tar_ratio=tar_ratio,
            query_ratio=query_ratio,
        )
        selected_ranges = _ranges_from_sorted_tokens(selected_tokens)

        window["selector"] = "infinipot-v"
        window["selector_status"] = score_info["selector_status"]
        window["selector_score_source"] = "local_stage_kv_cache"
        window["selected_token_sample"] = selected_tokens[:16]
        window["selected_token_ranges"] = selected_ranges
        window["selected_token_count"] = len(selected_tokens)
        window["candidate_token_sample"] = selected_tokens[:16]
        window["candidate_token_ranges"] = selected_ranges
        window["candidate_token_count"] = len(selected_tokens)
        window["score_layer_count"] = score_info["score_layer_count"]
        window["tar_score_layer_count"] = score_info["tar_score_layer_count"]
        window["tar_ratio"] = score_info["tar_ratio"]
        window["query_ratio"] = score_info["query_ratio"]
        window["tar_budget"] = score_info["tar_budget"]
        window["tar_query_token_count"] = score_info["tar_query_token_count"]
        window["tar_selected_token_sample"] = score_info["tar_selected_token_sample"]
        window["score_policy"] = score_info["score_policy"]
        window["tar_similarity_sign"] = score_info["tar_similarity_sign"]

        original_bytes = _safe_int(window.get("estimated_original_kv_bytes"))
        keep_bytes = _safe_int(window.get("estimated_keep_kv_bytes"))
        total_original_tokens += token_count
        total_keep_tokens += len(selected_tokens)
        total_original_bytes += original_bytes or 0
        total_keep_bytes += keep_bytes or 0
        materialized_windows.append(window)

    plan = dict(compression_plan)
    plan["windows"] = materialized_windows
    plan["selector_enabled"] = True
    plan["selector_materialized"] = True
    plan["selector_score_source"] = "local_stage_kv_cache"
    plan["infinipot_v_tar_ratio"] = float(tar_ratio)
    plan["infinipot_v_query_ratio"] = float(query_ratio)
    plan["total_original_tokens"] = total_original_tokens
    plan["total_keep_tokens"] = total_keep_tokens
    plan["total_drop_tokens"] = total_original_tokens - total_keep_tokens
    plan["expected_keep_ratio"] = _safe_ratio(total_keep_tokens, total_original_tokens)
    if plan.get("estimated_original_kv_bytes") is not None:
        plan["estimated_original_kv_bytes"] = total_original_bytes
    if plan.get("estimated_keep_kv_bytes") is not None:
        plan["estimated_keep_kv_bytes"] = total_keep_bytes
    if plan.get("estimated_savable_kv_bytes") is not None:
        plan["estimated_savable_kv_bytes"] = max(0, total_original_bytes - total_keep_bytes)
    plan["metadata_bytes"] = _json_size_bytes(plan)
    return plan


def build_video_kv_compression_contract(
    *,
    compression_plan: Mapping[str, Any] | None,
    prefill_seq_len: int,
    decoded_token_count: int = 0,
    query_len: int = 1,
) -> dict[str, Any] | None:
    """Build the physical/logical decode contract for KV compaction.

    The contract is metadata-only. It defines the lengths that a compacted
    StageKVCache and its decode mask must agree on.
    """

    if not compression_plan:
        return None
    original_prefill_len = int(prefill_seq_len)
    decoded_tokens = int(decoded_token_count)
    query_len = int(query_len)
    if original_prefill_len <= 0:
        raise ValueError(f"prefill_seq_len must be positive, got {prefill_seq_len}")
    if decoded_tokens < 0:
        raise ValueError(f"decoded_token_count must be >= 0, got {decoded_token_count}")
    if query_len <= 0:
        raise ValueError(f"query_len must be positive, got {query_len}")

    keep_indices = resolve_prefill_keep_token_indices(
        compression_plan=compression_plan,
        prefill_seq_len=original_prefill_len,
    )
    compact_prefill_len = len(keep_indices)
    dropped_prefill_tokens = original_prefill_len - compact_prefill_len
    physical_past_length = compact_prefill_len + decoded_tokens
    logical_past_length = original_prefill_len + decoded_tokens
    attention_mask_key_length = physical_past_length + query_len
    logical_key_length = logical_past_length + query_len
    requires_position_override = physical_past_length != logical_past_length
    keep_token_ranges = _ranges_from_sorted_tokens(keep_indices)

    contract = {
        "schema": VIDEO_KV_COMPRESSION_CONTRACT_SCHEMA,
        "contract_only": True,
        "mutates_kv": False,
        "method": str(compression_plan.get("method", "none")),
        "prefill": {
            "original_length": original_prefill_len,
            "compact_length": compact_prefill_len,
            "kept_token_count": compact_prefill_len,
            "dropped_token_count": dropped_prefill_tokens,
            "keep_token_sample": keep_indices[:16],
            "keep_token_range_sample": keep_token_ranges[:16],
            "keep_token_range_count": len(keep_token_ranges),
            "video_original_token_count": _safe_int(compression_plan.get("total_original_tokens")) or 0,
            "video_keep_token_count": _safe_int(compression_plan.get("total_keep_tokens")) or 0,
            "video_drop_token_count": _safe_int(compression_plan.get("total_drop_tokens")) or 0,
        },
        "decode": {
            "decoded_token_count_before_query": decoded_tokens,
            "query_len": query_len,
            "physical_past_length": physical_past_length,
            "logical_past_length": logical_past_length,
            "attention_mask_key_length": attention_mask_key_length,
            "logical_key_length": logical_key_length,
            "stage_kv_current_length_before_query": physical_past_length,
            "stage_kv_current_length_after_query": attention_mask_key_length,
            "decode_position_start": logical_past_length,
            "decode_position_end": logical_past_length + query_len,
            "requires_position_override": requires_position_override,
        },
        "rules": {
            "stage_kv_current_length": "physical_kv_tokens",
            "attention_mask_key_length": "StageKVCache.current_length_after_query",
            "past_length": "StageKVCache.current_length_before_query",
            "position_ids": "logical_uncompressed_positions",
            "compact_mask_must_not_drive_position_ids": requires_position_override,
        },
    }
    contract["metadata_bytes"] = _json_size_bytes(contract)
    return contract


def build_compact_prefill_attention_mask_2d(
    prefill_attention_mask_2d: torch.Tensor,
    *,
    compression_plan: Mapping[str, Any],
) -> torch.Tensor:
    """Select the prefill mask columns that remain after planned compaction."""

    if not torch.is_tensor(prefill_attention_mask_2d):
        raise TypeError("prefill_attention_mask_2d must be a torch.Tensor")
    if prefill_attention_mask_2d.ndim < 1:
        raise ValueError(
            "prefill_attention_mask_2d must have a sequence dimension, "
            f"got shape={tuple(prefill_attention_mask_2d.shape)}"
        )
    keep_indices = resolve_prefill_keep_token_indices(
        compression_plan=compression_plan,
        prefill_seq_len=int(prefill_attention_mask_2d.shape[-1]),
    )
    keep_index_tensor = torch.tensor(
        keep_indices,
        device=prefill_attention_mask_2d.device,
        dtype=torch.long,
    )
    return prefill_attention_mask_2d.index_select(-1, keep_index_tensor)


def compact_stage_kv_cache_for_video_plan(
    *,
    stage_kv_cache: StageKVCache,
    compression_plan: Mapping[str, Any],
    prefill_seq_len: int,
) -> dict[str, Any] | None:
    """Physically compact local StageKVCache according to a materialized plan."""

    method = str(compression_plan.get("method", "none"))
    if method not in VIDEO_KV_SELECTOR_METHODS:
        return None
    keep_indices = resolve_prefill_keep_token_indices(
        compression_plan=compression_plan,
        prefill_seq_len=int(prefill_seq_len),
    )
    original_length = int(prefill_seq_len)
    compact_length = len(keep_indices)
    if compact_length >= original_length:
        return None

    before_summary = stage_kv_cache.summary()
    compact_summary = stage_kv_cache.compact_prefix(
        keep_indices,
        original_length=original_length,
    )
    after_summary = stage_kv_cache.summary()
    keep_ranges = _ranges_from_sorted_tokens(keep_indices)
    payload = {
        "schema": VIDEO_KV_COMPACTION_SCHEMA,
        "applied": True,
        "mutates_kv": True,
        "compression_enabled": True,
        "method": method,
        "original_prefill_length": original_length,
        "compact_prefill_length": compact_length,
        "dropped_prefill_tokens": original_length - compact_length,
        "video_original_token_count": _safe_int(compression_plan.get("total_original_tokens")) or 0,
        "video_keep_token_count": _safe_int(compression_plan.get("total_keep_tokens")) or 0,
        "video_drop_token_count": _safe_int(compression_plan.get("total_drop_tokens")) or 0,
        "keep_token_sample": keep_indices[:16],
        "keep_token_range_sample": keep_ranges[:16],
        "keep_token_range_count": len(keep_ranges),
        "layer_count": compact_summary["layer_count"],
        "active_tensor_bytes_before": compact_summary["active_tensor_bytes_before"],
        "active_tensor_bytes_after": compact_summary["active_tensor_bytes_after"],
        "active_tensor_bytes_saved": compact_summary["active_tensor_bytes_saved"],
        "allocated_tensor_bytes_before": before_summary.get("tensor_bytes"),
        "allocated_tensor_bytes_after": after_summary.get("tensor_bytes"),
        "current_lengths_before": before_summary.get("current_lengths"),
        "current_lengths_after": after_summary.get("current_lengths"),
    }
    payload["metadata_bytes"] = _json_size_bytes(payload)
    return payload


def build_compact_decode_attention_mask_2d(
    prefill_attention_mask_2d: torch.Tensor,
    *,
    compression_plan: Mapping[str, Any],
    decoded_token_count: int = 0,
    query_len: int = 1,
) -> torch.Tensor:
    """Build a compact decode 2D mask whose key length matches compact KV."""

    compact_prefill_mask = build_compact_prefill_attention_mask_2d(
        prefill_attention_mask_2d,
        compression_plan=compression_plan,
    )
    decoded_tokens = int(decoded_token_count)
    query_len = int(query_len)
    if decoded_tokens < 0:
        raise ValueError(f"decoded_token_count must be >= 0, got {decoded_token_count}")
    if query_len <= 0:
        raise ValueError(f"query_len must be positive, got {query_len}")
    append_len = decoded_tokens + query_len
    append_mask = torch.ones(
        (*compact_prefill_mask.shape[:-1], append_len),
        device=compact_prefill_mask.device,
        dtype=compact_prefill_mask.dtype,
    )
    return torch.cat([compact_prefill_mask, append_mask], dim=-1)


def validate_video_kv_compression_decode_contract(
    *,
    attention_mask_2d: torch.Tensor,
    key_length: int,
    query_len: int,
    compression_contract: Mapping[str, Any],
) -> None:
    """Validate that compact decode mask, key length, and contract agree."""

    if not torch.is_tensor(attention_mask_2d):
        raise TypeError("attention_mask_2d must be a torch.Tensor")
    if attention_mask_2d.ndim < 1:
        raise ValueError(f"attention_mask_2d shape is invalid: {tuple(attention_mask_2d.shape)}")

    decode_contract = compression_contract.get("decode")
    if not isinstance(decode_contract, Mapping):
        raise ValueError("compression_contract missing decode section")

    expected_key_length = _safe_int(decode_contract.get("attention_mask_key_length"))
    expected_query_len = _safe_int(decode_contract.get("query_len"))
    expected_current_after = _safe_int(decode_contract.get("stage_kv_current_length_after_query"))
    mask_key_length = int(attention_mask_2d.shape[-1])
    key_length = int(key_length)
    query_len = int(query_len)

    if expected_key_length is None or expected_query_len is None or expected_current_after is None:
        raise ValueError("compression_contract decode lengths are incomplete")
    if expected_query_len != query_len:
        raise ValueError(
            "query_len does not match compression contract: "
            f"query_len={query_len} expected={expected_query_len}"
        )
    if key_length != expected_key_length or key_length != expected_current_after:
        raise ValueError(
            "key length does not match compression contract: "
            f"key_length={key_length} expected={expected_key_length} "
            f"current_after={expected_current_after}"
        )
    if mask_key_length != key_length:
        raise ValueError(
            "attention_mask/key length mismatch: "
            f"attention_mask_2d.shape={tuple(attention_mask_2d.shape)} key_length={key_length}"
        )


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


def _select_infinipot_v_tokens(
    *,
    stage_kv_cache: StageKVCache,
    token_start: int,
    token_end: int,
    keep_count: int,
    prefill_seq_len: int | None,
    tar_ratio: float,
    query_ratio: float,
) -> tuple[list[int], dict[str, Any]]:
    token_count = int(token_end) - int(token_start)
    keep_count = max(1, min(token_count, int(keep_count)))
    if token_count <= 0:
        raise ValueError(f"infinipot-v token range is empty: start={token_start} end={token_end}")
    if prefill_seq_len is not None and int(token_end) > int(prefill_seq_len):
        raise ValueError(
            "infinipot-v token range exceeds prefill length: "
            f"token_range={[token_start, token_end]} prefill_seq_len={prefill_seq_len}"
        )
    if keep_count >= token_count:
        return list(range(token_start, token_end)), {
            "selector_status": "all_tokens",
            "score_layer_count": 0,
            "tar_score_layer_count": 0,
            "tar_ratio": float(tar_ratio),
            "query_ratio": float(query_ratio),
            "tar_budget": 0,
            "tar_query_token_count": 0,
            "tar_selected_token_sample": [],
            "score_policy": "all_tokens_no_compaction",
            "tar_similarity_sign": "negative_cosine_like_infinipot_v_source",
        }

    cache_by_layer = stage_kv_cache.as_cache_by_layer()
    if not cache_by_layer:
        raise ValueError("infinipot-v selector requires at least one local KV layer")

    value_score: torch.Tensor | None = None
    tar_score: torch.Tensor | None = None
    score_layer_count = 0
    tar_score_layer_count = 0
    query_token_count = _resolve_infinipot_query_token_count(
        token_count=token_count,
        query_ratio=query_ratio,
    )

    for _layer_idx, (key, value) in sorted(cache_by_layer.items()):
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "infinipot-v expects KV tensors with shape [batch, heads, seq, head_dim], "
                f"got key={tuple(key.shape)} value={tuple(value.shape)}"
            )
        if int(key.shape[-2]) < token_end or int(value.shape[-2]) < token_end:
            continue
        key_window = key[..., token_start:token_end, :].detach()
        value_window = value[..., token_start:token_end, :].detach()
        if int(key_window.shape[-2]) != token_count or int(value_window.shape[-2]) != token_count:
            continue

        local_value_score = value_window.float().norm(dim=-1)
        local_value_score = local_value_score.mean(dim=tuple(range(local_value_score.ndim - 1)))
        if value_score is None:
            value_score = torch.zeros_like(local_value_score, dtype=torch.float32)
        value_score = value_score + local_value_score.to(dtype=torch.float32)
        score_layer_count += 1

        eligible_count = token_count - query_token_count
        if eligible_count <= 0:
            continue
        key_candidates = key_window[..., :eligible_count, :].float()
        query_tokens = key_window[..., eligible_count:, :].float()
        key_norm = torch.nn.functional.normalize(key_candidates, p=2, dim=-1, eps=1e-9)
        query_norm = torch.nn.functional.normalize(query_tokens, p=2, dim=-1, eps=1e-9)
        similarity = (key_norm.unsqueeze(-2) * query_norm.unsqueeze(-3)).sum(dim=-1)
        local_tar_score = -similarity.mean(dim=tuple(range(similarity.ndim - 2)) + (similarity.ndim - 1,))
        if tar_score is None:
            tar_score = torch.zeros(token_count, device=key.device, dtype=torch.float32)
        tar_score[:eligible_count] = tar_score[:eligible_count] + local_tar_score.to(dtype=torch.float32)
        tar_score_layer_count += 1

    if value_score is None or score_layer_count == 0:
        raise ValueError(
            "infinipot-v selector could not read local KV scores for "
            f"token_range={[token_start, token_end]}"
        )
    value_score = value_score / float(score_layer_count)
    if tar_score is not None and tar_score_layer_count > 0:
        tar_score = tar_score / float(tar_score_layer_count)

    tar_budget = _resolve_infinipot_tar_budget(
        keep_count=keep_count,
        tar_ratio=tar_ratio,
    )
    tar_positions = _select_infinipot_tar_positions(
        tar_score=tar_score,
        token_count=token_count,
        query_token_count=query_token_count,
        tar_budget=tar_budget,
    )
    final_score = value_score.clone()
    if tar_positions:
        tar_position_tensor = torch.tensor(
            tar_positions,
            device=final_score.device,
            dtype=torch.long,
        )
        score_bonus = final_score.max()
        if final_score.numel() > 1:
            score_bonus = score_bonus + final_score.std(unbiased=False).abs()
        final_score.index_fill_(0, tar_position_tensor, score_bonus + 1.0)

    selected_local = torch.topk(final_score, k=keep_count, dim=0).indices.sort().values.tolist()
    selected_tokens = [token_start + int(index) for index in selected_local]
    return selected_tokens, {
        "selector_status": "materialized_local_kv_scores",
        "score_layer_count": score_layer_count,
        "tar_score_layer_count": tar_score_layer_count,
        "tar_ratio": float(tar_ratio),
        "query_ratio": float(query_ratio),
        "tar_budget": tar_budget,
        "tar_query_token_count": query_token_count,
        "tar_selected_token_sample": [token_start + int(index) for index in tar_positions[:16]],
        "score_policy": "value_norm_topk_with_tar_recent_query_boost",
        "tar_similarity_sign": "negative_cosine_like_infinipot_v_source",
    }


def _resolve_infinipot_query_token_count(*, token_count: int, query_ratio: float) -> int:
    ratio = max(0.0, min(1.0, float(query_ratio)))
    return max(1, min(int(token_count), int(round(int(token_count) * ratio))))


def _resolve_infinipot_tar_budget(*, keep_count: int, tar_ratio: float) -> int:
    ratio = max(0.0, min(1.0, float(tar_ratio)))
    value_norm_budget = int(round((1.0 - ratio) * int(keep_count)))
    return max(0, min(int(keep_count), int(keep_count) - value_norm_budget))


def _select_infinipot_tar_positions(
    *,
    tar_score: torch.Tensor | None,
    token_count: int,
    query_token_count: int,
    tar_budget: int,
) -> list[int]:
    if tar_budget <= 0:
        return []
    recent_count = min(int(query_token_count), int(tar_budget), int(token_count))
    recent_positions = (
        list(range(int(token_count) - recent_count, int(token_count)))
        if recent_count > 0
        else []
    )
    remaining_budget = max(0, int(tar_budget) - len(recent_positions))
    if tar_score is None or remaining_budget <= 0:
        return sorted(set(recent_positions))

    eligible_count = max(0, int(token_count) - int(query_token_count))
    top_count = min(remaining_budget, eligible_count)
    if top_count <= 0:
        return sorted(set(recent_positions))
    top_positions = torch.topk(tar_score[:eligible_count], k=top_count, dim=0).indices.tolist()
    return sorted(set(int(index) for index in top_positions).union(recent_positions))


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


def resolve_prefill_keep_token_indices(
    *,
    compression_plan: Mapping[str, Any],
    prefill_seq_len: int,
) -> list[int]:
    if int(prefill_seq_len) <= 0:
        raise ValueError(f"prefill_seq_len must be positive, got {prefill_seq_len}")
    windows = compression_plan.get("windows")
    if not isinstance(windows, list) or not windows:
        return list(range(int(prefill_seq_len)))

    drop_tokens: set[int] = set()
    for raw_window in windows:
        if not isinstance(raw_window, Mapping):
            continue
        token_range = raw_window.get("token_range")
        if not isinstance(token_range, (list, tuple)) or len(token_range) != 2:
            raise ValueError(f"compression plan window has invalid token_range: {token_range!r}")
        token_start = int(token_range[0])
        token_end = int(token_range[1])
        if token_start < 0 or token_end > prefill_seq_len or token_end <= token_start:
            raise ValueError(
                "compression plan window token_range is out of prefill bounds: "
                f"token_range={token_range!r} prefill_seq_len={prefill_seq_len}"
            )

        selected_ranges = raw_window.get("selected_token_ranges")
        selected_tokens = _tokens_from_ranges(selected_ranges)
        selected_count = _safe_int(raw_window.get("selected_token_count"))
        if selected_count is not None and selected_count != len(selected_tokens):
            raise ValueError(
                "selected_token_count does not match selected_token_ranges: "
                f"selected_token_count={selected_count} ranges_count={len(selected_tokens)}"
            )
        keep_count = _safe_int(raw_window.get("keep_token_count"))
        if keep_count is not None and keep_count != len(selected_tokens):
            raise ValueError(
                "compression plan has no materialized selected ranges for its keep budget: "
                f"keep_token_count={keep_count} ranges_count={len(selected_tokens)}"
            )
        for token in selected_tokens:
            if token < token_start or token >= token_end:
                raise ValueError(
                    "selected token falls outside its window: "
                    f"token={token} token_range={token_range!r}"
                )
        window_tokens = set(range(token_start, token_end))
        drop_tokens.update(window_tokens.difference(selected_tokens))

    return [token for token in range(int(prefill_seq_len)) if token not in drop_tokens]


def _tokens_from_ranges(ranges: Any) -> set[int]:
    if not isinstance(ranges, list):
        raise ValueError(f"selected_token_ranges must be a list, got {type(ranges).__name__}")
    tokens: set[int] = set()
    for item in ranges:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"selected_token_ranges item is invalid: {item!r}")
        start = int(item[0])
        end = int(item[1])
        if end <= start:
            raise ValueError(f"selected_token_ranges item must be non-empty: {item!r}")
        tokens.update(range(start, end))
    return tokens


__all__ = [
    "VIDEO_KV_COMPACTION_SCHEMA",
    "VIDEO_KV_COMPRESSION_CONTRACT_SCHEMA",
    "VIDEO_KV_COMPRESSION_METHODS",
    "VIDEO_KV_COMPRESSION_PLAN_SCHEMA",
    "VIDEO_KV_SELECTOR_METHODS",
    "build_compact_decode_attention_mask_2d",
    "build_compact_prefill_attention_mask_2d",
    "compact_stage_kv_cache_for_video_plan",
    "materialize_video_kv_compression_plan",
    "resolve_prefill_keep_token_indices",
    "build_video_kv_compression_contract",
    "build_video_kv_compression_plan",
    "validate_video_kv_compression_decode_contract",
]
