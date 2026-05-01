"""Attention replay kernels for Qwen3-VL text decoder layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ....hexgen_core.distributed import all_reduce_cpu, broadcast_cpu
from .common import (
    _cast_optional_tensor,
    _resolve_tp_math_dtype,
    _slice_local_past_states,
)
from .kv_cache import LayerKVCache
from ..functional import apply_rope, attn_eager, rms_norm


def _validate_attention_mask_shape(
    attention_mask: torch.Tensor | None,
    *,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    context: str,
    hidden_states: torch.Tensor,
    past_key: torch.Tensor | None,
) -> None:
    if attention_mask is None:
        return
    if attention_mask.ndim < 2:
        raise RuntimeError(
            f"{context}: attention_mask 维度异常，"
            f"attention_mask_shape={tuple(attention_mask.shape)}"
        )

    query_len = int(query_states.shape[-2])
    key_len = int(key_states.shape[-2])
    mask_query_len = int(attention_mask.shape[-2])
    mask_key_len = int(attention_mask.shape[-1])
    if mask_query_len == query_len and mask_key_len == key_len:
        return

    past_key_shape = None if past_key is None else tuple(past_key.shape)
    raise RuntimeError(
        f"{context}: attention mask 与 key/query 长度不匹配，"
        f"hidden_states_shape={tuple(hidden_states.shape)} "
        f"query_shape={tuple(query_states.shape)} "
        f"key_shape={tuple(key_states.shape)} "
        f"attention_mask_shape={tuple(attention_mask.shape)} "
        f"past_key_shape={past_key_shape}"
    )


def forward_attention(hidden_states: torch.Tensor, bundle: dict) -> torch.Tensor:
    return trace_attention(hidden_states, bundle)["attn_output"]


def _concat_past_key_value(
    current_states: torch.Tensor,
    past_states: torch.Tensor | None,
) -> torch.Tensor:
    if past_states is None:
        return current_states
    past_states = past_states.to(device=current_states.device, dtype=current_states.dtype)
    if past_states.dim() != current_states.dim():
        raise ValueError(
            "past_states 和 current_states 维度不一致，"
            f"past_shape={tuple(past_states.shape)} current_shape={tuple(current_states.shape)}"
        )
    if past_states.shape[:-2] != current_states.shape[:-2] or past_states.shape[-1] != current_states.shape[-1]:
        raise ValueError(
            "past_states 和 current_states 的 batch/head/head_dim 不一致，"
            f"past_shape={tuple(past_states.shape)} current_shape={tuple(current_states.shape)}"
    )
    return torch.cat([past_states, current_states], dim=-2)


def _resolve_full_key_value(
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    *,
    past_key: torch.Tensor | None,
    past_value: torch.Tensor | None,
    layer_kv_cache: LayerKVCache | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if layer_kv_cache is None:
        return (
            _concat_past_key_value(current_key, past_key),
            _concat_past_key_value(current_value, past_value),
        )
    if past_key is not None or past_value is not None:
        raise ValueError("layer_kv_cache and past_key/past_value cannot be used together.")
    return layer_kv_cache.append(current_key, current_value)


def _is_tp_weight_sharded(bundle: dict) -> bool:
    return bool(bundle.get("tp_weight_sharded"))


def _resolve_tp_attention_layout(bundle: dict, world_size: int) -> tuple[int, int, int, int]:
    num_heads = int(bundle["num_attention_heads"])
    num_kv_heads = int(bundle["num_key_value_heads"])
    if _is_tp_weight_sharded(bundle):
        shard_world_size = int(bundle.get("tp_shard_world_size") or world_size)
        if shard_world_size != world_size:
            raise ValueError(
                "TP attention bundle 的 shard world_size 和当前 world_size 不一致，"
                f"bundle_world_size={shard_world_size} world_size={world_size}"
            )
        local_q_heads = int(bundle["tp_local_num_attention_heads"])
        local_kv_heads = int(bundle["tp_local_num_key_value_heads"])
        return num_heads, num_kv_heads, local_q_heads, local_kv_heads

    if num_heads % world_size != 0 or num_kv_heads % world_size != 0:
        raise ValueError("当前 TP attention 要求 num_heads 和 num_kv_heads 都能被 world_size 整除。")
    return num_heads, num_kv_heads, num_heads // world_size, num_kv_heads // world_size


def _resolve_tp_past_states(
    past_states: torch.Tensor | None,
    *,
    rank: int,
    world_size: int,
    full_num_heads: int,
    local_num_heads: int,
    device: torch.device,
    dtype: torch.dtype,
    tensor_name: str,
) -> torch.Tensor | None:
    if past_states is None:
        return None
    if past_states.shape[1] == local_num_heads:
        return past_states.to(device=device, dtype=dtype)
    return _slice_local_past_states(
        past_states,
        rank=rank,
        world_size=world_size,
        full_num_heads=full_num_heads,
        device=device,
        dtype=dtype,
        tensor_name=tensor_name,
    )


def _tp_collective_context(
    bundle: dict,
    *,
    phase: str,
    module: str,
    reason: str,
) -> dict:
    context = {
        "phase": phase,
        "module": module,
        "reason": reason,
    }
    if "layer_idx" in bundle:
        context["layer_idx"] = int(bundle["layer_idx"])
    return context


def _reduce_tp_attention_output(
    hidden_states: torch.Tensor,
    projected_attn_context: torch.Tensor,
    bundle: dict,
    *,
    sharded: bool,
    is_leader_rank: bool,
    tp_group,
    tp_src_rank: int,
    comm_dtype: torch.dtype,
    orig_dtype: torch.dtype,
    profile_phase: str,
) -> torch.Tensor:
    device = hidden_states.device
    o_bias = _cast_optional_tensor(bundle["o_bias"], device=device, dtype=orig_dtype)
    if sharded:
        local_o_weight = bundle["o_weight"].to(device=device, dtype=orig_dtype)
        local_output_partial = F.linear(projected_attn_context, local_o_weight, bias=None)
        attn_output = all_reduce_cpu(
            local_output_partial,
            target_device=device,
            target_dtype=orig_dtype,
            comm_dtype=comm_dtype,
            group=tp_group,
            profile_context=_tp_collective_context(
                bundle,
                phase=profile_phase,
                module="attention",
                reason="row_parallel_reduce",
            ),
        )
        if o_bias is not None:
            attn_output = attn_output + o_bias
        return attn_output

    full_o_weight = bundle["o_weight"].to(device=device, dtype=orig_dtype)
    leader_output = None
    if is_leader_rank:
        leader_output = F.linear(projected_attn_context, full_o_weight, o_bias)
    return broadcast_cpu(
        hidden_states,
        leader_output,
        src=tp_src_rank,
        comm_dtype=comm_dtype,
        group=tp_group,
        profile_context=_tp_collective_context(
            bundle,
            phase=profile_phase,
            module="attention",
            reason="full_weight_leader_broadcast",
        ),
    )


def trace_attention(hidden_states: torch.Tensor, bundle: dict) -> dict:
    input_shape = hidden_states.shape[:-1]
    head_dim = bundle["head_dim"]
    num_heads = bundle["num_attention_heads"]
    num_kv_heads = bundle["num_key_value_heads"]

    query_states = F.linear(hidden_states, bundle["q_weight"], bundle["q_bias"]).view(*input_shape, -1, head_dim)
    query_states = rms_norm(query_states, bundle["q_norm_weight"], bundle["rms_norm_eps"]).transpose(1, 2)

    key_states = F.linear(hidden_states, bundle["k_weight"], bundle["k_bias"]).view(
        *input_shape, num_kv_heads, head_dim
    )
    key_states = rms_norm(key_states, bundle["k_norm_weight"], bundle["rms_norm_eps"]).transpose(1, 2)

    value_states = F.linear(hidden_states, bundle["v_weight"], bundle["v_bias"]).view(
        *input_shape, num_kv_heads, head_dim
    ).transpose(1, 2)

    query_states, key_states = apply_rope(query_states, key_states, bundle["cos"], bundle["sin"])
    _validate_attention_mask_shape(
        bundle.get("attention_mask"),
        query_states=query_states,
        key_states=key_states,
        context="prefill_attention",
        hidden_states=hidden_states,
        past_key=None,
    )
    attn_output, _ = attn_eager(
        query_states,
        key_states,
        value_states,
        bundle["attention_mask"],
        num_key_value_groups=num_heads // num_kv_heads,
        scaling=bundle["scaling"],
    )
    attn_context = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = F.linear(attn_context, bundle["o_weight"], bundle["o_bias"])
    return {
        "attn_context": attn_context,
        "attn_output": attn_output,
    }


def forward_attention_cached(hidden_states: torch.Tensor, bundle: dict) -> torch.Tensor:
    return trace_attention_cached(hidden_states, bundle)["attn_output"]


def trace_attention_cached(hidden_states: torch.Tensor, bundle: dict) -> dict:
    return _trace_attention_cached_core(
        hidden_states,
        bundle,
        past_key=bundle.get("past_key"),
        past_value=bundle.get("past_value"),
        layer_kv_cache=bundle.get("layer_kv_cache"),
    )


def _trace_attention_cached_core(
    hidden_states: torch.Tensor,
    bundle: dict,
    *,
    past_key: torch.Tensor | None,
    past_value: torch.Tensor | None,
    layer_kv_cache: LayerKVCache | None,
) -> dict:
    input_shape = hidden_states.shape[:-1]
    head_dim = bundle["head_dim"]
    num_heads = bundle["num_attention_heads"]
    num_kv_heads = bundle["num_key_value_heads"]

    query_states = F.linear(hidden_states, bundle["q_weight"], bundle["q_bias"]).view(*input_shape, -1, head_dim)
    query_states = rms_norm(query_states, bundle["q_norm_weight"], bundle["rms_norm_eps"]).transpose(1, 2)

    key_states = F.linear(hidden_states, bundle["k_weight"], bundle["k_bias"]).view(
        *input_shape, num_kv_heads, head_dim
    )
    key_states = rms_norm(key_states, bundle["k_norm_weight"], bundle["rms_norm_eps"]).transpose(1, 2)

    value_states = F.linear(hidden_states, bundle["v_weight"], bundle["v_bias"]).view(
        *input_shape, num_kv_heads, head_dim
    ).transpose(1, 2)

    query_states, key_states = apply_rope(query_states, key_states, bundle["cos"], bundle["sin"])
    full_key_states, full_value_states = _resolve_full_key_value(
        key_states,
        value_states,
        past_key=past_key,
        past_value=past_value,
        layer_kv_cache=layer_kv_cache,
    )
    _validate_attention_mask_shape(
        bundle.get("attention_mask"),
        query_states=query_states,
        key_states=full_key_states,
        context="cached_attention",
        hidden_states=hidden_states,
        past_key=past_key,
    )

    attn_output, _ = attn_eager(
        query_states,
        full_key_states,
        full_value_states,
        bundle["attention_mask"],
        num_key_value_groups=num_heads // num_kv_heads,
        scaling=bundle["scaling"],
    )
    attn_context = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = F.linear(attn_context, bundle["o_weight"], bundle["o_bias"])
    return {
        "attn_context": attn_context,
        "attn_output": attn_output,
        "current_key": key_states,
        "current_value": value_states,
        "full_key": full_key_states,
        "full_value": full_value_states,
    }


def _trace_attention_tp_core(
    hidden_states: torch.Tensor,
    bundle: dict,
    *,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group,
    tp_src_rank: int,
    math_mode: str,
    past_key: torch.Tensor | None,
    past_value: torch.Tensor | None,
    layer_kv_cache: LayerKVCache | None,
) -> dict:
    full_num_heads, full_num_kv_heads, local_q_heads, local_kv_heads = _resolve_tp_attention_layout(
        bundle,
        world_size,
    )
    sharded = _is_tp_weight_sharded(bundle)
    head_dim = int(bundle["head_dim"])
    orig_dtype, _ = _resolve_tp_math_dtype(hidden_states, math_mode)
    device = hidden_states.device
    profile_phase = str(
        bundle.get("tp_profile_phase")
        or ("decode" if past_key is not None or past_value is not None else "prefill")
    )

    x = hidden_states.to(dtype=orig_dtype)
    q_weight = bundle["q_weight"].to(device=device, dtype=orig_dtype)
    q_bias = _cast_optional_tensor(bundle["q_bias"], device=device, dtype=orig_dtype)
    k_weight = bundle["k_weight"].to(device=device, dtype=orig_dtype)
    k_bias = _cast_optional_tensor(bundle["k_bias"], device=device, dtype=orig_dtype)
    v_weight = bundle["v_weight"].to(device=device, dtype=orig_dtype)
    v_bias = _cast_optional_tensor(bundle["v_bias"], device=device, dtype=orig_dtype)
    q_norm_weight = bundle["q_norm_weight"].to(device=device, dtype=orig_dtype)
    k_norm_weight = bundle["k_norm_weight"].to(device=device, dtype=orig_dtype)

    q_proj = F.linear(x, q_weight, q_bias)
    query_states = rms_norm(
        q_proj.view(*hidden_states.shape[:-1], local_q_heads if sharded else full_num_heads, head_dim),
        q_norm_weight,
        bundle["rms_norm_eps"],
    ).transpose(1, 2)

    k_proj = F.linear(x, k_weight, k_bias)
    current_key = rms_norm(
        k_proj.view(*hidden_states.shape[:-1], local_kv_heads if sharded else full_num_kv_heads, head_dim),
        k_norm_weight,
        bundle["rms_norm_eps"],
    ).transpose(1, 2)

    v_proj = F.linear(x, v_weight, v_bias)
    current_value = v_proj.view(
        *hidden_states.shape[:-1],
        local_kv_heads if sharded else full_num_kv_heads,
        head_dim,
    ).transpose(1, 2)

    query_states, current_key = apply_rope(query_states, current_key, bundle["cos"], bundle["sin"])
    local_past_key = _resolve_tp_past_states(
        past_key,
        rank=rank,
        world_size=world_size,
        full_num_heads=full_num_kv_heads,
        local_num_heads=local_kv_heads,
        device=device,
        dtype=orig_dtype,
        tensor_name="past_key",
    )
    local_past_value = _resolve_tp_past_states(
        past_value,
        rank=rank,
        world_size=world_size,
        full_num_heads=full_num_kv_heads,
        local_num_heads=local_kv_heads,
        device=device,
        dtype=orig_dtype,
        tensor_name="past_value",
    )
    full_key, full_value = _resolve_full_key_value(
        current_key,
        current_value,
        past_key=local_past_key,
        past_value=local_past_value,
        layer_kv_cache=layer_kv_cache,
    )
    _validate_attention_mask_shape(
        bundle.get("attention_mask"),
        query_states=query_states,
        key_states=full_key,
        context="cached_attention_tp" if past_key is not None or past_value is not None else "prefill_attention_tp",
        hidden_states=hidden_states,
        past_key=local_past_key,
    )

    attn_output, _ = attn_eager(
        query_states,
        full_key,
        full_value,
        bundle["attention_mask"],
        num_key_value_groups=(local_q_heads if sharded else full_num_heads)
        // (local_kv_heads if sharded else full_num_kv_heads),
        scaling=bundle["scaling"],
        compute_dtype=None,
        score_output_dtype=None,
        probabilities_dtype=orig_dtype,
        output_dtype=orig_dtype,
    )
    full_or_local_attn_context = attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous()
    if sharded:
        local_attn_context = full_or_local_attn_context
        projected_attn_context = local_attn_context
    else:
        q_head_start = rank * local_q_heads
        q_head_end = (rank + 1) * local_q_heads
        local_attn_context = full_or_local_attn_context[
            ...,
            q_head_start * head_dim : q_head_end * head_dim,
        ].contiguous()
        projected_attn_context = full_or_local_attn_context

    reduced_output = _reduce_tp_attention_output(
        hidden_states,
        projected_attn_context,
        bundle,
        sharded=sharded,
        is_leader_rank=rank == 0,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        comm_dtype=comm_dtype,
        orig_dtype=orig_dtype,
        profile_phase=profile_phase,
    )
    return {
        "attn_context": local_attn_context,
        "attn_output": reduced_output,
        "current_key": current_key,
        "current_value": current_value,
        "full_key": full_key,
        "full_value": full_value,
    }


def forward_attention_cached_tp(
    hidden_states: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    math_mode: str = "orig",
) -> torch.Tensor:
    return trace_attention_cached_tp(
        hidden_states,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        math_mode=math_mode,
    )["attn_output"]


def trace_attention_cached_tp(
    hidden_states: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    math_mode: str = "orig",
) -> dict:
    return _trace_attention_tp_core(
        hidden_states,
        bundle,
        rank=rank,
        world_size=world_size,
        comm_dtype=comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        math_mode=math_mode,
        past_key=bundle.get("past_key"),
        past_value=bundle.get("past_value"),
        layer_kv_cache=bundle.get("layer_kv_cache"),
    )


def forward_attention_tp(
    hidden_states: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    math_mode: str = "orig",
) -> torch.Tensor:
    return trace_attention_tp(
        hidden_states,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        math_mode=math_mode,
    )["attn_output"]


def trace_attention_tp(
    hidden_states: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    math_mode: str = "orig",
) -> dict:
    trace = _trace_attention_tp_core(
        hidden_states,
        bundle,
        rank=rank,
        world_size=world_size,
        comm_dtype=comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        math_mode=math_mode,
        past_key=None,
        past_value=None,
        layer_kv_cache=None,
    )
    return {
        "attn_context": trace["attn_context"],
        "attn_output": trace["attn_output"],
    }


__all__ = [
    "forward_attention",
    "trace_attention",
    "forward_attention_cached",
    "trace_attention_cached",
    "forward_attention_cached_tp",
    "trace_attention_cached_tp",
    "forward_attention_tp",
    "trace_attention_tp",
]
