"""Attention replay kernels for Qwen3-VL text decoder layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from qwen3vl_tp_runtime.hexgen_core.distributed import broadcast_cpu
from qwen3vl_tp_runtime.models.qwen3vl.execution.common import (
    _cast_optional_tensor,
    _resolve_tp_math_dtype,
)
from qwen3vl_tp_runtime.models.qwen3vl.functional import apply_rope, attn_eager, rms_norm


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
    )


def _trace_attention_cached_core(
    hidden_states: torch.Tensor,
    bundle: dict,
    *,
    past_key: torch.Tensor | None,
    past_value: torch.Tensor | None,
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
    full_key_states = _concat_past_key_value(key_states, past_key)
    full_value_states = _concat_past_key_value(value_states, past_value)
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
    num_heads = bundle["num_attention_heads"]
    num_kv_heads = bundle["num_key_value_heads"]
    head_dim = bundle["head_dim"]
    orig_dtype, _ = _resolve_tp_math_dtype(hidden_states, math_mode)
    device = hidden_states.device

    if num_heads % world_size != 0 or num_kv_heads % world_size != 0:
        raise ValueError("当前 TP attention 要求 num_heads 和 num_kv_heads 都能被 world_size 整除。")

    local_q_heads = num_heads // world_size
    q_head_start = rank * local_q_heads
    q_head_end = (rank + 1) * local_q_heads

    x = hidden_states.to(dtype=orig_dtype)
    q_weight = bundle["q_weight"].to(device=device, dtype=orig_dtype)
    q_bias = _cast_optional_tensor(bundle["q_bias"], device=device, dtype=orig_dtype)
    k_weight = bundle["k_weight"].to(device=device, dtype=orig_dtype)
    k_bias = _cast_optional_tensor(bundle["k_bias"], device=device, dtype=orig_dtype)
    v_weight = bundle["v_weight"].to(device=device, dtype=orig_dtype)
    v_bias = _cast_optional_tensor(bundle["v_bias"], device=device, dtype=orig_dtype)
    q_norm_weight = bundle["q_norm_weight"].to(device=device, dtype=orig_dtype)
    k_norm_weight = bundle["k_norm_weight"].to(device=device, dtype=orig_dtype)

    full_q_proj = F.linear(x, q_weight, q_bias)
    full_q = rms_norm(
        full_q_proj.view(*hidden_states.shape[:-1], num_heads, head_dim),
        q_norm_weight,
        bundle["rms_norm_eps"],
    ).transpose(1, 2)

    full_k_proj = F.linear(x, k_weight, k_bias)
    current_key = rms_norm(
        full_k_proj.view(*hidden_states.shape[:-1], num_kv_heads, head_dim),
        k_norm_weight,
        bundle["rms_norm_eps"],
    ).transpose(1, 2)

    full_v_proj = F.linear(x, v_weight, v_bias)
    current_value = full_v_proj.view(*hidden_states.shape[:-1], num_kv_heads, head_dim).transpose(1, 2)

    full_q, current_key = apply_rope(full_q, current_key, bundle["cos"], bundle["sin"])
    past_key = _cast_optional_tensor(bundle.get("past_key"), device=device, dtype=orig_dtype)
    past_value = _cast_optional_tensor(bundle.get("past_value"), device=device, dtype=orig_dtype)
    full_key = _concat_past_key_value(current_key, past_key)
    full_value = _concat_past_key_value(current_value, past_value)
    _validate_attention_mask_shape(
        bundle.get("attention_mask"),
        query_states=full_q,
        key_states=full_key,
        context="cached_attention_tp",
        hidden_states=hidden_states,
        past_key=past_key,
    )

    full_attn_output, _ = attn_eager(
        full_q,
        full_key,
        full_value,
        bundle["attention_mask"],
        num_key_value_groups=num_heads // num_kv_heads,
        scaling=bundle["scaling"],
        compute_dtype=None,
        score_output_dtype=None,
        probabilities_dtype=orig_dtype,
        output_dtype=orig_dtype,
    )
    full_attn_context = full_attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous()
    local_attn_context = full_attn_context[
        ...,
        q_head_start * head_dim : q_head_end * head_dim,
    ].contiguous()
    full_o_weight = bundle["o_weight"].to(device=device, dtype=orig_dtype)
    full_o_bias = _cast_optional_tensor(bundle["o_bias"], device=device, dtype=orig_dtype)
    leader_output = None
    if rank == 0:
        leader_output = F.linear(full_attn_context, full_o_weight, full_o_bias)
    attn_output = broadcast_cpu(
        hidden_states,
        leader_output,
        src=tp_src_rank,
        comm_dtype=comm_dtype,
        group=tp_group,
    )
    return {
        "attn_context": local_attn_context,
        "attn_output": attn_output,
        "current_key": current_key,
        "current_value": current_value,
        "full_key": full_key,
        "full_value": full_value,
    }


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
    num_heads = bundle["num_attention_heads"]
    num_kv_heads = bundle["num_key_value_heads"]
    head_dim = bundle["head_dim"]
    orig_dtype, _ = _resolve_tp_math_dtype(hidden_states, math_mode)
    device = hidden_states.device

    if num_heads % world_size != 0 or num_kv_heads % world_size != 0:
        raise ValueError("当前 TP attention 要求 num_heads 和 num_kv_heads 都能被 world_size 整除。")

    local_q_heads = num_heads // world_size
    q_head_start = rank * local_q_heads
    q_head_end = (rank + 1) * local_q_heads

    x = hidden_states.to(dtype=orig_dtype)
    q_weight = bundle["q_weight"].to(device=device, dtype=orig_dtype)
    q_bias = _cast_optional_tensor(bundle["q_bias"], device=device, dtype=orig_dtype)
    k_weight = bundle["k_weight"].to(device=device, dtype=orig_dtype)
    k_bias = _cast_optional_tensor(bundle["k_bias"], device=device, dtype=orig_dtype)
    v_weight = bundle["v_weight"].to(device=device, dtype=orig_dtype)
    v_bias = _cast_optional_tensor(bundle["v_bias"], device=device, dtype=orig_dtype)
    q_norm_weight = bundle["q_norm_weight"].to(device=device, dtype=orig_dtype)
    k_norm_weight = bundle["k_norm_weight"].to(device=device, dtype=orig_dtype)

    full_q_proj = F.linear(x, q_weight, q_bias)
    full_q = rms_norm(
        full_q_proj.view(*hidden_states.shape[:-1], num_heads, head_dim),
        q_norm_weight,
        bundle["rms_norm_eps"],
    ).transpose(1, 2)

    full_k_proj = F.linear(x, k_weight, k_bias)
    full_k = rms_norm(
        full_k_proj.view(*hidden_states.shape[:-1], num_kv_heads, head_dim),
        k_norm_weight,
        bundle["rms_norm_eps"],
    ).transpose(1, 2)

    full_v_proj = F.linear(x, v_weight, v_bias)
    full_v = full_v_proj.view(*hidden_states.shape[:-1], num_kv_heads, head_dim).transpose(1, 2)

    full_q, full_k = apply_rope(full_q, full_k, bundle["cos"], bundle["sin"])
    full_attn_output, _ = attn_eager(
        full_q,
        full_k,
        full_v,
        bundle["attention_mask"],
        num_key_value_groups=num_heads // num_kv_heads,
        scaling=bundle["scaling"],
        compute_dtype=None,
        score_output_dtype=None,
        probabilities_dtype=orig_dtype,
        output_dtype=orig_dtype,
    )
    full_attn_context = full_attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous()
    local_attn_context = full_attn_context[
        ...,
        q_head_start * head_dim : q_head_end * head_dim,
    ].contiguous()
    full_o_weight = bundle["o_weight"].to(device=device, dtype=orig_dtype)
    full_o_bias = _cast_optional_tensor(bundle["o_bias"], device=device, dtype=orig_dtype)
    leader_output = None
    if rank == 0:
        leader_output = F.linear(full_attn_context, full_o_weight, full_o_bias)
    attn_output = broadcast_cpu(
        hidden_states,
        leader_output,
        src=tp_src_rank,
        comm_dtype=comm_dtype,
        group=tp_group,
    )
    return {
        "attn_context": local_attn_context,
        "attn_output": attn_output,
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
