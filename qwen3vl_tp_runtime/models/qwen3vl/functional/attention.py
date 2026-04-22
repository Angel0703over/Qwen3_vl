"""Low-level eager attention helper."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from qwen3vl_tp_runtime.models.qwen3vl.functional.math_ops import repeat_kv


def attn_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_key_value_groups: int,
    scaling: float,
    compute_dtype: torch.dtype | None = None,
    score_output_dtype: torch.dtype | None = None,
    probabilities_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
):
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    score_query = query if compute_dtype is None else query.to(dtype=compute_dtype)
    score_key_states = key_states if compute_dtype is None else key_states.to(dtype=compute_dtype)
    attn_weights = torch.matmul(score_query, score_key_states.transpose(2, 3)) * scaling
    if score_output_dtype is not None and attn_weights.dtype != score_output_dtype:
        attn_weights = attn_weights.to(dtype=score_output_dtype)
    if attention_mask is not None:
        mask = attention_mask if attention_mask.dtype == attn_weights.dtype else attention_mask.to(dtype=attn_weights.dtype)
        attn_weights = attn_weights + mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_weights = attn_weights.to(dtype=query.dtype if probabilities_dtype is None else probabilities_dtype)

    value_for_output = value_states if value_states.dtype == attn_weights.dtype else value_states.to(attn_weights.dtype)
    attn_output = torch.matmul(attn_weights, value_for_output)
    if output_dtype is not None and attn_output.dtype != output_dtype:
        attn_output = attn_output.to(dtype=output_dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


__all__ = ["attn_eager"]
