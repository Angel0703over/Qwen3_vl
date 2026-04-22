"""Masking helpers for replay/runtime code paths."""

from __future__ import annotations

import torch


def cast_cpu(tensor: torch.Tensor | None, save_dtype: torch.dtype | None = None):
    if tensor is None:
        return None
    out = tensor.detach().clone().to("cpu")
    if save_dtype is not None:
        out = out.to(dtype=save_dtype)
    return out


def build_causal_mask(inputs_embeds: torch.Tensor, attention_mask_2d: torch.Tensor | None) -> torch.Tensor:
    batch_size, seq_len, _ = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    min_value = torch.finfo(dtype).min

    causal_mask = torch.full((batch_size, 1, seq_len, seq_len), min_value, device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)

    if attention_mask_2d is not None:
        attention_mask_2d = attention_mask_2d.to(device=device)
        key_padding_mask = attention_mask_2d[:, None, None, :] == 0
        causal_mask = causal_mask.masked_fill(key_padding_mask, min_value)

    return causal_mask


__all__ = [
    "cast_cpu",
    "build_causal_mask",
]
