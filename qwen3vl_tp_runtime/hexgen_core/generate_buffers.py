"""Small reusable buffers for runtime-only generate loops."""

from __future__ import annotations

import torch


def build_decode_attention_mask_buffer(
    prefill_attention_mask_2d: torch.Tensor,
    *,
    max_new_tokens: int,
) -> torch.Tensor:
    decode_steps = max(int(max_new_tokens) - 1, 0)
    if decode_steps == 0:
        return prefill_attention_mask_2d

    prefill_len = int(prefill_attention_mask_2d.shape[-1])
    total_len = prefill_len + decode_steps
    buffer_shape = (*prefill_attention_mask_2d.shape[:-1], total_len)
    buffer = torch.ones(
        buffer_shape,
        device=prefill_attention_mask_2d.device,
        dtype=prefill_attention_mask_2d.dtype,
    )
    buffer[..., :prefill_len].copy_(prefill_attention_mask_2d)
    return buffer


def decode_attention_mask_view(
    buffer: torch.Tensor,
    *,
    prefill_seq_len: int,
    step_idx: int,
) -> torch.Tensor:
    return buffer[..., : int(prefill_seq_len) + int(step_idx) + 1]


def fill_decode_input_ids(buffer: torch.Tensor, token_id: int) -> torch.Tensor:
    buffer.fill_(int(token_id))
    return buffer


__all__ = [
    "build_decode_attention_mask_buffer",
    "decode_attention_mask_view",
    "fill_decode_input_ids",
]
