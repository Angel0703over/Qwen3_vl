"""Shared types and helpers for live multimodal runtime."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qwen3vl_tp_runtime.models.qwen3vl.functional import dtype_from_name


@dataclass(slots=True)
class MultimodalRuntimeInputs:
    """Runtime tensors materialized from the live Qwen3-VL front-end."""

    input_ids: torch.Tensor | None
    attention_mask_2d: torch.Tensor | None
    position_ids: torch.Tensor | None
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    visual_pos_masks: torch.Tensor | None
    deepstack_by_layer: dict[int, torch.Tensor]
    mm_token_type_ids: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None
    pixel_values: torch.Tensor | None = None
    pixel_values_videos: torch.Tensor | None = None
    rope_deltas: torch.Tensor | None = None


def _resolve_compute_dtype(reference_tensor: torch.Tensor, compute_dtype_arg: str) -> torch.dtype:
    if compute_dtype_arg == "auto":
        return reference_tensor.dtype
    return dtype_from_name(compute_dtype_arg)


def _runtime_tensor(
    tensor: torch.Tensor | None,
    *,
    device: torch.device,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    out = tensor.detach().to(device=device)
    if compute_dtype is not None and out.is_floating_point():
        out = out.to(dtype=compute_dtype)
    return out


def _default_position_ids(
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    text_position_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, -1).expand(batch_size, -1)
    vision_position_ids = text_position_ids.view(1, batch_size, -1).repeat(3, 1, 1)
    full_position_ids = torch.cat([text_position_ids.unsqueeze(0), vision_position_ids], dim=0)
    return full_position_ids, text_position_ids, vision_position_ids


def _split_position_ids(
    position_ids: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if position_ids is None:
        return _default_position_ids(batch_size, seq_len, device)

    if position_ids.ndim == 2:
        text_position_ids = position_ids
        vision_position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1)
        full_position_ids = torch.cat([text_position_ids.unsqueeze(0), vision_position_ids], dim=0)
        return full_position_ids, text_position_ids, vision_position_ids

    if position_ids.ndim != 3:
        raise ValueError(f"position_ids 需要是 2 维或 3 维，当前拿到 shape={tuple(position_ids.shape)}")

    if position_ids.shape[0] == 4:
        return position_ids, position_ids[0], position_ids[1:]
    if position_ids.shape[0] == 3:
        full_position_ids = torch.cat([position_ids[0:1], position_ids], dim=0)
        return full_position_ids, None, position_ids

    raise ValueError(
        "position_ids 的第一维需要是 3 或 4，"
        f"当前拿到 shape={tuple(position_ids.shape)}"
    )


def _build_multimodal_decode_position_ids(
    *,
    decode_input_ids: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    rope_deltas: torch.Tensor,
    logical_position_start: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build single-step multimodal decode position ids from cached rope deltas."""

    batch_size, seq_length = decode_input_ids.shape
    if logical_position_start is None:
        text_position_ids = attention_mask_2d.long().cumsum(-1) - 1
        text_position_ids = text_position_ids.masked_fill(attention_mask_2d == 0, 0)
        text_position_ids = text_position_ids[:, -seq_length:].to(device=decode_input_ids.device)
    else:
        logical_start = int(logical_position_start)
        if logical_start < 0:
            raise ValueError(f"logical_position_start 不能小于 0，当前拿到 {logical_start}")
        text_position_ids = torch.arange(
            logical_start,
            logical_start + seq_length,
            device=decode_input_ids.device,
            dtype=torch.long,
        ).view(1, -1).expand(batch_size, -1)

    if rope_deltas.ndim == 1:
        rope_deltas = rope_deltas.unsqueeze(1)
    rope_deltas = rope_deltas.to(device=decode_input_ids.device)
    if batch_size % rope_deltas.shape[0] != 0:
        raise RuntimeError(
            "rope_deltas batch 维度和 decode batch 维度不兼容，"
            f"rope_deltas.shape={tuple(rope_deltas.shape)} decode_input_ids.shape={tuple(decode_input_ids.shape)}"
        )
    rope_deltas = rope_deltas.repeat_interleave(batch_size // rope_deltas.shape[0], dim=0)

    vision_position_ids = text_position_ids.view(1, batch_size, -1).repeat(3, 1, 1)
    vision_position_ids = vision_position_ids + rope_deltas.view(1, batch_size, 1)
    full_position_ids = torch.cat([text_position_ids.unsqueeze(0), vision_position_ids], dim=0)
    return full_position_ids, text_position_ids, vision_position_ids


__all__ = ["MultimodalRuntimeInputs"]
