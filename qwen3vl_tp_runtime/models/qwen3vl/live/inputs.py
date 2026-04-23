"""Live text and multimodal input materialization helpers."""

from __future__ import annotations

import torch
from transformers.masking_utils import create_causal_mask

from qwen3vl_tp_runtime.models.qwen3vl.functional import build_causal_mask
from qwen3vl_tp_runtime.models.qwen3vl.live.common import (
    MultimodalRuntimeInputs,
    _build_multimodal_decode_position_ids,
    _split_position_ids,
)
from qwen3vl_tp_runtime.models.qwen3vl.vision import materialize_visual_features


def _build_text_position_ids(
    *,
    attention_mask_2d: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_mask_2d is None:
        text_position_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, -1).expand(batch_size, -1)
    else:
        text_position_ids = attention_mask_2d.long().cumsum(-1) - 1
        text_position_ids = text_position_ids.masked_fill(attention_mask_2d == 0, 0).to(device=device)
        text_position_ids = text_position_ids[:, -seq_len:]

    full_position_ids = torch.cat(
        [
            text_position_ids.unsqueeze(0),
            text_position_ids.view(1, batch_size, -1).repeat(3, 1, 1),
        ],
        dim=0,
    )
    return full_position_ids, text_position_ids


def prepare_text_prefill_runtime_inputs(
    model,
    inputs: dict[str, torch.Tensor],
) -> MultimodalRuntimeInputs:
    """Build decoder-ready prefill tensors for text-only prompts."""

    language_model = model.model.language_model
    device = next(model.parameters()).device

    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise ValueError("text prefill 需要 input_ids。")

    input_ids = input_ids.to(device)
    attention_mask_2d = inputs.get("attention_mask")
    if attention_mask_2d is not None:
        attention_mask_2d = attention_mask_2d.to(device)

    inputs_embeds = language_model.get_input_embeddings()(input_ids)
    full_position_ids, text_position_ids = _build_text_position_ids(
        attention_mask_2d=attention_mask_2d,
        batch_size=inputs_embeds.shape[0],
        seq_len=inputs_embeds.shape[1],
        device=inputs_embeds.device,
    )

    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask_2d,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    if attention_mask is None:
        attention_mask = build_causal_mask(inputs_embeds, attention_mask_2d)

    cos, sin = language_model.rotary_emb(inputs_embeds, full_position_ids[1:])

    return MultimodalRuntimeInputs(
        input_ids=input_ids.detach(),
        attention_mask_2d=None if attention_mask_2d is None else attention_mask_2d.detach(),
        position_ids=full_position_ids.detach(),
        inputs_embeds=inputs_embeds.detach(),
        attention_mask=attention_mask.detach(),
        cos=cos.detach(),
        sin=sin.detach(),
        visual_pos_masks=None,
        deepstack_by_layer={},
    )


def prepare_text_decode_runtime_inputs(
    model,
    *,
    decode_input_ids: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    past_key_values,
) -> MultimodalRuntimeInputs:
    """Build incremental decode tensors for text-only prompts."""

    language_model = model.model.language_model
    device = next(model.parameters()).device

    decode_input_ids = decode_input_ids.to(device)
    attention_mask_2d = attention_mask_2d.to(device)
    inputs_embeds = language_model.get_input_embeddings()(decode_input_ids)

    full_position_ids, text_position_ids = _build_text_position_ids(
        attention_mask_2d=attention_mask_2d,
        batch_size=inputs_embeds.shape[0],
        seq_len=inputs_embeds.shape[1],
        device=inputs_embeds.device,
    )

    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask_2d,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )
    if attention_mask is None:
        attention_mask = build_causal_mask(inputs_embeds, attention_mask_2d)

    cos, sin = language_model.rotary_emb(inputs_embeds, full_position_ids[1:])

    return MultimodalRuntimeInputs(
        input_ids=decode_input_ids.detach(),
        attention_mask_2d=attention_mask_2d.detach(),
        position_ids=full_position_ids.detach(),
        inputs_embeds=inputs_embeds.detach(),
        attention_mask=attention_mask.detach(),
        cos=cos.detach(),
        sin=sin.detach(),
        visual_pos_masks=None,
        deepstack_by_layer={},
    )


def prepare_multimodal_prefill_runtime_inputs(
    model,
    inputs: dict[str, torch.Tensor],
) -> MultimodalRuntimeInputs:
    """Run the live vision front-end and build decoder-ready prefill tensors."""

    model_core = model.model
    language_model = model_core.language_model
    device = next(model.parameters()).device

    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise ValueError("multimodal prefill 需要 input_ids。")

    attention_mask_2d = inputs.get("attention_mask")
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    pixel_values = inputs.get("pixel_values")
    pixel_values_videos = inputs.get("pixel_values_videos")
    image_grid_thw = inputs.get("image_grid_thw")
    video_grid_thw = inputs.get("video_grid_thw")

    inputs_embeds = language_model.get_input_embeddings()(input_ids.to(device))
    inputs_embeds, visual_pos_masks, deepstack_by_layer = materialize_visual_features(
        model,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )

    position_ids = model_core.compute_3d_position_ids(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask_2d,
        past_key_values=None,
        mm_token_type_ids=mm_token_type_ids,
    )
    position_ids, text_position_ids, vision_position_ids = _split_position_ids(
        position_ids,
        batch_size=inputs_embeds.shape[0],
        seq_len=inputs_embeds.shape[1],
        device=inputs_embeds.device,
    )

    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask_2d,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    if attention_mask is None:
        attention_mask = build_causal_mask(inputs_embeds, attention_mask_2d)

    cos, sin = language_model.rotary_emb(inputs_embeds, vision_position_ids)

    return MultimodalRuntimeInputs(
        input_ids=input_ids.detach(),
        attention_mask_2d=None if attention_mask_2d is None else attention_mask_2d.detach(),
        position_ids=position_ids.detach(),
        inputs_embeds=inputs_embeds.detach(),
        attention_mask=attention_mask.detach(),
        cos=cos.detach(),
        sin=sin.detach(),
        visual_pos_masks=None if visual_pos_masks is None else visual_pos_masks.detach(),
        deepstack_by_layer={idx: embed.detach() for idx, embed in deepstack_by_layer.items()},
        mm_token_type_ids=None if mm_token_type_ids is None else mm_token_type_ids.detach(),
        image_grid_thw=None if image_grid_thw is None else image_grid_thw.detach(),
        video_grid_thw=None if video_grid_thw is None else video_grid_thw.detach(),
        pixel_values=None if pixel_values is None else pixel_values.detach(),
        pixel_values_videos=None if pixel_values_videos is None else pixel_values_videos.detach(),
        rope_deltas=None if model_core.rope_deltas is None else model_core.rope_deltas.detach(),
    )


def prepare_multimodal_decode_runtime_inputs(
    model,
    *,
    decode_input_ids: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    past_key_values,
) -> MultimodalRuntimeInputs:
    """Build incremental decode tensors using cached multimodal RoPE state."""

    model_core = model.model
    language_model = model_core.language_model
    device = next(model.parameters()).device

    decode_input_ids = decode_input_ids.to(device)
    attention_mask_2d = attention_mask_2d.to(device)
    inputs_embeds = language_model.get_input_embeddings()(decode_input_ids)
    rope_deltas = model_core.rope_deltas
    if rope_deltas is None:
        raise RuntimeError("multimodal decode 需要先完成 prefill，才能拿到 rope_deltas。")
    position_ids, text_position_ids, vision_position_ids = _build_multimodal_decode_position_ids(
        decode_input_ids=decode_input_ids,
        attention_mask_2d=attention_mask_2d,
        rope_deltas=rope_deltas,
    )

    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask_2d,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )
    if attention_mask is None:
        attention_mask = build_causal_mask(inputs_embeds, attention_mask_2d)

    cos, sin = language_model.rotary_emb(inputs_embeds, vision_position_ids)

    return MultimodalRuntimeInputs(
        input_ids=decode_input_ids.detach(),
        attention_mask_2d=attention_mask_2d.detach(),
        position_ids=position_ids.detach(),
        inputs_embeds=inputs_embeds.detach(),
        attention_mask=attention_mask.detach(),
        cos=cos.detach(),
        sin=sin.detach(),
        visual_pos_masks=None,
        deepstack_by_layer={},
        rope_deltas=rope_deltas.detach(),
    )


__all__ = [
    "prepare_text_prefill_runtime_inputs",
    "prepare_text_decode_runtime_inputs",
    "prepare_multimodal_prefill_runtime_inputs",
    "prepare_multimodal_decode_runtime_inputs",
]
