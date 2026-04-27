"""Multimodal frontend activation and runtime-state helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from qwen3vl_tp_runtime.models.qwen3vl.live.common import _split_position_ids
from qwen3vl_tp_runtime.models.qwen3vl.runtime_mm_stage import (
    MmRuntimeState,
    MmVisualState,
)
from qwen3vl_tp_runtime.models.qwen3vl.vision.frontend import (
    build_mm_frontend_attention_mask,
    build_mm_frontend_rotary,
    compute_mm_frontend_position_ids,
    embed_mm_frontend_inputs,
    merge_mm_frontend_visuals,
    model_device,
    move_frontend_inputs,
)


@dataclass(slots=True)
class MmFrontendVisualPlan:
    """Frontend visual injection result before position and attention planning."""

    inputs_embeds: torch.Tensor
    visual: MmVisualState

    @property
    def visual_pos_masks(self) -> torch.Tensor | None:
        return self.visual.visual_pos_masks

    @property
    def deepstack_by_layer(self) -> dict[int, torch.Tensor]:
        return self.visual.deepstack_by_layer


@dataclass(slots=True)
class MmFrontendPosPlan:
    """Frontend position and RoPE planning result."""

    position_ids: torch.Tensor
    text_position_ids: torch.Tensor | None
    vision_position_ids: torch.Tensor


@dataclass(slots=True)
class MmFrontendAttnPlan:
    """Frontend attention mask and rotary embedding outputs."""

    attention_mask: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


@dataclass(slots=True)
class MmFrontendPlan:
    """Explicit multimodal frontend activation plan for one prefill pass."""

    input_ids: torch.Tensor
    attention_mask_2d: torch.Tensor | None
    visual: MmFrontendVisualPlan
    pos: MmFrontendPosPlan
    attn: MmFrontendAttnPlan
    rope_deltas: torch.Tensor | None = None
    mm_token_type_ids: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None
    pixel_values: torch.Tensor | None = None
    pixel_values_videos: torch.Tensor | None = None

    @property
    def inputs_embeds(self) -> torch.Tensor:
        return self.visual.inputs_embeds

    @property
    def position_ids(self) -> torch.Tensor:
        return self.pos.position_ids

    @property
    def attention_mask(self) -> torch.Tensor:
        return self.attn.attention_mask

    @property
    def cos(self) -> torch.Tensor:
        return self.attn.cos

    @property
    def sin(self) -> torch.Tensor:
        return self.attn.sin

    @property
    def visual_pos_masks(self) -> torch.Tensor | None:
        return self.visual.visual_pos_masks

    @property
    def deepstack_by_layer(self) -> dict[int, torch.Tensor]:
        return self.visual.deepstack_by_layer


def _clone_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().clone()


def build_mm_frontend_visual_plan(
    model,
    *,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor,
    pixel_values: torch.Tensor | None,
    pixel_values_videos: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
) -> MmFrontendVisualPlan:
    inputs_embeds, visual_pos_masks, deepstack_by_layer = merge_mm_frontend_visuals(
        model,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )

    return MmFrontendVisualPlan(
        inputs_embeds=inputs_embeds,
        visual=MmVisualState(
            visual_pos_masks=visual_pos_masks,
            deepstack_by_layer=deepstack_by_layer,
        ),
    )


def build_mm_frontend_pos_plan(
    model,
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask_2d: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
) -> MmFrontendPosPlan:
    position_ids = compute_mm_frontend_position_ids(
        model,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask_2d=attention_mask_2d,
        mm_token_type_ids=mm_token_type_ids,
    )
    full_position_ids, text_position_ids, vision_position_ids = _split_position_ids(
        position_ids,
        batch_size=inputs_embeds.shape[0],
        seq_len=inputs_embeds.shape[1],
        device=inputs_embeds.device,
    )
    return MmFrontendPosPlan(
        position_ids=full_position_ids,
        text_position_ids=text_position_ids,
        vision_position_ids=vision_position_ids,
    )


def build_mm_frontend_attn_plan(
    model,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask_2d: torch.Tensor | None,
    pos_plan: MmFrontendPosPlan,
) -> MmFrontendAttnPlan:
    attention_mask = build_mm_frontend_attention_mask(
        model,
        inputs_embeds=inputs_embeds,
        attention_mask_2d=attention_mask_2d,
        text_position_ids=pos_plan.text_position_ids,
    )
    cos, sin = build_mm_frontend_rotary(
        model,
        inputs_embeds=inputs_embeds,
        vision_position_ids=pos_plan.vision_position_ids,
    )
    return MmFrontendAttnPlan(
        attention_mask=attention_mask,
        cos=cos,
        sin=sin,
    )


def build_mm_frontend_plan(
    model,
    inputs: dict[str, Any],
    *,
    inputs_on_device: bool = False,
) -> MmFrontendPlan:
    """Build one explicit multimodal frontend activation plan."""

    device = model_device(model)
    if not inputs_on_device:
        inputs = move_frontend_inputs(inputs, device=device)

    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise ValueError("multimodal prefill 需要 input_ids。")

    attention_mask_2d = inputs.get("attention_mask")
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    pixel_values = inputs.get("pixel_values")
    pixel_values_videos = inputs.get("pixel_values_videos")
    image_grid_thw = inputs.get("image_grid_thw")
    video_grid_thw = inputs.get("video_grid_thw")

    inputs_embeds = embed_mm_frontend_inputs(model, input_ids=input_ids)
    visual_plan = build_mm_frontend_visual_plan(
        model,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )
    pos_plan = build_mm_frontend_pos_plan(
        model,
        input_ids=input_ids,
        inputs_embeds=visual_plan.inputs_embeds,
        attention_mask_2d=attention_mask_2d,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )
    attn_plan = build_mm_frontend_attn_plan(
        model,
        inputs_embeds=visual_plan.inputs_embeds,
        attention_mask_2d=attention_mask_2d,
        pos_plan=pos_plan,
    )
    return MmFrontendPlan(
        input_ids=input_ids,
        attention_mask_2d=attention_mask_2d,
        visual=visual_plan,
        pos=pos_plan,
        attn=attn_plan,
        rope_deltas=getattr(model.model, "rope_deltas", None),
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
    )


def mm_state_from_frontend_plan(plan: MmFrontendPlan) -> MmRuntimeState:
    """Freeze one frontend activation plan into detached multimodal runtime state."""

    return MmRuntimeState(
        input_ids=plan.input_ids.detach(),
        attention_mask_2d=None if plan.attention_mask_2d is None else plan.attention_mask_2d.detach(),
        position_ids=plan.pos.position_ids.detach(),
        inputs_embeds=plan.inputs_embeds.detach(),
        attention_mask=plan.attn.attention_mask.detach(),
        cos=plan.attn.cos.detach(),
        sin=plan.attn.sin.detach(),
        visual=MmVisualState(
            visual_pos_masks=None if plan.visual.visual_pos_masks is None else plan.visual.visual_pos_masks.detach(),
            deepstack_by_layer={
                idx: embed.detach()
                for idx, embed in plan.visual.deepstack_by_layer.items()
            },
        ),
        mm_token_type_ids=None if plan.mm_token_type_ids is None else plan.mm_token_type_ids.detach(),
        image_grid_thw=None if plan.image_grid_thw is None else plan.image_grid_thw.detach(),
        video_grid_thw=None if plan.video_grid_thw is None else plan.video_grid_thw.detach(),
        pixel_values=None if plan.pixel_values is None else plan.pixel_values.detach(),
        pixel_values_videos=None if plan.pixel_values_videos is None else plan.pixel_values_videos.detach(),
        rope_deltas=None if plan.rope_deltas is None else plan.rope_deltas.detach(),
    )


def clone_mm_frontend_plan(plan: MmFrontendPlan) -> MmFrontendPlan:
    """Detach and clone one frontend activation plan."""

    return MmFrontendPlan(
        input_ids=plan.input_ids.detach().clone(),
        attention_mask_2d=_clone_tensor(plan.attention_mask_2d),
        visual=MmFrontendVisualPlan(
            inputs_embeds=plan.visual.inputs_embeds.detach().clone(),
            visual=MmVisualState(
                visual_pos_masks=_clone_tensor(plan.visual.visual_pos_masks),
                deepstack_by_layer={
                    int(layer_idx): embed.detach().clone()
                    for layer_idx, embed in plan.visual.deepstack_by_layer.items()
                },
            ),
        ),
        pos=MmFrontendPosPlan(
            position_ids=plan.pos.position_ids.detach().clone(),
            text_position_ids=_clone_tensor(plan.pos.text_position_ids),
            vision_position_ids=plan.pos.vision_position_ids.detach().clone(),
        ),
        attn=MmFrontendAttnPlan(
            attention_mask=plan.attn.attention_mask.detach().clone(),
            cos=plan.attn.cos.detach().clone(),
            sin=plan.attn.sin.detach().clone(),
        ),
        rope_deltas=_clone_tensor(plan.rope_deltas),
        mm_token_type_ids=_clone_tensor(plan.mm_token_type_ids),
        image_grid_thw=_clone_tensor(plan.image_grid_thw),
        video_grid_thw=_clone_tensor(plan.video_grid_thw),
        pixel_values=_clone_tensor(plan.pixel_values),
        pixel_values_videos=_clone_tensor(plan.pixel_values_videos),
    )


def move_mm_frontend_plan(
    plan: MmFrontendPlan,
    *,
    device: torch.device,
) -> MmFrontendPlan:
    """Move one frontend activation plan to the target device."""

    def _move_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.detach().to(device=device)

    return MmFrontendPlan(
        input_ids=plan.input_ids.detach().to(device=device),
        attention_mask_2d=_move_tensor(plan.attention_mask_2d),
        visual=MmFrontendVisualPlan(
            inputs_embeds=plan.visual.inputs_embeds.detach().to(device=device),
            visual=MmVisualState(
                visual_pos_masks=_move_tensor(plan.visual.visual_pos_masks),
                deepstack_by_layer={
                    int(layer_idx): embed.detach().to(device=device)
                    for layer_idx, embed in plan.visual.deepstack_by_layer.items()
                },
            ),
        ),
        pos=MmFrontendPosPlan(
            position_ids=plan.pos.position_ids.detach().to(device=device),
            text_position_ids=_move_tensor(plan.pos.text_position_ids),
            vision_position_ids=plan.pos.vision_position_ids.detach().to(device=device),
        ),
        attn=MmFrontendAttnPlan(
            attention_mask=plan.attn.attention_mask.detach().to(device=device),
            cos=plan.attn.cos.detach().to(device=device),
            sin=plan.attn.sin.detach().to(device=device),
        ),
        rope_deltas=_move_tensor(plan.rope_deltas),
        mm_token_type_ids=_move_tensor(plan.mm_token_type_ids),
        image_grid_thw=_move_tensor(plan.image_grid_thw),
        video_grid_thw=_move_tensor(plan.video_grid_thw),
        pixel_values=_move_tensor(plan.pixel_values),
        pixel_values_videos=_move_tensor(plan.pixel_values_videos),
    )


def prepare_mm_frontend_state(
    model,
    inputs: dict[str, Any],
    *,
    inputs_on_device: bool = False,
) -> MmRuntimeState:
    """Build explicit multimodal runtime state from one frontend activation."""

    return mm_state_from_frontend_plan(
        build_mm_frontend_plan(
            model,
            inputs,
            inputs_on_device=inputs_on_device,
        )
    )


__all__ = [
    "MmFrontendAttnPlan",
    "MmFrontendPlan",
    "MmFrontendPosPlan",
    "MmFrontendVisualPlan",
    "build_mm_frontend_attn_plan",
    "build_mm_frontend_plan",
    "build_mm_frontend_pos_plan",
    "build_mm_frontend_visual_plan",
    "clone_mm_frontend_plan",
    "model_device",
    "mm_state_from_frontend_plan",
    "move_mm_frontend_plan",
    "move_frontend_inputs",
    "prepare_mm_frontend_state",
]
