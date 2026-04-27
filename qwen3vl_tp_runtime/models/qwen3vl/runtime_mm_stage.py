"""Explicit multimodal runtime state used by direct stage builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask

from qwen3vl_tp_runtime.models.qwen3vl.live.common import (
    MultimodalRuntimeInputs,
    _build_multimodal_decode_position_ids,
    _split_position_ids,
    _runtime_tensor,
)
from qwen3vl_tp_runtime.models.qwen3vl.vision.frontend import (
    compute_mm_frontend_rope_index,
)
from qwen3vl_tp_runtime.models.qwen3vl.weights.text import (
    TextModelConfigSpec,
    build_text_causal_mask,
    build_text_hf_config,
    build_text_rotary_embedding,
)


@dataclass(slots=True)
class MmVisualState:
    """Visual frontend payload that decoder stages consume explicitly."""

    visual_pos_masks: torch.Tensor | None
    deepstack_by_layer: dict[int, torch.Tensor]


@dataclass(slots=True)
class MmRuntimeState:
    """Decoder-ready multimodal runtime tensors plus explicit visual state."""

    input_ids: torch.Tensor | None
    attention_mask_2d: torch.Tensor | None
    position_ids: torch.Tensor | None
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    visual: MmVisualState
    rope_deltas: torch.Tensor | None = None
    mm_token_type_ids: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None
    pixel_values: torch.Tensor | None = None
    pixel_values_videos: torch.Tensor | None = None

    @property
    def visual_pos_masks(self) -> torch.Tensor | None:
        return self.visual.visual_pos_masks

    @property
    def deepstack_by_layer(self) -> dict[int, torch.Tensor]:
        return self.visual.deepstack_by_layer


@dataclass(slots=True)
class MmFrontendSeed:
    """Thin cross-rank multimodal frontend payload: runtime tensors only."""

    input_ids: torch.Tensor | None
    attention_mask_2d: torch.Tensor | None
    position_ids: torch.Tensor | None
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    visual_pos_masks: torch.Tensor | None
    deepstack_by_layer: dict[int, torch.Tensor]
    rope_deltas: torch.Tensor | None = None
    mm_token_type_ids: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None


MmStateLike = MultimodalRuntimeInputs | MmRuntimeState | MmFrontendSeed


def mm_state_from_runtime_inputs(runtime_inputs: MultimodalRuntimeInputs) -> MmRuntimeState:
    return MmRuntimeState(
        input_ids=runtime_inputs.input_ids,
        attention_mask_2d=runtime_inputs.attention_mask_2d,
        position_ids=runtime_inputs.position_ids,
        inputs_embeds=runtime_inputs.inputs_embeds,
        attention_mask=runtime_inputs.attention_mask,
        cos=runtime_inputs.cos,
        sin=runtime_inputs.sin,
        visual=MmVisualState(
            visual_pos_masks=runtime_inputs.visual_pos_masks,
            deepstack_by_layer={
                int(layer_idx): deepstack
                for layer_idx, deepstack in runtime_inputs.deepstack_by_layer.items()
            },
        ),
        rope_deltas=runtime_inputs.rope_deltas,
        mm_token_type_ids=runtime_inputs.mm_token_type_ids,
        image_grid_thw=runtime_inputs.image_grid_thw,
        video_grid_thw=runtime_inputs.video_grid_thw,
        pixel_values=runtime_inputs.pixel_values,
        pixel_values_videos=runtime_inputs.pixel_values_videos,
    )


def coerce_mm_runtime_state(value: MmStateLike) -> MmRuntimeState:
    if isinstance(value, MmRuntimeState):
        return value
    if all(
        hasattr(value, attr)
        for attr in (
            "input_ids",
            "attention_mask_2d",
            "position_ids",
            "inputs_embeds",
            "attention_mask",
            "cos",
            "sin",
            "visual_pos_masks",
            "deepstack_by_layer",
        )
    ):
        return MmRuntimeState(
            input_ids=getattr(value, "input_ids"),
            attention_mask_2d=getattr(value, "attention_mask_2d"),
            position_ids=getattr(value, "position_ids"),
            inputs_embeds=getattr(value, "inputs_embeds"),
            attention_mask=getattr(value, "attention_mask"),
            cos=getattr(value, "cos"),
            sin=getattr(value, "sin"),
            visual=MmVisualState(
                visual_pos_masks=getattr(value, "visual_pos_masks"),
                deepstack_by_layer={
                    int(layer_idx): deepstack
                    for layer_idx, deepstack in getattr(value, "deepstack_by_layer").items()
                },
            ),
            rope_deltas=getattr(value, "rope_deltas", None),
            mm_token_type_ids=getattr(value, "mm_token_type_ids", None),
            image_grid_thw=getattr(value, "image_grid_thw", None),
            video_grid_thw=getattr(value, "video_grid_thw", None),
            pixel_values=getattr(value, "pixel_values", None),
            pixel_values_videos=getattr(value, "pixel_values_videos", None),
        )
    return mm_state_from_runtime_inputs(value)


def mm_runtime_inputs_from_state(state: MmStateLike) -> MultimodalRuntimeInputs:
    runtime_state = coerce_mm_runtime_state(state)
    return MultimodalRuntimeInputs(
        input_ids=runtime_state.input_ids,
        attention_mask_2d=runtime_state.attention_mask_2d,
        position_ids=runtime_state.position_ids,
        inputs_embeds=runtime_state.inputs_embeds,
        attention_mask=runtime_state.attention_mask,
        cos=runtime_state.cos,
        sin=runtime_state.sin,
        visual_pos_masks=runtime_state.visual_pos_masks,
        deepstack_by_layer={
            int(layer_idx): deepstack
            for layer_idx, deepstack in runtime_state.deepstack_by_layer.items()
        },
        mm_token_type_ids=runtime_state.mm_token_type_ids,
        image_grid_thw=runtime_state.image_grid_thw,
        video_grid_thw=runtime_state.video_grid_thw,
        pixel_values=runtime_state.pixel_values,
        pixel_values_videos=runtime_state.pixel_values_videos,
        rope_deltas=runtime_state.rope_deltas,
    )


def mm_frontend_seed_from_state(state: Any) -> MmFrontendSeed:
    runtime_state = coerce_mm_runtime_state(state)
    return MmFrontendSeed(
        input_ids=runtime_state.input_ids,
        attention_mask_2d=runtime_state.attention_mask_2d,
        position_ids=runtime_state.position_ids,
        inputs_embeds=runtime_state.inputs_embeds,
        attention_mask=runtime_state.attention_mask,
        cos=runtime_state.cos,
        sin=runtime_state.sin,
        visual_pos_masks=runtime_state.visual_pos_masks,
        deepstack_by_layer={
            int(layer_idx): deepstack
            for layer_idx, deepstack in runtime_state.deepstack_by_layer.items()
        },
        rope_deltas=runtime_state.rope_deltas,
        mm_token_type_ids=runtime_state.mm_token_type_ids,
        image_grid_thw=runtime_state.image_grid_thw,
        video_grid_thw=runtime_state.video_grid_thw,
    )


def _clone_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().clone()


def clone_mm_state(state: MmStateLike) -> MmRuntimeState:
    runtime_state = coerce_mm_runtime_state(state)
    return MmRuntimeState(
        input_ids=_clone_tensor(runtime_state.input_ids),
        attention_mask_2d=_clone_tensor(runtime_state.attention_mask_2d),
        position_ids=_clone_tensor(runtime_state.position_ids),
        inputs_embeds=runtime_state.inputs_embeds.detach().clone(),
        attention_mask=runtime_state.attention_mask.detach().clone(),
        cos=runtime_state.cos.detach().clone(),
        sin=runtime_state.sin.detach().clone(),
        visual=MmVisualState(
            visual_pos_masks=_clone_tensor(runtime_state.visual_pos_masks),
            deepstack_by_layer={
                int(layer_idx): deepstack.detach().clone()
                for layer_idx, deepstack in runtime_state.deepstack_by_layer.items()
            },
        ),
        rope_deltas=_clone_tensor(runtime_state.rope_deltas),
        mm_token_type_ids=_clone_tensor(runtime_state.mm_token_type_ids),
        image_grid_thw=_clone_tensor(runtime_state.image_grid_thw),
        video_grid_thw=_clone_tensor(runtime_state.video_grid_thw),
        pixel_values=_clone_tensor(runtime_state.pixel_values),
        pixel_values_videos=_clone_tensor(runtime_state.pixel_values_videos),
    )


def clone_mm_frontend_seed(state: Any) -> MmFrontendSeed:
    seed = mm_frontend_seed_from_state(state)
    return MmFrontendSeed(
        input_ids=_clone_tensor(seed.input_ids),
        attention_mask_2d=_clone_tensor(seed.attention_mask_2d),
        position_ids=_clone_tensor(seed.position_ids),
        inputs_embeds=seed.inputs_embeds.detach().clone(),
        attention_mask=seed.attention_mask.detach().clone(),
        cos=seed.cos.detach().clone(),
        sin=seed.sin.detach().clone(),
        visual_pos_masks=_clone_tensor(seed.visual_pos_masks),
        deepstack_by_layer={
            int(layer_idx): deepstack.detach().clone()
            for layer_idx, deepstack in seed.deepstack_by_layer.items()
        },
        rope_deltas=_clone_tensor(seed.rope_deltas),
        mm_token_type_ids=_clone_tensor(seed.mm_token_type_ids),
        image_grid_thw=_clone_tensor(seed.image_grid_thw),
        video_grid_thw=_clone_tensor(seed.video_grid_thw),
    )


def compact_mm_frontend_tensors(
    state: Any,
    *,
    include_derived: bool = False,
) -> dict[str, Any]:
    if isinstance(state, Mapping):
        compact_payload = {
            "inputs_embeds": state["inputs_embeds"].detach().clone(),
            "visual_pos_masks": _clone_tensor(state.get("visual_pos_masks")),
            "deepstack_by_layer": {
                int(layer_idx): deepstack.detach().clone()
                for layer_idx, deepstack in (state.get("deepstack_by_layer") or {}).items()
            },
        }
        if include_derived:
            if "attention_mask" in state:
                compact_payload["attention_mask"] = state["attention_mask"].detach().clone()
            if "cos" in state:
                compact_payload["cos"] = state["cos"].detach().clone()
            if "sin" in state:
                compact_payload["sin"] = state["sin"].detach().clone()
        return compact_payload

    seed = clone_mm_frontend_seed(state)
    return {
        "inputs_embeds": seed.inputs_embeds,
        "visual_pos_masks": seed.visual_pos_masks,
        "deepstack_by_layer": {
            int(layer_idx): deepstack
            for layer_idx, deepstack in seed.deepstack_by_layer.items()
        },
        **(
            {
                "attention_mask": seed.attention_mask,
                "cos": seed.cos,
                "sin": seed.sin,
            }
            if include_derived
            else {}
        ),
    }


def compact_mm_frontend_meta(
    state: Any,
    *,
    include_derived: bool = False,
) -> dict[str, Any]:
    if isinstance(state, Mapping):
        compact_payload = {
            "input_ids": _clone_tensor(state.get("input_ids")),
            "attention_mask_2d": _clone_tensor(state.get("attention_mask_2d")),
            "mm_token_type_ids": _clone_tensor(state.get("mm_token_type_ids")),
            "image_grid_thw": _clone_tensor(state.get("image_grid_thw")),
            "video_grid_thw": _clone_tensor(state.get("video_grid_thw")),
        }
        if include_derived:
            compact_payload["position_ids"] = _clone_tensor(state.get("position_ids"))
            compact_payload["rope_deltas"] = _clone_tensor(state.get("rope_deltas"))
        return compact_payload

    seed = clone_mm_frontend_seed(state)
    return {
        "input_ids": seed.input_ids,
        "attention_mask_2d": seed.attention_mask_2d,
        "mm_token_type_ids": seed.mm_token_type_ids,
        "image_grid_thw": seed.image_grid_thw,
        "video_grid_thw": seed.video_grid_thw,
        **(
            {
                "position_ids": seed.position_ids,
                "rope_deltas": seed.rope_deltas,
            }
            if include_derived
            else {}
        ),
    }


def compact_mm_frontend_seed(
    state: Any,
    *,
    include_derived: bool = False,
) -> dict[str, Any]:
    compact_payload = compact_mm_frontend_tensors(
        state,
        include_derived=include_derived,
    )
    compact_payload.update(
        compact_mm_frontend_meta(
            state,
            include_derived=include_derived,
        )
    )
    return compact_payload


def compact_mm_runtime_shared(state: Any) -> dict[str, Any]:
    """Keep only the shared multimodal decoder runtime tensors."""

    runtime_state = coerce_mm_runtime_state(state)
    return {
        "input_ids": _clone_tensor(runtime_state.input_ids),
        "attention_mask_2d": _clone_tensor(runtime_state.attention_mask_2d),
        "position_ids": _clone_tensor(runtime_state.position_ids),
        "attention_mask": _clone_tensor(runtime_state.attention_mask),
        "cos": _clone_tensor(runtime_state.cos),
        "sin": _clone_tensor(runtime_state.sin),
        "rope_deltas": _clone_tensor(runtime_state.rope_deltas),
        "mm_token_type_ids": _clone_tensor(runtime_state.mm_token_type_ids),
        "image_grid_thw": _clone_tensor(runtime_state.image_grid_thw),
        "video_grid_thw": _clone_tensor(runtime_state.video_grid_thw),
    }


def restore_mm_frontend_seed_tensors(
    payload: Mapping[str, Any],
    *,
    config_spec: TextModelConfigSpec | None = None,
    mm_config=None,
    device: torch.device | None = None,
    compute_dtype: torch.dtype | None = None,
    rotary_emb=None,
) -> MmFrontendSeed:
    deepstack_by_layer = payload.get("deepstack_by_layer") or {}
    inputs_embeds = payload["inputs_embeds"].detach().clone()
    restore_device = inputs_embeds.device if device is None else device
    restore_compute_dtype = inputs_embeds.dtype if compute_dtype is None else compute_dtype
    inputs_embeds = _runtime_tensor(
        inputs_embeds,
        device=restore_device,
        compute_dtype=restore_compute_dtype,
    )
    if inputs_embeds is None:
        raise RuntimeError("multimodal frontend seed 缺少 inputs_embeds。")

    input_ids = _runtime_tensor(payload.get("input_ids"), device=restore_device)
    attention_mask_2d = _runtime_tensor(payload.get("attention_mask_2d"), device=restore_device)
    position_ids = _runtime_tensor(payload.get("position_ids"), device=restore_device)
    visual_pos_masks = _runtime_tensor(payload.get("visual_pos_masks"), device=restore_device)
    rope_deltas = _runtime_tensor(payload.get("rope_deltas"), device=restore_device)
    mm_token_type_ids = _runtime_tensor(payload.get("mm_token_type_ids"), device=restore_device)
    image_grid_thw = _runtime_tensor(payload.get("image_grid_thw"), device=restore_device)
    video_grid_thw = _runtime_tensor(payload.get("video_grid_thw"), device=restore_device)
    restored_deepstack = {
        int(layer_idx): _runtime_tensor(
            deepstack,
            device=restore_device,
            compute_dtype=restore_compute_dtype,
        )
        for layer_idx, deepstack in deepstack_by_layer.items()
    }

    if (
        (position_ids is None or rope_deltas is None)
        and mm_config is not None
        and input_ids is not None
        and mm_token_type_ids is not None
        and (image_grid_thw is not None or video_grid_thw is not None)
    ):
        mm_position_ids, mm_rope_deltas = compute_mm_frontend_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask_2d,
            spatial_merge_size=int(mm_config.vision_config.spatial_merge_size),
        )
        if position_ids is None:
            position_ids = mm_position_ids
        if rope_deltas is None:
            rope_deltas = mm_rope_deltas

    full_position_ids, text_position_ids, vision_position_ids = _split_position_ids(
        position_ids,
        batch_size=inputs_embeds.shape[0],
        seq_len=inputs_embeds.shape[1],
        device=restore_device,
    )
    position_ids = full_position_ids.detach()

    attention_mask = payload.get("attention_mask")
    cos = payload.get("cos")
    sin = payload.get("sin")
    if attention_mask is None or cos is None or sin is None:
        if config_spec is None:
            raise RuntimeError(
                "restore multimodal frontend seed 需要 config_spec，"
                "因为当前 payload 没有 attention_mask/cos/sin。"
            )
        attention_mask = create_causal_mask(
            config=build_text_hf_config(config_spec),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_2d,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        if attention_mask is None:
            attention_mask = build_text_causal_mask(
                inputs_embeds,
                attention_mask_2d=attention_mask_2d,
                past_length=0,
            )
        rotary = rotary_emb or build_text_rotary_embedding(
            config_spec,
            device=restore_device,
        )
        rotary = rotary.to(device=restore_device)
        cos, sin = rotary(inputs_embeds, vision_position_ids)

    return MmFrontendSeed(
        input_ids=input_ids,
        attention_mask_2d=attention_mask_2d,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=_runtime_tensor(
            attention_mask,
            device=restore_device,
            compute_dtype=restore_compute_dtype,
        ),
        cos=_runtime_tensor(
            cos,
            device=restore_device,
            compute_dtype=restore_compute_dtype,
        ),
        sin=_runtime_tensor(
            sin,
            device=restore_device,
            compute_dtype=restore_compute_dtype,
        ),
        visual_pos_masks=visual_pos_masks,
        deepstack_by_layer=restored_deepstack,
        rope_deltas=rope_deltas,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )


def move_mm_state(
    state: MmStateLike,
    *,
    device: torch.device,
) -> MmRuntimeState:
    runtime_state = coerce_mm_runtime_state(state)
    return MmRuntimeState(
        input_ids=_runtime_tensor(runtime_state.input_ids, device=device),
        attention_mask_2d=_runtime_tensor(runtime_state.attention_mask_2d, device=device),
        position_ids=_runtime_tensor(runtime_state.position_ids, device=device),
        inputs_embeds=_runtime_tensor(runtime_state.inputs_embeds, device=device),
        attention_mask=_runtime_tensor(runtime_state.attention_mask, device=device),
        cos=_runtime_tensor(runtime_state.cos, device=device),
        sin=_runtime_tensor(runtime_state.sin, device=device),
        visual=MmVisualState(
            visual_pos_masks=_runtime_tensor(runtime_state.visual_pos_masks, device=device),
            deepstack_by_layer={
                int(layer_idx): _runtime_tensor(deepstack, device=device)
                for layer_idx, deepstack in runtime_state.deepstack_by_layer.items()
            },
        ),
        rope_deltas=_runtime_tensor(runtime_state.rope_deltas, device=device),
        mm_token_type_ids=_runtime_tensor(runtime_state.mm_token_type_ids, device=device),
        image_grid_thw=_runtime_tensor(runtime_state.image_grid_thw, device=device),
        video_grid_thw=_runtime_tensor(runtime_state.video_grid_thw, device=device),
        pixel_values=_runtime_tensor(runtime_state.pixel_values, device=device),
        pixel_values_videos=_runtime_tensor(runtime_state.pixel_values_videos, device=device),
    )


def move_mm_frontend_seed(
    state: Any,
    *,
    device: torch.device,
) -> MmFrontendSeed:
    seed = mm_frontend_seed_from_state(state)
    return MmFrontendSeed(
        input_ids=_runtime_tensor(seed.input_ids, device=device),
        attention_mask_2d=_runtime_tensor(seed.attention_mask_2d, device=device),
        position_ids=_runtime_tensor(seed.position_ids, device=device),
        inputs_embeds=_runtime_tensor(seed.inputs_embeds, device=device),
        attention_mask=_runtime_tensor(seed.attention_mask, device=device),
        cos=_runtime_tensor(seed.cos, device=device),
        sin=_runtime_tensor(seed.sin, device=device),
        visual_pos_masks=_runtime_tensor(seed.visual_pos_masks, device=device),
        deepstack_by_layer={
            int(layer_idx): _runtime_tensor(deepstack, device=device)
            for layer_idx, deepstack in seed.deepstack_by_layer.items()
        },
        rope_deltas=_runtime_tensor(seed.rope_deltas, device=device),
        mm_token_type_ids=_runtime_tensor(seed.mm_token_type_ids, device=device),
        image_grid_thw=_runtime_tensor(seed.image_grid_thw, device=device),
        video_grid_thw=_runtime_tensor(seed.video_grid_thw, device=device),
    )


def _decode_state_inputs_embeds(state: Mapping[str, Any]) -> torch.Tensor:
    inputs_embeds = state.get("inputs_embeds")
    if torch.is_tensor(inputs_embeds):
        return inputs_embeds

    hidden_states = state.get("hidden_states")
    if isinstance(hidden_states, (tuple, list)) and hidden_states:
        first_hidden = hidden_states[0]
        if torch.is_tensor(first_hidden):
            return first_hidden

    stage_handoffs = state.get("stage_handoffs")
    if isinstance(stage_handoffs, Mapping):
        preferred_payload = stage_handoffs.get(0)
        if isinstance(preferred_payload, Mapping) and torch.is_tensor(preferred_payload.get("stage_input")):
            return preferred_payload["stage_input"]
        for stage_idx in sorted(stage_handoffs):
            stage_payload = stage_handoffs[stage_idx]
            if isinstance(stage_payload, Mapping) and torch.is_tensor(stage_payload.get("stage_input")):
                return stage_payload["stage_input"]

    raise RuntimeError("decode state 缺少 inputs_embeds/hidden_states/stage_handoffs，无法恢复 multimodal state。")


def mm_state_from_decode_state(state: Mapping[str, Any]) -> MmRuntimeState:
    existing_state = state.get("mm_runtime_state")
    if existing_state is not None:
        return coerce_mm_runtime_state(existing_state)
    return MmRuntimeState(
        input_ids=state.get("decode_input_ids"),
        attention_mask_2d=state.get("attention_mask_2d"),
        position_ids=state.get("position_ids"),
        inputs_embeds=_decode_state_inputs_embeds(state),
        attention_mask=state["attention_mask"],
        cos=state["cos"],
        sin=state["sin"],
        visual=MmVisualState(
            visual_pos_masks=state.get("visual_pos_masks"),
            deepstack_by_layer={
                int(layer_idx): deepstack
                for layer_idx, deepstack in state.get("deepstack_by_layer", {}).items()
            },
        ),
    )


def mm_text_position_ids(state: MmStateLike) -> torch.Tensor | None:
    position_ids = coerce_mm_runtime_state(state).position_ids
    if position_ids is None:
        return None
    if position_ids.ndim == 2:
        return position_ids
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        return position_ids[0]
    return None


def mm_deepstack_embeds(state: MmStateLike) -> list[torch.Tensor | None] | None:
    runtime_state = coerce_mm_runtime_state(state)
    if not runtime_state.deepstack_by_layer:
        return None
    max_layer_idx = max(int(layer_idx) for layer_idx in runtime_state.deepstack_by_layer)
    return [runtime_state.deepstack_by_layer.get(layer_idx) for layer_idx in range(max_layer_idx + 1)]


def build_mm_stage_visual_payload(
    state: MmStateLike,
    *,
    start_idx: int,
    end_idx: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor | None, dict[int, torch.Tensor]]:
    runtime_state = coerce_mm_runtime_state(state)
    stage_deepstack = {
        int(layer_idx): _runtime_tensor(deepstack_embed, device=device, compute_dtype=compute_dtype)
        for layer_idx, deepstack_embed in runtime_state.deepstack_by_layer.items()
        if start_idx <= int(layer_idx) <= end_idx
    }
    visual_pos_masks = runtime_state.visual_pos_masks
    if not stage_deepstack:
        visual_pos_masks = None
    return _runtime_tensor(visual_pos_masks, device=device), stage_deepstack


def build_mm_stage_state(
    state: MmStateLike | Mapping[str, Any],
    *,
    stage_input: torch.Tensor,
    start_idx: int,
    end_idx: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    visual_pos_masks: torch.Tensor | None = None,
    deepstack_by_layer: dict[int, torch.Tensor] | None = None,
) -> MmRuntimeState:
    """Build one stage-scoped multimodal runtime state around handoff activation."""

    if isinstance(state, Mapping):
        shared_state = state
    else:
        shared_state = compact_mm_runtime_shared(state)

    if visual_pos_masks is None and deepstack_by_layer is None:
        base_visual_pos_masks, base_deepstack = build_mm_stage_visual_payload(
            state,
            start_idx=start_idx,
            end_idx=end_idx,
            device=device,
            compute_dtype=compute_dtype,
        )
        if visual_pos_masks is None:
            visual_pos_masks = base_visual_pos_masks
        if deepstack_by_layer is None:
            deepstack_by_layer = base_deepstack

    return MmRuntimeState(
        input_ids=_runtime_tensor(shared_state.get("input_ids"), device=device),
        attention_mask_2d=_runtime_tensor(shared_state.get("attention_mask_2d"), device=device),
        position_ids=_runtime_tensor(shared_state.get("position_ids"), device=device),
        inputs_embeds=_runtime_tensor(stage_input, device=device, compute_dtype=compute_dtype),
        attention_mask=_runtime_tensor(
            shared_state.get("attention_mask"),
            device=device,
            compute_dtype=compute_dtype,
        ),
        cos=_runtime_tensor(
            shared_state.get("cos"),
            device=device,
            compute_dtype=compute_dtype,
        ),
        sin=_runtime_tensor(
            shared_state.get("sin"),
            device=device,
            compute_dtype=compute_dtype,
        ),
        visual=MmVisualState(
            visual_pos_masks=_runtime_tensor(visual_pos_masks, device=device),
            deepstack_by_layer={
                int(layer_idx): _runtime_tensor(
                    deepstack,
                    device=device,
                    compute_dtype=compute_dtype,
                )
                for layer_idx, deepstack in (deepstack_by_layer or {}).items()
            },
        ),
        rope_deltas=_runtime_tensor(shared_state.get("rope_deltas"), device=device),
        mm_token_type_ids=_runtime_tensor(shared_state.get("mm_token_type_ids"), device=device),
        image_grid_thw=_runtime_tensor(shared_state.get("image_grid_thw"), device=device),
        video_grid_thw=_runtime_tensor(shared_state.get("video_grid_thw"), device=device),
    )


def build_mm_decode_state_from_weights(
    *,
    decode_input_ids: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    past_length: int,
    rope_deltas: torch.Tensor,
    embed_tokens_weight: torch.Tensor,
    config_spec: TextModelConfigSpec,
    device: torch.device,
    compute_dtype: torch.dtype,
    rotary_emb=None,
) -> MmRuntimeState:
    decode_input_ids = decode_input_ids.to(device=device)
    attention_mask_2d = attention_mask_2d.to(device=device)
    embed_weight = _runtime_tensor(
        embed_tokens_weight,
        device=device,
        compute_dtype=compute_dtype,
    )
    if embed_weight is None:
        raise RuntimeError("multimodal decode 需要 embed_tokens_weight。")

    inputs_embeds = F.embedding(decode_input_ids, embed_weight)
    position_ids, _text_position_ids, vision_position_ids = _build_multimodal_decode_position_ids(
        decode_input_ids=decode_input_ids,
        attention_mask_2d=attention_mask_2d,
        rope_deltas=rope_deltas,
    )
    attention_mask = build_text_causal_mask(
        inputs_embeds,
        attention_mask_2d=attention_mask_2d,
        past_length=past_length,
    )
    rotary = rotary_emb or build_text_rotary_embedding(config_spec, device=device)
    rotary = rotary.to(device=device)
    cos, sin = rotary(inputs_embeds, vision_position_ids)
    return MmRuntimeState(
        input_ids=decode_input_ids.detach(),
        attention_mask_2d=attention_mask_2d.detach(),
        position_ids=position_ids.detach(),
        inputs_embeds=inputs_embeds.detach(),
        attention_mask=attention_mask.detach(),
        cos=cos.detach(),
        sin=sin.detach(),
        visual=MmVisualState(
            visual_pos_masks=None,
            deepstack_by_layer={},
        ),
        rope_deltas=rope_deltas.detach().to(device=device),
    )


__all__ = [
    "MmFrontendSeed",
    "MmStateLike",
    "MmRuntimeState",
    "MmVisualState",
    "build_mm_stage_state",
    "build_mm_stage_visual_payload",
    "build_mm_decode_state_from_weights",
    "compact_mm_frontend_meta",
    "compact_mm_frontend_seed",
    "compact_mm_frontend_tensors",
    "compact_mm_runtime_shared",
    "clone_mm_frontend_seed",
    "clone_mm_state",
    "coerce_mm_runtime_state",
    "mm_frontend_seed_from_state",
    "mm_deepstack_embeds",
    "mm_runtime_inputs_from_state",
    "mm_state_from_decode_state",
    "mm_state_from_runtime_inputs",
    "mm_text_position_ids",
    "move_mm_frontend_seed",
    "move_mm_state",
    "restore_mm_frontend_seed_tensors",
]
