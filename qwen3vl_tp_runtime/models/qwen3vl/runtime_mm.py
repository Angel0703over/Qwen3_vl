"""Multimodal live-session helpers for direct runtime building."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from qwen3vl_tp_runtime.models.qwen3vl.execution import apply_deepstack
from qwen3vl_tp_runtime.models.qwen3vl.live import (
    prepare_multimodal_prefill_runtime_inputs,
)
from qwen3vl_tp_runtime.models.qwen3vl.live.common import (
    MultimodalRuntimeInputs,
    _runtime_tensor,
)
from qwen3vl_tp_runtime.models.qwen3vl.processing import (
    build_inputs,
    list_frames,
    load_model,
    load_processor,
)


@dataclass(slots=True)
class MmPrefillSession:
    """Live multimodal prefill session state used by the runtime builder."""

    model: torch.nn.Module
    raw_inputs: dict[str, Any]
    runtime_inputs: MultimodalRuntimeInputs
    extra: dict[str, Any]


def text_pos_ids(runtime_inputs: MultimodalRuntimeInputs) -> torch.Tensor | None:
    position_ids = runtime_inputs.position_ids
    if position_ids is None:
        return None
    if position_ids.ndim == 2:
        return position_ids
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        return position_ids[0]
    return None


def deepstack_embeds(
    runtime_inputs: MultimodalRuntimeInputs,
) -> list[torch.Tensor | None] | None:
    if not runtime_inputs.deepstack_by_layer:
        return None
    max_layer_idx = max(int(layer_idx) for layer_idx in runtime_inputs.deepstack_by_layer)
    return [runtime_inputs.deepstack_by_layer.get(layer_idx) for layer_idx in range(max_layer_idx + 1)]


def prepare_mm_session(runtime_config: dict[str, Any]) -> MmPrefillSession:
    """Prepare one live multimodal prefill session from model_path."""

    model_path = runtime_config["model_path"]
    model = load_model(model_path, attn_implementation="eager")
    processor = load_processor(model_path)
    num_frames = int(runtime_config.get("num_frames", 8))
    frame_paths = list_frames(num_frames, runtime_config.get("frame_dir"))
    raw_inputs = build_inputs(processor, frame_paths)
    raw_inputs = raw_inputs.to(model.device)
    runtime_inputs = prepare_multimodal_prefill_runtime_inputs(model, raw_inputs)
    return MmPrefillSession(
        model=model,
        raw_inputs=raw_inputs,
        runtime_inputs=runtime_inputs,
        extra={"num_frames": len(frame_paths), "frame_paths": frame_paths},
    )


def run_mm_prefill_ref(
    model,
    *,
    runtime_inputs: MultimodalRuntimeInputs,
    start_idx: int,
    end_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run multimodal prefill up to one stage boundary and capture stage tensors."""

    text_model = model.model.language_model
    hidden_states = runtime_inputs.inputs_embeds
    stage_input = hidden_states.detach().clone() if start_idx == 0 else None
    position_embeddings = (runtime_inputs.cos, runtime_inputs.sin)
    stage_text_pos_ids = text_pos_ids(runtime_inputs)

    with torch.inference_mode():
        for layer_idx in range(end_idx + 1):
            hidden_states = text_model.layers[layer_idx](
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=runtime_inputs.attention_mask,
                position_ids=stage_text_pos_ids,
                past_key_values=None,
                use_cache=False,
            )
            hidden_states = apply_deepstack(
                hidden_states,
                runtime_inputs.visual_pos_masks,
                runtime_inputs.deepstack_by_layer.get(layer_idx),
            )
            if layer_idx + 1 == start_idx:
                stage_input = hidden_states.detach().clone()

    if stage_input is None:
        raise RuntimeError(f"没有拿到 multimodal prefill stage_input，start_idx={start_idx} end_idx={end_idx}")
    return stage_input, hidden_states.detach().clone()


def run_mm_prefill(model, runtime_inputs: MultimodalRuntimeInputs):
    """Run the full multimodal prefill forward on the live decoder."""

    text_model = model.model.language_model
    with torch.inference_mode():
        return text_model(
            inputs_embeds=runtime_inputs.inputs_embeds,
            attention_mask=runtime_inputs.attention_mask_2d,
            position_ids=runtime_inputs.position_ids,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
            visual_pos_masks=runtime_inputs.visual_pos_masks,
            deepstack_visual_embeds=deepstack_embeds(runtime_inputs),
        )


def run_mm_decode(
    model,
    *,
    runtime_inputs: MultimodalRuntimeInputs,
    past_key_values,
    is_last_stage: bool,
):
    """Run one multimodal decode step on the live decoder."""

    text_model = model.model.language_model
    captured: dict[str, torch.Tensor] = {}

    handle = None
    if is_last_stage:
        def final_norm_pre_hook(_module, module_inputs):
            captured["hidden_stage_output"] = module_inputs[0].detach().clone()

        handle = text_model.norm.register_forward_pre_hook(final_norm_pre_hook)

    try:
        with torch.inference_mode():
            outputs = text_model(
                inputs_embeds=runtime_inputs.inputs_embeds,
                attention_mask=runtime_inputs.attention_mask_2d,
                position_ids=runtime_inputs.position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                visual_pos_masks=runtime_inputs.visual_pos_masks,
                deepstack_visual_embeds=deepstack_embeds(runtime_inputs),
            )
    finally:
        if handle is not None:
            handle.remove()

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("decode full forward 请求了 output_hidden_states=True，但没有拿到 hidden_states。")
    return outputs, hidden_states, captured.get("hidden_stage_output")


def build_mm_stage_visuals(
    runtime_inputs: MultimodalRuntimeInputs,
    *,
    start_idx: int,
    end_idx: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor | None, dict[int, torch.Tensor]]:
    """Select the multimodal visual payload needed by one decoder stage."""

    stage_deepstack = {
        int(layer_idx): _runtime_tensor(deepstack_embed, device=device, compute_dtype=compute_dtype)
        for layer_idx, deepstack_embed in runtime_inputs.deepstack_by_layer.items()
        if start_idx <= int(layer_idx) <= end_idx
    }
    visual_pos_masks = runtime_inputs.visual_pos_masks
    if not stage_deepstack:
        visual_pos_masks = None
    return _runtime_tensor(visual_pos_masks, device=device), stage_deepstack
