"""Helpers that convert live model state into replay bundles."""

from __future__ import annotations

import torch

from qwen3vl_tp_runtime.models.qwen3vl.capture import extract_past_key_values
from qwen3vl_tp_runtime.models.qwen3vl.live.common import (
    MultimodalRuntimeInputs,
    _resolve_compute_dtype,
    _runtime_tensor,
)


def extract_decoder_layer_params_live(
    layer,
    layer_idx: int,
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> dict:
    """Materialize one decoder layer's parameters directly from the live model."""

    attn = layer.self_attn
    mlp = layer.mlp

    return {
        "layer_idx": layer_idx,
        "hidden_act": mlp.config.hidden_act,
        "q_weight": _runtime_tensor(attn.q_proj.weight, device=device, compute_dtype=compute_dtype),
        "q_bias": _runtime_tensor(attn.q_proj.bias, device=device, compute_dtype=compute_dtype),
        "k_weight": _runtime_tensor(attn.k_proj.weight, device=device, compute_dtype=compute_dtype),
        "k_bias": _runtime_tensor(attn.k_proj.bias, device=device, compute_dtype=compute_dtype),
        "v_weight": _runtime_tensor(attn.v_proj.weight, device=device, compute_dtype=compute_dtype),
        "v_bias": _runtime_tensor(attn.v_proj.bias, device=device, compute_dtype=compute_dtype),
        "o_weight": _runtime_tensor(attn.o_proj.weight, device=device, compute_dtype=compute_dtype),
        "o_bias": _runtime_tensor(attn.o_proj.bias, device=device, compute_dtype=compute_dtype),
        "q_norm_weight": _runtime_tensor(attn.q_norm.weight, device=device, compute_dtype=compute_dtype),
        "k_norm_weight": _runtime_tensor(attn.k_norm.weight, device=device, compute_dtype=compute_dtype),
        "gate_weight": _runtime_tensor(mlp.gate_proj.weight, device=device, compute_dtype=compute_dtype),
        "gate_bias": _runtime_tensor(mlp.gate_proj.bias, device=device, compute_dtype=compute_dtype),
        "up_weight": _runtime_tensor(mlp.up_proj.weight, device=device, compute_dtype=compute_dtype),
        "up_bias": _runtime_tensor(mlp.up_proj.bias, device=device, compute_dtype=compute_dtype),
        "down_weight": _runtime_tensor(mlp.down_proj.weight, device=device, compute_dtype=compute_dtype),
        "down_bias": _runtime_tensor(mlp.down_proj.bias, device=device, compute_dtype=compute_dtype),
        "input_ln_weight": _runtime_tensor(layer.input_layernorm.weight, device=device, compute_dtype=compute_dtype),
        "input_ln_eps": layer.input_layernorm.variance_epsilon,
        "post_attn_ln_weight": _runtime_tensor(
            layer.post_attention_layernorm.weight,
            device=device,
            compute_dtype=compute_dtype,
        ),
        "post_attn_ln_eps": layer.post_attention_layernorm.variance_epsilon,
        "rms_norm_eps": attn.q_norm.variance_epsilon,
        "num_attention_heads": attn.config.num_attention_heads,
        "num_key_value_heads": attn.config.num_key_value_heads,
        "head_dim": attn.head_dim,
        "scaling": attn.scaling,
        "attn_implementation": attn.config._attn_implementation,
    }


def build_cache_by_layer_from_past_key_values(
    past_key_values,
    *,
    device: torch.device,
    compute_dtype: torch.dtype | None = None,
) -> dict[int, tuple[torch.Tensor | None, torch.Tensor | None]]:
    """Convert HF cache objects into the runtime's per-layer cache map."""

    cache_layers = extract_past_key_values(past_key_values)
    cache_by_layer = {}
    for layer_idx, (past_key, past_value) in enumerate(cache_layers):
        cache_by_layer[layer_idx] = (
            _runtime_tensor(past_key, device=device, compute_dtype=compute_dtype),
            _runtime_tensor(past_value, device=device, compute_dtype=compute_dtype),
        )
    return cache_by_layer


def build_live_multimodal_stage_bundle(
    model,
    *,
    start_idx: int,
    end_idx: int,
    runtime_inputs: MultimodalRuntimeInputs,
    phase: str,
    compute_dtype_arg: str = "auto",
    stage_input: torch.Tensor | None = None,
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
) -> tuple[dict, torch.dtype]:
    """Build a decoder stage bundle directly from the live model weights."""

    text_model = model.model.language_model
    layers = text_model.layers
    if start_idx > end_idx:
        raise ValueError("start_idx 不能大于 end_idx。")
    if end_idx >= len(layers):
        raise ValueError(f"end_idx={end_idx} 超出层数上限 {len(layers) - 1}。")

    device = next(model.parameters()).device
    compute_dtype = _resolve_compute_dtype(text_model.embed_tokens.weight, compute_dtype_arg)
    is_last_stage = end_idx == len(layers) - 1
    runtime_stage_input = runtime_inputs.inputs_embeds if stage_input is None else stage_input
    runtime_stage_input = _runtime_tensor(runtime_stage_input, device=device, compute_dtype=compute_dtype)

    if phase == "prefill":
        stage_type = "text_prefill_last" if is_last_stage else "text"
        module_name = "live_multimodal_prefill_stage"
    elif phase == "decode":
        stage_type = "text_decode_last" if is_last_stage else "text_decode"
        module_name = "live_multimodal_decode_stage"
    else:
        raise ValueError(f"不支持的 phase={phase!r}")

    filtered_deepstack = {
        layer_idx: _runtime_tensor(embed, device=device, compute_dtype=compute_dtype)
        for layer_idx, embed in runtime_inputs.deepstack_by_layer.items()
        if start_idx <= int(layer_idx) <= end_idx
    }

    layer_bundles = []
    current_cache = cache_by_layer or {}
    for layer_idx in range(start_idx, end_idx + 1):
        layer_bundle = extract_decoder_layer_params_live(
            layers[layer_idx],
            layer_idx,
            device=device,
            compute_dtype=compute_dtype,
        )
        if layer_idx in current_cache:
            past_key, past_value = current_cache[layer_idx]
            layer_bundle["past_key"] = _runtime_tensor(past_key, device=device, compute_dtype=compute_dtype)
            layer_bundle["past_value"] = _runtime_tensor(past_value, device=device, compute_dtype=compute_dtype)
        layer_bundles.append(layer_bundle)

    bundle = {
        "module_name": module_name,
        "stage_type": stage_type,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "save_dtype": str(compute_dtype).replace("torch.", ""),
        "original_input_dtype": str(runtime_inputs.inputs_embeds.dtype),
        "original_input_device": str(runtime_inputs.inputs_embeds.device),
        "stage_input": runtime_stage_input,
        "layer_input": runtime_stage_input,
        "attention_mask": _runtime_tensor(runtime_inputs.attention_mask, device=device),
        "cos": _runtime_tensor(runtime_inputs.cos, device=device, compute_dtype=compute_dtype),
        "sin": _runtime_tensor(runtime_inputs.sin, device=device, compute_dtype=compute_dtype),
        "visual_pos_masks": _runtime_tensor(runtime_inputs.visual_pos_masks, device=device),
        "deepstack_by_layer": filtered_deepstack,
        "deepstack_layer_indices": sorted(filtered_deepstack),
        "layers": layer_bundles,
        "position_ids": _runtime_tensor(runtime_inputs.position_ids, device=device),
        "attention_mask_2d": _runtime_tensor(runtime_inputs.attention_mask_2d, device=device),
        "multimodal_meta": {
            "mm_token_type_ids": _runtime_tensor(runtime_inputs.mm_token_type_ids, device=device),
            "image_grid_thw": _runtime_tensor(runtime_inputs.image_grid_thw, device=device),
            "video_grid_thw": _runtime_tensor(runtime_inputs.video_grid_thw, device=device),
            "rope_deltas": _runtime_tensor(runtime_inputs.rope_deltas, device=device),
        },
    }

    if runtime_inputs.input_ids is not None:
        if phase == "prefill":
            bundle["input_ids"] = _runtime_tensor(runtime_inputs.input_ids, device=device)
        else:
            bundle["decode_input_ids"] = _runtime_tensor(runtime_inputs.input_ids, device=device)
        if start_idx == 0:
            bundle["embed_tokens_weight"] = _runtime_tensor(
                text_model.embed_tokens.weight,
                device=device,
                compute_dtype=compute_dtype,
            )

    if is_last_stage:
        bundle["final_norm_weight"] = _runtime_tensor(text_model.norm.weight, device=device, compute_dtype=compute_dtype)
        bundle["final_norm_eps"] = text_model.norm.variance_epsilon
        bundle["lm_head_weight"] = _runtime_tensor(model.lm_head.weight, device=device, compute_dtype=compute_dtype)
        bundle["lm_head_bias"] = _runtime_tensor(model.lm_head.bias, device=device, compute_dtype=compute_dtype)

    return bundle, compute_dtype


__all__ = [
    "extract_decoder_layer_params_live",
    "build_cache_by_layer_from_past_key_values",
    "build_live_multimodal_stage_bundle",
]
