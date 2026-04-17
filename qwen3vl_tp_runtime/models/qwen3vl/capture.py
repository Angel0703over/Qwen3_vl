"""Capture utilities that extract replayable Qwen3-VL bundles from the real model."""

from pathlib import Path

import torch
from transformers.masking_utils import create_causal_mask

from qwen3vl_tp_runtime.hexgen_core.config import (
    FRAME_DIR,
    FULL_LAYER_BUNDLE_PATH,
    LAYER_RANGE_BUNDLE_PATH,
    MODEL_PATH,
    TEXT_STAGE_BUNDLE_PATH,
)
from qwen3vl_tp_runtime.models.qwen3vl.forward import (
    forward_decoder_layer,
    forward_layer_range,
    forward_text_stage,
)
from qwen3vl_tp_runtime.models.qwen3vl.inputs import build_inputs, list_frames, load_model, load_processor
from qwen3vl_tp_runtime.models.qwen3vl.ops import build_causal_mask, cast_cpu, resolve_save_dtype


def load_bundle(bundle_path: str = FULL_LAYER_BUNDLE_PATH):
    return torch.load(bundle_path, map_location="cpu")


def _move_bundle_value(value, device: torch.device, compute_dtype: torch.dtype):
    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(device=device, dtype=compute_dtype)
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_bundle_value(item, device, compute_dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_bundle_value(item, device, compute_dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_bundle_value(item, device, compute_dtype) for item in value)
    return value


def move_bundle(bundle: dict, device: torch.device, compute_dtype: torch.dtype) -> dict:
    return _move_bundle_value(bundle, device, compute_dtype)


def capture_decoder_layer_params(layer, layer_idx: int, save_dtype_value: torch.dtype) -> dict:
    attn = layer.self_attn
    mlp = layer.mlp

    return {
        "layer_idx": layer_idx,
        "hidden_act": mlp.config.hidden_act,
        "q_weight": cast_cpu(attn.q_proj.weight, save_dtype_value),
        "q_bias": cast_cpu(attn.q_proj.bias, save_dtype_value),
        "k_weight": cast_cpu(attn.k_proj.weight, save_dtype_value),
        "k_bias": cast_cpu(attn.k_proj.bias, save_dtype_value),
        "v_weight": cast_cpu(attn.v_proj.weight, save_dtype_value),
        "v_bias": cast_cpu(attn.v_proj.bias, save_dtype_value),
        "o_weight": cast_cpu(attn.o_proj.weight, save_dtype_value),
        "o_bias": cast_cpu(attn.o_proj.bias, save_dtype_value),
        "q_norm_weight": cast_cpu(attn.q_norm.weight, save_dtype_value),
        "k_norm_weight": cast_cpu(attn.k_norm.weight, save_dtype_value),
        "gate_weight": cast_cpu(mlp.gate_proj.weight, save_dtype_value),
        "gate_bias": cast_cpu(mlp.gate_proj.bias, save_dtype_value),
        "up_weight": cast_cpu(mlp.up_proj.weight, save_dtype_value),
        "up_bias": cast_cpu(mlp.up_proj.bias, save_dtype_value),
        "down_weight": cast_cpu(mlp.down_proj.weight, save_dtype_value),
        "down_bias": cast_cpu(mlp.down_proj.bias, save_dtype_value),
        "input_ln_weight": cast_cpu(layer.input_layernorm.weight, save_dtype_value),
        "input_ln_eps": layer.input_layernorm.variance_epsilon,
        "post_attn_ln_weight": cast_cpu(layer.post_attention_layernorm.weight, save_dtype_value),
        "post_attn_ln_eps": layer.post_attention_layernorm.variance_epsilon,
        "rms_norm_eps": attn.q_norm.variance_epsilon,
        "num_attention_heads": attn.config.num_attention_heads,
        "num_key_value_heads": attn.config.num_key_value_heads,
        "head_dim": attn.head_dim,
        "scaling": attn.scaling,
        "attn_implementation": attn.config._attn_implementation,
    }


def resolve_runtime_tensors(
    model,
    inputs,
    hidden_states: torch.Tensor,
    captured: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention_mask = captured.get("attention_mask")
    if attention_mask is None:
        attention_mask = create_causal_mask(
            config=model.model.language_model.config,
            inputs_embeds=hidden_states,
            attention_mask=inputs.get("attention_mask"),
            past_key_values=None,
            position_ids=None,
        )
    if attention_mask is None:
        attention_mask = build_causal_mask(hidden_states, inputs.get("attention_mask"))

    cos = captured.get("cos")
    sin = captured.get("sin")
    if cos is None or sin is None:
        raise RuntimeError("没有捕获到 position_embeddings，对应的 cos/sin 不能为空。")

    return attention_mask.detach().clone(), cos.detach().clone(), sin.detach().clone()


def run_forward_with_runtime_hook(model, inputs, layer):
    captured = {}

    def attn_pre_hook(_module, module_args, module_kwargs):
        position_embeddings = module_kwargs.get("position_embeddings")
        if position_embeddings is None and len(module_args) > 1:
            position_embeddings = module_args[1]
        if position_embeddings is None:
            raise RuntimeError("没有在 self_attn pre-hook 中拿到 position_embeddings。")

        attention_mask = module_kwargs.get("attention_mask")
        if attention_mask is None and len(module_args) > 2:
            attention_mask = module_args[2]
        if attention_mask is not None:
            captured["attention_mask"] = attention_mask.detach().clone()

        cos, sin = position_embeddings
        captured["cos"] = cos.detach().clone()
        captured["sin"] = sin.detach().clone()

    handle = layer.self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        handle.remove()
    return outputs, captured


def capture_full_layer_bundle(
    *,
    layer_idx: int = 11,
    num_frames: int = 8,
    bundle_path: str = FULL_LAYER_BUNDLE_PATH,
    save_dtype: str = "auto",
    model_path: str = MODEL_PATH,
    frame_dir: str = FRAME_DIR,
) -> dict:
    frame_paths = list_frames(num_frames, frame_dir)

    print("Loading model...")
    model = load_model(model_path, attn_implementation="eager")

    print("Loading processor...")
    processor = load_processor(model_path)

    inputs = build_inputs(processor, frame_paths)
    inputs = inputs.to(model.device)

    layer = model.model.language_model.layers[layer_idx]
    captured = {}

    def input_ln_pre_hook(_module, module_inputs):
        captured["layer_input"] = module_inputs[0].detach().clone()

    def input_ln_forward_hook(_module, _module_inputs, module_output):
        captured["attn_input"] = module_output.detach().clone()

    def attn_pre_hook(_module, module_args, module_kwargs):
        position_embeddings = module_kwargs.get("position_embeddings")
        if position_embeddings is None and len(module_args) > 1:
            position_embeddings = module_args[1]
        if position_embeddings is None:
            raise RuntimeError("没有在 self_attn pre-hook 中拿到 position_embeddings。")

        attention_mask = module_kwargs.get("attention_mask")
        if attention_mask is None and len(module_args) > 2:
            attention_mask = module_args[2]
        if attention_mask is not None:
            captured["attention_mask"] = attention_mask.detach().clone()

        cos, sin = position_embeddings
        captured["cos"] = cos.detach().clone()
        captured["sin"] = sin.detach().clone()

    def attn_forward_hook(_module, _module_inputs, module_output):
        if isinstance(module_output, tuple):
            captured["attn_output"] = module_output[0].detach().clone()
        else:
            captured["attn_output"] = module_output.detach().clone()

    def post_attn_ln_forward_hook(_module, _module_inputs, module_output):
        captured["mlp_input"] = module_output.detach().clone()

    def layer_forward_hook(_module, _module_inputs, module_output):
        captured["layer_output"] = module_output.detach().clone()

    handles = [
        layer.input_layernorm.register_forward_pre_hook(input_ln_pre_hook),
        layer.input_layernorm.register_forward_hook(input_ln_forward_hook),
        layer.self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True),
        layer.self_attn.register_forward_hook(attn_forward_hook),
        layer.post_attention_layernorm.register_forward_hook(post_attn_ln_forward_hook),
        layer.register_forward_hook(layer_forward_hook),
    ]
    try:
        with torch.inference_mode():
            _ = model(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    required = {
        "layer_input",
        "attn_input",
        "cos",
        "sin",
        "attn_output",
        "mlp_input",
        "layer_output",
    }
    if not required.issubset(captured):
        missing = required - set(captured)
        raise RuntimeError(f"没有捕获到完整 layer replay 的必要输入: {missing}")

    attention_mask, cos, sin = resolve_runtime_tensors(model, inputs, captured["attn_input"], captured)
    save_dtype_value = resolve_save_dtype(save_dtype, captured["layer_input"])

    bundle = {
        "layer_idx": layer_idx,
        "module_name": "full_layer",
        "save_dtype": str(save_dtype_value).replace("torch.", ""),
        "original_input_dtype": str(captured["layer_input"].dtype),
        "original_output_dtype": str(captured["layer_output"].dtype),
        "original_input_device": str(captured["layer_input"].device),
        "original_output_device": str(captured["layer_output"].device),
        "layer_input": cast_cpu(captured["layer_input"], save_dtype_value),
        "attn_input": cast_cpu(captured["attn_input"], save_dtype_value),
        "attention_mask": cast_cpu(attention_mask, None),
        "cos": cast_cpu(cos, save_dtype_value),
        "sin": cast_cpu(sin, save_dtype_value),
        "attn_output": cast_cpu(captured["attn_output"], save_dtype_value),
        "mlp_input": cast_cpu(captured["mlp_input"], save_dtype_value),
        "layer_output": cast_cpu(captured["layer_output"], save_dtype_value),
        **capture_decoder_layer_params(layer, layer_idx, save_dtype_value),
        "frame_paths": frame_paths,
    }

    moved_bundle = move_bundle(bundle, model.device, captured["layer_input"].dtype)
    direct_output = forward_decoder_layer(bundle["layer_input"].to(model.device), moved_bundle)
    reference_output = bundle["layer_output"].to(model.device)
    bundle["sanity_max_diff"] = (direct_output - reference_output).abs().max().item()
    bundle["sanity_mean_diff"] = (direct_output - reference_output).abs().mean().item()

    save_path = Path(bundle_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, save_path)
    return bundle


def capture_layer_range_bundle(
    *,
    start_idx: int = 11,
    end_idx: int = 12,
    num_frames: int = 8,
    bundle_path: str = LAYER_RANGE_BUNDLE_PATH,
    save_dtype: str = "auto",
    model_path: str = MODEL_PATH,
    frame_dir: str = FRAME_DIR,
) -> dict:
    if start_idx > end_idx:
        raise ValueError("start_idx 不能大于 end_idx。")

    frame_paths = list_frames(num_frames, frame_dir)

    print("Loading model...")
    model = load_model(model_path, attn_implementation="eager")

    print("Loading processor...")
    processor = load_processor(model_path)

    inputs = build_inputs(processor, frame_paths)
    inputs = inputs.to(model.device)

    layers = model.model.language_model.layers
    start_layer = layers[start_idx]
    outputs, captured = run_forward_with_runtime_hook(model, inputs, start_layer)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("output_hidden_states=True 但没有拿到 hidden_states。")
    if end_idx + 1 >= len(hidden_states):
        raise ValueError("给定的 end_idx 超出了 hidden_states 可用范围。")

    layer_input = hidden_states[start_idx].detach().clone()
    reference_output = hidden_states[end_idx + 1].detach().clone()
    attention_mask, cos, sin = resolve_runtime_tensors(model, inputs, layer_input, captured)
    save_dtype_value = resolve_save_dtype(save_dtype, layer_input)

    layer_bundles = []
    for layer_idx in range(start_idx, end_idx + 1):
        layer_bundles.append(capture_decoder_layer_params(layers[layer_idx], layer_idx, save_dtype_value))

    bundle = {
        "module_name": "layer_range",
        "start_idx": start_idx,
        "end_idx": end_idx,
        "save_dtype": str(save_dtype_value).replace("torch.", ""),
        "original_input_dtype": str(layer_input.dtype),
        "original_output_dtype": str(reference_output.dtype),
        "original_input_device": str(layer_input.device),
        "original_output_device": str(reference_output.device),
        "layer_input": cast_cpu(layer_input, save_dtype_value),
        "layer_output": cast_cpu(reference_output, save_dtype_value),
        "attention_mask": cast_cpu(attention_mask, None),
        "cos": cast_cpu(cos, save_dtype_value),
        "sin": cast_cpu(sin, save_dtype_value),
        "layers": layer_bundles,
        "frame_paths": frame_paths,
    }

    moved_bundle = move_bundle(bundle, model.device, layer_input.dtype)
    direct_output = forward_layer_range(bundle["layer_input"].to(model.device), moved_bundle)
    reference_output = bundle["layer_output"].to(model.device)
    bundle["sanity_max_diff"] = (direct_output - reference_output).abs().max().item()
    bundle["sanity_mean_diff"] = (direct_output - reference_output).abs().mean().item()

    save_path = Path(bundle_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, save_path)
    return bundle


def capture_text_stage_bundle(
    *,
    start_idx: int = 0,
    end_idx: int = 2,
    num_frames: int = 8,
    bundle_path: str = TEXT_STAGE_BUNDLE_PATH,
    save_dtype: str = "auto",
    model_path: str = MODEL_PATH,
    frame_dir: str = FRAME_DIR,
) -> dict:
    if start_idx > end_idx:
        raise ValueError("start_idx 不能大于 end_idx。")

    frame_paths = list_frames(num_frames, frame_dir)

    print("Loading model...")
    model = load_model(model_path, attn_implementation="eager")

    print("Loading processor...")
    processor = load_processor(model_path)

    inputs = build_inputs(processor, frame_paths)
    inputs = inputs.to(model.device)

    layers = model.model.language_model.layers
    text_model = model.model.language_model
    start_layer = layers[start_idx]
    captured = {}

    def stage_input_hook(_module, module_inputs):
        captured["stage_input"] = module_inputs[0].detach().clone()

    def stage_output_hook(_module, module_inputs):
        captured["stage_output"] = module_inputs[0].detach().clone()

    def text_pre_hook(_module, module_args, module_kwargs):
        visual_pos_masks = module_kwargs.get("visual_pos_masks")
        deepstack_visual_embeds = module_kwargs.get("deepstack_visual_embeds")

        if visual_pos_masks is not None:
            captured["visual_pos_masks"] = visual_pos_masks.detach().clone()

        if deepstack_visual_embeds is not None:
            captured["deepstack_visual_embeds"] = [embed.detach().clone() for embed in deepstack_visual_embeds]

    def attn_pre_hook(_module, module_args, module_kwargs):
        position_embeddings = module_kwargs.get("position_embeddings")
        if position_embeddings is None and len(module_args) > 1:
            position_embeddings = module_args[1]
        if position_embeddings is None:
            raise RuntimeError("没有在 self_attn pre-hook 中拿到 position_embeddings。")

        attention_mask = module_kwargs.get("attention_mask")
        if attention_mask is None and len(module_args) > 2:
            attention_mask = module_args[2]
        if attention_mask is not None:
            captured["attention_mask"] = attention_mask.detach().clone()

        cos, sin = position_embeddings
        captured["cos"] = cos.detach().clone()
        captured["sin"] = sin.detach().clone()

    stage_input_handle = start_layer.input_layernorm.register_forward_pre_hook(stage_input_hook)
    text_handle = text_model.register_forward_pre_hook(text_pre_hook, with_kwargs=True)
    attn_handle = start_layer.self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True)
    if end_idx + 1 < len(layers):
        stage_output_handle = layers[end_idx + 1].input_layernorm.register_forward_pre_hook(stage_output_hook)
    else:
        stage_output_handle = text_model.norm.register_forward_pre_hook(stage_output_hook)
    try:
        with torch.inference_mode():
            _ = model(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        stage_input_handle.remove()
        text_handle.remove()
        attn_handle.remove()
        stage_output_handle.remove()

    if "stage_input" not in captured or "stage_output" not in captured:
        raise RuntimeError("没有捕获到 text stage 的真实输入/输出。")

    layer_input = captured["stage_input"].detach().clone()
    reference_output = captured["stage_output"].detach().clone()
    attention_mask, cos, sin = resolve_runtime_tensors(model, inputs, layer_input, captured)
    save_dtype_value = resolve_save_dtype(save_dtype, layer_input)

    layer_bundles = []
    for layer_idx in range(start_idx, end_idx + 1):
        layer_bundles.append(capture_decoder_layer_params(layers[layer_idx], layer_idx, save_dtype_value))

    deepstack_by_layer = {}
    deepstack_visual_embeds = captured.get("deepstack_visual_embeds")
    if deepstack_visual_embeds is not None:
        for layer_idx, deepstack_embed in enumerate(deepstack_visual_embeds):
            if start_idx <= layer_idx <= end_idx:
                deepstack_by_layer[layer_idx] = cast_cpu(deepstack_embed, save_dtype_value)

    visual_pos_masks = captured.get("visual_pos_masks")
    if deepstack_by_layer and visual_pos_masks is None:
        raise RuntimeError("捕获到了 deepstack_visual_embeds，但没有拿到 visual_pos_masks。")

    bundle = {
        "module_name": "text_stage",
        "stage_type": "text",
        "start_idx": start_idx,
        "end_idx": end_idx,
        "save_dtype": str(save_dtype_value).replace("torch.", ""),
        "original_input_dtype": str(layer_input.dtype),
        "original_output_dtype": str(reference_output.dtype),
        "original_input_device": str(layer_input.device),
        "original_output_device": str(reference_output.device),
        "stage_input": cast_cpu(layer_input, save_dtype_value),
        "stage_output": cast_cpu(reference_output, save_dtype_value),
        "layer_input": cast_cpu(layer_input, save_dtype_value),
        "layer_output": cast_cpu(reference_output, save_dtype_value),
        "attention_mask": cast_cpu(attention_mask, None),
        "cos": cast_cpu(cos, save_dtype_value),
        "sin": cast_cpu(sin, save_dtype_value),
        "visual_pos_masks": cast_cpu(visual_pos_masks, None),
        "deepstack_by_layer": deepstack_by_layer,
        "deepstack_layer_indices": sorted(deepstack_by_layer.keys()),
        "layers": layer_bundles,
        "frame_paths": frame_paths,
    }

    moved_bundle = move_bundle(bundle, model.device, layer_input.dtype)
    direct_output = forward_text_stage(bundle["layer_input"].to(model.device), moved_bundle)
    reference_output = bundle["layer_output"].to(model.device)
    bundle["sanity_max_diff"] = (direct_output - reference_output).abs().max().item()
    bundle["sanity_mean_diff"] = (direct_output - reference_output).abs().mean().item()

    save_path = Path(bundle_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, save_path)
    return bundle
