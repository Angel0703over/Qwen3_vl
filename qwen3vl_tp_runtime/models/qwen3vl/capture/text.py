"""Text-only capture flows for Qwen3-VL replay bundles."""

from pathlib import Path

import torch

from qwen3vl_tp_runtime.hexgen_core.config import (
    MODEL_PATH,
    TEXT_DECODE_BUNDLE_PATH,
    TEXT_GENERATE_BUNDLE_PATH,
    TEXT_PREFILL_BUNDLE_PATH,
    TEXT_STAGE_BUNDLE_PATH,
)
from qwen3vl_tp_runtime.models.qwen3vl.capture.common import (
    capture_decoder_layer_params,
    extract_past_key_values,
    move_bundle,
    resolve_runtime_tensors,
)
from qwen3vl_tp_runtime.models.qwen3vl.execution import (
    forward_text_decode_stage,
    forward_text_stage,
    trace_text_decode_logits,
    trace_text_prefill_logits,
    trace_text_prefill_stage_logits,
)
from qwen3vl_tp_runtime.models.qwen3vl.functional import cast_cpu, resolve_save_dtype
from qwen3vl_tp_runtime.models.qwen3vl.processing import build_text_inputs, load_model, load_processor


def _capture_text_prefill_reference(
    model,
    inputs,
):
    text_model = model.model.language_model
    first_layer = text_model.layers[0]
    captured = {}

    def stage_input_hook(_module, module_inputs):
        captured["layer_input"] = module_inputs[0].detach().clone()

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

    def final_norm_pre_hook(_module, module_inputs):
        captured["stage_output"] = module_inputs[0].detach().clone()

    def final_norm_forward_hook(_module, _module_inputs, module_output):
        captured["norm_output"] = module_output.detach().clone()

    handles = [
        first_layer.input_layernorm.register_forward_pre_hook(stage_input_hook),
        first_layer.self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True),
        text_model.norm.register_forward_pre_hook(final_norm_pre_hook),
        text_model.norm.register_forward_hook(final_norm_forward_hook),
    ]
    try:
        with torch.inference_mode():
            outputs = model(
                **inputs,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    required = {
        "layer_input",
        "stage_output",
        "norm_output",
        "attention_mask",
        "cos",
        "sin",
    }
    if not required.issubset(captured):
        missing = required - set(captured)
        raise RuntimeError(f"没有捕获到 text generate prefill 的必要输入: {missing}")

    return outputs, captured


def _capture_text_decode_reference_step(
    model,
    *,
    decode_input_ids: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    past_key_values,
):
    text_model = model.model.language_model
    first_layer = text_model.layers[0]
    captured = {}

    def stage_input_hook(_module, module_inputs):
        captured["layer_input"] = module_inputs[0].detach().clone()

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

    def final_norm_pre_hook(_module, module_inputs):
        captured["stage_output"] = module_inputs[0].detach().clone()

    def final_norm_forward_hook(_module, _module_inputs, module_output):
        captured["norm_output"] = module_output.detach().clone()

    handles = [
        first_layer.input_layernorm.register_forward_pre_hook(stage_input_hook),
        first_layer.self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True),
        text_model.norm.register_forward_pre_hook(final_norm_pre_hook),
        text_model.norm.register_forward_hook(final_norm_forward_hook),
    ]
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=decode_input_ids,
                attention_mask=attention_mask_2d,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    required = {
        "layer_input",
        "stage_output",
        "norm_output",
        "attention_mask",
        "cos",
        "sin",
    }
    if not required.issubset(captured):
        missing = required - set(captured)
        raise RuntimeError(f"没有捕获到 text generate decode step 的必要输入: {missing}")

    return outputs, captured


def capture_text_decode_bundle(
    *,
    prompt: str = "请用中文简要介绍一下人工智能。",
    decode_token_id: int | None = None,
    bundle_path: str = TEXT_DECODE_BUNDLE_PATH,
    save_dtype: str = "auto",
    model_path: str = MODEL_PATH,
) -> dict:
    print("Loading model...")
    model = load_model(model_path, attn_implementation="eager")

    print("Loading processor...")
    processor = load_processor(model_path)

    prefill_inputs = build_text_inputs(processor, prompt)
    prefill_inputs = prefill_inputs.to(model.device)

    text_model = model.model.language_model
    layers = text_model.layers
    first_layer = layers[0]

    with torch.inference_mode():
        prefill_outputs = model(
            **prefill_inputs,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )

    prefill_cache_layers = extract_past_key_values(prefill_outputs.past_key_values)
    if len(prefill_cache_layers) != len(layers):
        raise RuntimeError(
            "prefill cache 层数和 decoder layer 数量不一致，"
            f"cache_layers={len(prefill_cache_layers)} decoder_layers={len(layers)}"
        )

    greedy_decode_token_id = int(prefill_outputs.logits[0, -1].argmax().item())
    decode_token_id_value = greedy_decode_token_id if decode_token_id is None else int(decode_token_id)
    decode_source = "greedy_from_prefill" if decode_token_id is None else "provided"

    prefill_input_ids = prefill_inputs["input_ids"]
    prefill_attention_mask_2d = prefill_inputs.get("attention_mask")
    if prefill_attention_mask_2d is None:
        prefill_attention_mask_2d = torch.ones_like(prefill_input_ids)

    decode_input_ids = torch.tensor(
        [[decode_token_id_value]],
        device=model.device,
        dtype=prefill_input_ids.dtype,
    )
    decode_attention_mask_2d = torch.cat(
        [
            prefill_attention_mask_2d,
            torch.ones(
                (prefill_attention_mask_2d.shape[0], 1),
                device=model.device,
                dtype=prefill_attention_mask_2d.dtype,
            ),
        ],
        dim=-1,
    )

    captured = {}

    def stage_input_hook(_module, module_inputs):
        captured["layer_input"] = module_inputs[0].detach().clone()

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

    def final_norm_pre_hook(_module, module_inputs):
        captured["stage_output"] = module_inputs[0].detach().clone()

    def final_norm_forward_hook(_module, _module_inputs, module_output):
        captured["norm_output"] = module_output.detach().clone()

    handles = [
        first_layer.input_layernorm.register_forward_pre_hook(stage_input_hook),
        first_layer.self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True),
        text_model.norm.register_forward_pre_hook(final_norm_pre_hook),
        text_model.norm.register_forward_hook(final_norm_forward_hook),
    ]
    try:
        with torch.inference_mode():
            decode_outputs = model(
                input_ids=decode_input_ids,
                attention_mask=decode_attention_mask_2d,
                past_key_values=prefill_outputs.past_key_values,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    required = {
        "layer_input",
        "stage_output",
        "norm_output",
        "attention_mask",
        "cos",
        "sin",
    }
    if not required.issubset(captured):
        missing = required - set(captured)
        raise RuntimeError(f"没有捕获到 text decode logits 的必要输入: {missing}")

    logits = decode_outputs.logits.detach().clone()
    layer_input = captured["layer_input"].detach().clone()
    save_dtype_value = resolve_save_dtype(save_dtype, layer_input)

    layer_bundles = []
    for layer_idx, layer in enumerate(layers):
        past_key, past_value = prefill_cache_layers[layer_idx]
        layer_bundle = capture_decoder_layer_params(layer, layer_idx, save_dtype_value)
        layer_bundle["past_key"] = cast_cpu(past_key, save_dtype_value)
        layer_bundle["past_value"] = cast_cpu(past_value, save_dtype_value)
        layer_bundles.append(layer_bundle)

    bundle = {
        "module_name": "text_decode",
        "stage_type": "text_decode",
        "start_idx": 0,
        "end_idx": len(layers) - 1,
        "save_dtype": str(save_dtype_value).replace("torch.", ""),
        "original_input_dtype": str(layer_input.dtype),
        "original_output_dtype": str(logits.dtype),
        "original_input_device": str(layer_input.device),
        "original_output_device": str(logits.device),
        "prompt": prompt,
        "decode_source": decode_source,
        "decode_token_id": decode_token_id_value,
        "prefill_seq_len": int(prefill_input_ids.shape[-1]),
        "total_seq_len": int(decode_attention_mask_2d.shape[-1]),
        "prefill_input_ids": cast_cpu(prefill_input_ids, None),
        "decode_input_ids": cast_cpu(decode_input_ids, None),
        "prefill_attention_mask_2d": cast_cpu(prefill_attention_mask_2d, None),
        "attention_mask_2d": cast_cpu(decode_attention_mask_2d, None),
        "embed_tokens_weight": cast_cpu(text_model.embed_tokens.weight, save_dtype_value),
        "stage_input": cast_cpu(layer_input, save_dtype_value),
        "layer_input": cast_cpu(layer_input, save_dtype_value),
        "stage_output": cast_cpu(captured["stage_output"], save_dtype_value),
        "layer_output": cast_cpu(captured["stage_output"], save_dtype_value),
        "norm_output": cast_cpu(captured["norm_output"], save_dtype_value),
        "logits": cast_cpu(logits, save_dtype_value),
        "attention_mask": cast_cpu(captured["attention_mask"], None),
        "cos": cast_cpu(captured["cos"], save_dtype_value),
        "sin": cast_cpu(captured["sin"], save_dtype_value),
        "visual_pos_masks": None,
        "deepstack_by_layer": {},
        "deepstack_layer_indices": [],
        "final_norm_weight": cast_cpu(text_model.norm.weight, save_dtype_value),
        "final_norm_eps": text_model.norm.variance_epsilon,
        "lm_head_weight": cast_cpu(model.lm_head.weight, save_dtype_value),
        "lm_head_bias": cast_cpu(model.lm_head.bias, save_dtype_value),
        "layers": layer_bundles,
    }

    moved_bundle = move_bundle(bundle, model.device, layer_input.dtype)
    direct_trace = trace_text_decode_logits(
        bundle["layer_input"].to(model.device),
        moved_bundle,
    )
    reference_stage_output = bundle["stage_output"].to(model.device)
    reference_norm_output = bundle["norm_output"].to(model.device)
    reference_logits = bundle["logits"].to(model.device)

    bundle["stage_sanity_max_diff"] = (direct_trace["stage_output"] - reference_stage_output).abs().max().item()
    bundle["stage_sanity_mean_diff"] = (direct_trace["stage_output"] - reference_stage_output).abs().mean().item()
    bundle["norm_sanity_max_diff"] = (direct_trace["norm_output"] - reference_norm_output).abs().max().item()
    bundle["norm_sanity_mean_diff"] = (direct_trace["norm_output"] - reference_norm_output).abs().mean().item()
    bundle["logits_sanity_max_diff"] = (direct_trace["logits"] - reference_logits).abs().max().item()
    bundle["logits_sanity_mean_diff"] = (direct_trace["logits"] - reference_logits).abs().mean().item()
    bundle["sanity_max_diff"] = bundle["logits_sanity_max_diff"]
    bundle["sanity_mean_diff"] = bundle["logits_sanity_mean_diff"]

    save_path = Path(bundle_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, save_path)
    return bundle


def capture_text_generate_bundle(
    *,
    prompt: str = "请用中文简要介绍一下人工智能。",
    max_new_tokens: int = 4,
    bundle_path: str = TEXT_GENERATE_BUNDLE_PATH,
    save_dtype: str = "auto",
    model_path: str = MODEL_PATH,
) -> dict:
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens 必须大于 0，当前拿到 {max_new_tokens}。")

    print("Loading model...")
    model = load_model(model_path, attn_implementation="eager")

    print("Loading processor...")
    processor = load_processor(model_path)

    prefill_inputs = build_text_inputs(processor, prompt)
    prefill_inputs = prefill_inputs.to(model.device)

    text_model = model.model.language_model
    layers = text_model.layers

    prefill_outputs, prefill_captured = _capture_text_prefill_reference(model, prefill_inputs)
    prefill_layer_input = prefill_captured["layer_input"].detach().clone()
    save_dtype_value = resolve_save_dtype(save_dtype, prefill_layer_input)

    prefill_input_ids = prefill_inputs["input_ids"]
    prefill_attention_mask_2d = prefill_inputs.get("attention_mask")
    if prefill_attention_mask_2d is None:
        prefill_attention_mask_2d = torch.ones_like(prefill_input_ids)

    layer_bundles = [
        capture_decoder_layer_params(layer, layer_idx, save_dtype_value)
        for layer_idx, layer in enumerate(layers)
    ]
    prefill_cache_layers = extract_past_key_values(prefill_outputs.past_key_values)
    if len(prefill_cache_layers) != len(layer_bundles):
        raise RuntimeError(
            "prefill cache 层数和 decoder layer 数量不一致，"
            f"cache_layers={len(prefill_cache_layers)} decoder_layers={len(layer_bundles)}"
        )

    generated_token_ids = [int(prefill_outputs.logits[0, -1].argmax().item())]
    decode_steps = []
    current_attention_mask_2d = prefill_attention_mask_2d
    current_past_key_values = prefill_outputs.past_key_values

    for step_idx in range(max_new_tokens - 1):
        decode_input_ids = torch.tensor(
            [[generated_token_ids[-1]]],
            device=model.device,
            dtype=prefill_input_ids.dtype,
        )
        current_attention_mask_2d = torch.cat(
            [
                current_attention_mask_2d,
                torch.ones(
                    (current_attention_mask_2d.shape[0], 1),
                    device=model.device,
                    dtype=current_attention_mask_2d.dtype,
                ),
            ],
            dim=-1,
        )
        decode_outputs, decode_captured = _capture_text_decode_reference_step(
            model,
            decode_input_ids=decode_input_ids,
            attention_mask_2d=current_attention_mask_2d,
            past_key_values=current_past_key_values,
        )
        next_token_id = int(decode_outputs.logits[0, -1].argmax().item())
        generated_token_ids.append(next_token_id)
        decode_steps.append(
            {
                "step_idx": step_idx,
                "decode_input_ids": cast_cpu(decode_input_ids, None),
                "attention_mask_2d": cast_cpu(current_attention_mask_2d, None),
                "total_seq_len": int(current_attention_mask_2d.shape[-1]),
                "layer_input": cast_cpu(decode_captured["layer_input"], save_dtype_value),
                "stage_output": cast_cpu(decode_captured["stage_output"], save_dtype_value),
                "norm_output": cast_cpu(decode_captured["norm_output"], save_dtype_value),
                "logits": cast_cpu(decode_outputs.logits, save_dtype_value),
                "attention_mask": cast_cpu(decode_captured["attention_mask"], None),
                "cos": cast_cpu(decode_captured["cos"], save_dtype_value),
                "sin": cast_cpu(decode_captured["sin"], save_dtype_value),
                "output_token_id": next_token_id,
            }
        )
        current_past_key_values = decode_outputs.past_key_values

    bundle = {
        "module_name": "text_generate",
        "stage_type": "text_generate",
        "save_dtype": str(save_dtype_value).replace("torch.", ""),
        "original_input_dtype": str(prefill_layer_input.dtype),
        "original_output_dtype": str(prefill_outputs.logits.dtype),
        "original_input_device": str(prefill_layer_input.device),
        "original_output_device": str(prefill_outputs.logits.device),
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "prefill_seq_len": int(prefill_input_ids.shape[-1]),
        "input_ids": cast_cpu(prefill_input_ids, None),
        "attention_mask_2d": cast_cpu(prefill_attention_mask_2d, None),
        "embed_tokens_weight": cast_cpu(text_model.embed_tokens.weight, save_dtype_value),
        "visual_pos_masks": None,
        "deepstack_by_layer": {},
        "deepstack_layer_indices": [],
        "final_norm_weight": cast_cpu(text_model.norm.weight, save_dtype_value),
        "final_norm_eps": text_model.norm.variance_epsilon,
        "lm_head_weight": cast_cpu(model.lm_head.weight, save_dtype_value),
        "lm_head_bias": cast_cpu(model.lm_head.bias, save_dtype_value),
        "layers": layer_bundles,
        "prefill": {
            "layer_input": cast_cpu(prefill_captured["layer_input"], save_dtype_value),
            "stage_output": cast_cpu(prefill_captured["stage_output"], save_dtype_value),
            "norm_output": cast_cpu(prefill_captured["norm_output"], save_dtype_value),
            "logits": cast_cpu(prefill_outputs.logits, save_dtype_value),
            "attention_mask": cast_cpu(prefill_captured["attention_mask"], None),
            "cos": cast_cpu(prefill_captured["cos"], save_dtype_value),
            "sin": cast_cpu(prefill_captured["sin"], save_dtype_value),
        },
        "prefill_cache_layers": [
            {
                "layer_idx": layer_idx,
                "past_key": cast_cpu(past_key, save_dtype_value),
                "past_value": cast_cpu(past_value, save_dtype_value),
            }
            for layer_idx, (past_key, past_value) in enumerate(prefill_cache_layers)
        ],
        "generated_token_ids": cast_cpu(
            torch.tensor([generated_token_ids], device=model.device, dtype=prefill_input_ids.dtype),
            None,
        ),
        "decode_steps": decode_steps,
    }

    save_path = Path(bundle_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, save_path)
    return bundle


def _capture_text_prefill_stage_reference(
    model,
    inputs,
    *,
    start_idx: int,
    end_idx: int,
):
    text_model = model.model.language_model
    layers = text_model.layers
    start_layer = layers[start_idx]
    is_last_stage = end_idx == len(layers) - 1
    captured = {}

    def stage_input_hook(_module, module_inputs):
        captured["stage_input"] = module_inputs[0].detach().clone()

    def stage_output_hook(_module, module_inputs):
        captured["stage_output"] = module_inputs[0].detach().clone()

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

    def final_norm_forward_hook(_module, _module_inputs, module_output):
        captured["norm_output"] = module_output.detach().clone()

    handles = [
        start_layer.input_layernorm.register_forward_pre_hook(stage_input_hook),
        start_layer.self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True),
    ]
    if is_last_stage:
        handles.append(text_model.norm.register_forward_pre_hook(stage_output_hook))
        handles.append(text_model.norm.register_forward_hook(final_norm_forward_hook))
    else:
        handles.append(layers[end_idx + 1].input_layernorm.register_forward_pre_hook(stage_output_hook))

    try:
        with torch.inference_mode():
            outputs = model(
                **inputs,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    required = {
        "stage_input",
        "stage_output",
        "cos",
        "sin",
    }
    if is_last_stage:
        required.add("norm_output")
    if not required.issubset(captured):
        missing = required - set(captured)
        raise RuntimeError(f"没有捕获到 text generate prefill stage 的必要输入: {missing}")

    attention_mask, cos, sin = resolve_runtime_tensors(model, inputs, captured["stage_input"], captured)
    result = {
        "stage_input": captured["stage_input"].detach().clone(),
        "stage_output": captured["stage_output"].detach().clone(),
        "attention_mask": attention_mask,
        "cos": cos,
        "sin": sin,
    }
    if is_last_stage:
        result["norm_output"] = captured["norm_output"].detach().clone()
        result["logits"] = outputs.logits.detach().clone()
    return outputs, result


def _capture_text_decode_stage_reference_step(
    model,
    *,
    start_idx: int,
    end_idx: int,
    decode_input_ids: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    past_key_values,
):
    text_model = model.model.language_model
    layers = text_model.layers
    start_layer = layers[start_idx]
    is_last_stage = end_idx == len(layers) - 1
    captured = {}

    def stage_input_hook(_module, module_inputs):
        captured["stage_input"] = module_inputs[0].detach().clone()

    def stage_output_hook(_module, module_inputs):
        captured["stage_output"] = module_inputs[0].detach().clone()

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

    def final_norm_forward_hook(_module, _module_inputs, module_output):
        captured["norm_output"] = module_output.detach().clone()

    handles = [
        start_layer.input_layernorm.register_forward_pre_hook(stage_input_hook),
        start_layer.self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True),
    ]
    if is_last_stage:
        handles.append(text_model.norm.register_forward_pre_hook(stage_output_hook))
        handles.append(text_model.norm.register_forward_hook(final_norm_forward_hook))
    else:
        handles.append(layers[end_idx + 1].input_layernorm.register_forward_pre_hook(stage_output_hook))

    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=decode_input_ids,
                attention_mask=attention_mask_2d,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    required = {
        "stage_input",
        "stage_output",
        "attention_mask",
        "cos",
        "sin",
    }
    if is_last_stage:
        required.add("norm_output")
    if not required.issubset(captured):
        missing = required - set(captured)
        raise RuntimeError(f"没有捕获到 text generate decode stage 的必要输入: {missing}")

    result = {
        "stage_input": captured["stage_input"].detach().clone(),
        "stage_output": captured["stage_output"].detach().clone(),
        "attention_mask": captured["attention_mask"].detach().clone(),
        "cos": captured["cos"].detach().clone(),
        "sin": captured["sin"].detach().clone(),
    }
    if is_last_stage:
        result["norm_output"] = captured["norm_output"].detach().clone()
        result["logits"] = outputs.logits.detach().clone()
    return outputs, result


def capture_text_generate_stage_bundle(
    *,
    start_idx: int = 0,
    end_idx: int = 35,
    prompt: str = "请用中文简要介绍一下人工智能。",
    max_new_tokens: int = 4,
    bundle_path: str | None = TEXT_STAGE_BUNDLE_PATH,
    save_dtype: str = "auto",
    model_path: str = MODEL_PATH,
) -> dict:
    if start_idx > end_idx:
        raise ValueError("start_idx 不能大于 end_idx。")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens 必须大于 0，当前拿到 {max_new_tokens}。")

    print("Loading model...")
    model = load_model(model_path, attn_implementation="eager")

    print("Loading processor...")
    processor = load_processor(model_path)

    prefill_inputs = build_text_inputs(processor, prompt)
    prefill_inputs = prefill_inputs.to(model.device)

    text_model = model.model.language_model
    layers = text_model.layers
    if end_idx >= len(layers):
        raise ValueError(f"end_idx={end_idx} 超出层数上限 {len(layers) - 1}。")

    is_last_stage = end_idx == len(layers) - 1
    prefill_outputs, prefill_reference = _capture_text_prefill_stage_reference(
        model,
        prefill_inputs,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    prefill_cache_layers = extract_past_key_values(prefill_outputs.past_key_values)

    prefill_input_ids = prefill_inputs["input_ids"]
    prefill_attention_mask_2d = prefill_inputs.get("attention_mask")
    if prefill_attention_mask_2d is None:
        prefill_attention_mask_2d = torch.ones_like(prefill_input_ids)

    save_dtype_value = resolve_save_dtype(save_dtype, prefill_reference["stage_input"])
    layer_bundles = []
    for layer_idx in range(start_idx, end_idx + 1):
        past_key, past_value = prefill_cache_layers[layer_idx]
        layer_bundle = capture_decoder_layer_params(layers[layer_idx], layer_idx, save_dtype_value)
        layer_bundle["past_key"] = cast_cpu(past_key, save_dtype_value)
        layer_bundle["past_value"] = cast_cpu(past_value, save_dtype_value)
        layer_bundles.append(layer_bundle)

    generated_token_ids = [int(prefill_outputs.logits[0, -1].argmax().item())]
    decode_steps = []
    current_attention_mask_2d = prefill_attention_mask_2d
    current_past_key_values = prefill_outputs.past_key_values
    for step_idx in range(max_new_tokens - 1):
        decode_input_ids = torch.tensor(
            [[generated_token_ids[-1]]],
            device=model.device,
            dtype=prefill_input_ids.dtype,
        )
        current_attention_mask_2d = torch.cat(
            [
                current_attention_mask_2d,
                torch.ones(
                    (current_attention_mask_2d.shape[0], 1),
                    device=model.device,
                    dtype=current_attention_mask_2d.dtype,
                ),
            ],
            dim=-1,
        )
        decode_outputs, decode_reference = _capture_text_decode_stage_reference_step(
            model,
            start_idx=start_idx,
            end_idx=end_idx,
            decode_input_ids=decode_input_ids,
            attention_mask_2d=current_attention_mask_2d,
            past_key_values=current_past_key_values,
        )
        next_token_id = int(decode_outputs.logits[0, -1].argmax().item())
        generated_token_ids.append(next_token_id)

        step_payload = {
            "step_idx": step_idx,
            "decode_input_ids": cast_cpu(decode_input_ids, None),
            "attention_mask_2d": cast_cpu(current_attention_mask_2d, None),
            "total_seq_len": int(current_attention_mask_2d.shape[-1]),
            "stage_input": cast_cpu(decode_reference["stage_input"], save_dtype_value),
            "attention_mask": cast_cpu(decode_reference["attention_mask"], None),
            "cos": cast_cpu(decode_reference["cos"], save_dtype_value),
            "sin": cast_cpu(decode_reference["sin"], save_dtype_value),
        }
        if is_last_stage:
            step_payload.update(
                {
                    "stage_output": cast_cpu(decode_reference["logits"], save_dtype_value),
                    "hidden_stage_output": cast_cpu(decode_reference["stage_output"], save_dtype_value),
                    "norm_output": cast_cpu(decode_reference["norm_output"], save_dtype_value),
                    "logits": cast_cpu(decode_reference["logits"], save_dtype_value),
                    "output_token_id": next_token_id,
                }
            )
        else:
            step_payload["stage_output"] = cast_cpu(decode_reference["stage_output"], save_dtype_value)
        decode_steps.append(step_payload)
        current_past_key_values = decode_outputs.past_key_values

    bundle = {
        "module_name": "text_generate_stage",
        "stage_type": "text_generate_last" if is_last_stage else "text_generate",
        "start_idx": start_idx,
        "end_idx": end_idx,
        "save_dtype": str(save_dtype_value).replace("torch.", ""),
        "original_input_dtype": str(prefill_reference["stage_input"].dtype),
        "original_input_device": str(prefill_reference["stage_input"].device),
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "prefill_seq_len": int(prefill_input_ids.shape[-1]),
        "generated_token_ids": cast_cpu(
            torch.tensor([generated_token_ids], device=model.device, dtype=prefill_input_ids.dtype),
            None,
        ),
        "prefill": {
            "attention_mask_2d": cast_cpu(prefill_attention_mask_2d, None),
            "stage_input": cast_cpu(prefill_reference["stage_input"], save_dtype_value),
            "attention_mask": cast_cpu(prefill_reference["attention_mask"], None),
            "cos": cast_cpu(prefill_reference["cos"], save_dtype_value),
            "sin": cast_cpu(prefill_reference["sin"], save_dtype_value),
        },
        "visual_pos_masks": None,
        "deepstack_by_layer": {},
        "deepstack_layer_indices": [],
        "layers": layer_bundles,
        "decode_steps": decode_steps,
    }
    if start_idx == 0:
        bundle["input_ids"] = cast_cpu(prefill_input_ids, None)
        bundle["embed_tokens_weight"] = cast_cpu(text_model.embed_tokens.weight, save_dtype_value)

    if is_last_stage:
        bundle.update(
            {
                "final_norm_weight": cast_cpu(text_model.norm.weight, save_dtype_value),
                "final_norm_eps": text_model.norm.variance_epsilon,
                "lm_head_weight": cast_cpu(model.lm_head.weight, save_dtype_value),
                "lm_head_bias": cast_cpu(model.lm_head.bias, save_dtype_value),
            }
        )
        bundle["prefill"].update(
            {
                "stage_output": cast_cpu(prefill_reference["logits"], save_dtype_value),
                "hidden_stage_output": cast_cpu(prefill_reference["stage_output"], save_dtype_value),
                "norm_output": cast_cpu(prefill_reference["norm_output"], save_dtype_value),
                "logits": cast_cpu(prefill_reference["logits"], save_dtype_value),
                "output_token_id": generated_token_ids[0],
            }
        )
    else:
        bundle["prefill"]["stage_output"] = cast_cpu(prefill_reference["stage_output"], save_dtype_value)

    if bundle_path is not None:
        save_path = Path(bundle_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, save_path)
    return bundle


def capture_text_decode_stage_bundle(
    *,
    start_idx: int = 0,
    end_idx: int = 35,
    prompt: str = "请用中文简要介绍一下人工智能。",
    decode_token_id: int | None = None,
    bundle_path: str | None = TEXT_STAGE_BUNDLE_PATH,
    save_dtype: str = "auto",
    model_path: str = MODEL_PATH,
) -> dict:
    if start_idx > end_idx:
        raise ValueError("start_idx 不能大于 end_idx。")

    print("Loading model...")
    model = load_model(model_path, attn_implementation="eager")

    print("Loading processor...")
    processor = load_processor(model_path)

    prefill_inputs = build_text_inputs(processor, prompt)
    prefill_inputs = prefill_inputs.to(model.device)

    text_model = model.model.language_model
    layers = text_model.layers
    if end_idx >= len(layers):
        raise ValueError(f"end_idx={end_idx} 超出层数上限 {len(layers) - 1}。")

    with torch.inference_mode():
        prefill_outputs = model(
            **prefill_inputs,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )

    prefill_cache_layers = extract_past_key_values(prefill_outputs.past_key_values)
    if len(prefill_cache_layers) != len(layers):
        raise RuntimeError(
            "prefill cache 层数和 decoder layer 数量不一致，"
            f"cache_layers={len(prefill_cache_layers)} decoder_layers={len(layers)}"
        )

    greedy_decode_token_id = int(prefill_outputs.logits[0, -1].argmax().item())
    decode_token_id_value = greedy_decode_token_id if decode_token_id is None else int(decode_token_id)
    decode_source = "greedy_from_prefill" if decode_token_id is None else "provided"

    prefill_input_ids = prefill_inputs["input_ids"]
    prefill_attention_mask_2d = prefill_inputs.get("attention_mask")
    if prefill_attention_mask_2d is None:
        prefill_attention_mask_2d = torch.ones_like(prefill_input_ids)

    decode_input_ids = torch.tensor(
        [[decode_token_id_value]],
        device=model.device,
        dtype=prefill_input_ids.dtype,
    )
    decode_attention_mask_2d = torch.cat(
        [
            prefill_attention_mask_2d,
            torch.ones(
                (prefill_attention_mask_2d.shape[0], 1),
                device=model.device,
                dtype=prefill_attention_mask_2d.dtype,
            ),
        ],
        dim=-1,
    )

    start_layer = layers[start_idx]
    is_last_stage = end_idx == len(layers) - 1
    captured = {}

    def stage_input_hook(_module, module_inputs):
        captured["stage_input"] = module_inputs[0].detach().clone()

    def stage_output_hook(_module, module_inputs):
        captured["stage_output"] = module_inputs[0].detach().clone()

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

    def final_norm_forward_hook(_module, _module_inputs, module_output):
        captured["norm_output"] = module_output.detach().clone()

    handles = [
        start_layer.input_layernorm.register_forward_pre_hook(stage_input_hook),
        start_layer.self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True),
    ]
    if is_last_stage:
        handles.append(text_model.norm.register_forward_pre_hook(stage_output_hook))
        handles.append(text_model.norm.register_forward_hook(final_norm_forward_hook))
    else:
        handles.append(layers[end_idx + 1].input_layernorm.register_forward_pre_hook(stage_output_hook))

    try:
        with torch.inference_mode():
            decode_outputs = model(
                input_ids=decode_input_ids,
                attention_mask=decode_attention_mask_2d,
                past_key_values=prefill_outputs.past_key_values,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    required = {
        "stage_input",
        "stage_output",
        "attention_mask",
        "cos",
        "sin",
    }
    if is_last_stage:
        required.add("norm_output")
    if not required.issubset(captured):
        missing = required - set(captured)
        raise RuntimeError(f"没有捕获到 text decode stage 的必要输入: {missing}")

    layer_input = captured["stage_input"].detach().clone()
    reference_hidden_output = captured["stage_output"].detach().clone()
    save_dtype_value = resolve_save_dtype(save_dtype, layer_input)

    layer_bundles = []
    for layer_idx in range(start_idx, end_idx + 1):
        past_key, past_value = prefill_cache_layers[layer_idx]
        layer_bundle = capture_decoder_layer_params(layers[layer_idx], layer_idx, save_dtype_value)
        layer_bundle["past_key"] = cast_cpu(past_key, save_dtype_value)
        layer_bundle["past_value"] = cast_cpu(past_value, save_dtype_value)
        layer_bundles.append(layer_bundle)

    bundle = {
        "module_name": "text_decode_stage",
        "stage_type": "text_decode_last" if is_last_stage else "text_decode",
        "start_idx": start_idx,
        "end_idx": end_idx,
        "save_dtype": str(save_dtype_value).replace("torch.", ""),
        "original_input_dtype": str(layer_input.dtype),
        "original_input_device": str(layer_input.device),
        "prompt": prompt,
        "decode_source": decode_source,
        "decode_token_id": decode_token_id_value,
        "prefill_seq_len": int(prefill_input_ids.shape[-1]),
        "total_seq_len": int(decode_attention_mask_2d.shape[-1]),
        "prefill_input_ids": cast_cpu(prefill_input_ids, None),
        "decode_input_ids": cast_cpu(decode_input_ids, None),
        "prefill_attention_mask_2d": cast_cpu(prefill_attention_mask_2d, None),
        "attention_mask_2d": cast_cpu(decode_attention_mask_2d, None),
        "stage_input": cast_cpu(layer_input, save_dtype_value),
        "layer_input": cast_cpu(layer_input, save_dtype_value),
        "attention_mask": cast_cpu(captured["attention_mask"], None),
        "cos": cast_cpu(captured["cos"], save_dtype_value),
        "sin": cast_cpu(captured["sin"], save_dtype_value),
        "visual_pos_masks": None,
        "deepstack_by_layer": {},
        "deepstack_layer_indices": [],
        "layers": layer_bundles,
    }

    if is_last_stage:
        logits = decode_outputs.logits.detach().clone()
        norm_output = captured["norm_output"].detach().clone()
        bundle.update(
            {
                "original_output_dtype": str(logits.dtype),
                "original_output_device": str(logits.device),
                "stage_output": cast_cpu(logits, save_dtype_value),
                "layer_output": cast_cpu(logits, save_dtype_value),
                "hidden_stage_output": cast_cpu(reference_hidden_output, save_dtype_value),
                "norm_output": cast_cpu(norm_output, save_dtype_value),
                "logits": cast_cpu(logits, save_dtype_value),
                "final_norm_weight": cast_cpu(text_model.norm.weight, save_dtype_value),
                "final_norm_eps": text_model.norm.variance_epsilon,
                "lm_head_weight": cast_cpu(model.lm_head.weight, save_dtype_value),
                "lm_head_bias": cast_cpu(model.lm_head.bias, save_dtype_value),
            }
        )
    else:
        bundle.update(
            {
                "original_output_dtype": str(reference_hidden_output.dtype),
                "original_output_device": str(reference_hidden_output.device),
                "stage_output": cast_cpu(reference_hidden_output, save_dtype_value),
                "layer_output": cast_cpu(reference_hidden_output, save_dtype_value),
            }
        )

    moved_bundle = move_bundle(bundle, model.device, layer_input.dtype)
    if is_last_stage:
        direct_trace = trace_text_decode_logits(
            bundle["layer_input"].to(model.device),
            moved_bundle,
        )
        reference_norm_output = bundle["norm_output"].to(model.device)
        reference_logits = bundle["logits"].to(model.device)
        reference_hidden_output = bundle["hidden_stage_output"].to(model.device)

        bundle["hidden_stage_sanity_max_diff"] = (
            direct_trace["stage_output"] - reference_hidden_output
        ).abs().max().item()
        bundle["hidden_stage_sanity_mean_diff"] = (
            direct_trace["stage_output"] - reference_hidden_output
        ).abs().mean().item()
        bundle["norm_sanity_max_diff"] = (direct_trace["norm_output"] - reference_norm_output).abs().max().item()
        bundle["norm_sanity_mean_diff"] = (direct_trace["norm_output"] - reference_norm_output).abs().mean().item()
        bundle["logits_sanity_max_diff"] = (direct_trace["logits"] - reference_logits).abs().max().item()
        bundle["logits_sanity_mean_diff"] = (direct_trace["logits"] - reference_logits).abs().mean().item()
        bundle["sanity_max_diff"] = bundle["logits_sanity_max_diff"]
        bundle["sanity_mean_diff"] = bundle["logits_sanity_mean_diff"]
    else:
        direct_output = forward_text_decode_stage(bundle["layer_input"].to(model.device), moved_bundle)
        reference_output = bundle["stage_output"].to(model.device)
        bundle["sanity_max_diff"] = (direct_output - reference_output).abs().max().item()
        bundle["sanity_mean_diff"] = (direct_output - reference_output).abs().mean().item()

    if bundle_path is not None:
        save_path = Path(bundle_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, save_path)
    return bundle


def capture_text_prefill_bundle(
    *,
    prompt: str = "请用中文简要介绍一下人工智能。",
    bundle_path: str = TEXT_PREFILL_BUNDLE_PATH,
    save_dtype: str = "auto",
    model_path: str = MODEL_PATH,
) -> dict:
    print("Loading model...")
    model = load_model(model_path, attn_implementation="eager")

    print("Loading processor...")
    processor = load_processor(model_path)

    inputs = build_text_inputs(processor, prompt)
    inputs = inputs.to(model.device)

    text_model = model.model.language_model
    layers = text_model.layers
    first_layer = layers[0]
    captured = {}

    def stage_input_hook(_module, module_inputs):
        captured["layer_input"] = module_inputs[0].detach().clone()

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

    def final_norm_pre_hook(_module, module_inputs):
        captured["stage_output"] = module_inputs[0].detach().clone()

    def final_norm_forward_hook(_module, _module_inputs, module_output):
        captured["norm_output"] = module_output.detach().clone()

    handles = [
        first_layer.input_layernorm.register_forward_pre_hook(stage_input_hook),
        first_layer.self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True),
        text_model.norm.register_forward_pre_hook(final_norm_pre_hook),
        text_model.norm.register_forward_hook(final_norm_forward_hook),
    ]
    try:
        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    required = {
        "layer_input",
        "stage_output",
        "norm_output",
        "cos",
        "sin",
    }
    if not required.issubset(captured):
        missing = required - set(captured)
        raise RuntimeError(f"没有捕获到 text prefill logits 的必要输入: {missing}")

    logits = outputs.logits.detach().clone()
    layer_input = captured["layer_input"].detach().clone()
    attention_mask, cos, sin = resolve_runtime_tensors(model, inputs, layer_input, captured)
    save_dtype_value = resolve_save_dtype(save_dtype, layer_input)

    layer_bundles = []
    for layer_idx, layer in enumerate(layers):
        layer_bundles.append(capture_decoder_layer_params(layer, layer_idx, save_dtype_value))

    bundle = {
        "module_name": "text_prefill",
        "stage_type": "text",
        "start_idx": 0,
        "end_idx": len(layers) - 1,
        "save_dtype": str(save_dtype_value).replace("torch.", ""),
        "original_input_dtype": str(layer_input.dtype),
        "original_output_dtype": str(logits.dtype),
        "original_input_device": str(layer_input.device),
        "original_output_device": str(logits.device),
        "prompt": prompt,
        "input_ids": cast_cpu(inputs["input_ids"], None),
        "attention_mask_2d": cast_cpu(inputs.get("attention_mask"), None),
        "embed_tokens_weight": cast_cpu(text_model.embed_tokens.weight, save_dtype_value),
        "stage_input": cast_cpu(layer_input, save_dtype_value),
        "layer_input": cast_cpu(layer_input, save_dtype_value),
        "stage_output": cast_cpu(captured["stage_output"], save_dtype_value),
        "layer_output": cast_cpu(captured["stage_output"], save_dtype_value),
        "norm_output": cast_cpu(captured["norm_output"], save_dtype_value),
        "logits": cast_cpu(logits, save_dtype_value),
        "attention_mask": cast_cpu(attention_mask, None),
        "cos": cast_cpu(cos, save_dtype_value),
        "sin": cast_cpu(sin, save_dtype_value),
        "visual_pos_masks": None,
        "deepstack_by_layer": {},
        "deepstack_layer_indices": [],
        "final_norm_weight": cast_cpu(text_model.norm.weight, save_dtype_value),
        "final_norm_eps": text_model.norm.variance_epsilon,
        "lm_head_weight": cast_cpu(model.lm_head.weight, save_dtype_value),
        "lm_head_bias": cast_cpu(model.lm_head.bias, save_dtype_value),
        "layers": layer_bundles,
    }

    moved_bundle = move_bundle(bundle, model.device, layer_input.dtype)
    direct_trace = trace_text_prefill_logits(
        bundle["layer_input"].to(model.device),
        moved_bundle,
    )
    reference_stage_output = bundle["stage_output"].to(model.device)
    reference_norm_output = bundle["norm_output"].to(model.device)
    reference_logits = bundle["logits"].to(model.device)

    bundle["stage_sanity_max_diff"] = (direct_trace["stage_output"] - reference_stage_output).abs().max().item()
    bundle["stage_sanity_mean_diff"] = (direct_trace["stage_output"] - reference_stage_output).abs().mean().item()
    bundle["norm_sanity_max_diff"] = (direct_trace["norm_output"] - reference_norm_output).abs().max().item()
    bundle["norm_sanity_mean_diff"] = (direct_trace["norm_output"] - reference_norm_output).abs().mean().item()
    bundle["logits_sanity_max_diff"] = (direct_trace["logits"] - reference_logits).abs().max().item()
    bundle["logits_sanity_mean_diff"] = (direct_trace["logits"] - reference_logits).abs().mean().item()
    bundle["sanity_max_diff"] = bundle["logits_sanity_max_diff"]
    bundle["sanity_mean_diff"] = bundle["logits_sanity_mean_diff"]

    save_path = Path(bundle_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, save_path)
    return bundle


def capture_text_prefill_stage_bundle(
    *,
    start_idx: int = 0,
    end_idx: int = 35,
    prompt: str = "请用中文简要介绍一下人工智能。",
    bundle_path: str | None = TEXT_STAGE_BUNDLE_PATH,
    save_dtype: str = "auto",
    model_path: str = MODEL_PATH,
) -> dict:
    if start_idx > end_idx:
        raise ValueError("start_idx 不能大于 end_idx。")

    print("Loading model...")
    model = load_model(model_path, attn_implementation="eager")

    print("Loading processor...")
    processor = load_processor(model_path)

    inputs = build_text_inputs(processor, prompt)
    inputs = inputs.to(model.device)

    text_model = model.model.language_model
    layers = text_model.layers
    if end_idx >= len(layers):
        raise ValueError(f"end_idx={end_idx} 超出层数上限 {len(layers) - 1}。")

    start_layer = layers[start_idx]
    is_last_stage = end_idx == len(layers) - 1
    captured = {}

    def stage_input_hook(_module, module_inputs):
        captured["stage_input"] = module_inputs[0].detach().clone()

    def stage_output_hook(_module, module_inputs):
        captured["stage_output"] = module_inputs[0].detach().clone()

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

    def final_norm_forward_hook(_module, _module_inputs, module_output):
        captured["norm_output"] = module_output.detach().clone()

    handles = [
        start_layer.input_layernorm.register_forward_pre_hook(stage_input_hook),
        start_layer.self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True),
    ]
    if is_last_stage:
        handles.append(text_model.norm.register_forward_pre_hook(stage_output_hook))
        handles.append(text_model.norm.register_forward_hook(final_norm_forward_hook))
    else:
        handles.append(layers[end_idx + 1].input_layernorm.register_forward_pre_hook(stage_output_hook))

    try:
        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    required = {
        "stage_input",
        "stage_output",
        "cos",
        "sin",
    }
    if is_last_stage:
        required.add("norm_output")
    if not required.issubset(captured):
        missing = required - set(captured)
        raise RuntimeError(f"没有捕获到 text prefill stage 的必要输入: {missing}")

    layer_input = captured["stage_input"].detach().clone()
    reference_hidden_output = captured["stage_output"].detach().clone()
    attention_mask, cos, sin = resolve_runtime_tensors(model, inputs, layer_input, captured)
    save_dtype_value = resolve_save_dtype(save_dtype, layer_input)

    layer_bundles = []
    for layer_idx in range(start_idx, end_idx + 1):
        layer_bundles.append(capture_decoder_layer_params(layers[layer_idx], layer_idx, save_dtype_value))

    bundle = {
        "module_name": "text_prefill_stage",
        "stage_type": "text_prefill_last" if is_last_stage else "text",
        "start_idx": start_idx,
        "end_idx": end_idx,
        "save_dtype": str(save_dtype_value).replace("torch.", ""),
        "original_input_dtype": str(layer_input.dtype),
        "original_input_device": str(layer_input.device),
        "prompt": prompt,
        "input_ids": cast_cpu(inputs["input_ids"], None),
        "attention_mask_2d": cast_cpu(inputs.get("attention_mask"), None),
        "stage_input": cast_cpu(layer_input, save_dtype_value),
        "layer_input": cast_cpu(layer_input, save_dtype_value),
        "attention_mask": cast_cpu(attention_mask, None),
        "cos": cast_cpu(cos, save_dtype_value),
        "sin": cast_cpu(sin, save_dtype_value),
        "visual_pos_masks": None,
        "deepstack_by_layer": {},
        "deepstack_layer_indices": [],
        "layers": layer_bundles,
    }

    if is_last_stage:
        logits = outputs.logits.detach().clone()
        norm_output = captured["norm_output"].detach().clone()
        bundle.update(
            {
                "original_output_dtype": str(logits.dtype),
                "original_output_device": str(logits.device),
                "stage_output": cast_cpu(logits, save_dtype_value),
                "layer_output": cast_cpu(logits, save_dtype_value),
                "hidden_stage_output": cast_cpu(reference_hidden_output, save_dtype_value),
                "norm_output": cast_cpu(norm_output, save_dtype_value),
                "logits": cast_cpu(logits, save_dtype_value),
                "final_norm_weight": cast_cpu(text_model.norm.weight, save_dtype_value),
                "final_norm_eps": text_model.norm.variance_epsilon,
                "lm_head_weight": cast_cpu(model.lm_head.weight, save_dtype_value),
                "lm_head_bias": cast_cpu(model.lm_head.bias, save_dtype_value),
            }
        )
    else:
        bundle.update(
            {
                "original_output_dtype": str(reference_hidden_output.dtype),
                "original_output_device": str(reference_hidden_output.device),
                "stage_output": cast_cpu(reference_hidden_output, save_dtype_value),
                "layer_output": cast_cpu(reference_hidden_output, save_dtype_value),
            }
        )

    moved_bundle = move_bundle(bundle, model.device, layer_input.dtype)
    if is_last_stage:
        direct_trace = trace_text_prefill_stage_logits(
            bundle["layer_input"].to(model.device),
            moved_bundle,
        )
        reference_norm_output = bundle["norm_output"].to(model.device)
        reference_logits = bundle["logits"].to(model.device)
        reference_hidden_output = bundle["hidden_stage_output"].to(model.device)

        bundle["hidden_stage_sanity_max_diff"] = (
            direct_trace["hidden_stage_output"] - reference_hidden_output
        ).abs().max().item()
        bundle["hidden_stage_sanity_mean_diff"] = (
            direct_trace["hidden_stage_output"] - reference_hidden_output
        ).abs().mean().item()
        bundle["norm_sanity_max_diff"] = (direct_trace["norm_output"] - reference_norm_output).abs().max().item()
        bundle["norm_sanity_mean_diff"] = (direct_trace["norm_output"] - reference_norm_output).abs().mean().item()
        bundle["logits_sanity_max_diff"] = (direct_trace["logits"] - reference_logits).abs().max().item()
        bundle["logits_sanity_mean_diff"] = (direct_trace["logits"] - reference_logits).abs().mean().item()
        bundle["sanity_max_diff"] = bundle["logits_sanity_max_diff"]
        bundle["sanity_mean_diff"] = bundle["logits_sanity_mean_diff"]
    else:
        direct_output = forward_text_stage(bundle["layer_input"].to(model.device), moved_bundle)
        reference_output = bundle["stage_output"].to(model.device)
        bundle["sanity_max_diff"] = (direct_output - reference_output).abs().max().item()
        bundle["sanity_mean_diff"] = (direct_output - reference_output).abs().mean().item()

    if bundle_path is not None:
        save_path = Path(bundle_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, save_path)
    return bundle


__all__ = [
    "capture_text_decode_bundle",
    "capture_text_decode_stage_bundle",
    "capture_text_generate_bundle",
    "capture_text_generate_stage_bundle",
    "capture_text_prefill_bundle",
    "capture_text_prefill_stage_bundle",
]
