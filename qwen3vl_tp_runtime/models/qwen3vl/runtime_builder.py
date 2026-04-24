"""Direct runtime builders that construct stage bundles from a live model_path session."""

from __future__ import annotations

import gc
from typing import Any

import torch
import torch.nn.functional as F

from qwen3vl_tp_runtime.hexgen_core.distributed import startup_log, startup_timer
from qwen3vl_tp_runtime.hexgen_core.gen_hetero_groups import build_hybrid_layout, parse_tp_degrees
from qwen3vl_tp_runtime.hexgen_core.schema import StageSpec, TextHybridManifest, TextPipelineManifest
from qwen3vl_tp_runtime.models.qwen3vl.execution import (
    apply_deepstack,
    trace_text_decode_stage_with_runtime_cache,
)
from qwen3vl_tp_runtime.models.qwen3vl.functional import rms_norm
from qwen3vl_tp_runtime.models.qwen3vl.live import (
    build_cache_by_layer_from_past_key_values,
    extract_decoder_layer_params_live,
    prepare_multimodal_decode_runtime_inputs,
    prepare_multimodal_prefill_runtime_inputs,
    prepare_text_decode_runtime_inputs,
    prepare_text_prefill_runtime_inputs,
)
from qwen3vl_tp_runtime.models.qwen3vl.live.common import MultimodalRuntimeInputs, _resolve_compute_dtype, _runtime_tensor
from qwen3vl_tp_runtime.models.qwen3vl.processing import (
    build_inputs,
    build_text_inputs,
    list_frames,
    load_model,
    load_processor,
    load_text_tokenizer,
    load_text_tokenizer_backend,
)
from qwen3vl_tp_runtime.models.qwen3vl.weights import (
    TextModelConfigSpec,
    TextStageWeightBundle,
    build_text_rotary_embedding,
    build_text_runtime_aux_tensors,
    load_model_weight_index,
    load_tensors_from_index,
    load_text_decoder_stage_weight_bundle,
    load_text_model_config_spec,
    prepare_text_decode_runtime_inputs_from_weights,
    prepare_text_prefill_runtime_inputs_from_weights,
)


def _pipeline_type(modality: str, mode: str) -> str:
    if modality == "text":
        return f"text_{mode}"
    return f"multimodal_{mode}"


def _runtime_name(modality: str, mode: str, backend: str) -> str:
    return f"{_pipeline_type(modality, mode)}_{backend}"


def _build_runtime_config(
    *,
    modality: str,
    mode: str,
    model_path: str,
    save_dtype: str,
    prompt: str | None = None,
    decode_token_id: int | None = None,
    max_new_tokens: int | None = None,
    num_frames: int | None = None,
    frame_dir: str | None = None,
    include_runtime_reference: bool | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "modality": modality,
        "mode": mode,
        "model_path": model_path,
        "save_dtype": save_dtype,
    }
    if prompt is not None:
        config["prompt"] = prompt
    if decode_token_id is not None:
        config["decode_token_id"] = int(decode_token_id)
    if max_new_tokens is not None:
        config["max_new_tokens"] = int(max_new_tokens)
    if num_frames is not None:
        config["num_frames"] = int(num_frames)
    if frame_dir is not None:
        config["frame_dir"] = frame_dir
    if include_runtime_reference is not None:
        config["include_runtime_reference"] = bool(include_runtime_reference)
    return config


def _save_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _default_attention_mask_2d(
    input_ids: torch.Tensor | None,
    attention_mask_2d: torch.Tensor | None,
) -> torch.Tensor | None:
    if attention_mask_2d is not None:
        return attention_mask_2d
    if input_ids is None:
        return None
    return torch.ones_like(input_ids)


def _extract_text_position_ids(runtime_inputs: MultimodalRuntimeInputs) -> torch.Tensor | None:
    position_ids = runtime_inputs.position_ids
    if position_ids is None:
        return None
    if position_ids.ndim == 2:
        return position_ids
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        return position_ids[0]
    return None


def _build_deepstack_visual_embeds(runtime_inputs: MultimodalRuntimeInputs) -> list[torch.Tensor | None] | None:
    if not runtime_inputs.deepstack_by_layer:
        return None
    max_layer_idx = max(int(layer_idx) for layer_idx in runtime_inputs.deepstack_by_layer)
    return [runtime_inputs.deepstack_by_layer.get(layer_idx) for layer_idx in range(max_layer_idx + 1)]


def _default_runtime_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _strip_text_phase_runtime_inputs(phase_payload: dict[str, Any]) -> dict[str, Any]:
    compact_payload = dict(phase_payload)
    compact_payload.pop("attention_mask", None)
    compact_payload.pop("cos", None)
    compact_payload.pop("sin", None)
    compact_payload.pop("position_ids", None)
    return compact_payload


def _compact_text_generate_scaffold(bundle: dict[str, Any]) -> dict[str, Any]:
    scaffold = dict(bundle)
    scaffold["runtime_inputs_local_rebuild"] = True
    scaffold["runtime_prefill_cache_policy"] = "recompute"
    scaffold.pop("cache_by_layer", None)
    scaffold["prefill"] = _strip_text_phase_runtime_inputs(scaffold["prefill"])
    scaffold["decode_steps"] = [
        _strip_text_phase_runtime_inputs(step_payload)
        for step_payload in scaffold["decode_steps"]
    ]
    return scaffold


def _compact_runtime_only_text_generate_scaffold(bundle: dict[str, Any]) -> dict[str, Any]:
    scaffold = dict(bundle)
    scaffold["runtime_only_prompt_local_rebuild"] = True
    scaffold.pop("input_ids", None)
    scaffold.pop("prefill_attention_mask_2d", None)
    scaffold.pop("prefill_seq_len", None)
    scaffold.pop("batch_size", None)
    scaffold.pop("token_id_dtype", None)
    return scaffold


def _build_runtime_only_text_generate_bundle(
    *,
    spec: StageSpec,
    bundle_device: torch.device,
    compute_dtype: torch.dtype,
    prefill_attention_mask_2d: torch.Tensor,
    prefill_seq_len: int,
    batch_size: int,
    token_id_dtype: torch.dtype,
    hidden_size: int,
    layers: list[dict[str, Any]],
    text_stage_weights: TextStageWeightBundle | None,
) -> dict[str, Any]:
    bundle = {
        "module_name": "text_generate_stage",
        "stage_type": "text_generate_runtime_only",
        "runtime_only_generate": True,
        "start_idx": spec.start_idx,
        "end_idx": spec.end_idx,
        "save_dtype": _save_dtype_name(compute_dtype),
        "prefill_seq_len": int(prefill_seq_len),
        "max_new_tokens": 0,
        "prefill_attention_mask_2d": _runtime_tensor(prefill_attention_mask_2d, device=bundle_device),
        "batch_size": int(batch_size),
        "token_id_dtype": _save_dtype_name(token_id_dtype),
        "hidden_size": int(hidden_size),
        "layers": layers,
        "tp_weight_sharded": False if text_stage_weights is None else text_stage_weights.tp_weight_sharded,
        "tp_shard_rank": None if text_stage_weights is None else text_stage_weights.tp_shard_rank,
        "tp_shard_world_size": None if text_stage_weights is None else text_stage_weights.tp_shard_world_size,
    }
    return bundle


def _restore_text_phase_runtime_inputs(
    phase_payload: dict[str, Any],
    *,
    config_spec: TextModelConfigSpec,
    compute_dtype: torch.dtype,
    rotary_emb,
) -> dict[str, Any]:
    if (
        phase_payload.get("attention_mask") is not None
        and phase_payload.get("cos") is not None
        and phase_payload.get("sin") is not None
    ):
        return phase_payload

    stage_input = phase_payload.get("stage_input")
    if not torch.is_tensor(stage_input):
        raise RuntimeError("text scaffold 缺少 stage_input，无法本地重建 runtime inputs。")

    attention_mask_2d = phase_payload.get("attention_mask_2d")
    if attention_mask_2d is not None and not torch.is_tensor(attention_mask_2d):
        raise RuntimeError("text scaffold 的 attention_mask_2d 不是 tensor，无法本地重建 runtime inputs。")

    batch_size, seq_len, _hidden_size = stage_input.shape
    total_seq_len = int(attention_mask_2d.shape[-1]) if attention_mask_2d is not None else seq_len
    past_length = total_seq_len - seq_len
    runtime_aux = build_text_runtime_aux_tensors(
        attention_mask_2d=attention_mask_2d,
        batch_size=batch_size,
        seq_len=seq_len,
        past_length=past_length,
        config_spec=config_spec,
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
        rotary_emb=rotary_emb,
    )
    restored_payload = dict(phase_payload)
    restored_payload["attention_mask"] = _runtime_tensor(runtime_aux["attention_mask"], device=torch.device("cpu"))
    restored_payload["position_ids"] = _runtime_tensor(runtime_aux["position_ids"], device=torch.device("cpu"))
    restored_payload["cos"] = _runtime_tensor(
        runtime_aux["cos"],
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
    )
    restored_payload["sin"] = _runtime_tensor(
        runtime_aux["sin"],
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
    )
    return restored_payload


def _restore_text_generate_bundle_runtime_inputs(
    bundle: dict[str, Any],
    *,
    config_spec: TextModelConfigSpec,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    if not bundle.get("runtime_inputs_local_rebuild"):
        return bundle

    rotary_emb = build_text_rotary_embedding(config_spec, device=torch.device("cpu"))
    restored_bundle = dict(bundle)
    restored_bundle["prefill"] = _restore_text_phase_runtime_inputs(
        restored_bundle["prefill"],
        config_spec=config_spec,
        compute_dtype=compute_dtype,
        rotary_emb=rotary_emb,
    )
    restored_bundle["decode_steps"] = [
        _restore_text_phase_runtime_inputs(
            step_payload,
            config_spec=config_spec,
            compute_dtype=compute_dtype,
            rotary_emb=rotary_emb,
        )
        for step_payload in restored_bundle["decode_steps"]
    ]
    return restored_bundle


def _prepare_prefill_session(
    runtime_config: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, torch.Tensor], MultimodalRuntimeInputs, dict[str, Any]]:
    modality = runtime_config["modality"]
    model_path = runtime_config["model_path"]

    model = load_model(model_path, attn_implementation="eager")
    processor = load_processor(model_path)

    if modality == "text":
        prompt = runtime_config.get("prompt", "请用中文简要介绍一下人工智能。")
        raw_inputs = build_text_inputs(processor, prompt)
        raw_inputs = raw_inputs.to(model.device)
        runtime_inputs = prepare_text_prefill_runtime_inputs(model, raw_inputs)
        extra = {"prompt": prompt}
    elif modality == "multimodal":
        num_frames = int(runtime_config.get("num_frames", 8))
        frame_paths = list_frames(num_frames, runtime_config.get("frame_dir"))
        raw_inputs = build_inputs(processor, frame_paths)
        raw_inputs = raw_inputs.to(model.device)
        runtime_inputs = prepare_multimodal_prefill_runtime_inputs(model, raw_inputs)
        extra = {"num_frames": len(frame_paths), "frame_paths": frame_paths}
    else:
        raise ValueError(f"不支持的 modality={modality!r}")

    return model, raw_inputs, runtime_inputs, extra


def _load_text_compute_dtype_reference_tensor(
    weight_index,
    *,
    device: torch.device,
) -> torch.Tensor:
    candidate_names = (
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.norm.weight",
        "model.language_model.embed_tokens.weight",
    )
    for tensor_name in candidate_names:
        if not weight_index.has_tensor(tensor_name):
            continue
        loaded = load_tensors_from_index(
            weight_index,
            [tensor_name],
            device=device,
            compute_dtype=None,
            strict=True,
        )
        tensor = loaded.get(tensor_name)
        if tensor is not None:
            return tensor
    raise RuntimeError("无法为 text runtime-only session 找到可用于推断 compute dtype 的参考权重。")


def _build_text_prompt_messages(prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]


def _build_qwen_text_generation_prompt(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def compact_runtime_only_text_prompt_metadata_for_broadcast(
    prompt_metadata: dict[str, torch.Tensor] | None,
) -> dict[str, object] | None:
    if prompt_metadata is None:
        return None

    input_ids = prompt_metadata["input_ids"]
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise RuntimeError(
            "runtime-only text prompt metadata 当前只支持 batch_size=1 的 input_ids 广播压缩。"
        )

    compact_payload: dict[str, object] = {
        "input_ids_list": [int(token_id) for token_id in input_ids[0].tolist()],
    }
    attention_mask = prompt_metadata.get("attention_mask")
    if attention_mask is not None:
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise RuntimeError("runtime-only text attention_mask 形状和 input_ids 不匹配，无法压缩广播。")
        if not torch.all(attention_mask == 1):
            compact_payload["attention_mask_list"] = [int(mask) for mask in attention_mask[0].tolist()]
    return compact_payload


def restore_runtime_only_text_prompt_metadata_from_broadcast(
    prompt_metadata: dict[str, object],
) -> dict[str, torch.Tensor]:
    if "input_ids" in prompt_metadata:
        restored = {
            "input_ids": prompt_metadata["input_ids"],
        }
        if prompt_metadata.get("attention_mask") is not None:
            restored["attention_mask"] = prompt_metadata["attention_mask"]
        return restored

    input_ids_list = prompt_metadata.get("input_ids_list")
    if input_ids_list is None:
        raise RuntimeError("runtime-only text prompt metadata 广播负载缺少 input_ids 或 input_ids_list。")

    input_ids = torch.tensor([input_ids_list], dtype=torch.int64)
    restored = {"input_ids": input_ids}
    attention_mask_list = prompt_metadata.get("attention_mask_list")
    if attention_mask_list is not None:
        restored["attention_mask"] = torch.tensor([attention_mask_list], dtype=torch.int64)
    return restored


def _normalize_runtime_only_text_prompt_metadata(
    raw_inputs: dict[str, Any],
) -> dict[str, torch.Tensor]:
    input_ids = _runtime_tensor(raw_inputs["input_ids"], device=torch.device("cpu"))
    attention_mask = _runtime_tensor(raw_inputs.get("attention_mask"), device=torch.device("cpu"))
    if attention_mask is not None and torch.all(attention_mask == 1):
        attention_mask = None
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def prepare_runtime_only_text_generate_prompt_metadata(
    runtime_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    model_path = runtime_config["model_path"]
    prompt = runtime_config.get("prompt", "请用中文简要介绍一下人工智能。")
    try:
        tokenizer_backend = load_text_tokenizer_backend(model_path)
        encoded = tokenizer_backend.encode(_build_qwen_text_generation_prompt(prompt))
        return {
            "input_ids": torch.tensor([encoded.ids], dtype=torch.int64),
            "attention_mask": None,
        }
    except Exception:
        try:
            tokenizer = load_text_tokenizer(model_path)
            raw_inputs = tokenizer.apply_chat_template(
                _build_text_prompt_messages(prompt),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            processor = load_processor(model_path)
            raw_inputs = build_text_inputs(processor, prompt)
    return _normalize_runtime_only_text_prompt_metadata(raw_inputs)


def _resolve_runtime_only_text_prompt_metadata(
    runtime_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    input_ids = runtime_config.get("_runtime_only_input_ids")
    if input_ids is None:
        return prepare_runtime_only_text_generate_prompt_metadata(runtime_config)
    return {
        "input_ids": _runtime_tensor(input_ids, device=torch.device("cpu")),
        "attention_mask": _runtime_tensor(
            runtime_config.get("_runtime_only_attention_mask"),
            device=torch.device("cpu"),
        ),
    }


def _restore_runtime_only_text_generate_prompt_metadata(
    bundle: dict[str, Any],
    *,
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    if not bundle.get("runtime_only_generate"):
        return bundle

    needs_restore = bool(bundle.pop("runtime_only_prompt_local_rebuild", False))
    is_first_stage = int(bundle["start_idx"]) == 0
    if (
        not needs_restore
        and bundle.get("prefill_attention_mask_2d") is not None
        and (not is_first_stage or bundle.get("input_ids") is not None)
    ):
        return bundle

    prompt_metadata = _resolve_runtime_only_text_prompt_metadata(runtime_config)
    input_ids = _runtime_tensor(prompt_metadata["input_ids"], device=torch.device("cpu"))
    attention_mask_2d = _default_attention_mask_2d(
        input_ids,
        prompt_metadata.get("attention_mask"),
    )

    restored = dict(bundle)
    restored["prefill_attention_mask_2d"] = _runtime_tensor(
        attention_mask_2d,
        device=torch.device("cpu"),
    )
    restored["prefill_seq_len"] = int(input_ids.shape[-1])
    restored["batch_size"] = int(input_ids.shape[0])
    restored["token_id_dtype"] = _save_dtype_name(input_ids.dtype)
    if is_first_stage:
        restored["input_ids"] = input_ids
    return restored


def _prepare_runtime_only_text_generate_session(
    weight_index,
    runtime_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], torch.dtype, dict[str, Any], torch.device]:
    raw_inputs = _resolve_runtime_only_text_prompt_metadata(runtime_config)
    compute_dtype_reference = _load_text_compute_dtype_reference_tensor(
        weight_index,
        device=torch.device("cpu"),
    )
    compute_dtype = _resolve_compute_dtype(
        compute_dtype_reference,
        runtime_config.get("save_dtype", "auto"),
    )
    return raw_inputs, compute_dtype, {}, _default_runtime_device()


def _run_live_prefill_stage_reference(
    model,
    *,
    runtime_inputs: MultimodalRuntimeInputs,
    start_idx: int,
    end_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    text_model = model.model.language_model
    text_position_ids = _extract_text_position_ids(runtime_inputs)
    position_embeddings = (runtime_inputs.cos, runtime_inputs.sin)

    hidden_states = runtime_inputs.inputs_embeds
    stage_input = hidden_states.detach().clone() if start_idx == 0 else None

    with torch.inference_mode():
        for layer_idx in range(end_idx + 1):
            hidden_states = text_model.layers[layer_idx](
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=runtime_inputs.attention_mask,
                position_ids=text_position_ids,
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
        raise RuntimeError(f"没有拿到 prefill stage_input，start_idx={start_idx} end_idx={end_idx}")
    return stage_input, hidden_states.detach().clone()


def _run_live_prefill_full(model, runtime_inputs: MultimodalRuntimeInputs):
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
            deepstack_visual_embeds=_build_deepstack_visual_embeds(runtime_inputs),
        )


def _run_live_decode_full(
    model,
    *,
    runtime_inputs: MultimodalRuntimeInputs,
    past_key_values,
    is_last_stage: bool,
):
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
                deepstack_visual_embeds=_build_deepstack_visual_embeds(runtime_inputs),
            )
    finally:
        if handle is not None:
            handle.remove()

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("decode full forward 请求了 output_hidden_states=True，但没有拿到 hidden_states。")
    return outputs, hidden_states, captured.get("hidden_stage_output")


def _compute_logits(model, norm_output: torch.Tensor) -> torch.Tensor:
    return F.linear(norm_output, model.lm_head.weight, model.lm_head.bias)


def _build_stage_deepstack_payload(
    runtime_inputs: MultimodalRuntimeInputs,
    *,
    start_idx: int,
    end_idx: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor | None, dict[int, torch.Tensor]]:
    deepstack_by_layer = {
        int(layer_idx): _runtime_tensor(deepstack_embeds, device=device, compute_dtype=compute_dtype)
        for layer_idx, deepstack_embeds in runtime_inputs.deepstack_by_layer.items()
        if start_idx <= int(layer_idx) <= end_idx
    }
    visual_pos_masks = runtime_inputs.visual_pos_masks
    if not deepstack_by_layer:
        visual_pos_masks = None
    return _runtime_tensor(visual_pos_masks, device=device), deepstack_by_layer


def _build_stage_layer_bundles(
    model,
    *,
    start_idx: int,
    end_idx: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
) -> list[dict[str, Any]]:
    text_model = model.model.language_model
    bundles = []
    for layer_idx in range(start_idx, end_idx + 1):
        layer_bundle = extract_decoder_layer_params_live(
            text_model.layers[layer_idx],
            layer_idx,
            device=device,
            compute_dtype=compute_dtype,
        )
        if cache_by_layer is not None and layer_idx in cache_by_layer:
            past_key, past_value = cache_by_layer[layer_idx]
            layer_bundle["past_key"] = _runtime_tensor(past_key, device=device, compute_dtype=compute_dtype)
            layer_bundle["past_value"] = _runtime_tensor(past_value, device=device, compute_dtype=compute_dtype)
        bundles.append(layer_bundle)
    return bundles


def _build_prefill_stage_bundle(
    runtime_config: dict[str, Any],
    *,
    start_idx: int,
    end_idx: int,
) -> dict[str, Any]:
    model, raw_inputs, runtime_inputs, extra = _prepare_prefill_session(runtime_config)
    device = next(model.parameters()).device
    compute_dtype = _resolve_compute_dtype(runtime_inputs.inputs_embeds, runtime_config.get("save_dtype", "auto"))

    stage_input, hidden_stage_output = _run_live_prefill_stage_reference(
        model,
        runtime_inputs=runtime_inputs,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    is_last_stage = end_idx == len(model.model.language_model.layers) - 1
    layer_bundles = _build_stage_layer_bundles(
        model,
        start_idx=start_idx,
        end_idx=end_idx,
        device=device,
        compute_dtype=compute_dtype,
    )
    visual_pos_masks, deepstack_by_layer = _build_stage_deepstack_payload(
        runtime_inputs,
        start_idx=start_idx,
        end_idx=end_idx,
        device=device,
        compute_dtype=compute_dtype,
    )

    bundle = {
        "module_name": f"{runtime_config['modality']}_prefill_stage",
        "stage_type": "text_prefill_last" if is_last_stage else "text",
        "start_idx": start_idx,
        "end_idx": end_idx,
        "save_dtype": _save_dtype_name(compute_dtype),
        "original_input_dtype": str(stage_input.dtype),
        "original_input_device": str(stage_input.device),
        "attention_mask_2d": _runtime_tensor(runtime_inputs.attention_mask_2d, device=device),
        "stage_input": _runtime_tensor(stage_input, device=device, compute_dtype=compute_dtype),
        "layer_input": _runtime_tensor(stage_input, device=device, compute_dtype=compute_dtype),
        "attention_mask": _runtime_tensor(runtime_inputs.attention_mask, device=device),
        "cos": _runtime_tensor(runtime_inputs.cos, device=device, compute_dtype=compute_dtype),
        "sin": _runtime_tensor(runtime_inputs.sin, device=device, compute_dtype=compute_dtype),
        "visual_pos_masks": visual_pos_masks,
        "deepstack_by_layer": deepstack_by_layer,
        "deepstack_layer_indices": sorted(deepstack_by_layer),
        "layers": layer_bundles,
    }

    if runtime_config["modality"] == "text":
        bundle["prompt"] = extra["prompt"]
        bundle["input_ids"] = _runtime_tensor(runtime_inputs.input_ids, device=device)
    else:
        bundle["num_frames"] = extra["num_frames"]
        bundle["frame_paths"] = extra["frame_paths"]
        bundle["input_ids"] = _runtime_tensor(raw_inputs.get("input_ids"), device=device)

    if is_last_stage:
        text_model = model.model.language_model
        norm_output = text_model.norm(hidden_stage_output)
        logits = _compute_logits(model, norm_output)
        bundle.update(
            {
                "original_output_dtype": str(logits.dtype),
                "original_output_device": str(logits.device),
                "stage_output": _runtime_tensor(logits, device=device, compute_dtype=compute_dtype),
                "layer_output": _runtime_tensor(logits, device=device, compute_dtype=compute_dtype),
                "hidden_stage_output": _runtime_tensor(hidden_stage_output, device=device, compute_dtype=compute_dtype),
                "norm_output": _runtime_tensor(norm_output, device=device, compute_dtype=compute_dtype),
                "logits": _runtime_tensor(logits, device=device, compute_dtype=compute_dtype),
                "final_norm_weight": _runtime_tensor(text_model.norm.weight, device=device, compute_dtype=compute_dtype),
                "final_norm_eps": text_model.norm.variance_epsilon,
                "lm_head_weight": _runtime_tensor(model.lm_head.weight, device=device, compute_dtype=compute_dtype),
                "lm_head_bias": _runtime_tensor(model.lm_head.bias, device=device, compute_dtype=compute_dtype),
            }
        )
    else:
        bundle.update(
            {
                "original_output_dtype": str(hidden_stage_output.dtype),
                "original_output_device": str(hidden_stage_output.device),
                "stage_output": _runtime_tensor(hidden_stage_output, device=device, compute_dtype=compute_dtype),
                "layer_output": _runtime_tensor(hidden_stage_output, device=device, compute_dtype=compute_dtype),
            }
        )

    return bundle


def _build_decode_stage_bundle(
    runtime_config: dict[str, Any],
    *,
    start_idx: int,
    end_idx: int,
) -> dict[str, Any]:
    model, _raw_inputs, prefill_runtime_inputs, extra = _prepare_prefill_session(runtime_config)
    device = next(model.parameters()).device
    prefill_outputs = _run_live_prefill_full(model, prefill_runtime_inputs)
    prefill_norm_output = prefill_outputs.last_hidden_state.detach().clone()
    prefill_logits = _compute_logits(model, prefill_norm_output)

    prefill_input_ids = prefill_runtime_inputs.input_ids
    prefill_attention_mask_2d = _default_attention_mask_2d(
        prefill_input_ids,
        prefill_runtime_inputs.attention_mask_2d,
    )
    if prefill_input_ids is None or prefill_attention_mask_2d is None:
        raise RuntimeError("decode stage builder 需要 prefill input_ids 和 attention_mask_2d。")

    prefill_cache_by_layer = build_cache_by_layer_from_past_key_values(
        prefill_outputs.past_key_values,
        device=device,
        compute_dtype=_resolve_compute_dtype(prefill_runtime_inputs.inputs_embeds, runtime_config.get("save_dtype", "auto")),
    )

    greedy_decode_token_id = int(prefill_logits[0, -1].argmax().item())
    decode_token_id_value = runtime_config.get("decode_token_id")
    if decode_token_id_value is None:
        decode_token_id_value = greedy_decode_token_id
        decode_source = "greedy_from_prefill"
    else:
        decode_token_id_value = int(decode_token_id_value)
        decode_source = "provided"

    decode_input_ids = torch.tensor([[decode_token_id_value]], device=device, dtype=prefill_input_ids.dtype)
    decode_attention_mask_2d = torch.cat(
        [
            prefill_attention_mask_2d,
            torch.ones(
                (prefill_attention_mask_2d.shape[0], 1),
                device=device,
                dtype=prefill_attention_mask_2d.dtype,
            ),
        ],
        dim=-1,
    )

    if runtime_config["modality"] == "text":
        decode_runtime_inputs = prepare_text_decode_runtime_inputs(
            model,
            decode_input_ids=decode_input_ids,
            attention_mask_2d=decode_attention_mask_2d,
            past_key_values=prefill_outputs.past_key_values,
        )
    else:
        decode_runtime_inputs = prepare_multimodal_decode_runtime_inputs(
            model,
            decode_input_ids=decode_input_ids,
            attention_mask_2d=decode_attention_mask_2d,
            past_key_values=prefill_outputs.past_key_values,
        )

    is_last_stage = end_idx == len(model.model.language_model.layers) - 1
    decode_outputs, hidden_states, hidden_stage_output = _run_live_decode_full(
        model,
        runtime_inputs=decode_runtime_inputs,
        past_key_values=prefill_outputs.past_key_values,
        is_last_stage=is_last_stage,
    )
    stage_input = hidden_states[start_idx].detach().clone()
    compute_dtype = _resolve_compute_dtype(stage_input, runtime_config.get("save_dtype", "auto"))
    cache_by_layer = {
        layer_idx: (
            None if past_key is None else _runtime_tensor(past_key, device=device, compute_dtype=compute_dtype),
            None if past_value is None else _runtime_tensor(past_value, device=device, compute_dtype=compute_dtype),
        )
        for layer_idx, (past_key, past_value) in prefill_cache_by_layer.items()
    }
    layer_bundles = _build_stage_layer_bundles(
        model,
        start_idx=start_idx,
        end_idx=end_idx,
        device=device,
        compute_dtype=compute_dtype,
        cache_by_layer=cache_by_layer,
    )
    visual_pos_masks, deepstack_by_layer = _build_stage_deepstack_payload(
        decode_runtime_inputs,
        start_idx=start_idx,
        end_idx=end_idx,
        device=device,
        compute_dtype=compute_dtype,
    )

    bundle = {
        "module_name": f"{runtime_config['modality']}_decode_stage",
        "stage_type": "text_decode_last" if is_last_stage else "text_decode",
        "start_idx": start_idx,
        "end_idx": end_idx,
        "save_dtype": _save_dtype_name(compute_dtype),
        "original_input_dtype": str(stage_input.dtype),
        "original_input_device": str(stage_input.device),
        "decode_source": decode_source,
        "decode_token_id": decode_token_id_value,
        "prefill_seq_len": int(prefill_input_ids.shape[-1]),
        "total_seq_len": int(decode_attention_mask_2d.shape[-1]),
        "prefill_input_ids": _runtime_tensor(prefill_input_ids, device=device),
        "decode_input_ids": _runtime_tensor(decode_input_ids, device=device),
        "prefill_attention_mask_2d": _runtime_tensor(prefill_attention_mask_2d, device=device),
        "attention_mask_2d": _runtime_tensor(decode_attention_mask_2d, device=device),
        "stage_input": _runtime_tensor(stage_input, device=device, compute_dtype=compute_dtype),
        "layer_input": _runtime_tensor(stage_input, device=device, compute_dtype=compute_dtype),
        "attention_mask": _runtime_tensor(decode_runtime_inputs.attention_mask, device=device),
        "cos": _runtime_tensor(decode_runtime_inputs.cos, device=device, compute_dtype=compute_dtype),
        "sin": _runtime_tensor(decode_runtime_inputs.sin, device=device, compute_dtype=compute_dtype),
        "visual_pos_masks": visual_pos_masks,
        "deepstack_by_layer": deepstack_by_layer,
        "deepstack_layer_indices": sorted(deepstack_by_layer),
        "layers": layer_bundles,
    }

    if runtime_config["modality"] == "text":
        bundle["prompt"] = extra["prompt"]
    else:
        bundle["num_frames"] = extra["num_frames"]
        bundle["frame_paths"] = extra["frame_paths"]
        bundle["position_ids"] = _runtime_tensor(decode_runtime_inputs.position_ids, device=device)

    if start_idx == 0:
        bundle["embed_tokens_weight"] = _runtime_tensor(
            model.model.language_model.embed_tokens.weight,
            device=device,
            compute_dtype=compute_dtype,
        )

    if is_last_stage:
        if hidden_stage_output is None:
            raise RuntimeError("decode last stage 没有拿到 final norm 前的 hidden_stage_output。")
        norm_output = decode_outputs.last_hidden_state.detach().clone()
        logits = _compute_logits(model, norm_output)
        bundle.update(
            {
                "original_output_dtype": str(logits.dtype),
                "original_output_device": str(logits.device),
                "stage_output": _runtime_tensor(logits, device=device, compute_dtype=compute_dtype),
                "layer_output": _runtime_tensor(logits, device=device, compute_dtype=compute_dtype),
                "hidden_stage_output": _runtime_tensor(hidden_stage_output, device=device, compute_dtype=compute_dtype),
                "norm_output": _runtime_tensor(norm_output, device=device, compute_dtype=compute_dtype),
                "logits": _runtime_tensor(logits, device=device, compute_dtype=compute_dtype),
                "final_norm_weight": _runtime_tensor(
                    model.model.language_model.norm.weight,
                    device=device,
                    compute_dtype=compute_dtype,
                ),
                "final_norm_eps": model.model.language_model.norm.variance_epsilon,
                "lm_head_weight": _runtime_tensor(model.lm_head.weight, device=device, compute_dtype=compute_dtype),
                "lm_head_bias": _runtime_tensor(model.lm_head.bias, device=device, compute_dtype=compute_dtype),
            }
        )
    else:
        stage_output = hidden_states[end_idx + 1].detach().clone()
        bundle.update(
            {
                "original_output_dtype": str(stage_output.dtype),
                "original_output_device": str(stage_output.device),
                "stage_output": _runtime_tensor(stage_output, device=device, compute_dtype=compute_dtype),
                "layer_output": _runtime_tensor(stage_output, device=device, compute_dtype=compute_dtype),
            }
        )

    return bundle


def _build_generate_stage_bundle(
    runtime_config: dict[str, Any],
    *,
    start_idx: int,
    end_idx: int,
) -> dict[str, Any]:
    max_new_tokens = int(runtime_config.get("max_new_tokens", 4))
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens 必须大于 0，当前拿到 {max_new_tokens}。")

    model, _raw_inputs, prefill_runtime_inputs, extra = _prepare_prefill_session(runtime_config)
    device = next(model.parameters()).device
    prefill_outputs = _run_live_prefill_full(model, prefill_runtime_inputs)
    prefill_norm_output = prefill_outputs.last_hidden_state.detach().clone()
    prefill_logits = _compute_logits(model, prefill_norm_output)
    prefill_stage_input, prefill_hidden_stage_output = _run_live_prefill_stage_reference(
        model,
        runtime_inputs=prefill_runtime_inputs,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    prefill_input_ids = prefill_runtime_inputs.input_ids
    prefill_attention_mask_2d = _default_attention_mask_2d(
        prefill_input_ids,
        prefill_runtime_inputs.attention_mask_2d,
    )
    if prefill_input_ids is None or prefill_attention_mask_2d is None:
        raise RuntimeError("generate stage builder 需要 prefill input_ids 和 attention_mask_2d。")

    compute_dtype = _resolve_compute_dtype(prefill_stage_input, runtime_config.get("save_dtype", "auto"))
    cache_by_layer = build_cache_by_layer_from_past_key_values(
        prefill_outputs.past_key_values,
        device=device,
        compute_dtype=compute_dtype,
    )
    layer_bundles = _build_stage_layer_bundles(
        model,
        start_idx=start_idx,
        end_idx=end_idx,
        device=device,
        compute_dtype=compute_dtype,
        cache_by_layer=cache_by_layer,
    )
    visual_pos_masks, deepstack_by_layer = _build_stage_deepstack_payload(
        prefill_runtime_inputs,
        start_idx=start_idx,
        end_idx=end_idx,
        device=device,
        compute_dtype=compute_dtype,
    )

    generated_token_ids = [int(prefill_logits[0, -1].argmax().item())]
    is_last_stage = end_idx == len(model.model.language_model.layers) - 1

    prefill_payload = {
        "attention_mask_2d": _runtime_tensor(prefill_attention_mask_2d, device=device),
        "stage_input": _runtime_tensor(prefill_stage_input, device=device, compute_dtype=compute_dtype),
        "attention_mask": _runtime_tensor(prefill_runtime_inputs.attention_mask, device=device),
        "cos": _runtime_tensor(prefill_runtime_inputs.cos, device=device, compute_dtype=compute_dtype),
        "sin": _runtime_tensor(prefill_runtime_inputs.sin, device=device, compute_dtype=compute_dtype),
    }
    if is_last_stage:
        prefill_payload.update(
            {
                "stage_output": _runtime_tensor(prefill_logits, device=device, compute_dtype=compute_dtype),
                "hidden_stage_output": _runtime_tensor(
                    prefill_hidden_stage_output,
                    device=device,
                    compute_dtype=compute_dtype,
                ),
                "norm_output": _runtime_tensor(prefill_norm_output, device=device, compute_dtype=compute_dtype),
                "logits": _runtime_tensor(prefill_logits, device=device, compute_dtype=compute_dtype),
                "output_token_id": generated_token_ids[0],
            }
        )
    else:
        prefill_payload["stage_output"] = _runtime_tensor(
            prefill_hidden_stage_output,
            device=device,
            compute_dtype=compute_dtype,
        )

    decode_steps = []
    current_attention_mask_2d = prefill_attention_mask_2d
    current_past_key_values = prefill_outputs.past_key_values

    for step_idx in range(max_new_tokens - 1):
        decode_input_ids = torch.tensor(
            [[generated_token_ids[-1]]],
            device=device,
            dtype=prefill_input_ids.dtype,
        )
        current_attention_mask_2d = torch.cat(
            [
                current_attention_mask_2d,
                torch.ones(
                    (current_attention_mask_2d.shape[0], 1),
                    device=device,
                    dtype=current_attention_mask_2d.dtype,
                ),
            ],
            dim=-1,
        )

        if runtime_config["modality"] == "text":
            decode_runtime_inputs = prepare_text_decode_runtime_inputs(
                model,
                decode_input_ids=decode_input_ids,
                attention_mask_2d=current_attention_mask_2d,
                past_key_values=current_past_key_values,
            )
        else:
            decode_runtime_inputs = prepare_multimodal_decode_runtime_inputs(
                model,
                decode_input_ids=decode_input_ids,
                attention_mask_2d=current_attention_mask_2d,
                past_key_values=current_past_key_values,
            )

        decode_outputs, hidden_states, hidden_stage_output = _run_live_decode_full(
            model,
            runtime_inputs=decode_runtime_inputs,
            past_key_values=current_past_key_values,
            is_last_stage=is_last_stage,
        )
        stage_input = hidden_states[start_idx].detach().clone()
        norm_output = decode_outputs.last_hidden_state.detach().clone()
        logits = _compute_logits(model, norm_output)
        next_token_id = int(logits[0, -1].argmax().item())
        generated_token_ids.append(next_token_id)

        step_payload = {
            "step_idx": step_idx,
            "decode_input_ids": _runtime_tensor(decode_input_ids, device=device),
            "attention_mask_2d": _runtime_tensor(current_attention_mask_2d, device=device),
            "total_seq_len": int(current_attention_mask_2d.shape[-1]),
            "stage_input": _runtime_tensor(stage_input, device=device, compute_dtype=compute_dtype),
            "attention_mask": _runtime_tensor(decode_runtime_inputs.attention_mask, device=device),
            "cos": _runtime_tensor(decode_runtime_inputs.cos, device=device, compute_dtype=compute_dtype),
            "sin": _runtime_tensor(decode_runtime_inputs.sin, device=device, compute_dtype=compute_dtype),
            "visual_pos_masks": None,
            "deepstack_by_layer": {},
            "deepstack_layer_indices": [],
        }
        if runtime_config["modality"] == "multimodal":
            step_payload["position_ids"] = _runtime_tensor(decode_runtime_inputs.position_ids, device=device)

        if is_last_stage:
            if hidden_stage_output is None:
                raise RuntimeError("generate decode last stage 没有拿到 final norm 前的 hidden_stage_output。")
            step_payload.update(
                {
                    "stage_output": _runtime_tensor(logits, device=device, compute_dtype=compute_dtype),
                    "hidden_stage_output": _runtime_tensor(
                        hidden_stage_output,
                        device=device,
                        compute_dtype=compute_dtype,
                    ),
                    "norm_output": _runtime_tensor(norm_output, device=device, compute_dtype=compute_dtype),
                    "logits": _runtime_tensor(logits, device=device, compute_dtype=compute_dtype),
                    "output_token_id": next_token_id,
                }
            )
        else:
            stage_output = hidden_states[end_idx + 1].detach().clone()
            step_payload["stage_output"] = _runtime_tensor(stage_output, device=device, compute_dtype=compute_dtype)

        decode_steps.append(step_payload)
        current_past_key_values = decode_outputs.past_key_values

    bundle = {
        "module_name": f"{runtime_config['modality']}_generate_stage",
        "stage_type": f"{runtime_config['modality']}_generate_last" if is_last_stage else f"{runtime_config['modality']}_generate",
        "start_idx": start_idx,
        "end_idx": end_idx,
        "save_dtype": _save_dtype_name(compute_dtype),
        "original_input_dtype": str(prefill_stage_input.dtype),
        "original_input_device": str(prefill_stage_input.device),
        "max_new_tokens": max_new_tokens,
        "prefill_seq_len": int(prefill_input_ids.shape[-1]),
        "prefill_input_ids": _runtime_tensor(prefill_input_ids, device=device),
        "generated_token_ids": torch.tensor([generated_token_ids], device=device, dtype=prefill_input_ids.dtype),
        "prefill": prefill_payload,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_by_layer": deepstack_by_layer,
        "deepstack_layer_indices": sorted(deepstack_by_layer),
        "layers": layer_bundles,
        "decode_steps": decode_steps,
    }

    if runtime_config["modality"] == "text":
        bundle["prompt"] = extra["prompt"]
        if start_idx == 0:
            bundle["input_ids"] = _runtime_tensor(prefill_input_ids, device=device)
    else:
        bundle["num_frames"] = extra["num_frames"]
        bundle["frame_paths"] = extra["frame_paths"]

    if start_idx == 0:
        bundle["embed_tokens_weight"] = _runtime_tensor(
            model.model.language_model.embed_tokens.weight,
            device=device,
            compute_dtype=compute_dtype,
        )

    if is_last_stage:
        bundle.update(
            {
                "final_norm_weight": _runtime_tensor(
                    model.model.language_model.norm.weight,
                    device=device,
                    compute_dtype=compute_dtype,
                ),
                "final_norm_eps": model.model.language_model.norm.variance_epsilon,
                "lm_head_weight": _runtime_tensor(model.lm_head.weight, device=device, compute_dtype=compute_dtype),
                "lm_head_bias": _runtime_tensor(model.lm_head.bias, device=device, compute_dtype=compute_dtype),
            }
        )

    return bundle


class DirectStageBundleBuilder:
    """Reuse one builder session to materialize multiple direct stage bundles."""

    def __init__(
        self,
        *,
        stage_specs: list[StageSpec],
        runtime_config: dict[str, Any],
        tp_shard_rank: int | None = None,
        tp_shard_world_size: int | None = None,
        include_text_weights: bool = True,
    ) -> None:
        if not stage_specs:
            raise ValueError("DirectStageBundleBuilder 至少需要一个 stage spec。")

        self.stage_specs = sorted(stage_specs, key=lambda spec: spec.stage_idx)
        self.stage_specs_by_idx = {spec.stage_idx: spec for spec in self.stage_specs}
        if len(self.stage_specs_by_idx) != len(self.stage_specs):
            raise ValueError("stage_idx 必须唯一。")

        self.runtime_config = runtime_config
        self.tp_shard_rank = tp_shard_rank
        self.tp_shard_world_size = tp_shard_world_size
        self.include_text_weights = include_text_weights
        self.include_runtime_reference = bool(runtime_config.get("include_runtime_reference", True))
        self.modality = runtime_config["modality"]
        self.mode = runtime_config["mode"]
        self.runtime_only_text_generate = (
            self.modality == "text" and self.mode == "generate" and not self.include_runtime_reference
        )
        self.stage_summary = ",".join(
            f"{spec.stage_idx}:{spec.start_idx}-{spec.end_idx}" for spec in self.stage_specs
        )
        self.log_component = f"direct-builder:{self.modality}_{self.mode}"
        self.bundle_device = torch.device("cpu")
        self.session_kind = (
            "runtime-only-text"
            if self.runtime_only_text_generate
            else "file-backed-text"
            if self.modality == "text"
            else "live-model"
        )
        self._text_weight_index = None
        self._text_model_config: TextModelConfigSpec | None = None
        self._text_stage_static_weights: dict[int, TextStageWeightBundle] = {}
        self._text_reference_layer_static_bundles: dict[int, dict[str, Any]] = {}
        self._text_last_stage_static_weights: TextStageWeightBundle | None = None
        self._text_embed_tokens_weight: torch.Tensor | None = None
        self._text_rotary_emb = None
        self._prefill_stage_inputs_by_stage: dict[int, torch.Tensor] = {}
        self._prefill_stage_outputs_by_stage: dict[int, torch.Tensor] = {}
        self._prefill_full_state: dict[str, Any] | None = None
        self._decode_state: dict[str, Any] | None = None
        self._generate_state: dict[str, Any] | None = None

        if self.modality == "text":
            prepare_message = (
                f"prepare runtime-only text session stages=[{self.stage_summary}]"
                if self.runtime_only_text_generate
                else f"prepare file-backed text session stages=[{self.stage_summary}]"
            )
            with startup_timer(self.log_component, prepare_message):
                self._text_weight_index = load_model_weight_index(self.runtime_config["model_path"])
                self._text_model_config = load_text_model_config_spec(self.runtime_config["model_path"])
                if self.runtime_only_text_generate:
                    self.raw_inputs, self.compute_dtype, self.extra, self.device = _prepare_runtime_only_text_generate_session(
                        self._text_weight_index,
                        runtime_config
                    )
                else:
                    processor = load_processor(self.runtime_config["model_path"])
                    prompt = self.runtime_config.get("prompt", "请用中文简要介绍一下人工智能。")
                    self.raw_inputs = build_text_inputs(processor, prompt)
                    self.extra = {"prompt": prompt}
                    self.device = _default_runtime_device()

                    embed_tokens_weight = load_tensors_from_index(
                        self._text_weight_index,
                        ["model.language_model.embed_tokens.weight"],
                        device=self.bundle_device,
                        compute_dtype=None,
                        strict=True,
                    )["model.language_model.embed_tokens.weight"]
                    self.compute_dtype = _resolve_compute_dtype(
                        embed_tokens_weight,
                        runtime_config.get("save_dtype", "auto"),
                    )
                    self._text_embed_tokens_weight = _runtime_tensor(
                        embed_tokens_weight,
                        device=self.bundle_device,
                        compute_dtype=self.compute_dtype,
                    )
                    self._text_rotary_emb = build_text_rotary_embedding(
                        self._text_model_config,
                        device=self.device,
                    )
                    self.prefill_runtime_inputs = prepare_text_prefill_runtime_inputs_from_weights(
                        input_ids=self.raw_inputs["input_ids"],
                        attention_mask_2d=self.raw_inputs.get("attention_mask"),
                        embed_tokens_weight=self._text_embed_tokens_weight,
                        config_spec=self._text_model_config,
                        device=self.device,
                        compute_dtype=self.compute_dtype,
                        rotary_emb=self._text_rotary_emb,
                    )
                self.num_layers = self._text_model_config.num_hidden_layers
        else:
            with startup_timer(
                self.log_component,
                f"prepare live session stages=[{self.stage_summary}]",
            ):
                self.model, self.raw_inputs, self.prefill_runtime_inputs, self.extra = _prepare_prefill_session(runtime_config)
            self.device = next(self.model.parameters()).device
            self.num_layers = len(self.model.model.language_model.layers)
            self.compute_dtype = _resolve_compute_dtype(
                self.prefill_runtime_inputs.inputs_embeds,
                runtime_config.get("save_dtype", "auto"),
            )

        self.has_last_stage = any(spec.end_idx == self.num_layers - 1 for spec in self.stage_specs)
        if self.runtime_only_text_generate:
            self.prefill_input_ids = self.raw_inputs.get("input_ids")
            self.prefill_attention_mask_2d_raw = self.raw_inputs.get("attention_mask")
        else:
            self.prefill_input_ids = self.prefill_runtime_inputs.input_ids
            self.prefill_attention_mask_2d_raw = self.prefill_runtime_inputs.attention_mask_2d
        self.prefill_attention_mask_2d = _default_attention_mask_2d(
            self.prefill_input_ids,
            self.prefill_attention_mask_2d_raw,
        )
        if self.mode in {"decode", "generate"} and (
            self.prefill_input_ids is None or self.prefill_attention_mask_2d is None
        ):
            raise RuntimeError(f"{self.mode} stage builder 需要 prefill input_ids 和 attention_mask_2d。")

        startup_log(
            self.log_component,
            f"builder session ready session_kind={self.session_kind} "
            f"device={self.device} bundle_device={self.bundle_device} "
            f"compute_dtype={self.compute_dtype} num_layers={self.num_layers} "
            f"tp_shard_rank={self.tp_shard_rank} tp_shard_world_size={self.tp_shard_world_size} "
            f"include_text_weights={self.include_text_weights} "
            f"include_runtime_reference={self.include_runtime_reference}",
        )
        if self.runtime_only_text_generate:
            startup_log(
                self.log_component,
                f"skip prefill boundary capture for runtime-only text generate stages=[{self.stage_summary}]",
            )
        else:
            with startup_timer(
                self.log_component,
                f"capture prefill stage boundaries stages=[{self.stage_summary}]",
            ):
                self._capture_prefill_stage_boundaries()

    def __enter__(self) -> "DirectStageBundleBuilder":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def close(self) -> None:
        startup_log(self.log_component, f"releasing builder session session_kind={self.session_kind}")
        self._prefill_full_state = None
        self._decode_state = None
        self._generate_state = None
        self._prefill_stage_inputs_by_stage.clear()
        self._prefill_stage_outputs_by_stage.clear()
        self._text_stage_static_weights.clear()
        self._text_reference_layer_static_bundles.clear()
        self._text_last_stage_static_weights = None
        self._text_embed_tokens_weight = None
        self._text_rotary_emb = None

        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "raw_inputs"):
            del self.raw_inputs
        if hasattr(self, "prefill_runtime_inputs"):
            del self.prefill_runtime_inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _capture_prefill_stage_boundaries(self) -> None:
        if self.modality == "text" and not hasattr(self, "model"):
            prefill_state = self._ensure_prefill_full_state()
            hidden_states = prefill_state["hidden_states"]
            for spec in self.stage_specs:
                self._prefill_stage_inputs_by_stage[spec.stage_idx] = hidden_states[spec.start_idx].detach().clone()
                self._prefill_stage_outputs_by_stage[spec.stage_idx] = hidden_states[spec.end_idx + 1].detach().clone()
            return

        start_stage_ids: dict[int, list[int]] = {}
        end_stage_ids: dict[int, list[int]] = {}
        for spec in self.stage_specs:
            start_stage_ids.setdefault(spec.start_idx, []).append(spec.stage_idx)
            end_stage_ids.setdefault(spec.end_idx, []).append(spec.stage_idx)

        runtime_inputs = self.prefill_runtime_inputs
        text_model = self.model.model.language_model
        text_position_ids = _extract_text_position_ids(runtime_inputs)
        position_embeddings = (runtime_inputs.cos, runtime_inputs.sin)

        hidden_states = runtime_inputs.inputs_embeds
        for stage_idx in start_stage_ids.get(0, []):
            self._prefill_stage_inputs_by_stage[stage_idx] = hidden_states.detach().clone()

        with torch.inference_mode():
            for layer_idx in range(self.num_layers):
                hidden_states = text_model.layers[layer_idx](
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=runtime_inputs.attention_mask,
                    position_ids=text_position_ids,
                    past_key_values=None,
                    use_cache=False,
                )
                hidden_states = apply_deepstack(
                    hidden_states,
                    runtime_inputs.visual_pos_masks,
                    runtime_inputs.deepstack_by_layer.get(layer_idx),
                )
                for stage_idx in end_stage_ids.get(layer_idx, []):
                    self._prefill_stage_outputs_by_stage[stage_idx] = hidden_states.detach().clone()
                for stage_idx in start_stage_ids.get(layer_idx + 1, []):
                    self._prefill_stage_inputs_by_stage[stage_idx] = hidden_states.detach().clone()

        missing_inputs = set(self.stage_specs_by_idx) - set(self._prefill_stage_inputs_by_stage)
        if missing_inputs:
            raise RuntimeError(f"prefill stage input 捕获不完整，缺少 stage_idx={sorted(missing_inputs)}")
        missing_outputs = set(self.stage_specs_by_idx) - set(self._prefill_stage_outputs_by_stage)
        if missing_outputs:
            raise RuntimeError(f"prefill stage output 捕获不完整，缺少 stage_idx={sorted(missing_outputs)}")

    def _get_text_last_stage_static_weights(self) -> TextStageWeightBundle:
        if self._text_last_stage_static_weights is not None:
            return self._text_last_stage_static_weights
        if self._text_weight_index is None or self._text_model_config is None:
            raise RuntimeError("text weight index/config 尚未初始化。")

        last_layer_idx = self.num_layers - 1
        self._text_last_stage_static_weights = load_text_decoder_stage_weight_bundle(
            model_path=self.runtime_config["model_path"],
            start_idx=last_layer_idx,
            end_idx=last_layer_idx,
            is_first_stage=False,
            is_last_stage=True,
            device=self.bundle_device,
            compute_dtype=self.compute_dtype,
            weight_index=self._text_weight_index,
            config_spec=self._text_model_config,
        )
        return self._text_last_stage_static_weights

    def _get_text_reference_layer_static_bundle(self, layer_idx: int) -> dict[str, Any]:
        if layer_idx in self._text_reference_layer_static_bundles:
            return self._text_reference_layer_static_bundles[layer_idx]
        if self._text_weight_index is None or self._text_model_config is None:
            raise RuntimeError("text weight index/config 尚未初始化。")

        stage_weights = load_text_decoder_stage_weight_bundle(
            model_path=self.runtime_config["model_path"],
            start_idx=layer_idx,
            end_idx=layer_idx,
            is_first_stage=False,
            is_last_stage=False,
            device=self.bundle_device,
            compute_dtype=self.compute_dtype,
            weight_index=self._text_weight_index,
            config_spec=self._text_model_config,
        )
        layer_bundle = dict(stage_weights.layer_bundles[0])
        self._text_reference_layer_static_bundles[layer_idx] = layer_bundle
        return layer_bundle

    def _build_text_reference_single_layer_stage_bundle(
        self,
        layer_idx: int,
        runtime_inputs: MultimodalRuntimeInputs,
    ) -> dict[str, Any]:
        static_layer_bundle = self._get_text_reference_layer_static_bundle(layer_idx)
        runtime_layer_bundle = {
            key: (
                _runtime_tensor(value, device=self.device, compute_dtype=self.compute_dtype)
                if torch.is_tensor(value)
                else value
            )
            for key, value in static_layer_bundle.items()
        }
        return {
            "module_name": "text_reference_layer",
            "stage_type": "text_decode",
            "start_idx": layer_idx,
            "end_idx": layer_idx,
            "attention_mask": _runtime_tensor(runtime_inputs.attention_mask, device=self.device),
            "cos": _runtime_tensor(runtime_inputs.cos, device=self.device, compute_dtype=self.compute_dtype),
            "sin": _runtime_tensor(runtime_inputs.sin, device=self.device, compute_dtype=self.compute_dtype),
            "visual_pos_masks": None,
            "deepstack_by_layer": {},
            "deepstack_layer_indices": [],
            "layers": [runtime_layer_bundle],
        }

    def _get_text_final_runtime_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
        last_stage_weights = self._get_text_last_stage_static_weights()
        final_norm_weight = _runtime_tensor(
            last_stage_weights.final_norm_weight,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )
        lm_head_weight = _runtime_tensor(
            last_stage_weights.lm_head_weight,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )
        lm_head_bias = _runtime_tensor(
            last_stage_weights.lm_head_bias,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )
        if final_norm_weight is None or lm_head_weight is None or last_stage_weights.final_norm_eps is None:
            raise RuntimeError("text last stage 缺少 final norm / lm_head 权重，无法做 file-backed reference。")
        return final_norm_weight, lm_head_weight, lm_head_bias, float(last_stage_weights.final_norm_eps)

    def _normalize_cache_by_layer(
        self,
        cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]],
        *,
        device: torch.device,
    ) -> dict[int, tuple[torch.Tensor | None, torch.Tensor | None]]:
        return {
            int(layer_idx): (
                _runtime_tensor(past_key, device=device, compute_dtype=self.compute_dtype),
                _runtime_tensor(past_value, device=device, compute_dtype=self.compute_dtype),
            )
            for layer_idx, (past_key, past_value) in cache_by_layer.items()
        }

    def _infer_cache_seq_len(
        self,
        cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]],
    ) -> int:
        lengths = {
            int(past_key.shape[-2])
            for past_key, _past_value in cache_by_layer.values()
            if past_key is not None
        }
        if not lengths:
            return 0
        if len(lengths) != 1:
            raise RuntimeError(f"cache_by_layer 的 past length 不一致: {sorted(lengths)}")
        return lengths.pop()

    def _run_text_file_backed_prefill(self) -> dict[str, Any]:
        hidden_states = _runtime_tensor(
            self.prefill_runtime_inputs.inputs_embeds,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )
        if hidden_states is None:
            raise RuntimeError("text prefill 缺少 inputs_embeds。")

        hidden_state_list = [hidden_states.detach().clone()]
        cache_by_layer_runtime: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] = {}

        for layer_idx in range(self.num_layers):
            layer_stage_bundle = self._build_text_reference_single_layer_stage_bundle(
                layer_idx,
                self.prefill_runtime_inputs,
            )
            trace = trace_text_decode_stage_with_runtime_cache(
                hidden_states,
                layer_stage_bundle,
                cache_by_layer=cache_by_layer_runtime,
            )
            hidden_states = trace["stage_output"].detach().clone()
            hidden_state_list.append(hidden_states)
            cache_by_layer_runtime[layer_idx] = trace["cache_by_layer"][layer_idx]

        final_norm_weight, lm_head_weight, lm_head_bias, final_norm_eps = self._get_text_final_runtime_weights()
        norm_output = rms_norm(hidden_states, final_norm_weight, final_norm_eps)
        logits = F.linear(norm_output, lm_head_weight, lm_head_bias)

        return {
            "hidden_states": tuple(
                _runtime_tensor(hidden, device=self.bundle_device, compute_dtype=self.compute_dtype)
                for hidden in hidden_state_list
            ),
            "norm_output": _runtime_tensor(norm_output, device=self.bundle_device, compute_dtype=self.compute_dtype),
            "logits": _runtime_tensor(logits, device=self.bundle_device, compute_dtype=self.compute_dtype),
            "prefill_input_ids": _runtime_tensor(self.prefill_input_ids, device=self.bundle_device),
            "prefill_attention_mask_2d": _runtime_tensor(self.prefill_attention_mask_2d, device=self.bundle_device),
            "cache_by_layer": self._normalize_cache_by_layer(cache_by_layer_runtime, device=self.bundle_device),
        }

    def _run_text_file_backed_decode(
        self,
        *,
        decode_input_ids: torch.Tensor,
        attention_mask_2d: torch.Tensor,
        cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]],
    ) -> dict[str, Any]:
        if self._text_model_config is None or self._text_embed_tokens_weight is None:
            raise RuntimeError("text config/embed_tokens_weight 尚未初始化。")

        decode_runtime_inputs = prepare_text_decode_runtime_inputs_from_weights(
            decode_input_ids=decode_input_ids,
            attention_mask_2d=attention_mask_2d,
            past_length=self._infer_cache_seq_len(cache_by_layer),
            embed_tokens_weight=self._text_embed_tokens_weight,
            config_spec=self._text_model_config,
            device=self.device,
            compute_dtype=self.compute_dtype,
            rotary_emb=self._text_rotary_emb,
        )
        hidden_states = _runtime_tensor(
            decode_runtime_inputs.inputs_embeds,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )
        if hidden_states is None:
            raise RuntimeError("text decode 缺少 inputs_embeds。")

        hidden_state_list = [hidden_states.detach().clone()]
        cache_by_layer_runtime = self._normalize_cache_by_layer(cache_by_layer, device=self.device)

        for layer_idx in range(self.num_layers):
            layer_stage_bundle = self._build_text_reference_single_layer_stage_bundle(
                layer_idx,
                decode_runtime_inputs,
            )
            trace = trace_text_decode_stage_with_runtime_cache(
                hidden_states,
                layer_stage_bundle,
                cache_by_layer=cache_by_layer_runtime,
            )
            hidden_states = trace["stage_output"].detach().clone()
            hidden_state_list.append(hidden_states)
            cache_by_layer_runtime[layer_idx] = trace["cache_by_layer"][layer_idx]

        final_norm_weight, lm_head_weight, lm_head_bias, final_norm_eps = self._get_text_final_runtime_weights()
        norm_output = rms_norm(hidden_states, final_norm_weight, final_norm_eps)
        logits = F.linear(norm_output, lm_head_weight, lm_head_bias)

        return {
            "decode_input_ids": _runtime_tensor(decode_input_ids, device=self.bundle_device),
            "attention_mask_2d": _runtime_tensor(attention_mask_2d, device=self.bundle_device),
            "attention_mask": _runtime_tensor(decode_runtime_inputs.attention_mask, device=self.bundle_device),
            "cos": _runtime_tensor(decode_runtime_inputs.cos, device=self.bundle_device, compute_dtype=self.compute_dtype),
            "sin": _runtime_tensor(decode_runtime_inputs.sin, device=self.bundle_device, compute_dtype=self.compute_dtype),
            "position_ids": _runtime_tensor(decode_runtime_inputs.position_ids, device=self.bundle_device),
            "hidden_states": tuple(
                _runtime_tensor(hidden, device=self.bundle_device, compute_dtype=self.compute_dtype)
                for hidden in hidden_state_list
            ),
            "hidden_stage_output": _runtime_tensor(
                hidden_states,
                device=self.bundle_device,
                compute_dtype=self.compute_dtype,
            ),
            "norm_output": _runtime_tensor(norm_output, device=self.bundle_device, compute_dtype=self.compute_dtype),
            "logits": _runtime_tensor(logits, device=self.bundle_device, compute_dtype=self.compute_dtype),
            "cache_by_layer": self._normalize_cache_by_layer(cache_by_layer_runtime, device=self.bundle_device),
        }

    def _ensure_prefill_full_state(self) -> dict[str, Any]:
        if self._prefill_full_state is not None:
            return self._prefill_full_state

        if self.modality == "text" and not hasattr(self, "model"):
            with startup_timer(self.log_component, "run file-backed text prefill reference"):
                self._prefill_full_state = self._run_text_file_backed_prefill()
            return self._prefill_full_state

        with startup_timer(self.log_component, "run full prefill reference"):
            outputs = _run_live_prefill_full(self.model, self.prefill_runtime_inputs)
            norm_output = outputs.last_hidden_state.detach().clone()
            logits = _compute_logits(self.model, norm_output).detach().clone()
            prefill_cache_by_layer = build_cache_by_layer_from_past_key_values(
                outputs.past_key_values,
                device=self.bundle_device,
                compute_dtype=self.compute_dtype,
            )
            self._prefill_full_state = {
                "past_key_values": outputs.past_key_values,
                "norm_output": norm_output,
                "logits": logits,
                "prefill_input_ids": self.prefill_input_ids,
                "prefill_attention_mask_2d": self.prefill_attention_mask_2d,
                "cache_by_layer": prefill_cache_by_layer,
            }
        return self._prefill_full_state

    def _ensure_decode_state(self) -> dict[str, Any]:
        if self._decode_state is not None:
            return self._decode_state

        if self.modality == "text" and not hasattr(self, "model"):
            with startup_timer(self.log_component, "prepare file-backed text decode reference state"):
                prefill_state = self._ensure_prefill_full_state()
                prefill_logits = prefill_state["logits"]
                greedy_decode_token_id = int(prefill_logits[0, -1].argmax().item())
                decode_token_id_value = self.runtime_config.get("decode_token_id")
                if decode_token_id_value is None:
                    decode_token_id_value = greedy_decode_token_id
                    decode_source = "greedy_from_prefill"
                else:
                    decode_token_id_value = int(decode_token_id_value)
                    decode_source = "provided"

                decode_input_ids = torch.tensor(
                    [[decode_token_id_value]],
                    device=self.bundle_device,
                    dtype=self.prefill_input_ids.dtype,
                )
                decode_attention_mask_2d = torch.cat(
                    [
                        prefill_state["prefill_attention_mask_2d"],
                        torch.ones(
                            (prefill_state["prefill_attention_mask_2d"].shape[0], 1),
                            device=self.bundle_device,
                            dtype=prefill_state["prefill_attention_mask_2d"].dtype,
                        ),
                    ],
                    dim=-1,
                )
                decode_result = self._run_text_file_backed_decode(
                    decode_input_ids=decode_input_ids,
                    attention_mask_2d=decode_attention_mask_2d,
                    cache_by_layer=prefill_state["cache_by_layer"],
                )
                self._decode_state = {
                    "decode_source": decode_source,
                    "decode_token_id": decode_token_id_value,
                    "decode_input_ids": decode_result["decode_input_ids"],
                    "attention_mask_2d": decode_result["attention_mask_2d"],
                    "attention_mask": decode_result["attention_mask"],
                    "cos": decode_result["cos"],
                    "sin": decode_result["sin"],
                    "position_ids": decode_result["position_ids"],
                    "visual_pos_masks": None,
                    "deepstack_by_layer": {},
                    "hidden_states": decode_result["hidden_states"],
                    "hidden_stage_output": decode_result["hidden_stage_output"],
                    "norm_output": decode_result["norm_output"],
                    "logits": decode_result["logits"],
                    "cache_by_layer": prefill_state["cache_by_layer"],
                }
            return self._decode_state

        with startup_timer(self.log_component, "prepare decode reference state"):
            prefill_state = self._ensure_prefill_full_state()
            prefill_logits = prefill_state["logits"]
            greedy_decode_token_id = int(prefill_logits[0, -1].argmax().item())
            decode_token_id_value = self.runtime_config.get("decode_token_id")
            if decode_token_id_value is None:
                decode_token_id_value = greedy_decode_token_id
                decode_source = "greedy_from_prefill"
            else:
                decode_token_id_value = int(decode_token_id_value)
                decode_source = "provided"

            decode_input_ids = torch.tensor(
                [[decode_token_id_value]],
                device=self.device,
                dtype=self.prefill_input_ids.dtype,
            )
            decode_attention_mask_2d = torch.cat(
                [
                    self.prefill_attention_mask_2d,
                    torch.ones(
                        (self.prefill_attention_mask_2d.shape[0], 1),
                        device=self.device,
                        dtype=self.prefill_attention_mask_2d.dtype,
                    ),
                ],
                dim=-1,
            )

            if self.modality == "text":
                decode_runtime_inputs = prepare_text_decode_runtime_inputs(
                    self.model,
                    decode_input_ids=decode_input_ids,
                    attention_mask_2d=decode_attention_mask_2d,
                    past_key_values=prefill_state["past_key_values"],
                )
            else:
                decode_runtime_inputs = prepare_multimodal_decode_runtime_inputs(
                    self.model,
                    decode_input_ids=decode_input_ids,
                    attention_mask_2d=decode_attention_mask_2d,
                    past_key_values=prefill_state["past_key_values"],
                )

            decode_outputs, hidden_states, hidden_stage_output = _run_live_decode_full(
                self.model,
                runtime_inputs=decode_runtime_inputs,
                past_key_values=prefill_state["past_key_values"],
                is_last_stage=self.has_last_stage,
            )
            self._decode_state = {
                "decode_source": decode_source,
                "decode_token_id": decode_token_id_value,
                "decode_input_ids": decode_input_ids.detach().clone(),
                "attention_mask_2d": decode_attention_mask_2d.detach().clone(),
                "attention_mask": decode_runtime_inputs.attention_mask.detach().clone(),
                "cos": decode_runtime_inputs.cos.detach().clone(),
                "sin": decode_runtime_inputs.sin.detach().clone(),
                "position_ids": None
                if decode_runtime_inputs.position_ids is None
                else decode_runtime_inputs.position_ids.detach().clone(),
                "visual_pos_masks": None
                if decode_runtime_inputs.visual_pos_masks is None
                else decode_runtime_inputs.visual_pos_masks.detach().clone(),
                "deepstack_by_layer": {
                    int(layer_idx): deepstack_embeds.detach().clone()
                    for layer_idx, deepstack_embeds in decode_runtime_inputs.deepstack_by_layer.items()
                },
                "hidden_states": tuple(hidden.detach().clone() for hidden in hidden_states),
                "hidden_stage_output": None
                if hidden_stage_output is None
                else hidden_stage_output.detach().clone(),
                "norm_output": decode_outputs.last_hidden_state.detach().clone(),
                "logits": _compute_logits(self.model, decode_outputs.last_hidden_state).detach().clone(),
                "cache_by_layer": prefill_state["cache_by_layer"],
            }
        return self._decode_state

    def _ensure_generate_state(self) -> dict[str, Any]:
        if self._generate_state is not None:
            return self._generate_state

        if self.modality == "text" and not hasattr(self, "model"):
            with startup_timer(self.log_component, "prepare file-backed text generate reference state"):
                prefill_state = self._ensure_prefill_full_state()
                current_attention_mask_2d = prefill_state["prefill_attention_mask_2d"]
                current_cache_by_layer = prefill_state["cache_by_layer"]
                generated_token_ids = [int(prefill_state["logits"][0, -1].argmax().item())]
                step_results: list[dict[str, Any]] = []
                max_new_tokens = int(self.runtime_config.get("max_new_tokens", 4))
                if max_new_tokens <= 0:
                    raise ValueError(f"max_new_tokens 必须大于 0，当前拿到 {max_new_tokens}。")

                startup_log(
                    self.log_component,
                    f"file-backed generate decode planning max_new_tokens={max_new_tokens} "
                    f"decode_steps={max_new_tokens - 1}",
                )
                for step_idx in range(max_new_tokens - 1):
                    decode_input_ids = torch.tensor(
                        [[generated_token_ids[-1]]],
                        device=self.bundle_device,
                        dtype=self.prefill_input_ids.dtype,
                    )
                    current_attention_mask_2d = torch.cat(
                        [
                            current_attention_mask_2d,
                            torch.ones(
                                (current_attention_mask_2d.shape[0], 1),
                                device=self.bundle_device,
                                dtype=current_attention_mask_2d.dtype,
                            ),
                        ],
                        dim=-1,
                    )
                    decode_result = self._run_text_file_backed_decode(
                        decode_input_ids=decode_input_ids,
                        attention_mask_2d=current_attention_mask_2d,
                        cache_by_layer=current_cache_by_layer,
                    )
                    next_token_id = int(decode_result["logits"][0, -1].argmax().item())
                    generated_token_ids.append(next_token_id)
                    step_results.append(
                        {
                            "step_idx": step_idx,
                            "decode_input_ids": decode_result["decode_input_ids"],
                            "attention_mask_2d": decode_result["attention_mask_2d"],
                            "attention_mask": decode_result["attention_mask"],
                            "cos": decode_result["cos"],
                            "sin": decode_result["sin"],
                            "position_ids": decode_result["position_ids"],
                            "hidden_states": decode_result["hidden_states"],
                            "hidden_stage_output": decode_result["hidden_stage_output"],
                            "norm_output": decode_result["norm_output"],
                            "logits": decode_result["logits"],
                            "output_token_id": next_token_id,
                        }
                    )
                    current_cache_by_layer = decode_result["cache_by_layer"]

                self._generate_state = {
                    "max_new_tokens": max_new_tokens,
                    "generated_token_ids": generated_token_ids,
                    "prefill_norm_output": prefill_state["norm_output"],
                    "prefill_logits": prefill_state["logits"],
                    "cache_by_layer": prefill_state["cache_by_layer"],
                    "step_results": step_results,
                }
            return self._generate_state

        with startup_timer(self.log_component, "prepare generate reference state"):
            prefill_state = self._ensure_prefill_full_state()
            current_attention_mask_2d = self.prefill_attention_mask_2d
            current_past_key_values = prefill_state["past_key_values"]
            generated_token_ids = [int(prefill_state["logits"][0, -1].argmax().item())]
            step_results: list[dict[str, Any]] = []
            max_new_tokens = int(self.runtime_config.get("max_new_tokens", 4))
            if max_new_tokens <= 0:
                raise ValueError(f"max_new_tokens 必须大于 0，当前拿到 {max_new_tokens}。")

            startup_log(
                self.log_component,
                f"generate decode planning max_new_tokens={max_new_tokens} decode_steps={max_new_tokens - 1}",
            )
            for step_idx in range(max_new_tokens - 1):
                startup_log(
                    self.log_component,
                    f"generate decode step {step_idx + 1}/{max_new_tokens - 1}",
                )
                decode_input_ids = torch.tensor(
                    [[generated_token_ids[-1]]],
                    device=self.device,
                    dtype=self.prefill_input_ids.dtype,
                )
                current_attention_mask_2d = torch.cat(
                    [
                        current_attention_mask_2d,
                        torch.ones(
                            (current_attention_mask_2d.shape[0], 1),
                            device=self.device,
                            dtype=current_attention_mask_2d.dtype,
                        ),
                    ],
                    dim=-1,
                )

                if self.modality == "text":
                    decode_runtime_inputs = prepare_text_decode_runtime_inputs(
                        self.model,
                        decode_input_ids=decode_input_ids,
                        attention_mask_2d=current_attention_mask_2d,
                        past_key_values=current_past_key_values,
                    )
                else:
                    decode_runtime_inputs = prepare_multimodal_decode_runtime_inputs(
                        self.model,
                        decode_input_ids=decode_input_ids,
                        attention_mask_2d=current_attention_mask_2d,
                        past_key_values=current_past_key_values,
                    )

                decode_outputs, hidden_states, hidden_stage_output = _run_live_decode_full(
                    self.model,
                    runtime_inputs=decode_runtime_inputs,
                    past_key_values=current_past_key_values,
                    is_last_stage=self.has_last_stage,
                )
                norm_output = decode_outputs.last_hidden_state.detach().clone()
                logits = _compute_logits(self.model, norm_output).detach().clone()
                next_token_id = int(logits[0, -1].argmax().item())
                generated_token_ids.append(next_token_id)

                step_results.append(
                    {
                        "step_idx": step_idx,
                        "decode_input_ids": decode_input_ids.detach().clone(),
                        "attention_mask_2d": current_attention_mask_2d.detach().clone(),
                        "attention_mask": decode_runtime_inputs.attention_mask.detach().clone(),
                        "cos": decode_runtime_inputs.cos.detach().clone(),
                        "sin": decode_runtime_inputs.sin.detach().clone(),
                        "position_ids": None
                        if decode_runtime_inputs.position_ids is None
                        else decode_runtime_inputs.position_ids.detach().clone(),
                        "hidden_states": tuple(hidden.detach().clone() for hidden in hidden_states),
                        "hidden_stage_output": None
                        if hidden_stage_output is None
                        else hidden_stage_output.detach().clone(),
                        "norm_output": norm_output,
                        "logits": logits,
                        "output_token_id": next_token_id,
                    }
                )
                current_past_key_values = decode_outputs.past_key_values

            self._generate_state = {
                "max_new_tokens": max_new_tokens,
                "generated_token_ids": generated_token_ids,
                "prefill_norm_output": prefill_state["norm_output"],
                "prefill_logits": prefill_state["logits"],
                "cache_by_layer": prefill_state["cache_by_layer"],
                "step_results": step_results,
            }
        return self._generate_state

    def _build_layers_for_stage(
        self,
        spec: StageSpec,
        *,
        cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
    ) -> list[dict[str, Any]]:
        if self.modality == "text":
            if not self.include_text_weights:
                return []
            static_weights = self._get_text_stage_static_weights(spec)
            layer_bundles = [dict(layer_bundle) for layer_bundle in static_weights.layer_bundles]
            if cache_by_layer is not None:
                for layer_bundle in layer_bundles:
                    layer_idx = int(layer_bundle["layer_idx"])
                    if layer_idx not in cache_by_layer:
                        continue
                    past_key, past_value = cache_by_layer[layer_idx]
                    layer_bundle["past_key"] = _runtime_tensor(
                        past_key,
                        device=self.bundle_device,
                        compute_dtype=self.compute_dtype,
                    )
                    layer_bundle["past_value"] = _runtime_tensor(
                        past_value,
                        device=self.bundle_device,
                        compute_dtype=self.compute_dtype,
                    )
            return layer_bundles

        return _build_stage_layer_bundles(
            self.model,
            start_idx=spec.start_idx,
            end_idx=spec.end_idx,
            device=self.bundle_device,
            compute_dtype=self.compute_dtype,
            cache_by_layer=cache_by_layer,
        )

    def _get_text_stage_static_weights(self, spec: StageSpec) -> TextStageWeightBundle:
        if spec.stage_idx in self._text_stage_static_weights:
            return self._text_stage_static_weights[spec.stage_idx]
        if self._text_weight_index is None or self._text_model_config is None:
            raise RuntimeError("text weight index/config 尚未初始化。")

        with startup_timer(
            self.log_component,
            f"load text stage weights stage_idx={spec.stage_idx} range={spec.start_idx}:{spec.end_idx}",
        ):
            stage_weights = load_text_decoder_stage_weight_bundle(
                model_path=self.runtime_config["model_path"],
                start_idx=spec.start_idx,
                end_idx=spec.end_idx,
                is_first_stage=spec.start_idx == 0,
                is_last_stage=spec.end_idx == self.num_layers - 1,
                device=self.bundle_device,
                compute_dtype=self.compute_dtype,
                weight_index=self._text_weight_index,
                config_spec=self._text_model_config,
                tp_shard_rank=self.tp_shard_rank,
                tp_shard_world_size=self.tp_shard_world_size,
            )
        self._text_stage_static_weights[spec.stage_idx] = stage_weights
        return stage_weights

    def _build_stage_visual_payload(
        self,
        spec: StageSpec,
        runtime_inputs: MultimodalRuntimeInputs,
    ) -> tuple[torch.Tensor | None, dict[int, torch.Tensor]]:
        return _build_stage_deepstack_payload(
            runtime_inputs,
            start_idx=spec.start_idx,
            end_idx=spec.end_idx,
            device=self.bundle_device,
            compute_dtype=self.compute_dtype,
        )

    def _build_prefill_bundle(self, spec: StageSpec) -> dict[str, Any]:
        stage_input = self._prefill_stage_inputs_by_stage[spec.stage_idx]
        hidden_stage_output = self._prefill_stage_outputs_by_stage[spec.stage_idx]
        is_last_stage = spec.end_idx == self.num_layers - 1
        layer_bundles = self._build_layers_for_stage(spec)
        visual_pos_masks, deepstack_by_layer = self._build_stage_visual_payload(spec, self.prefill_runtime_inputs)
        text_stage_weights = (
            self._get_text_stage_static_weights(spec)
            if self.modality == "text" and self.include_text_weights
            else None
        )

        bundle = {
            "module_name": f"{self.modality}_prefill_stage",
            "stage_type": "text_prefill_last" if is_last_stage else "text",
            "start_idx": spec.start_idx,
            "end_idx": spec.end_idx,
            "save_dtype": _save_dtype_name(self.compute_dtype),
            "original_input_dtype": str(stage_input.dtype),
            "original_input_device": str(stage_input.device),
            "attention_mask_2d": _runtime_tensor(self.prefill_attention_mask_2d_raw, device=self.bundle_device),
            "stage_input": _runtime_tensor(stage_input, device=self.bundle_device, compute_dtype=self.compute_dtype),
            "layer_input": _runtime_tensor(stage_input, device=self.bundle_device, compute_dtype=self.compute_dtype),
            "attention_mask": _runtime_tensor(self.prefill_runtime_inputs.attention_mask, device=self.bundle_device),
            "cos": _runtime_tensor(
                self.prefill_runtime_inputs.cos,
                device=self.bundle_device,
                compute_dtype=self.compute_dtype,
            ),
            "sin": _runtime_tensor(
                self.prefill_runtime_inputs.sin,
                device=self.bundle_device,
                compute_dtype=self.compute_dtype,
            ),
            "visual_pos_masks": visual_pos_masks,
            "deepstack_by_layer": deepstack_by_layer,
            "deepstack_layer_indices": sorted(deepstack_by_layer),
            "layers": layer_bundles,
        }

        if self.modality == "text":
            bundle["prompt"] = self.extra["prompt"]
            bundle["input_ids"] = _runtime_tensor(self.prefill_runtime_inputs.input_ids, device=self.bundle_device)
            bundle["tp_weight_sharded"] = False if text_stage_weights is None else text_stage_weights.tp_weight_sharded
            bundle["tp_shard_rank"] = None if text_stage_weights is None else text_stage_weights.tp_shard_rank
            bundle["tp_shard_world_size"] = (
                None if text_stage_weights is None else text_stage_weights.tp_shard_world_size
            )
            if spec.start_idx == 0 and text_stage_weights is not None:
                bundle["embed_tokens_weight"] = text_stage_weights.embed_tokens_weight
        else:
            bundle["num_frames"] = self.extra["num_frames"]
            bundle["frame_paths"] = self.extra["frame_paths"]
            bundle["input_ids"] = _runtime_tensor(self.raw_inputs.get("input_ids"), device=self.bundle_device)

        if is_last_stage:
            if self.modality == "text" and not hasattr(self, "model"):
                prefill_state = self._ensure_prefill_full_state()
                norm_output = prefill_state["norm_output"]
                logits = prefill_state["logits"]
            else:
                text_model = self.model.model.language_model
                norm_output = text_model.norm(hidden_stage_output).detach().clone()
                logits = _compute_logits(self.model, norm_output).detach().clone()
            bundle.update(
                {
                    "original_output_dtype": str(logits.dtype),
                    "original_output_device": str(logits.device),
                    "stage_output": _runtime_tensor(logits, device=self.bundle_device, compute_dtype=self.compute_dtype),
                    "layer_output": _runtime_tensor(logits, device=self.bundle_device, compute_dtype=self.compute_dtype),
                    "hidden_stage_output": _runtime_tensor(
                        hidden_stage_output,
                        device=self.bundle_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "norm_output": _runtime_tensor(norm_output, device=self.bundle_device, compute_dtype=self.compute_dtype),
                    "logits": _runtime_tensor(logits, device=self.bundle_device, compute_dtype=self.compute_dtype),
                }
            )
            if text_stage_weights is not None or self.modality != "text":
                bundle.update(
                    {
                        "final_norm_weight": (
                            text_stage_weights.final_norm_weight
                            if text_stage_weights is not None
                            else _runtime_tensor(
                                self.model.model.language_model.norm.weight,
                                device=self.bundle_device,
                                compute_dtype=self.compute_dtype,
                            )
                        ),
                        "final_norm_eps": (
                            text_stage_weights.final_norm_eps
                            if text_stage_weights is not None
                            else self.model.model.language_model.norm.variance_epsilon
                        ),
                        "lm_head_weight": (
                            text_stage_weights.lm_head_weight
                            if text_stage_weights is not None
                            else _runtime_tensor(
                                self.model.lm_head.weight,
                                device=self.bundle_device,
                                compute_dtype=self.compute_dtype,
                            )
                        ),
                        "lm_head_bias": (
                            text_stage_weights.lm_head_bias
                            if text_stage_weights is not None
                            else _runtime_tensor(
                                self.model.lm_head.bias,
                                device=self.bundle_device,
                                compute_dtype=self.compute_dtype,
                            )
                        ),
                    }
                )
        else:
            bundle.update(
                {
                    "original_output_dtype": str(hidden_stage_output.dtype),
                    "original_output_device": str(hidden_stage_output.device),
                    "stage_output": _runtime_tensor(
                        hidden_stage_output,
                        device=self.bundle_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "layer_output": _runtime_tensor(
                        hidden_stage_output,
                        device=self.bundle_device,
                        compute_dtype=self.compute_dtype,
                    ),
                }
            )

        return bundle

    def _build_decode_bundle(self, spec: StageSpec) -> dict[str, Any]:
        state = self._ensure_decode_state()
        stage_input = state["hidden_states"][spec.start_idx]
        is_last_stage = spec.end_idx == self.num_layers - 1
        layer_bundles = self._build_layers_for_stage(spec, cache_by_layer=state["cache_by_layer"])
        decode_runtime_inputs = MultimodalRuntimeInputs(
            input_ids=state["decode_input_ids"],
            attention_mask_2d=state["attention_mask_2d"],
            position_ids=state["position_ids"],
            inputs_embeds=state["hidden_states"][0],
            attention_mask=state["attention_mask"],
            cos=state["cos"],
            sin=state["sin"],
            visual_pos_masks=state["visual_pos_masks"],
            deepstack_by_layer=state["deepstack_by_layer"],
        )
        visual_pos_masks, deepstack_by_layer = self._build_stage_visual_payload(spec, decode_runtime_inputs)
        text_stage_weights = (
            self._get_text_stage_static_weights(spec)
            if self.modality == "text" and self.include_text_weights
            else None
        )

        bundle = {
            "module_name": f"{self.modality}_decode_stage",
            "stage_type": "text_decode_last" if is_last_stage else "text_decode",
            "start_idx": spec.start_idx,
            "end_idx": spec.end_idx,
            "save_dtype": _save_dtype_name(self.compute_dtype),
            "original_input_dtype": str(stage_input.dtype),
            "original_input_device": str(stage_input.device),
            "decode_source": state["decode_source"],
            "decode_token_id": state["decode_token_id"],
            "prefill_seq_len": int(self.prefill_input_ids.shape[-1]),
            "total_seq_len": int(state["attention_mask_2d"].shape[-1]),
            "prefill_input_ids": _runtime_tensor(self.prefill_input_ids, device=self.bundle_device),
            "decode_input_ids": _runtime_tensor(state["decode_input_ids"], device=self.bundle_device),
            "prefill_attention_mask_2d": _runtime_tensor(self.prefill_attention_mask_2d, device=self.bundle_device),
            "attention_mask_2d": _runtime_tensor(state["attention_mask_2d"], device=self.bundle_device),
            "stage_input": _runtime_tensor(stage_input, device=self.bundle_device, compute_dtype=self.compute_dtype),
            "layer_input": _runtime_tensor(stage_input, device=self.bundle_device, compute_dtype=self.compute_dtype),
            "attention_mask": _runtime_tensor(state["attention_mask"], device=self.bundle_device),
            "cos": _runtime_tensor(state["cos"], device=self.bundle_device, compute_dtype=self.compute_dtype),
            "sin": _runtime_tensor(state["sin"], device=self.bundle_device, compute_dtype=self.compute_dtype),
            "visual_pos_masks": visual_pos_masks,
            "deepstack_by_layer": deepstack_by_layer,
            "deepstack_layer_indices": sorted(deepstack_by_layer),
            "layers": layer_bundles,
        }
        if self.modality == "text":
            bundle["cache_by_layer"] = state["cache_by_layer"]

        if self.modality == "text":
            bundle["prompt"] = self.extra["prompt"]
            bundle["tp_weight_sharded"] = False if text_stage_weights is None else text_stage_weights.tp_weight_sharded
            bundle["tp_shard_rank"] = None if text_stage_weights is None else text_stage_weights.tp_shard_rank
            bundle["tp_shard_world_size"] = (
                None if text_stage_weights is None else text_stage_weights.tp_shard_world_size
            )
        else:
            bundle["num_frames"] = self.extra["num_frames"]
            bundle["frame_paths"] = self.extra["frame_paths"]
            bundle["position_ids"] = _runtime_tensor(state["position_ids"], device=self.bundle_device)

        if spec.start_idx == 0:
            if text_stage_weights is not None or self.modality != "text":
                bundle["embed_tokens_weight"] = (
                    text_stage_weights.embed_tokens_weight
                    if text_stage_weights is not None
                    else _runtime_tensor(
                        self.model.model.language_model.embed_tokens.weight,
                        device=self.bundle_device,
                        compute_dtype=self.compute_dtype,
                    )
                )

        if is_last_stage:
            hidden_stage_output = state["hidden_stage_output"]
            if hidden_stage_output is None:
                raise RuntimeError("decode last stage 没有拿到 final norm 前的 hidden_stage_output。")
            logits = state["logits"]
            norm_output = state["norm_output"]
            bundle.update(
                {
                    "original_output_dtype": str(logits.dtype),
                    "original_output_device": str(logits.device),
                    "stage_output": _runtime_tensor(logits, device=self.bundle_device, compute_dtype=self.compute_dtype),
                    "layer_output": _runtime_tensor(logits, device=self.bundle_device, compute_dtype=self.compute_dtype),
                    "hidden_stage_output": _runtime_tensor(
                        hidden_stage_output,
                        device=self.bundle_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "norm_output": _runtime_tensor(norm_output, device=self.bundle_device, compute_dtype=self.compute_dtype),
                    "logits": _runtime_tensor(logits, device=self.bundle_device, compute_dtype=self.compute_dtype),
                }
            )
            if text_stage_weights is not None or self.modality != "text":
                bundle.update(
                    {
                        "final_norm_weight": (
                            text_stage_weights.final_norm_weight
                            if text_stage_weights is not None
                            else _runtime_tensor(
                                self.model.model.language_model.norm.weight,
                                device=self.bundle_device,
                                compute_dtype=self.compute_dtype,
                            )
                        ),
                        "final_norm_eps": (
                            text_stage_weights.final_norm_eps
                            if text_stage_weights is not None
                            else self.model.model.language_model.norm.variance_epsilon
                        ),
                        "lm_head_weight": (
                            text_stage_weights.lm_head_weight
                            if text_stage_weights is not None
                            else _runtime_tensor(
                                self.model.lm_head.weight,
                                device=self.bundle_device,
                                compute_dtype=self.compute_dtype,
                            )
                        ),
                        "lm_head_bias": (
                            text_stage_weights.lm_head_bias
                            if text_stage_weights is not None
                            else _runtime_tensor(
                                self.model.lm_head.bias,
                                device=self.bundle_device,
                                compute_dtype=self.compute_dtype,
                            )
                        ),
                    }
                )
        else:
            stage_output = state["hidden_states"][spec.end_idx + 1]
            bundle.update(
                {
                    "original_output_dtype": str(stage_output.dtype),
                    "original_output_device": str(stage_output.device),
                    "stage_output": _runtime_tensor(
                        stage_output,
                        device=self.bundle_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "layer_output": _runtime_tensor(
                        stage_output,
                        device=self.bundle_device,
                        compute_dtype=self.compute_dtype,
                    ),
                }
            )

        return bundle

    def _build_generate_bundle(self, spec: StageSpec) -> dict[str, Any]:
        state = self._ensure_generate_state()
        is_last_stage = spec.end_idx == self.num_layers - 1
        prefill_stage_input = self._prefill_stage_inputs_by_stage[spec.stage_idx]
        prefill_hidden_stage_output = self._prefill_stage_outputs_by_stage[spec.stage_idx]
        layer_bundles = self._build_layers_for_stage(spec, cache_by_layer=state["cache_by_layer"])
        visual_pos_masks, deepstack_by_layer = self._build_stage_visual_payload(spec, self.prefill_runtime_inputs)
        text_stage_weights = (
            self._get_text_stage_static_weights(spec)
            if self.modality == "text" and self.include_text_weights
            else None
        )

        prefill_payload = {
            "attention_mask_2d": _runtime_tensor(self.prefill_attention_mask_2d, device=self.bundle_device),
            "stage_input": _runtime_tensor(
                prefill_stage_input,
                device=self.bundle_device,
                compute_dtype=self.compute_dtype,
            ),
            "attention_mask": _runtime_tensor(self.prefill_runtime_inputs.attention_mask, device=self.bundle_device),
            "cos": _runtime_tensor(
                self.prefill_runtime_inputs.cos,
                device=self.bundle_device,
                compute_dtype=self.compute_dtype,
            ),
            "sin": _runtime_tensor(
                self.prefill_runtime_inputs.sin,
                device=self.bundle_device,
                compute_dtype=self.compute_dtype,
            ),
        }
        if is_last_stage:
            prefill_payload.update(
                {
                    "stage_output": _runtime_tensor(
                        state["prefill_logits"],
                        device=self.bundle_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "hidden_stage_output": _runtime_tensor(
                        prefill_hidden_stage_output,
                        device=self.bundle_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "norm_output": _runtime_tensor(
                        state["prefill_norm_output"],
                        device=self.bundle_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "logits": _runtime_tensor(
                        state["prefill_logits"],
                        device=self.bundle_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "output_token_id": state["generated_token_ids"][0],
                }
            )
        else:
            prefill_payload["stage_output"] = _runtime_tensor(
                prefill_hidden_stage_output,
                device=self.bundle_device,
                compute_dtype=self.compute_dtype,
            )

        decode_steps = []
        for step_result in state["step_results"]:
            step_payload = {
                "step_idx": step_result["step_idx"],
                "decode_input_ids": _runtime_tensor(step_result["decode_input_ids"], device=self.bundle_device),
                "attention_mask_2d": _runtime_tensor(step_result["attention_mask_2d"], device=self.bundle_device),
                "total_seq_len": int(step_result["attention_mask_2d"].shape[-1]),
                "stage_input": _runtime_tensor(
                    step_result["hidden_states"][spec.start_idx],
                    device=self.bundle_device,
                    compute_dtype=self.compute_dtype,
                ),
                "attention_mask": _runtime_tensor(step_result["attention_mask"], device=self.bundle_device),
                "cos": _runtime_tensor(
                    step_result["cos"],
                    device=self.bundle_device,
                    compute_dtype=self.compute_dtype,
                ),
                "sin": _runtime_tensor(
                    step_result["sin"],
                    device=self.bundle_device,
                    compute_dtype=self.compute_dtype,
                ),
                "visual_pos_masks": None,
                "deepstack_by_layer": {},
                "deepstack_layer_indices": [],
            }
            if self.modality == "multimodal":
                step_payload["position_ids"] = _runtime_tensor(step_result["position_ids"], device=self.bundle_device)

            if is_last_stage:
                hidden_stage_output = step_result["hidden_stage_output"]
                if hidden_stage_output is None:
                    raise RuntimeError("generate decode last stage 没有拿到 final norm 前的 hidden_stage_output。")
                step_payload.update(
                    {
                        "stage_output": _runtime_tensor(
                            step_result["logits"],
                            device=self.bundle_device,
                            compute_dtype=self.compute_dtype,
                        ),
                        "hidden_stage_output": _runtime_tensor(
                            hidden_stage_output,
                            device=self.bundle_device,
                            compute_dtype=self.compute_dtype,
                        ),
                        "norm_output": _runtime_tensor(
                            step_result["norm_output"],
                            device=self.bundle_device,
                            compute_dtype=self.compute_dtype,
                        ),
                        "logits": _runtime_tensor(
                            step_result["logits"],
                            device=self.bundle_device,
                            compute_dtype=self.compute_dtype,
                        ),
                        "output_token_id": step_result["output_token_id"],
                    }
                )
            else:
                step_payload["stage_output"] = _runtime_tensor(
                    step_result["hidden_states"][spec.end_idx + 1],
                    device=self.bundle_device,
                    compute_dtype=self.compute_dtype,
                )
            decode_steps.append(step_payload)

        bundle = {
            "module_name": f"{self.modality}_generate_stage",
            "stage_type": f"{self.modality}_generate_last" if is_last_stage else f"{self.modality}_generate",
            "start_idx": spec.start_idx,
            "end_idx": spec.end_idx,
            "save_dtype": _save_dtype_name(self.compute_dtype),
            "original_input_dtype": str(prefill_stage_input.dtype),
            "original_input_device": str(prefill_stage_input.device),
            "max_new_tokens": state["max_new_tokens"],
            "prefill_seq_len": int(self.prefill_input_ids.shape[-1]),
            "prefill_input_ids": _runtime_tensor(self.prefill_input_ids, device=self.bundle_device),
            "generated_token_ids": torch.tensor(
                [state["generated_token_ids"]],
                device=self.bundle_device,
                dtype=self.prefill_input_ids.dtype,
            ),
            "prefill": prefill_payload,
            "visual_pos_masks": visual_pos_masks,
            "deepstack_by_layer": deepstack_by_layer,
            "deepstack_layer_indices": sorted(deepstack_by_layer),
            "layers": layer_bundles,
            "decode_steps": decode_steps,
        }
        if self.modality == "text":
            bundle["cache_by_layer"] = state["cache_by_layer"]

        if self.modality == "text":
            bundle["prompt"] = self.extra["prompt"]
            bundle["tp_weight_sharded"] = False if text_stage_weights is None else text_stage_weights.tp_weight_sharded
            bundle["tp_shard_rank"] = None if text_stage_weights is None else text_stage_weights.tp_shard_rank
            bundle["tp_shard_world_size"] = (
                None if text_stage_weights is None else text_stage_weights.tp_shard_world_size
            )
            if spec.start_idx == 0:
                bundle["input_ids"] = _runtime_tensor(self.prefill_input_ids, device=self.bundle_device)
        else:
            bundle["num_frames"] = self.extra["num_frames"]
            bundle["frame_paths"] = self.extra["frame_paths"]

        if spec.start_idx == 0:
            if text_stage_weights is not None or self.modality != "text":
                bundle["embed_tokens_weight"] = (
                    text_stage_weights.embed_tokens_weight
                    if text_stage_weights is not None
                    else _runtime_tensor(
                        self.model.model.language_model.embed_tokens.weight,
                        device=self.bundle_device,
                        compute_dtype=self.compute_dtype,
                    )
                )

        if is_last_stage:
            if text_stage_weights is not None or self.modality != "text":
                bundle.update(
                    {
                        "final_norm_weight": (
                            text_stage_weights.final_norm_weight
                            if text_stage_weights is not None
                            else _runtime_tensor(
                                self.model.model.language_model.norm.weight,
                                device=self.bundle_device,
                                compute_dtype=self.compute_dtype,
                            )
                        ),
                        "final_norm_eps": (
                            text_stage_weights.final_norm_eps
                            if text_stage_weights is not None
                            else self.model.model.language_model.norm.variance_epsilon
                        ),
                        "lm_head_weight": (
                            text_stage_weights.lm_head_weight
                            if text_stage_weights is not None
                            else _runtime_tensor(
                                self.model.lm_head.weight,
                                device=self.bundle_device,
                                compute_dtype=self.compute_dtype,
                            )
                        ),
                        "lm_head_bias": (
                            text_stage_weights.lm_head_bias
                            if text_stage_weights is not None
                            else _runtime_tensor(
                                self.model.lm_head.bias,
                                device=self.bundle_device,
                                compute_dtype=self.compute_dtype,
                            )
                        ),
                    }
                )

        return bundle

    def _build_generate_bundle_runtime_only(self, spec: StageSpec) -> dict[str, Any]:
        if self.modality != "text":
            raise RuntimeError("runtime-only generate bundle 当前只支持 text modality。")

        text_stage_weights = (
            self._get_text_stage_static_weights(spec)
            if self.include_text_weights
            else None
        )
        layer_bundles = self._build_layers_for_stage(spec)
        bundle = _build_runtime_only_text_generate_bundle(
            spec=spec,
            bundle_device=self.bundle_device,
            compute_dtype=self.compute_dtype,
            prefill_attention_mask_2d=self.prefill_attention_mask_2d,
            prefill_seq_len=int(self.prefill_input_ids.shape[-1]),
            batch_size=int(self.prefill_input_ids.shape[0]),
            token_id_dtype=self.prefill_input_ids.dtype,
            hidden_size=self._text_model_config.hidden_size if self._text_model_config is not None else 0,
            layers=layer_bundles,
            text_stage_weights=text_stage_weights,
        )
        bundle["max_new_tokens"] = int(self.runtime_config.get("max_new_tokens", 4))

        if spec.start_idx == 0:
            bundle["input_ids"] = _runtime_tensor(self.prefill_input_ids, device=self.bundle_device)
            if text_stage_weights is not None and text_stage_weights.embed_tokens_weight is not None:
                bundle["embed_tokens_weight"] = text_stage_weights.embed_tokens_weight

        if spec.end_idx == self.num_layers - 1 and text_stage_weights is not None:
            if text_stage_weights.final_norm_weight is not None:
                bundle["final_norm_weight"] = text_stage_weights.final_norm_weight
            if text_stage_weights.final_norm_eps is not None:
                bundle["final_norm_eps"] = text_stage_weights.final_norm_eps
            if text_stage_weights.lm_head_weight is not None:
                bundle["lm_head_weight"] = text_stage_weights.lm_head_weight
            bundle["lm_head_bias"] = text_stage_weights.lm_head_bias

        return bundle

    def build_stage_bundle(self, stage_idx: int) -> dict[str, Any]:
        spec = self.stage_specs_by_idx[stage_idx]
        with startup_timer(
            self.log_component,
            f"materialize stage_idx={stage_idx} range={spec.start_idx}:{spec.end_idx}",
        ):
            if self.mode == "prefill":
                bundle = self._build_prefill_bundle(spec)
            elif self.mode == "decode":
                bundle = self._build_decode_bundle(spec)
            elif self.mode == "generate":
                if self.modality == "text" and not self.include_runtime_reference:
                    bundle = self._build_generate_bundle_runtime_only(spec)
                else:
                    bundle = self._build_generate_bundle(spec)
            else:
                raise ValueError(
                    f"不支持的 direct stage 构造组合: modality={self.modality!r} mode={self.mode!r} stage_idx={stage_idx}"
                )

            if (
                self.modality == "text"
                and self.mode == "generate"
                and not self.include_text_weights
            ):
                if self.include_runtime_reference:
                    return _compact_text_generate_scaffold(bundle)
                if bundle.get("runtime_only_generate"):
                    return _compact_runtime_only_text_generate_scaffold(bundle)
            return bundle


def build_direct_stage_bundle(
    *,
    stage_idx: int,
    start_idx: int,
    end_idx: int,
    runtime_config: dict[str, Any],
    tp_shard_rank: int | None = None,
    tp_shard_world_size: int | None = None,
    include_text_weights: bool = True,
) -> dict:
    with DirectStageBundleBuilder(
        stage_specs=[
            StageSpec(
                stage_idx=stage_idx,
                start_idx=start_idx,
                end_idx=end_idx,
                num_layers=end_idx - start_idx + 1,
                save_dtype=runtime_config.get("save_dtype", "auto"),
                bundle_path=None,
            )
        ],
        runtime_config=runtime_config,
        tp_shard_rank=tp_shard_rank,
        tp_shard_world_size=tp_shard_world_size,
        include_text_weights=include_text_weights,
    ) as builder:
        return builder.build_stage_bundle(stage_idx)


def materialize_direct_text_stage_bundle_from_scaffold(
    *,
    stage_bundle_scaffold: dict[str, Any],
    runtime_config: dict[str, Any],
    compute_dtype: torch.dtype,
    tp_shard_rank: int | None = None,
    tp_shard_world_size: int | None = None,
) -> dict[str, Any]:
    bundle = _restore_runtime_only_text_generate_prompt_metadata(
        dict(stage_bundle_scaffold),
        runtime_config=runtime_config,
    )
    model_path = runtime_config["model_path"]
    start_idx = int(bundle["start_idx"])
    end_idx = int(bundle["end_idx"])

    weight_index = load_model_weight_index(model_path)
    config_spec = load_text_model_config_spec(model_path)
    stage_weights = load_text_decoder_stage_weight_bundle(
        model_path=model_path,
        start_idx=start_idx,
        end_idx=end_idx,
        is_first_stage=start_idx == 0,
        is_last_stage=end_idx == config_spec.num_hidden_layers - 1,
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
        weight_index=weight_index,
        config_spec=config_spec,
        tp_shard_rank=tp_shard_rank,
        tp_shard_world_size=tp_shard_world_size,
    )

    bundle["layers"] = [dict(layer_bundle) for layer_bundle in stage_weights.layer_bundles]
    bundle["tp_weight_sharded"] = stage_weights.tp_weight_sharded
    bundle["tp_shard_rank"] = stage_weights.tp_shard_rank
    bundle["tp_shard_world_size"] = stage_weights.tp_shard_world_size
    bundle.pop("cache_by_layer", None)

    if start_idx == 0 and stage_weights.embed_tokens_weight is not None:
        bundle["embed_tokens_weight"] = stage_weights.embed_tokens_weight
    if end_idx == config_spec.num_hidden_layers - 1:
        if stage_weights.final_norm_weight is not None:
            bundle["final_norm_weight"] = stage_weights.final_norm_weight
        if stage_weights.final_norm_eps is not None:
            bundle["final_norm_eps"] = stage_weights.final_norm_eps
        if stage_weights.lm_head_weight is not None:
            bundle["lm_head_weight"] = stage_weights.lm_head_weight
        bundle["lm_head_bias"] = stage_weights.lm_head_bias
    bundle = _restore_text_generate_bundle_runtime_inputs(
        bundle,
        config_spec=config_spec,
        compute_dtype=compute_dtype,
    )
    return bundle


def build_direct_pipeline_manifest(
    *,
    modality: str,
    mode: str,
    stage_ranges: list[tuple[int, int]],
    model_path: str,
    save_dtype: str,
    prompt: str | None = None,
    decode_token_id: int | None = None,
    max_new_tokens: int | None = None,
    num_frames: int | None = None,
    frame_dir: str | None = None,
    include_runtime_reference: bool | None = None,
) -> TextPipelineManifest:
    runtime_config = _build_runtime_config(
        modality=modality,
        mode=mode,
        model_path=model_path,
        save_dtype=save_dtype,
        prompt=prompt,
        decode_token_id=decode_token_id,
        max_new_tokens=max_new_tokens,
        num_frames=num_frames,
        frame_dir=frame_dir,
        include_runtime_reference=include_runtime_reference,
    )
    stages = [
        StageSpec(
            stage_idx=stage_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            num_layers=end_idx - start_idx + 1,
            save_dtype=save_dtype,
            bundle_path=None,
        )
        for stage_idx, (start_idx, end_idx) in enumerate(stage_ranges)
    ]
    return TextPipelineManifest(
        pipeline_type=_pipeline_type(modality, mode),
        num_stages=len(stages),
        stage_ranges=stage_ranges,
        bundle_dir="<direct>",
        stages=stages,
        boundaries=[],
        num_frames=0 if modality == "text" else int(num_frames or 8),
        save_dtype=save_dtype,
        runtime_config=runtime_config,
    )


def build_direct_hybrid_manifest(
    *,
    modality: str,
    mode: str,
    stage_ranges: list[tuple[int, int]],
    tp_degrees: list[int],
    model_path: str,
    save_dtype: str,
    prompt: str | None = None,
    decode_token_id: int | None = None,
    max_new_tokens: int | None = None,
    num_frames: int | None = None,
    frame_dir: str | None = None,
    backend: str = "hybrid",
    include_runtime_reference: bool | None = None,
) -> TextHybridManifest:
    pipeline_manifest = build_direct_pipeline_manifest(
        modality=modality,
        mode=mode,
        stage_ranges=stage_ranges,
        model_path=model_path,
        save_dtype=save_dtype,
        prompt=prompt,
        decode_token_id=decode_token_id,
        max_new_tokens=max_new_tokens,
        num_frames=num_frames,
        frame_dir=frame_dir,
        include_runtime_reference=include_runtime_reference,
    )
    parsed_tp_degrees = parse_tp_degrees(tp_degrees)
    if len(parsed_tp_degrees) != pipeline_manifest.num_stages:
        raise ValueError(
            f"stage 数是 {pipeline_manifest.num_stages}，但 TP 度数拿到 {len(parsed_tp_degrees)} 个。"
        )
    layout = build_hybrid_layout(parsed_tp_degrees)
    return TextHybridManifest.from_pipeline_manifest(
        pipeline_manifest,
        layout,
        runtime=_runtime_name(modality, mode, backend),
    )


__all__ = [
    "DirectStageBundleBuilder",
    "build_direct_stage_bundle",
    "materialize_direct_text_stage_bundle_from_scaffold",
    "build_direct_pipeline_manifest",
    "build_direct_hybrid_manifest",
    "compact_runtime_only_text_prompt_metadata_for_broadcast",
    "prepare_runtime_only_text_generate_prompt_metadata",
    "restore_runtime_only_text_prompt_metadata_from_broadcast",
]
