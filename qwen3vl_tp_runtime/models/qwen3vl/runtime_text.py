"""Text runtime helpers for prompt metadata and runtime-only stage restore."""

from __future__ import annotations

import sys
from typing import Any

import torch

from qwen3vl_tp_runtime.models.qwen3vl.live.common import _resolve_compute_dtype, _runtime_tensor
from qwen3vl_tp_runtime.models.qwen3vl.processing import (
    build_text_inputs,
    load_processor,
    load_text_tokenizer,
    load_text_tokenizer_backend,
)
from qwen3vl_tp_runtime.models.qwen3vl.weights import load_tensors_from_index


def _builder_dep(name: str, fallback):
    builder_mod = sys.modules.get("qwen3vl_tp_runtime.models.qwen3vl.runtime_builder")
    if builder_mod is not None and hasattr(builder_mod, name):
        return getattr(builder_mod, name)
    return fallback


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _default_mask_2d(
    input_ids: torch.Tensor | None,
    attention_mask_2d: torch.Tensor | None,
) -> torch.Tensor | None:
    if attention_mask_2d is not None:
        return attention_mask_2d
    if input_ids is None:
        return None
    return torch.ones_like(input_ids)


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _load_text_compute_dtype_ref(
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
        loaded = _builder_dep("load_tensors_from_index", load_tensors_from_index)(
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


def _build_text_msgs(prompt: str) -> list[dict[str, Any]]:
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


def _build_qwen_text_prompt(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def compact_text_prompt_meta(
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


def restore_text_prompt_meta(
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


def _norm_text_prompt_meta(
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


def prepare_text_prompt_meta(
    runtime_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    model_path = runtime_config["model_path"]
    prompt = runtime_config.get("prompt", "请用中文简要介绍一下人工智能。")
    try:
        tokenizer_backend = _builder_dep("load_text_tokenizer_backend", load_text_tokenizer_backend)(model_path)
        encoded = tokenizer_backend.encode(_build_qwen_text_prompt(prompt))
        return {
            "input_ids": torch.tensor([encoded.ids], dtype=torch.int64),
            "attention_mask": None,
        }
    except Exception:
        try:
            tokenizer = _builder_dep("load_text_tokenizer", load_text_tokenizer)(model_path)
            raw_inputs = tokenizer.apply_chat_template(
                _build_text_msgs(prompt),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            processor = _builder_dep("load_processor", load_processor)(model_path)
            raw_inputs = build_text_inputs(processor, prompt)
    return _norm_text_prompt_meta(raw_inputs)


def _resolve_text_prompt_meta(
    runtime_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    input_ids = runtime_config.get("_runtime_only_input_ids")
    if input_ids is None:
        return prepare_text_prompt_meta(runtime_config)
    return {
        "input_ids": _runtime_tensor(input_ids, device=torch.device("cpu")),
        "attention_mask": _runtime_tensor(
            runtime_config.get("_runtime_only_attention_mask"),
            device=torch.device("cpu"),
        ),
    }


def _restore_text_prompt_bundle(
    bundle: dict[str, Any],
    *,
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    if not bundle.get("runtime_only_generate"):
        return bundle

    if bundle.get("modality") == "multimodal":
        restored = dict(bundle)
        restored.pop("runtime_only_prompt_local_rebuild", None)
        if restored.get("prefill_attention_mask_2d") is None:
            raise RuntimeError("multimodal runtime-only generate scaffold 缺少 prefill_attention_mask_2d。")
        return restored

    needs_restore = bool(bundle.pop("runtime_only_prompt_local_rebuild", False))
    is_first_stage = int(bundle["start_idx"]) == 0
    if (
        not needs_restore
        and bundle.get("prefill_attention_mask_2d") is not None
        and (not is_first_stage or bundle.get("input_ids") is not None)
    ):
        return bundle

    prompt_metadata = _resolve_text_prompt_meta(runtime_config)
    input_ids = _runtime_tensor(prompt_metadata["input_ids"], device=torch.device("cpu"))
    attention_mask_2d = _default_mask_2d(
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
    restored["token_id_dtype"] = _dtype_name(input_ids.dtype)
    if is_first_stage:
        restored["input_ids"] = input_ids
    return restored


def _prep_rt_text_session(
    weight_index,
    runtime_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], torch.dtype, dict[str, Any], torch.device]:
    raw_inputs = _resolve_text_prompt_meta(runtime_config)
    compute_dtype_ref = _load_text_compute_dtype_ref(
        weight_index,
        device=torch.device("cpu"),
    )
    compute_dtype = _resolve_compute_dtype(
        compute_dtype_ref,
        runtime_config.get("save_dtype", "auto"),
    )
    return raw_inputs, compute_dtype, {}, _default_device()


__all__ = [
    "compact_text_prompt_meta",
    "prepare_text_prompt_meta",
    "restore_text_prompt_meta",
]
