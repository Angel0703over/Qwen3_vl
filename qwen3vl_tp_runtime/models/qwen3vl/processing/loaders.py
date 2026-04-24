"""Local model and processor loading helpers."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from qwen3vl_tp_runtime.hexgen_core.config import MODEL_PATH
from qwen3vl_tp_runtime.models.qwen3vl.weights import (
    ModelWeightIndex,
    load_model_weight_index,
    load_tensors_by_name,
)


def load_model(
    model_path: str = MODEL_PATH,
    *,
    attn_implementation: str = "eager",
):
    return Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=attn_implementation,
        local_files_only=True,
    ).eval()


def load_processor(model_path: str = MODEL_PATH):
    return AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )


def load_text_tokenizer(model_path: str = MODEL_PATH):
    return AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )


def load_text_tokenizer_backend(model_path: str = MODEL_PATH):
    try:
        from tokenizers import Tokenizer
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("当前环境没有可用的 tokenizers 后端。") from exc

    tokenizer_path = Path(model_path) / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"找不到 tokenizer.json: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


def inspect_model_weights(model_path: str = MODEL_PATH) -> ModelWeightIndex:
    return load_model_weight_index(model_path)


__all__ = [
    "inspect_model_weights",
    "load_model",
    "load_processor",
    "load_text_tokenizer",
    "load_text_tokenizer_backend",
    "load_model_weight_index",
    "load_tensors_by_name",
]
