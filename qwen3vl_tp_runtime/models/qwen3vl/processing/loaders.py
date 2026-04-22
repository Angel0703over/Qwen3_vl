"""Local model and processor loading helpers."""

from __future__ import annotations

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwen3vl_tp_runtime.hexgen_core.config import MODEL_PATH


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


__all__ = [
    "load_model",
    "load_processor",
]
