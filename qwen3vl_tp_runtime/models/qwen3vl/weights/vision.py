"""Vision/frontend weight loading helpers for multimodal direct runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

from qwen3vl_tp_runtime.models.qwen3vl.weights.index import (
    ModelWeightIndex,
    load_model_weight_index,
)
from qwen3vl_tp_runtime.models.qwen3vl.weights.loader import load_tensors_from_index


MM_FRONTEND_VISUAL_PREFIX = "model.visual."
MM_FRONTEND_EMBED_TOKENS_NAME = "model.language_model.embed_tokens.weight"


@dataclass(slots=True)
class MmFrontendWeightBundle:
    """Selective frontend weights needed by the lightweight vision shell."""

    compute_dtype: torch.dtype
    visual_state: dict[str, torch.Tensor]
    embed_tokens_weight: torch.Tensor


def load_mm_frontend_config(model_path: str) -> Qwen3VLConfig:
    """Load the HF config used by the lightweight multimodal frontend shell."""

    return Qwen3VLConfig.from_pretrained(model_path, local_files_only=True)


def build_mm_frontend_parameter_names(
    visual_parameter_names: Iterable[str],
) -> list[str]:
    """Resolve the checkpoint parameter names needed by the frontend shell."""

    resolved = [f"{MM_FRONTEND_VISUAL_PREFIX}{name}" for name in visual_parameter_names]
    resolved.append(MM_FRONTEND_EMBED_TOKENS_NAME)
    return resolved


def load_mm_frontend_weight_bundle(
    *,
    model_path: str,
    visual_parameter_names: Iterable[str],
    device: torch.device | str,
    weight_index: ModelWeightIndex | None = None,
) -> MmFrontendWeightBundle:
    """Selectively load frontend-only weights for vision tower + token embeddings."""

    index = weight_index or load_model_weight_index(model_path)
    requested_names = build_mm_frontend_parameter_names(visual_parameter_names)
    if not requested_names:
        raise RuntimeError("frontend selective load 没有拿到任何 parameter name。")

    sample_name = (
        MM_FRONTEND_EMBED_TOKENS_NAME
        if index.has_tensor(MM_FRONTEND_EMBED_TOKENS_NAME)
        else requested_names[0]
    )
    sample_tensors = load_tensors_from_index(
        index,
        [sample_name],
        device="cpu",
        compute_dtype=None,
        strict=True,
    )
    target_dtype = _first_float_dtype(sample_tensors)
    loaded_tensors = load_tensors_from_index(
        index,
        requested_names,
        device=device,
        compute_dtype=target_dtype,
        strict=True,
    )
    return MmFrontendWeightBundle(
        compute_dtype=target_dtype,
        visual_state=_strip_prefix(loaded_tensors, MM_FRONTEND_VISUAL_PREFIX),
        embed_tokens_weight=loaded_tensors[MM_FRONTEND_EMBED_TOKENS_NAME],
    )


def _strip_prefix(
    tensors: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    return {
        name[len(prefix):]: tensor
        for name, tensor in tensors.items()
        if name.startswith(prefix)
    }


def _first_float_dtype(tensors: dict[str, torch.Tensor]) -> torch.dtype:
    for tensor in tensors.values():
        if tensor.is_floating_point():
            return tensor.dtype
    return torch.float32


__all__ = [
    "MM_FRONTEND_EMBED_TOKENS_NAME",
    "MM_FRONTEND_VISUAL_PREFIX",
    "MmFrontendWeightBundle",
    "build_mm_frontend_parameter_names",
    "load_mm_frontend_config",
    "load_mm_frontend_weight_bundle",
]
