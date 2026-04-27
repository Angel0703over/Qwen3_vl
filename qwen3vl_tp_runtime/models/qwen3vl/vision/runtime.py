"""Multimodal frontend runtime orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from qwen3vl_tp_runtime.models.qwen3vl.processing import (
    build_inputs,
    list_frames,
    load_processor,
)
from qwen3vl_tp_runtime.models.qwen3vl.runtime_mm_stage import (
    MmFrontendSeed,
    MmRuntimeState,
    move_mm_frontend_seed,
)
from qwen3vl_tp_runtime.models.qwen3vl.vision.frontend import (
    load_mm_frontend_model,
    model_device,
    move_frontend_inputs,
)
from qwen3vl_tp_runtime.models.qwen3vl.vision.state import (
    MmFrontendPlan,
    build_mm_frontend_plan,
    prepare_mm_frontend_state,
)


@dataclass(slots=True)
class MmFrontendBatch:
    """Raw multimodal inputs prepared for one frontend activation."""

    raw_inputs: dict[str, Any]
    frame_paths: list[str]

    @property
    def num_frames(self) -> int:
        return len(self.frame_paths)


def build_mm_frontend_batch(runtime_config: dict[str, Any]) -> MmFrontendBatch:
    """Prepare processor inputs for one multimodal frontend pass."""

    model_path = runtime_config["model_path"]
    processor = load_processor(model_path)
    num_frames = int(runtime_config.get("num_frames", 8))
    frame_paths = list_frames(num_frames, runtime_config.get("frame_dir"))
    raw_inputs = build_inputs(processor, frame_paths)
    return MmFrontendBatch(
        raw_inputs=raw_inputs,
        frame_paths=frame_paths,
    )


def _prepare_mm_frontend_batch_on_device(
    runtime_config: dict[str, Any],
    *,
    model: torch.nn.Module | None = None,
) -> tuple[torch.nn.Module, MmFrontendBatch]:
    frontend_model = (
        load_mm_frontend_model(
            runtime_config["model_path"],
            weight_index=runtime_config.get("_mm_weight_index"),
        )
        if model is None
        else model
    )
    frontend_batch = build_mm_frontend_batch(runtime_config)
    raw_inputs = move_frontend_inputs(
        frontend_batch.raw_inputs,
        device=model_device(frontend_model),
    )
    return frontend_model, MmFrontendBatch(
        raw_inputs=raw_inputs,
        frame_paths=list(frontend_batch.frame_paths),
    )


def prepare_mm_frontend_parts(
    runtime_config: dict[str, Any],
    *,
    model: torch.nn.Module | None = None,
) -> tuple[torch.nn.Module, MmFrontendBatch, MmRuntimeState]:
    """Prepare the frontend model, raw inputs, and explicit runtime state."""

    frontend_model, frontend_batch = _prepare_mm_frontend_batch_on_device(
        runtime_config,
        model=model,
    )
    frontend_state = prepare_mm_frontend_state(
        frontend_model,
        frontend_batch.raw_inputs,
        inputs_on_device=True,
    )
    return frontend_model, frontend_batch, frontend_state


def prepare_mm_frontend_seed_parts(
    runtime_config: dict[str, Any],
    *,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, MmFrontendBatch, MmFrontendSeed]:
    """Prepare the frontend model, raw inputs, and thin runtime seed."""

    frontend_model, frontend_batch, frontend_state = prepare_mm_frontend_parts(
        runtime_config,
        model=model,
    )
    seed_device = torch.device("cpu") if device is None else device
    frontend_seed = move_mm_frontend_seed(frontend_state, device=seed_device)
    return frontend_model, frontend_batch, frontend_seed


def prepare_mm_frontend_plan_parts(
    runtime_config: dict[str, Any],
    *,
    model: torch.nn.Module | None = None,
) -> tuple[torch.nn.Module, MmFrontendBatch, MmFrontendPlan]:
    """Legacy compat path that prepares the frontend model, inputs, and plan."""

    frontend_model, frontend_batch = _prepare_mm_frontend_batch_on_device(
        runtime_config,
        model=model,
    )
    frontend_plan = build_mm_frontend_plan(
        frontend_model,
        frontend_batch.raw_inputs,
        inputs_on_device=True,
    )
    return frontend_model, frontend_batch, frontend_plan


__all__ = [
    "MmFrontendBatch",
    "build_mm_frontend_batch",
    "prepare_mm_frontend_parts",
    "prepare_mm_frontend_seed_parts",
    "prepare_mm_frontend_plan_parts",
]
