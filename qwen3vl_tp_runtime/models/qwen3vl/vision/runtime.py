"""Multimodal frontend runtime orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from qwen3vl_tp_runtime.models.qwen3vl.processing import (
    build_inputs_with_metadata,
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
    video_input_metadata: dict[str, Any] | None = None

    @property
    def num_frames(self) -> int:
        metadata = self.video_input_metadata if isinstance(self.video_input_metadata, dict) else {}
        frame_count = metadata.get("frame_count")
        if frame_count is not None:
            try:
                return int(frame_count)
            except (TypeError, ValueError):
                pass
        return len(self.frame_paths)


def build_mm_frontend_batch(runtime_config: dict[str, Any]) -> MmFrontendBatch:
    """Prepare processor inputs for one multimodal frontend pass."""

    model_path = runtime_config["model_path"]
    processor = load_processor(model_path)
    num_frames = int(runtime_config.get("num_frames", 8))
    video_path = runtime_config.get("video_path")
    video_url = runtime_config.get("video_url")
    if video_path is None and video_url is None:
        frame_paths = list_frames(num_frames, runtime_config.get("frame_dir"))
        frame_paths_for_builder = frame_paths
    else:
        frame_paths = []
        frame_paths_for_builder = None
    raw_inputs, video_input_metadata = build_inputs_with_metadata(
        processor,
        frame_paths_for_builder,
        video_path=video_path,
        video_url=video_url,
        sample_fps=runtime_config.get("sample_fps", 1),
        video_fps=runtime_config.get("video_fps"),
        video_nframes=runtime_config.get("video_nframes"),
        video_start=runtime_config.get("video_start"),
        video_end=runtime_config.get("video_end"),
        video_min_frames=runtime_config.get("video_min_frames"),
        video_max_frames=runtime_config.get("video_max_frames"),
    )
    return MmFrontendBatch(
        raw_inputs=raw_inputs,
        frame_paths=frame_paths,
        video_input_metadata=video_input_metadata,
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
        video_input_metadata=frontend_batch.video_input_metadata,
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
