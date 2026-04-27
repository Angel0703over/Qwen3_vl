"""Vision front-end bridge for live Qwen3-VL runtime inputs."""

from __future__ import annotations

import torch

from qwen3vl_tp_runtime.models.qwen3vl.vision.frontend import (
    merge_mm_frontend_visuals,
)


def materialize_visual_features(
    model,
    *,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor,
    pixel_values: torch.Tensor | None,
    pixel_values_videos: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[int, torch.Tensor]]:
    """Run the upstream vision encoder and merge visual features into decoder inputs."""
    return merge_mm_frontend_visuals(
        model,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )


__all__ = ["materialize_visual_features"]
