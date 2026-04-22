"""Wrappers around the upstream Qwen3-VL vision encoder."""

from __future__ import annotations

import torch


def encode_image_features(
    model,
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    output_device: torch.device,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the upstream image vision encoder and return decoder-ready embeddings."""

    image_outputs = model.model.get_image_features(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        return_dict=True,
    )
    image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(output_device, output_dtype)
    return image_embeds, image_outputs.deepstack_features


def encode_video_features(
    model,
    *,
    pixel_values_videos: torch.Tensor,
    video_grid_thw: torch.Tensor | None,
    output_device: torch.device,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the upstream video vision encoder and return decoder-ready embeddings."""

    video_outputs = model.model.get_video_features(
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
        return_dict=True,
    )
    video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(output_device, output_dtype)
    return video_embeds, video_outputs.deepstack_features


__all__ = [
    "encode_image_features",
    "encode_video_features",
]
