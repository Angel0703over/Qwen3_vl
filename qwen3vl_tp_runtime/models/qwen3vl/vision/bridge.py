"""Vision front-end bridge for live Qwen3-VL runtime inputs."""

from __future__ import annotations

import torch

from qwen3vl_tp_runtime.models.qwen3vl.vision.encoder import (
    encode_image_features,
    encode_video_features,
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

    model_core = model.model
    image_mask = None
    video_mask = None
    deepstack_image_embeds = None
    deepstack_video_embeds = None

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = encode_image_features(
            model,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_device=inputs_embeds.device,
            output_dtype=inputs_embeds.dtype,
        )
        image_mask, _ = model_core.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = encode_video_features(
            model,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            output_device=inputs_embeds.device,
            output_dtype=inputs_embeds.dtype,
        )
        _, video_mask = model_core.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            video_features=video_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_by_layer: dict[int, torch.Tensor] = {}
    if image_mask is not None and video_mask is not None:
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for layer_idx, (img_embed, vid_embed) in enumerate(zip(deepstack_image_embeds, deepstack_video_embeds)):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_by_layer[layer_idx] = embed_joint
    elif image_mask is not None:
        visual_pos_masks = image_mask[..., 0]
        for layer_idx, image_embed in enumerate(deepstack_image_embeds):
            deepstack_by_layer[layer_idx] = image_embed
    elif video_mask is not None:
        visual_pos_masks = video_mask[..., 0]
        for layer_idx, video_embed in enumerate(deepstack_video_embeds):
            deepstack_by_layer[layer_idx] = video_embed

    return inputs_embeds, visual_pos_masks, deepstack_by_layer


__all__ = ["materialize_visual_features"]
