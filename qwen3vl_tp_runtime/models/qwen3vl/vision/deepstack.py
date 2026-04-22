"""DeepStack-style visual feature injection helpers."""

from __future__ import annotations

import torch


def apply_deepstack(
    hidden_states: torch.Tensor,
    visual_pos_masks: torch.Tensor | None,
    visual_embeds: torch.Tensor | None,
) -> torch.Tensor:
    if visual_pos_masks is None or visual_embeds is None:
        return hidden_states

    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

    expected_mask_shape = hidden_states.shape[:2]
    if tuple(visual_pos_masks.shape) != tuple(expected_mask_shape):
        return hidden_states

    if visual_pos_masks.dtype != torch.bool:
        visual_pos_masks = visual_pos_masks.to(torch.bool)

    if visual_embeds.shape[0] != int(visual_pos_masks.sum().item()):
        return hidden_states

    hidden_states = hidden_states.clone()
    local_this = hidden_states[visual_pos_masks, :] + visual_embeds
    hidden_states[visual_pos_masks, :] = local_this
    return hidden_states


def get_deepstack_embeds(stage_bundle: dict, layer_idx: int) -> torch.Tensor | None:
    deepstack_by_layer = stage_bundle.get("deepstack_by_layer")
    if deepstack_by_layer is None:
        return None
    return deepstack_by_layer.get(layer_idx)


__all__ = [
    "apply_deepstack",
    "get_deepstack_embeds",
]
