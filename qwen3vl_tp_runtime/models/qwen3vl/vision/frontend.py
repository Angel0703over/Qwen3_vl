"""Multimodal frontend model shell and selective weight helpers."""

from __future__ import annotations

import itertools
from typing import Any

import torch
import torch.nn as nn
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithDeepstackFeatures,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
)

from qwen3vl_tp_runtime.models.qwen3vl.functional import build_causal_mask
from qwen3vl_tp_runtime.models.qwen3vl.weights import (
    load_mm_frontend_config,
    load_mm_frontend_weight_bundle,
)
from qwen3vl_tp_runtime.models.qwen3vl.vision.encoder import (
    encode_image_features,
    encode_video_features,
)


def model_device(model: torch.nn.Module) -> torch.device:
    """Resolve the device used by one lightweight frontend shell."""

    if isinstance(model, nn.Module):
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass

    model_dict = getattr(model, "__dict__", None)
    if isinstance(model_dict, dict):
        resolved = model_dict.get("device")
        if isinstance(resolved, torch.device):
            return resolved

    resolved = getattr(model, "device", None)
    if isinstance(resolved, torch.device):
        return resolved
    raise RuntimeError(f"无法解析 frontend model device: {type(model)!r}")


def move_frontend_inputs(
    inputs: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Move one multimodal frontend batch to the target device."""

    if hasattr(inputs, "to"):
        moved = inputs.to(device)
        if isinstance(moved, dict):
            return moved

    moved_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved_inputs[key] = value.to(device)
        else:
            moved_inputs[key] = value
    return moved_inputs


def embed_mm_frontend_inputs(
    model,
    *,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Embed multimodal prompt tokens with the lightweight frontend shell."""

    language_model = model.model.language_model
    return language_model.get_input_embeddings()(input_ids.to(model_device(model)))


def _merge_deepstack_features(
    image_mask: torch.Tensor | None,
    video_mask: torch.Tensor | None,
    deepstack_image_embeds: list[torch.Tensor] | None,
    deepstack_video_embeds: list[torch.Tensor] | None,
) -> tuple[torch.Tensor | None, dict[int, torch.Tensor]]:
    visual_pos_masks = None
    deepstack_by_layer: dict[int, torch.Tensor] = {}
    if image_mask is not None and video_mask is not None:
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        if deepstack_image_embeds is None or deepstack_video_embeds is None:
            raise RuntimeError("multimodal deepstack 合并缺少 image/video embeds。")
        for layer_idx, (img_embed, vid_embed) in enumerate(zip(deepstack_image_embeds, deepstack_video_embeds)):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_by_layer[layer_idx] = embed_joint
    elif image_mask is not None:
        visual_pos_masks = image_mask[..., 0]
        if deepstack_image_embeds is None:
            raise RuntimeError("multimodal image deepstack 缺少 image embeds。")
        for layer_idx, image_embed in enumerate(deepstack_image_embeds):
            deepstack_by_layer[layer_idx] = image_embed
    elif video_mask is not None:
        visual_pos_masks = video_mask[..., 0]
        if deepstack_video_embeds is None:
            raise RuntimeError("multimodal video deepstack 缺少 video embeds。")
        for layer_idx, video_embed in enumerate(deepstack_video_embeds):
            deepstack_by_layer[layer_idx] = video_embed
    return visual_pos_masks, deepstack_by_layer


def merge_mm_frontend_visuals(
    model,
    *,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor,
    pixel_values: torch.Tensor | None,
    pixel_values_videos: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[int, torch.Tensor]]:
    """Run the frontend visual tower once and merge visual features into decoder embeds."""

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

    visual_pos_masks, deepstack_by_layer = _merge_deepstack_features(
        image_mask,
        video_mask,
        deepstack_image_embeds,
        deepstack_video_embeds,
    )
    return inputs_embeds, visual_pos_masks, deepstack_by_layer


def compute_mm_frontend_position_ids(
    model,
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask_2d: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
) -> torch.Tensor:
    """Compute the full multimodal 3D position ids for one frontend activation."""

    return model.model.compute_3d_position_ids(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask_2d,
        past_key_values=None,
        mm_token_type_ids=mm_token_type_ids,
    )


def build_mm_frontend_attention_mask(
    model,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask_2d: torch.Tensor | None,
    text_position_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Build the decoder attention mask for one frontend activation."""

    language_model = model.model.language_model
    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask_2d,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    if attention_mask is None:
        attention_mask = build_causal_mask(inputs_embeds, attention_mask_2d)
    return attention_mask


def build_mm_frontend_rotary(
    model,
    *,
    inputs_embeds: torch.Tensor,
    vision_position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build rotary cos/sin tensors for one frontend activation."""

    return model.model.language_model.rotary_emb(inputs_embeds, vision_position_ids)


def _build_mm_vision_pos_ids(
    start_position: int,
    grid_thw: list[int] | torch.Tensor,
    *,
    spatial_merge_size: int,
    temp_merge_size: int = 1,
    time_interval: int = 1,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    llm_grid_t, llm_grid_h, llm_grid_w = (
        grid_thw[0].item() // temp_merge_size,
        grid_thw[1].item() // spatial_merge_size,
        grid_thw[2].item() // spatial_merge_size,
    )

    image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
    position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
        llm_grid_h * llm_grid_t
    )
    position_height = torch.arange(
        start_position,
        start_position + llm_grid_h,
        device=device,
    ).repeat_interleave(llm_grid_w * llm_grid_t)
    position_temporal = torch.full(
        (image_seq_length,),
        start_position,
        device=device,
        dtype=torch.long,
    )
    position_temporal = position_temporal * time_interval
    return torch.stack([position_temporal, position_height, position_width], dim=0)


def compute_mm_frontend_rope_index(
    *,
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    spatial_merge_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute multimodal prefill position ids and rope deltas from thin metadata."""

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }

    for batch_idx, current_input_ids in enumerate(input_ids):
        input_token_type = mm_token_type_ids[batch_idx]
        if attention_mask is not None:
            current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
            input_token_type = input_token_type[attention_mask[batch_idx].bool()]

        input_type_group = []
        for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
            else:
                grid_iter = grid_iters.get(modality_type)
                if grid_iter is None:
                    raise ValueError(
                        "multimodal frontend rope planning 缺少 grid_thw，"
                        f"modality_type={modality_type}"
                    )
                grid_thw = next(grid_iter)
                vision_position_ids = _build_mm_vision_pos_ids(
                    current_pos,
                    grid_thw,
                    spatial_merge_size=spatial_merge_size,
                    device=input_ids.device,
                )
                llm_pos_ids_list.append(vision_position_ids)
                current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if attention_mask is not None:
            position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
        else:
            position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
    rope_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, rope_deltas


class _MmFrontendText(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value


class _MmFrontendCore(nn.Module):
    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__()
        self.config = config
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = _MmFrontendText(config.text_config)
        self.rope_deltas = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.language_model.set_input_embeddings(value)

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: list[int] | torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        return _build_mm_vision_pos_ids(
            start_position,
            grid_thw,
            spatial_merge_size=spatial_merge_size,
            temp_merge_size=temp_merge_size,
            time_interval=time_interval,
            device=device,
        )

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_mm_frontend_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
        )

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        pixel_values = pixel_values.type(self.visual.dtype)
        # The frontend shell always consumes structured vision outputs here, so
        # we keep return_dict pinned to True and avoid passing it twice when
        # callers forward upstream kwargs through get_video_features().
        kwargs.pop("return_dict", None)
        vision_output: BaseModelOutputWithDeepstackFeatures = self.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            return_dict=True,
            **kwargs,
        )
        image_embeds = vision_output.pooler_output
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        vision_output.pooler_output = image_embeds
        return vision_output

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = int(special_image_mask.sum().item())
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise RuntimeError(
                "Image features and image tokens do not match, "
                f"tokens: {n_image_tokens}, features: {image_features.shape[0]}"
            )

        n_video_tokens = int(special_video_mask.sum().item())
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise RuntimeError(
                "Video features and video tokens do not match, "
                f"tokens: {n_video_tokens}, features: {video_features.shape[0]}"
            )
        return special_image_mask, special_video_mask

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
    ) -> torch.Tensor | None:
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        has_multimodal = image_grid_thw is not None or video_grid_thw is not None
        if has_multimodal and mm_token_type_ids is None and input_ids is not None:
            raise ValueError(
                "Multimodal data was passed but `mm_token_type_ids` is missing."
            )
        can_compute_mrope = input_ids is not None and mm_token_type_ids is not None and has_multimodal

        if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
        elif self.rope_deltas is not None and (past_key_values_length > 0 or input_ids is None):
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
            else:
                position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            position_ids = position_ids + delta.to(device=inputs_embeds.device)
        else:
            position_ids = None
        return position_ids


class MmFrontendModel(nn.Module):
    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__()
        self.config = config
        self.model = _MmFrontendCore(config)

    @property
    def device(self) -> torch.device:
        return model_device(self)


def _default_frontend_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_mm_frontend_model(
    model_path: str,
    *,
    device: torch.device | str | None = None,
    weight_index=None,
) -> torch.nn.Module:
    """Load a lightweight frontend shell with only visual tower and token embeddings."""

    target_device = _default_frontend_device() if device is None else torch.device(device)
    config = load_mm_frontend_config(model_path)
    frontend = MmFrontendModel(config)
    weight_bundle = load_mm_frontend_weight_bundle(
        model_path=model_path,
        visual_parameter_names=frontend.model.visual.state_dict().keys(),
        device=target_device,
        weight_index=weight_index,
    )
    frontend = frontend.to(device=target_device, dtype=weight_bundle.compute_dtype)
    frontend.model.visual.load_state_dict(weight_bundle.visual_state, strict=True)
    frontend.model.language_model.embed_tokens.load_state_dict(
        {"weight": weight_bundle.embed_tokens_weight},
        strict=True,
    )
    frontend.eval()
    return frontend

__all__ = [
    "build_mm_frontend_attention_mask",
    "build_mm_frontend_rotary",
    "compute_mm_frontend_rope_index",
    "compute_mm_frontend_position_ids",
    "embed_mm_frontend_inputs",
    "load_mm_frontend_model",
    "merge_mm_frontend_visuals",
    "model_device",
    "move_frontend_inputs",
]
