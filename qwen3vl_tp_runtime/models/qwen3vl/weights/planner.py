"""Stage-local parameter planning for stage-only and shard-only loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from qwen3vl_tp_runtime.models.qwen3vl.weights.index import ModelWeightIndex

if TYPE_CHECKING:
    from qwen3vl_tp_runtime.models.qwen3vl.weights.text import TextModelConfigSpec


_TEXT_LAYER_SUFFIXES = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.weight",
    "self_attn.o_proj.bias",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.gate_proj.bias",
    "mlp.up_proj.weight",
    "mlp.up_proj.bias",
    "mlp.down_proj.weight",
    "mlp.down_proj.bias",
    "post_attention_layernorm.weight",
)

_TEXT_TP_SHARDED_LAYER_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.gate_proj.bias",
    "mlp.up_proj.weight",
    "mlp.up_proj.bias",
    "mlp.down_proj.weight",
)


@dataclass(slots=True)
class TextDecoderStageWeightPlan:
    """Resolved tensor list for one text decoder stage."""

    start_idx: int
    end_idx: int
    is_first_stage: bool
    is_last_stage: bool
    requested_parameter_names: tuple[str, ...]
    resolved_parameter_names: tuple[str, ...]
    shard_files: tuple[str, ...]
    shared_parameter_aliases: dict[str, str]


@dataclass(slots=True, frozen=True)
class TensorSliceSpec:
    """One contiguous slice on a tensor dimension."""

    dim: int
    start: int
    end: int


@dataclass(slots=True)
class TextTensorParallelShardPlan:
    """Per-rank TP slice plan for one text decoder stage range."""

    rank: int
    world_size: int
    local_num_attention_heads: int
    local_num_key_value_heads: int
    local_intermediate_size: int
    tensor_slices: dict[str, tuple[TensorSliceSpec, ...]]


def build_text_decoder_stage_weight_plan(
    index: ModelWeightIndex,
    *,
    start_idx: int,
    end_idx: int,
    is_first_stage: bool,
    is_last_stage: bool,
) -> TextDecoderStageWeightPlan:
    requested_parameter_names = build_text_decoder_stage_parameter_names(
        start_idx=start_idx,
        end_idx=end_idx,
        include_embeddings=is_first_stage,
        include_final_norm=is_last_stage,
        include_lm_head=is_last_stage,
    )

    resolved_parameter_names: list[str] = []
    shared_parameter_aliases: dict[str, str] = {}
    for parameter_name in requested_parameter_names:
        if index.has_tensor(parameter_name):
            resolved_parameter_names.append(parameter_name)
            continue

        # Qwen3-VL 的 lm_head 权重大概率和 embed_tokens tie，索引里可能不会单独存一份。
        if parameter_name == "lm_head.weight" and index.has_tensor("model.language_model.embed_tokens.weight"):
            shared_parameter_aliases[parameter_name] = "model.language_model.embed_tokens.weight"
            if "model.language_model.embed_tokens.weight" not in resolved_parameter_names:
                resolved_parameter_names.append("model.language_model.embed_tokens.weight")

    return TextDecoderStageWeightPlan(
        start_idx=start_idx,
        end_idx=end_idx,
        is_first_stage=is_first_stage,
        is_last_stage=is_last_stage,
        requested_parameter_names=tuple(requested_parameter_names),
        resolved_parameter_names=tuple(resolved_parameter_names),
        shard_files=index.files_for_tensors(resolved_parameter_names),
        shared_parameter_aliases=shared_parameter_aliases,
    )


def build_text_decoder_stage_parameter_names(
    *,
    start_idx: int,
    end_idx: int,
    include_embeddings: bool,
    include_final_norm: bool,
    include_lm_head: bool,
) -> list[str]:
    if start_idx > end_idx:
        raise ValueError("start_idx 不能大于 end_idx。")

    parameter_names: list[str] = []
    if include_embeddings:
        parameter_names.append("model.language_model.embed_tokens.weight")

    for layer_idx in range(start_idx, end_idx + 1):
        prefix = f"model.language_model.layers.{layer_idx}"
        parameter_names.extend(f"{prefix}.{suffix}" for suffix in _TEXT_LAYER_SUFFIXES)

    if include_final_norm:
        parameter_names.append("model.language_model.norm.weight")
    if include_lm_head:
        parameter_names.extend(["lm_head.weight", "lm_head.bias"])
    return parameter_names


def build_text_decoder_stage_tp_sharded_parameter_names(
    *,
    start_idx: int,
    end_idx: int,
) -> list[str]:
    """Return the tensor names that must be slice-loaded for shard-only TP."""

    if start_idx > end_idx:
        raise ValueError("start_idx 不能大于 end_idx。")

    parameter_names: list[str] = []
    for layer_idx in range(start_idx, end_idx + 1):
        prefix = f"model.language_model.layers.{layer_idx}"
        parameter_names.extend(f"{prefix}.{suffix}" for suffix in _TEXT_TP_SHARDED_LAYER_SUFFIXES)
    return parameter_names


def build_text_decoder_stage_tp_shard_plan(
    config_spec: "TextModelConfigSpec",
    *,
    start_idx: int,
    end_idx: int,
    rank: int,
    world_size: int,
) -> TextTensorParallelShardPlan:
    if start_idx > end_idx:
        raise ValueError("start_idx 不能大于 end_idx。")
    if world_size <= 0:
        raise ValueError(f"world_size 必须是正整数，当前拿到 {world_size}。")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank={rank} 不在 [0, {world_size}) 范围内。")

    local_num_attention_heads = _shard_extent(
        config_spec.num_attention_heads,
        rank=rank,
        world_size=world_size,
        tensor_name="num_attention_heads",
    )
    local_num_key_value_heads = _shard_extent(
        config_spec.num_key_value_heads,
        rank=rank,
        world_size=world_size,
        tensor_name="num_key_value_heads",
    )
    local_intermediate_size = _shard_extent(
        config_spec.intermediate_size,
        rank=rank,
        world_size=world_size,
        tensor_name="intermediate_size",
    )

    q_hidden = local_num_attention_heads * config_spec.head_dim
    kv_hidden = local_num_key_value_heads * config_spec.head_dim
    q_slice = _build_shard_slice(
        config_spec.num_attention_heads * config_spec.head_dim,
        rank=rank,
        world_size=world_size,
        tensor_name="attention_hidden_size",
    )
    kv_slice = _build_shard_slice(
        config_spec.num_key_value_heads * config_spec.head_dim,
        rank=rank,
        world_size=world_size,
        tensor_name="kv_hidden_size",
    )
    mlp_slice = _build_shard_slice(
        config_spec.intermediate_size,
        rank=rank,
        world_size=world_size,
        tensor_name="intermediate_size",
    )

    tensor_slices: dict[str, tuple[TensorSliceSpec, ...]] = {}
    for layer_idx in range(start_idx, end_idx + 1):
        prefix = f"model.language_model.layers.{layer_idx}"
        tensor_slices[f"{prefix}.self_attn.q_proj.weight"] = (TensorSliceSpec(dim=0, start=q_slice.start, end=q_slice.end),)
        tensor_slices[f"{prefix}.self_attn.q_proj.bias"] = (TensorSliceSpec(dim=0, start=q_slice.start, end=q_slice.end),)
        tensor_slices[f"{prefix}.self_attn.k_proj.weight"] = (TensorSliceSpec(dim=0, start=kv_slice.start, end=kv_slice.end),)
        tensor_slices[f"{prefix}.self_attn.k_proj.bias"] = (TensorSliceSpec(dim=0, start=kv_slice.start, end=kv_slice.end),)
        tensor_slices[f"{prefix}.self_attn.v_proj.weight"] = (TensorSliceSpec(dim=0, start=kv_slice.start, end=kv_slice.end),)
        tensor_slices[f"{prefix}.self_attn.v_proj.bias"] = (TensorSliceSpec(dim=0, start=kv_slice.start, end=kv_slice.end),)
        tensor_slices[f"{prefix}.self_attn.o_proj.weight"] = (TensorSliceSpec(dim=1, start=q_slice.start, end=q_slice.end),)
        tensor_slices[f"{prefix}.mlp.gate_proj.weight"] = (TensorSliceSpec(dim=0, start=mlp_slice.start, end=mlp_slice.end),)
        tensor_slices[f"{prefix}.mlp.gate_proj.bias"] = (TensorSliceSpec(dim=0, start=mlp_slice.start, end=mlp_slice.end),)
        tensor_slices[f"{prefix}.mlp.up_proj.weight"] = (TensorSliceSpec(dim=0, start=mlp_slice.start, end=mlp_slice.end),)
        tensor_slices[f"{prefix}.mlp.up_proj.bias"] = (TensorSliceSpec(dim=0, start=mlp_slice.start, end=mlp_slice.end),)
        tensor_slices[f"{prefix}.mlp.down_proj.weight"] = (TensorSliceSpec(dim=1, start=mlp_slice.start, end=mlp_slice.end),)

    if q_slice.end - q_slice.start != q_hidden:
        raise RuntimeError("q hidden shard 规划结果和 local_num_attention_heads 不一致。")
    if kv_slice.end - kv_slice.start != kv_hidden:
        raise RuntimeError("kv hidden shard 规划结果和 local_num_key_value_heads 不一致。")

    return TextTensorParallelShardPlan(
        rank=rank,
        world_size=world_size,
        local_num_attention_heads=local_num_attention_heads,
        local_num_key_value_heads=local_num_key_value_heads,
        local_intermediate_size=local_intermediate_size,
        tensor_slices=tensor_slices,
    )


def _build_shard_slice(
    total_size: int,
    *,
    rank: int,
    world_size: int,
    tensor_name: str,
) -> TensorSliceSpec:
    shard = _shard_extent(total_size, rank=rank, world_size=world_size, tensor_name=tensor_name)
    start = rank * shard
    end = start + shard
    return TensorSliceSpec(dim=0, start=start, end=end)


def _shard_extent(
    total_size: int,
    *,
    rank: int,
    world_size: int,
    tensor_name: str,
) -> int:
    del rank
    if total_size % world_size != 0:
        raise ValueError(
            f"{tensor_name}={total_size} 不能被 TP world_size={world_size} 整除，"
            "当前 shard-only TP 仍要求均匀切分。"
        )
    return total_size // world_size


__all__ = [
    "TensorSliceSpec",
    "TextDecoderStageWeightPlan",
    "TextTensorParallelShardPlan",
    "build_text_decoder_stage_parameter_names",
    "build_text_decoder_stage_tp_sharded_parameter_names",
    "build_text_decoder_stage_tp_shard_plan",
    "build_text_decoder_stage_weight_plan",
]
