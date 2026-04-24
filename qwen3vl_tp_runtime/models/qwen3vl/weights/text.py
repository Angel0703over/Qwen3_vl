"""Text-decoder weight loading primitives for stage-only and shard-only runtime work."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from qwen3vl_tp_runtime.models.qwen3vl.weights.index import ModelWeightIndex, load_model_weight_index
from qwen3vl_tp_runtime.models.qwen3vl.weights.loader import load_tensors_from_index
from qwen3vl_tp_runtime.models.qwen3vl.weights.planner import (
    TextDecoderStageWeightPlan,
    TextTensorParallelShardPlan,
    build_text_decoder_stage_tp_shard_plan,
    build_text_decoder_stage_weight_plan,
)


@dataclass(slots=True)
class TextModelConfigSpec:
    """Minimal text-decoder config needed to build runtime bundles without live modules."""

    model_path: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    hidden_act: str
    tie_word_embeddings: bool
    vocab_size: int
    head_dim: int
    max_position_embeddings: int
    rope_parameters: dict[str, Any]
    attention_bias: bool
    attention_dropout: float
    use_cache: bool
    pad_token_id: int | None
    scaling: float
    attn_implementation: str


@dataclass(slots=True)
class TextStageWeightBundle:
    """Stage-local tensors loaded directly from checkpoint shards."""

    plan: TextDecoderStageWeightPlan
    layer_bundles: list[dict[str, Any]]
    embed_tokens_weight: torch.Tensor | None
    final_norm_weight: torch.Tensor | None
    final_norm_eps: float | None
    lm_head_weight: torch.Tensor | None
    lm_head_bias: torch.Tensor | None
    tp_weight_sharded: bool = False
    tp_shard_rank: int | None = None
    tp_shard_world_size: int | None = None
    tp_local_num_attention_heads: int | None = None
    tp_local_num_key_value_heads: int | None = None
    tp_local_intermediate_size: int | None = None


def load_text_model_config_spec(
    model_path: str,
    *,
    attn_implementation: str = "eager",
) -> TextModelConfigSpec:
    model_dir = Path(model_path)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"没有找到 config.json: {config_path}")

    payload = json.loads(config_path.read_text())
    text_config = payload.get("text_config", payload)
    hidden_size = int(text_config["hidden_size"])
    intermediate_size = int(text_config["intermediate_size"])
    num_hidden_layers = int(text_config["num_hidden_layers"])
    num_attention_heads = int(text_config["num_attention_heads"])
    num_key_value_heads = int(text_config["num_key_value_heads"])
    rms_norm_eps = float(text_config["rms_norm_eps"])
    hidden_act = str(text_config["hidden_act"])
    tie_word_embeddings = bool(payload.get("tie_word_embeddings", text_config.get("tie_word_embeddings", False)))
    vocab_size = int(text_config["vocab_size"])
    head_dim = int(text_config.get("head_dim") or (hidden_size // num_attention_heads))
    max_position_embeddings = int(text_config.get("max_position_embeddings", 128000))
    rope_parameters = dict(text_config.get("rope_parameters") or text_config.get("rope_scaling") or {})
    if "rope_theta" not in rope_parameters:
        rope_parameters["rope_theta"] = float(text_config.get("rope_theta", 10000.0))
    attention_bias = bool(text_config.get("attention_bias", False))
    attention_dropout = float(text_config.get("attention_dropout", 0.0))
    use_cache = bool(text_config.get("use_cache", True))
    pad_token_id_value = payload.get("pad_token_id", text_config.get("pad_token_id"))
    pad_token_id = None if pad_token_id_value is None else int(pad_token_id_value)
    scaling = 1.0 / (head_dim ** 0.5)
    return TextModelConfigSpec(
        model_path=str(model_dir),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        rms_norm_eps=rms_norm_eps,
        hidden_act=hidden_act,
        tie_word_embeddings=tie_word_embeddings,
        vocab_size=vocab_size,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        rope_parameters=rope_parameters,
        attention_bias=attention_bias,
        attention_dropout=attention_dropout,
        use_cache=use_cache,
        pad_token_id=pad_token_id,
        scaling=scaling,
        attn_implementation=attn_implementation,
    )


def load_text_decoder_stage_weight_bundle(
    *,
    model_path: str,
    start_idx: int,
    end_idx: int,
    is_first_stage: bool,
    is_last_stage: bool,
    device: torch.device,
    compute_dtype: torch.dtype,
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
    weight_index: ModelWeightIndex | None = None,
    config_spec: TextModelConfigSpec | None = None,
    tp_shard_rank: int | None = None,
    tp_shard_world_size: int | None = None,
) -> TextStageWeightBundle:
    index = weight_index or load_model_weight_index(model_path)
    spec = config_spec or load_text_model_config_spec(model_path)
    shard_plan = _maybe_build_text_tp_shard_plan(
        spec,
        start_idx=start_idx,
        end_idx=end_idx,
        tp_shard_rank=tp_shard_rank,
        tp_shard_world_size=tp_shard_world_size,
    )
    plan = build_text_decoder_stage_weight_plan(
        index,
        start_idx=start_idx,
        end_idx=end_idx,
        is_first_stage=is_first_stage,
        is_last_stage=is_last_stage,
    )
    loaded_tensors = load_tensors_from_index(
        index,
        plan.resolved_parameter_names,
        device=device,
        compute_dtype=compute_dtype,
        strict=True,
        tensor_slices=None if shard_plan is None else shard_plan.tensor_slices,
    )

    layer_bundles = []
    current_cache = cache_by_layer or {}
    for layer_idx in range(start_idx, end_idx + 1):
        layer_bundles.append(
            _build_text_layer_bundle(
                loaded_tensors,
                spec=spec,
                layer_idx=layer_idx,
                device=device,
                compute_dtype=compute_dtype,
                past_key_value=current_cache.get(layer_idx),
                tp_shard_plan=shard_plan,
            )
        )

    embed_tokens_weight = _resolve_tensor_alias(
        loaded_tensors,
        plan,
        "model.language_model.embed_tokens.weight",
    )
    final_norm_weight = _resolve_tensor_alias(
        loaded_tensors,
        plan,
        "model.language_model.norm.weight",
    )
    lm_head_weight = _resolve_tensor_alias(
        loaded_tensors,
        plan,
        "lm_head.weight",
    )
    lm_head_bias = _resolve_tensor_alias(
        loaded_tensors,
        plan,
        "lm_head.bias",
    )
    final_norm_eps = spec.rms_norm_eps if is_last_stage else None
    if is_first_stage and embed_tokens_weight is None:
        raise RuntimeError("text first stage 缺少 embed_tokens_weight，无法做 stage0 embedding。")
    if is_last_stage and final_norm_weight is None:
        raise RuntimeError("text last stage 缺少 final_norm_weight。")
    if is_last_stage and lm_head_weight is None:
        raise RuntimeError("text last stage 缺少 lm_head_weight。")
    return TextStageWeightBundle(
        plan=plan,
        layer_bundles=layer_bundles,
        embed_tokens_weight=embed_tokens_weight,
        final_norm_weight=final_norm_weight,
        final_norm_eps=final_norm_eps,
        lm_head_weight=lm_head_weight,
        lm_head_bias=lm_head_bias,
        tp_weight_sharded=shard_plan is not None,
        tp_shard_rank=None if shard_plan is None else shard_plan.rank,
        tp_shard_world_size=None if shard_plan is None else shard_plan.world_size,
        tp_local_num_attention_heads=None if shard_plan is None else shard_plan.local_num_attention_heads,
        tp_local_num_key_value_heads=None if shard_plan is None else shard_plan.local_num_key_value_heads,
        tp_local_intermediate_size=None if shard_plan is None else shard_plan.local_intermediate_size,
    )


def _build_text_layer_bundle(
    loaded_tensors: dict[str, torch.Tensor],
    *,
    spec: TextModelConfigSpec,
    layer_idx: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    past_key_value: tuple[torch.Tensor | None, torch.Tensor | None] | None,
    tp_shard_plan: TextTensorParallelShardPlan | None,
) -> dict[str, Any]:
    prefix = f"model.language_model.layers.{layer_idx}"
    layer_bundle = {
        "layer_idx": layer_idx,
        "hidden_act": spec.hidden_act,
        "q_weight": _require_tensor(loaded_tensors, f"{prefix}.self_attn.q_proj.weight"),
        "q_bias": loaded_tensors.get(f"{prefix}.self_attn.q_proj.bias"),
        "k_weight": _require_tensor(loaded_tensors, f"{prefix}.self_attn.k_proj.weight"),
        "k_bias": loaded_tensors.get(f"{prefix}.self_attn.k_proj.bias"),
        "v_weight": _require_tensor(loaded_tensors, f"{prefix}.self_attn.v_proj.weight"),
        "v_bias": loaded_tensors.get(f"{prefix}.self_attn.v_proj.bias"),
        "o_weight": _require_tensor(loaded_tensors, f"{prefix}.self_attn.o_proj.weight"),
        "o_bias": loaded_tensors.get(f"{prefix}.self_attn.o_proj.bias"),
        "q_norm_weight": _require_tensor(loaded_tensors, f"{prefix}.self_attn.q_norm.weight"),
        "k_norm_weight": _require_tensor(loaded_tensors, f"{prefix}.self_attn.k_norm.weight"),
        "gate_weight": _require_tensor(loaded_tensors, f"{prefix}.mlp.gate_proj.weight"),
        "gate_bias": loaded_tensors.get(f"{prefix}.mlp.gate_proj.bias"),
        "up_weight": _require_tensor(loaded_tensors, f"{prefix}.mlp.up_proj.weight"),
        "up_bias": loaded_tensors.get(f"{prefix}.mlp.up_proj.bias"),
        "down_weight": _require_tensor(loaded_tensors, f"{prefix}.mlp.down_proj.weight"),
        "down_bias": loaded_tensors.get(f"{prefix}.mlp.down_proj.bias"),
        "input_ln_weight": _require_tensor(loaded_tensors, f"{prefix}.input_layernorm.weight"),
        "input_ln_eps": spec.rms_norm_eps,
        "post_attn_ln_weight": _require_tensor(loaded_tensors, f"{prefix}.post_attention_layernorm.weight"),
        "post_attn_ln_eps": spec.rms_norm_eps,
        "rms_norm_eps": spec.rms_norm_eps,
        "num_attention_heads": spec.num_attention_heads,
        "num_key_value_heads": spec.num_key_value_heads,
        "tp_weight_sharded": tp_shard_plan is not None,
        "tp_shard_rank": None if tp_shard_plan is None else tp_shard_plan.rank,
        "tp_shard_world_size": None if tp_shard_plan is None else tp_shard_plan.world_size,
        "tp_local_num_attention_heads": (
            spec.num_attention_heads if tp_shard_plan is None else tp_shard_plan.local_num_attention_heads
        ),
        "tp_local_num_key_value_heads": (
            spec.num_key_value_heads if tp_shard_plan is None else tp_shard_plan.local_num_key_value_heads
        ),
        "tp_local_intermediate_size": (
            spec.intermediate_size if tp_shard_plan is None else tp_shard_plan.local_intermediate_size
        ),
        "head_dim": spec.head_dim,
        "scaling": spec.scaling,
        "attn_implementation": spec.attn_implementation,
    }
    if past_key_value is not None:
        past_key, past_value = past_key_value
        layer_bundle["past_key"] = _to_runtime_tensor(past_key, device=device, compute_dtype=compute_dtype)
        layer_bundle["past_value"] = _to_runtime_tensor(past_value, device=device, compute_dtype=compute_dtype)
    return layer_bundle


def _maybe_build_text_tp_shard_plan(
    config_spec: TextModelConfigSpec,
    *,
    start_idx: int,
    end_idx: int,
    tp_shard_rank: int | None,
    tp_shard_world_size: int | None,
) -> TextTensorParallelShardPlan | None:
    if tp_shard_rank is None and tp_shard_world_size is None:
        return None
    if tp_shard_rank is None or tp_shard_world_size is None:
        raise ValueError("tp_shard_rank 和 tp_shard_world_size 需要一起传。")
    if tp_shard_world_size <= 1:
        return None
    return build_text_decoder_stage_tp_shard_plan(
        config_spec,
        start_idx=start_idx,
        end_idx=end_idx,
        rank=tp_shard_rank,
        world_size=tp_shard_world_size,
    )


def _resolve_tensor_alias(
    loaded_tensors: dict[str, torch.Tensor],
    plan: TextDecoderStageWeightPlan,
    tensor_name: str,
) -> torch.Tensor | None:
    if tensor_name in loaded_tensors:
        return loaded_tensors[tensor_name]
    alias_name = plan.shared_parameter_aliases.get(tensor_name)
    if alias_name is None:
        return None
    return loaded_tensors.get(alias_name)


def _require_tensor(loaded_tensors: dict[str, torch.Tensor], tensor_name: str) -> torch.Tensor:
    try:
        return loaded_tensors[tensor_name]
    except KeyError as exc:
        raise KeyError(f"缺少构建 text decoder layer 所需的 tensor={tensor_name!r}") from exc


def _to_runtime_tensor(
    tensor: torch.Tensor | None,
    *,
    device: torch.device,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    out = tensor.detach().to(device=device)
    if compute_dtype is not None and out.is_floating_point():
        out = out.to(dtype=compute_dtype)
    return out


def build_text_hf_config(config_spec: TextModelConfigSpec) -> Qwen3VLTextConfig:
    return Qwen3VLTextConfig(
        vocab_size=config_spec.vocab_size,
        hidden_size=config_spec.hidden_size,
        intermediate_size=config_spec.intermediate_size,
        num_hidden_layers=config_spec.num_hidden_layers,
        num_attention_heads=config_spec.num_attention_heads,
        num_key_value_heads=config_spec.num_key_value_heads,
        head_dim=config_spec.head_dim,
        hidden_act=config_spec.hidden_act,
        max_position_embeddings=config_spec.max_position_embeddings,
        rms_norm_eps=config_spec.rms_norm_eps,
        rope_parameters=dict(config_spec.rope_parameters),
        attention_bias=config_spec.attention_bias,
        attention_dropout=config_spec.attention_dropout,
        use_cache=config_spec.use_cache,
        pad_token_id=config_spec.pad_token_id,
    )


def build_text_rotary_embedding(
    config_spec: TextModelConfigSpec,
    *,
    device: torch.device | None = None,
) -> Qwen3VLTextRotaryEmbedding:
    return Qwen3VLTextRotaryEmbedding(
        config=build_text_hf_config(config_spec),
        device=device,
    )


def build_text_runtime_aux_tensors(
    *,
    attention_mask_2d: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    past_length: int,
    config_spec: TextModelConfigSpec,
    device: torch.device,
    compute_dtype: torch.dtype,
    rotary_emb: Qwen3VLTextRotaryEmbedding | None = None,
) -> dict[str, torch.Tensor | None]:
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须大于 0，当前拿到 {batch_size}")
    if seq_len <= 0:
        raise ValueError(f"seq_len 必须大于 0，当前拿到 {seq_len}")
    if past_length < 0:
        raise ValueError(f"past_length 不能小于 0，当前拿到 {past_length}")

    attention_mask_2d = None if attention_mask_2d is None else attention_mask_2d.to(device=device)
    inputs_embeds = torch.zeros(
        (batch_size, seq_len, config_spec.hidden_size),
        device=device,
        dtype=compute_dtype,
    )
    full_position_ids, _text_position_ids = _build_text_position_ids(
        attention_mask_2d=attention_mask_2d,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )
    attention_mask = build_text_causal_mask(
        inputs_embeds,
        attention_mask_2d=attention_mask_2d,
        past_length=past_length,
    )
    cos, sin = _build_text_rotary_tensors(
        inputs_embeds,
        full_position_ids,
        config_spec=config_spec,
        device=device,
        rotary_emb=rotary_emb,
    )
    return {
        "attention_mask": attention_mask.detach(),
        "position_ids": full_position_ids.detach(),
        "cos": cos.detach(),
        "sin": sin.detach(),
    }


def prepare_text_prefill_runtime_inputs_from_weights(
    *,
    input_ids: torch.Tensor,
    attention_mask_2d: torch.Tensor | None,
    embed_tokens_weight: torch.Tensor,
    config_spec: TextModelConfigSpec,
    device: torch.device,
    compute_dtype: torch.dtype,
    rotary_emb: Qwen3VLTextRotaryEmbedding | None = None,
) -> "MultimodalRuntimeInputs":
    from qwen3vl_tp_runtime.models.qwen3vl.live.common import MultimodalRuntimeInputs

    input_ids = input_ids.to(device=device)
    attention_mask_2d = None if attention_mask_2d is None else attention_mask_2d.to(device=device)
    inputs_embeds = _embed_input_ids(
        input_ids,
        embed_tokens_weight,
        device=device,
        compute_dtype=compute_dtype,
    )
    full_position_ids, _text_position_ids = _build_text_position_ids(
        attention_mask_2d=attention_mask_2d,
        batch_size=inputs_embeds.shape[0],
        seq_len=inputs_embeds.shape[1],
        device=device,
    )
    attention_mask = build_text_causal_mask(
        inputs_embeds,
        attention_mask_2d=attention_mask_2d,
        past_length=0,
    )
    cos, sin = _build_text_rotary_tensors(
        inputs_embeds,
        full_position_ids,
        config_spec=config_spec,
        device=device,
        rotary_emb=rotary_emb,
    )
    return MultimodalRuntimeInputs(
        input_ids=input_ids.detach(),
        attention_mask_2d=None if attention_mask_2d is None else attention_mask_2d.detach(),
        position_ids=full_position_ids.detach(),
        inputs_embeds=inputs_embeds.detach(),
        attention_mask=attention_mask.detach(),
        cos=cos.detach(),
        sin=sin.detach(),
        visual_pos_masks=None,
        deepstack_by_layer={},
    )


def prepare_text_decode_runtime_inputs_from_weights(
    *,
    decode_input_ids: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    past_length: int,
    embed_tokens_weight: torch.Tensor,
    config_spec: TextModelConfigSpec,
    device: torch.device,
    compute_dtype: torch.dtype,
    rotary_emb: Qwen3VLTextRotaryEmbedding | None = None,
) -> "MultimodalRuntimeInputs":
    from qwen3vl_tp_runtime.models.qwen3vl.live.common import MultimodalRuntimeInputs

    if past_length < 0:
        raise ValueError(f"past_length 不能小于 0，当前拿到 {past_length}")

    decode_input_ids = decode_input_ids.to(device=device)
    attention_mask_2d = attention_mask_2d.to(device=device)
    inputs_embeds = _embed_input_ids(
        decode_input_ids,
        embed_tokens_weight,
        device=device,
        compute_dtype=compute_dtype,
    )
    full_position_ids, _text_position_ids = _build_text_position_ids(
        attention_mask_2d=attention_mask_2d,
        batch_size=inputs_embeds.shape[0],
        seq_len=inputs_embeds.shape[1],
        device=device,
    )
    attention_mask = build_text_causal_mask(
        inputs_embeds,
        attention_mask_2d=attention_mask_2d,
        past_length=past_length,
    )
    cos, sin = _build_text_rotary_tensors(
        inputs_embeds,
        full_position_ids,
        config_spec=config_spec,
        device=device,
        rotary_emb=rotary_emb,
    )
    return MultimodalRuntimeInputs(
        input_ids=decode_input_ids.detach(),
        attention_mask_2d=attention_mask_2d.detach(),
        position_ids=full_position_ids.detach(),
        inputs_embeds=inputs_embeds.detach(),
        attention_mask=attention_mask.detach(),
        cos=cos.detach(),
        sin=sin.detach(),
        visual_pos_masks=None,
        deepstack_by_layer={},
    )


def build_text_causal_mask(
    inputs_embeds: torch.Tensor,
    *,
    attention_mask_2d: torch.Tensor | None,
    past_length: int,
) -> torch.Tensor:
    if past_length < 0:
        raise ValueError(f"past_length 不能小于 0，当前拿到 {past_length}")

    batch_size, query_len, _ = inputs_embeds.shape
    total_kv_len = past_length + query_len
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    min_value = torch.finfo(dtype).min

    query_positions = torch.arange(
        past_length,
        total_kv_len,
        device=device,
        dtype=torch.long,
    )
    key_positions = torch.arange(total_kv_len, device=device, dtype=torch.long)
    future_mask = key_positions.view(1, 1, 1, total_kv_len) > query_positions.view(1, 1, query_len, 1)

    causal_mask = torch.zeros((batch_size, 1, query_len, total_kv_len), device=device, dtype=dtype)
    causal_mask = causal_mask.masked_fill(future_mask, min_value)

    if attention_mask_2d is not None:
        if attention_mask_2d.shape[-1] != total_kv_len:
            raise ValueError(
                "attention_mask_2d 的长度和 past/query 长度不匹配，"
                f"attention_mask_2d.shape={tuple(attention_mask_2d.shape)} "
                f"past_length={past_length} query_len={query_len}"
            )
        key_padding_mask = attention_mask_2d.to(device=device)[:, None, None, :] == 0
        causal_mask = causal_mask.masked_fill(key_padding_mask, min_value)

    return causal_mask


def _embed_input_ids(
    input_ids: torch.Tensor,
    embed_tokens_weight: torch.Tensor,
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    embed_weight = _to_runtime_tensor(
        embed_tokens_weight,
        device=device,
        compute_dtype=compute_dtype,
    )
    if embed_weight is None:
        raise RuntimeError("embed_tokens_weight 不能为空。")
    return F.embedding(input_ids, embed_weight)


def _build_text_position_ids(
    *,
    attention_mask_2d: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_mask_2d is None:
        text_position_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, -1).expand(batch_size, -1)
    else:
        text_position_ids = attention_mask_2d.long().cumsum(-1) - 1
        text_position_ids = text_position_ids.masked_fill(attention_mask_2d == 0, 0).to(device=device)
        text_position_ids = text_position_ids[:, -seq_len:]

    full_position_ids = torch.cat(
        [
            text_position_ids.unsqueeze(0),
            text_position_ids.view(1, batch_size, -1).repeat(3, 1, 1),
        ],
        dim=0,
    )
    return full_position_ids, text_position_ids


def _build_text_rotary_tensors(
    inputs_embeds: torch.Tensor,
    full_position_ids: torch.Tensor,
    *,
    config_spec: TextModelConfigSpec,
    device: torch.device,
    rotary_emb: Qwen3VLTextRotaryEmbedding | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary = rotary_emb or build_text_rotary_embedding(config_spec, device=device)
    rotary = rotary.to(device=device)
    return rotary(inputs_embeds, full_position_ids[1:])


__all__ = [
    "build_text_runtime_aux_tensors",
    "TextModelConfigSpec",
    "TextStageWeightBundle",
    "build_text_causal_mask",
    "build_text_hf_config",
    "build_text_rotary_embedding",
    "load_text_decoder_stage_weight_bundle",
    "load_text_model_config_spec",
    "prepare_text_decode_runtime_inputs_from_weights",
    "prepare_text_prefill_runtime_inputs_from_weights",
]
