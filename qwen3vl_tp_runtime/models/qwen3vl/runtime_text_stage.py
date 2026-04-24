"""Text stage scaffold helpers and local materialization."""

from __future__ import annotations

import sys
from typing import Any

import torch

from qwen3vl_tp_runtime.hexgen_core.schema import StageSpec
from qwen3vl_tp_runtime.models.qwen3vl.live.common import _runtime_tensor
from qwen3vl_tp_runtime.models.qwen3vl.runtime_text import _restore_text_prompt_bundle
from qwen3vl_tp_runtime.models.qwen3vl.weights import (
    TextModelConfigSpec,
    TextStageWeightBundle,
    build_text_rotary_embedding,
    build_text_runtime_aux_tensors,
    load_model_weight_index,
    load_text_decoder_stage_weight_bundle,
    load_text_model_config_spec,
)


def _builder_dep(name: str, fallback):
    builder_mod = sys.modules.get("qwen3vl_tp_runtime.models.qwen3vl.runtime_builder")
    if builder_mod is not None and hasattr(builder_mod, name):
        return getattr(builder_mod, name)
    return fallback


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _strip_text_phase(phase_payload: dict[str, Any]) -> dict[str, Any]:
    compact_payload = dict(phase_payload)
    compact_payload.pop("attention_mask", None)
    compact_payload.pop("cos", None)
    compact_payload.pop("sin", None)
    compact_payload.pop("position_ids", None)
    return compact_payload


def compact_text_scaffold(bundle: dict[str, Any]) -> dict[str, Any]:
    scaffold = dict(bundle)
    scaffold["runtime_inputs_local_rebuild"] = True
    scaffold["runtime_prefill_cache_policy"] = "recompute"
    scaffold.pop("cache_by_layer", None)
    scaffold["prefill"] = _strip_text_phase(scaffold["prefill"])
    scaffold["decode_steps"] = [
        _strip_text_phase(step_payload)
        for step_payload in scaffold["decode_steps"]
    ]
    return scaffold


def compact_rt_text_scaffold(bundle: dict[str, Any]) -> dict[str, Any]:
    scaffold = dict(bundle)
    scaffold["runtime_only_prompt_local_rebuild"] = True
    scaffold.pop("input_ids", None)
    scaffold.pop("prefill_attention_mask_2d", None)
    scaffold.pop("prefill_seq_len", None)
    scaffold.pop("batch_size", None)
    scaffold.pop("token_id_dtype", None)
    return scaffold


def build_rt_text_bundle(
    *,
    spec: StageSpec,
    bundle_device: torch.device,
    compute_dtype: torch.dtype,
    prefill_attention_mask_2d: torch.Tensor,
    prefill_seq_len: int,
    batch_size: int,
    token_id_dtype: torch.dtype,
    hidden_size: int,
    layers: list[dict[str, Any]],
    text_stage_weights: TextStageWeightBundle | None,
) -> dict[str, Any]:
    return {
        "module_name": "text_generate_stage",
        "stage_type": "text_generate_runtime_only",
        "runtime_only_generate": True,
        "start_idx": spec.start_idx,
        "end_idx": spec.end_idx,
        "save_dtype": _dtype_name(compute_dtype),
        "prefill_seq_len": int(prefill_seq_len),
        "max_new_tokens": 0,
        "prefill_attention_mask_2d": _runtime_tensor(prefill_attention_mask_2d, device=bundle_device),
        "batch_size": int(batch_size),
        "token_id_dtype": _dtype_name(token_id_dtype),
        "hidden_size": int(hidden_size),
        "layers": layers,
        "tp_weight_sharded": False if text_stage_weights is None else text_stage_weights.tp_weight_sharded,
        "tp_shard_rank": None if text_stage_weights is None else text_stage_weights.tp_shard_rank,
        "tp_shard_world_size": None if text_stage_weights is None else text_stage_weights.tp_shard_world_size,
    }


def _restore_text_phase(
    phase_payload: dict[str, Any],
    *,
    config_spec: TextModelConfigSpec,
    compute_dtype: torch.dtype,
    rotary_emb,
) -> dict[str, Any]:
    if (
        phase_payload.get("attention_mask") is not None
        and phase_payload.get("cos") is not None
        and phase_payload.get("sin") is not None
    ):
        return phase_payload

    stage_input = phase_payload.get("stage_input")
    if not torch.is_tensor(stage_input):
        raise RuntimeError("text scaffold 缺少 stage_input，无法本地重建 runtime inputs。")

    attention_mask_2d = phase_payload.get("attention_mask_2d")
    if attention_mask_2d is not None and not torch.is_tensor(attention_mask_2d):
        raise RuntimeError("text scaffold 的 attention_mask_2d 不是 tensor，无法本地重建 runtime inputs。")

    batch_size, seq_len, _hidden_size = stage_input.shape
    total_seq_len = int(attention_mask_2d.shape[-1]) if attention_mask_2d is not None else seq_len
    past_length = total_seq_len - seq_len
    runtime_aux = _builder_dep("build_text_runtime_aux_tensors", build_text_runtime_aux_tensors)(
        attention_mask_2d=attention_mask_2d,
        batch_size=batch_size,
        seq_len=seq_len,
        past_length=past_length,
        config_spec=config_spec,
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
        rotary_emb=rotary_emb,
    )
    restored_payload = dict(phase_payload)
    restored_payload["attention_mask"] = _runtime_tensor(runtime_aux["attention_mask"], device=torch.device("cpu"))
    restored_payload["position_ids"] = _runtime_tensor(runtime_aux["position_ids"], device=torch.device("cpu"))
    restored_payload["cos"] = _runtime_tensor(
        runtime_aux["cos"],
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
    )
    restored_payload["sin"] = _runtime_tensor(
        runtime_aux["sin"],
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
    )
    return restored_payload


def restore_text_stage_inputs(
    bundle: dict[str, Any],
    *,
    config_spec: TextModelConfigSpec,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    if not bundle.get("runtime_inputs_local_rebuild"):
        return bundle

    rotary_emb = _builder_dep("build_text_rotary_embedding", build_text_rotary_embedding)(
        config_spec,
        device=torch.device("cpu"),
    )
    restored_bundle = dict(bundle)
    restored_bundle["prefill"] = _restore_text_phase(
        restored_bundle["prefill"],
        config_spec=config_spec,
        compute_dtype=compute_dtype,
        rotary_emb=rotary_emb,
    )
    restored_bundle["decode_steps"] = [
        _restore_text_phase(
            step_payload,
            config_spec=config_spec,
            compute_dtype=compute_dtype,
            rotary_emb=rotary_emb,
        )
        for step_payload in restored_bundle["decode_steps"]
    ]
    return restored_bundle


def materialize_text_stage_bundle(
    *,
    stage_bundle_scaffold: dict[str, Any],
    runtime_config: dict[str, Any],
    compute_dtype: torch.dtype,
    tp_shard_rank: int | None = None,
    tp_shard_world_size: int | None = None,
) -> dict[str, Any]:
    bundle = _restore_text_prompt_bundle(
        dict(stage_bundle_scaffold),
        runtime_config=runtime_config,
    )
    model_path = runtime_config["model_path"]
    start_idx = int(bundle["start_idx"])
    end_idx = int(bundle["end_idx"])

    weight_index = _builder_dep("load_model_weight_index", load_model_weight_index)(model_path)
    config_spec = _builder_dep("load_text_model_config_spec", load_text_model_config_spec)(model_path)
    stage_weights = _builder_dep(
        "load_text_decoder_stage_weight_bundle",
        load_text_decoder_stage_weight_bundle,
    )(
        model_path=model_path,
        start_idx=start_idx,
        end_idx=end_idx,
        is_first_stage=start_idx == 0,
        is_last_stage=end_idx == config_spec.num_hidden_layers - 1,
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
        weight_index=weight_index,
        config_spec=config_spec,
        tp_shard_rank=tp_shard_rank,
        tp_shard_world_size=tp_shard_world_size,
    )

    bundle["layers"] = [dict(layer_bundle) for layer_bundle in stage_weights.layer_bundles]
    bundle["tp_weight_sharded"] = stage_weights.tp_weight_sharded
    bundle["tp_shard_rank"] = stage_weights.tp_shard_rank
    bundle["tp_shard_world_size"] = stage_weights.tp_shard_world_size
    bundle.pop("cache_by_layer", None)

    if start_idx == 0 and stage_weights.embed_tokens_weight is not None:
        bundle["embed_tokens_weight"] = stage_weights.embed_tokens_weight
    if end_idx == config_spec.num_hidden_layers - 1:
        if stage_weights.final_norm_weight is not None:
            bundle["final_norm_weight"] = stage_weights.final_norm_weight
        if stage_weights.final_norm_eps is not None:
            bundle["final_norm_eps"] = stage_weights.final_norm_eps
        if stage_weights.lm_head_weight is not None:
            bundle["lm_head_weight"] = stage_weights.lm_head_weight
        bundle["lm_head_bias"] = stage_weights.lm_head_bias
    return restore_text_stage_inputs(
        bundle,
        config_spec=config_spec,
        compute_dtype=compute_dtype,
    )

