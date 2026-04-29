"""Text stage scaffold helpers and local materialization."""

from __future__ import annotations

import sys
from typing import Any

import torch

from ...hexgen_core.schema import StageSpec, StageState
from .live.common import _runtime_tensor
from .runtime_mm_stage import build_mm_stage_state
from .runtime_text import _restore_text_prompt_stage_state
from .weights import (
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


_TEXT_TENSOR_REF = "__tensor_ref__"
_TEXT_TUPLE_REF = "__tuple_ref__"

_TEXT_TOP_LEVEL_WEIGHT_KEYS = (
    "embed_tokens_weight",
    "final_norm_weight",
    "lm_head_weight",
    "lm_head_bias",
)

_TEXT_LAYER_WEIGHT_KEYS = (
    "q_weight",
    "q_bias",
    "k_weight",
    "k_bias",
    "v_weight",
    "v_bias",
    "o_weight",
    "o_bias",
    "q_norm_weight",
    "k_norm_weight",
    "gate_weight",
    "gate_bias",
    "up_weight",
    "up_bias",
    "down_weight",
    "down_bias",
    "input_ln_weight",
    "post_attn_ln_weight",
)


def _clone_tensor_cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return _runtime_tensor(tensor, device=torch.device("cpu"))


def _weight_tensor_key(tensor: torch.Tensor) -> tuple[str, int, tuple[int, ...], str]:
    return (str(tensor.device), int(tensor.data_ptr()), tuple(tensor.shape), str(tensor.dtype))


_TransportTensorAliasKey = tuple[str, int, int, tuple[int, ...], tuple[int, ...], str]


def _transport_tensor_alias_key(tensor: torch.Tensor) -> _TransportTensorAliasKey:
    return (
        str(tensor.device),
        int(tensor.data_ptr()),
        int(tensor.storage_offset()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
    )


def _weight_tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


_TP_SHARDED_PROJECTION_SPECS = (
    ("q_weight", "local_attention_hidden_size", "hidden_size"),
    ("k_weight", "local_kv_hidden_size", "hidden_size"),
    ("v_weight", "local_kv_hidden_size", "hidden_size"),
    ("o_weight", "hidden_size", "local_attention_hidden_size"),
    ("gate_weight", "local_intermediate_size", "hidden_size"),
    ("up_weight", "local_intermediate_size", "hidden_size"),
    ("down_weight", "hidden_size", "local_intermediate_size"),
)


def _tp_shard_projection_shape_checks(stage_state: dict[str, Any]) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int | None],
]:
    """Return shape proof/mismatches for tensors that must be rank-local TP shards."""

    if not bool(stage_state.get("tp_weight_sharded", False)):
        return [], [], {
            "local_attention_hidden_size": None,
            "local_kv_hidden_size": None,
            "local_intermediate_size": None,
        }

    checks: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    local_dims: dict[str, int | None] = {
        "local_attention_hidden_size": None,
        "local_kv_hidden_size": None,
        "local_intermediate_size": None,
    }
    for layer_pos, layer_bundle in enumerate(stage_state.get("layers", [])):
        if not isinstance(layer_bundle, dict):
            continue
        layer_idx = layer_bundle.get("layer_idx", layer_pos)
        head_dim = layer_bundle.get("head_dim")
        local_num_attention_heads = layer_bundle.get("tp_local_num_attention_heads")
        local_num_key_value_heads = layer_bundle.get("tp_local_num_key_value_heads")
        local_intermediate_size = layer_bundle.get("tp_local_intermediate_size")
        hidden_size = None
        input_ln_weight = layer_bundle.get("input_ln_weight")
        if torch.is_tensor(input_ln_weight):
            hidden_size = int(input_ln_weight.numel())
        if not all(
            value is not None
            for value in (
                head_dim,
                local_num_attention_heads,
                local_num_key_value_heads,
                local_intermediate_size,
                hidden_size,
            )
        ):
            continue
        current_dims = {
            "hidden_size": int(hidden_size),
            "local_attention_hidden_size": int(local_num_attention_heads) * int(head_dim),
            "local_kv_hidden_size": int(local_num_key_value_heads) * int(head_dim),
            "local_intermediate_size": int(local_intermediate_size),
        }
        local_dims["local_attention_hidden_size"] = current_dims["local_attention_hidden_size"]
        local_dims["local_kv_hidden_size"] = current_dims["local_kv_hidden_size"]
        local_dims["local_intermediate_size"] = current_dims["local_intermediate_size"]
        for tensor_name, dim0_key, dim1_key in _TP_SHARDED_PROJECTION_SPECS:
            tensor = layer_bundle.get(tensor_name)
            if not torch.is_tensor(tensor):
                continue
            expected_shape = [current_dims[dim0_key], current_dims[dim1_key]]
            actual_shape = list(tensor.shape)
            check = {
                "name": f"layers.{layer_idx}.{tensor_name}",
                "actual_shape": actual_shape,
                "expected_shape": expected_shape,
                "ok": actual_shape == expected_shape,
            }
            checks.append(check)
            if not check["ok"]:
                mismatches.append(check)
    return checks, mismatches, local_dims


def summarize_text_weight_load(stage_state: dict[str, Any]) -> dict[str, Any]:
    """Return compact, JSON-safe evidence for rank-local text weight materialization."""

    named_tensors: list[tuple[str, torch.Tensor]] = []
    loaded_top_level_weight_names: list[str] = []
    for key in _TEXT_TOP_LEVEL_WEIGHT_KEYS:
        value = stage_state.get(key)
        if torch.is_tensor(value):
            loaded_top_level_weight_names.append(key)
            named_tensors.append((key, value))

    loaded_layer_indices: list[int] = []
    for layer_pos, layer_bundle in enumerate(stage_state.get("layers", [])):
        if not isinstance(layer_bundle, dict):
            continue
        layer_idx = layer_bundle.get("layer_idx", layer_pos)
        try:
            loaded_layer_indices.append(int(layer_idx))
        except (TypeError, ValueError):
            pass
        for key in _TEXT_LAYER_WEIGHT_KEYS:
            value = layer_bundle.get(key)
            if torch.is_tensor(value):
                named_tensors.append((f"layers.{layer_idx}.{key}", value))

    unique_tensors: dict[tuple[str, int, tuple[int, ...], str], tuple[str, torch.Tensor]] = {}
    for name, tensor in named_tensors:
        unique_tensors.setdefault(_weight_tensor_key(tensor), (name, tensor))

    tensor_examples = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes": _weight_tensor_bytes(tensor),
        }
        for name, tensor in list(unique_tensors.values())[:8]
    ]
    sharded_parameter_names = tuple(stage_state.get("tp_sharded_parameter_names") or ())
    replicated_parameter_names = tuple(stage_state.get("tp_replicated_parameter_names") or ())
    loaded_weight_tensor_bytes = sum(_weight_tensor_bytes(tensor) for _name, tensor in unique_tensors.values())
    stage_start_idx = stage_state.get("start_idx")
    stage_end_idx = stage_state.get("end_idx")
    try:
        stage_start_idx = None if stage_start_idx is None else int(stage_start_idx)
        stage_end_idx = None if stage_end_idx is None else int(stage_end_idx)
    except (TypeError, ValueError):
        stage_start_idx = None
        stage_end_idx = None
    loaded_layer_indices = sorted(set(loaded_layer_indices))
    unexpected_layer_indices: list[int] = []
    if stage_start_idx is not None and stage_end_idx is not None:
        unexpected_layer_indices = [
            layer_idx
            for layer_idx in loaded_layer_indices
            if layer_idx < stage_start_idx or layer_idx > stage_end_idx
        ]
    tp_shape_checks, tp_shape_mismatches, tp_local_dims = _tp_shard_projection_shape_checks(stage_state)
    tp_weight_sharded = bool(stage_state.get("tp_weight_sharded", False))
    tp_shard_shape_ok = (not tp_weight_sharded or bool(tp_shape_checks)) and not tp_shape_mismatches
    tp_stage_loaded_weight_tensor_bytes = stage_state.get("_tp_stage_loaded_weight_tensor_bytes")
    if tp_stage_loaded_weight_tensor_bytes is None:
        tp_stage_loaded_weight_tensor_bytes = stage_state.get("tp_stage_loaded_weight_tensor_bytes")
    if tp_stage_loaded_weight_tensor_bytes is not None:
        tp_stage_loaded_weight_tensor_bytes = [int(value) for value in tp_stage_loaded_weight_tensor_bytes]
    tp_stage_loaded_weight_tensor_bytes_equal = stage_state.get("_tp_stage_loaded_weight_tensor_bytes_equal")
    if tp_stage_loaded_weight_tensor_bytes_equal is None:
        tp_stage_loaded_weight_tensor_bytes_equal = stage_state.get("tp_stage_loaded_weight_tensor_bytes_equal")
    tp_stage_loaded_weight_tensor_bytes_checked = stage_state.get("_tp_stage_loaded_weight_tensor_bytes_checked")
    if tp_stage_loaded_weight_tensor_bytes_checked is None:
        tp_stage_loaded_weight_tensor_bytes_checked = stage_state.get("tp_stage_loaded_weight_tensor_bytes_checked")
    return {
        "tp_weight_sharded": tp_weight_sharded,
        "tp_shard_rank": stage_state.get("tp_shard_rank"),
        "tp_shard_world_size": stage_state.get("tp_shard_world_size"),
        "tp_local_dims": tp_local_dims,
        "tp_shard_shape_ok": tp_shard_shape_ok,
        "tp_sharded_projection_check_count": len(tp_shape_checks),
        "tp_shard_shape_mismatches": tp_shape_mismatches,
        "tp_sharded_projection_examples": tp_shape_checks[:8],
        "tp_stage_loaded_weight_tensor_bytes": tp_stage_loaded_weight_tensor_bytes,
        "tp_stage_loaded_weight_tensor_bytes_equal": tp_stage_loaded_weight_tensor_bytes_equal,
        "tp_stage_loaded_weight_tensor_bytes_checked": tp_stage_loaded_weight_tensor_bytes_checked,
        "stage_start_idx": stage_start_idx,
        "stage_end_idx": stage_end_idx,
        "loaded_layer_indices": loaded_layer_indices,
        "loaded_layer_count": len(loaded_layer_indices),
        "loaded_top_level_weight_names": loaded_top_level_weight_names,
        "unexpected_layer_indices": unexpected_layer_indices,
        "stage_weight_scope_ok": not unexpected_layer_indices,
        "loaded_weight_tensor_count": len(unique_tensors),
        "loaded_weight_tensor_bytes": loaded_weight_tensor_bytes,
        "loaded_weight_tensor_mib": round(loaded_weight_tensor_bytes / (1024 * 1024), 3),
        "loaded_weight_tensor_examples": tensor_examples,
        "tp_sharded_parameter_count": len(sharded_parameter_names),
        "tp_replicated_parameter_count": len(replicated_parameter_names),
        "tp_sharded_parameter_examples": list(sharded_parameter_names[:8]),
        "tp_replicated_parameter_examples": list(replicated_parameter_names[:8]),
    }


def assert_text_weight_scope(stage_state: dict[str, Any]) -> None:
    """Fail fast if a StageState contains decoder layer weights outside its own range."""

    weight_load = summarize_text_weight_load(stage_state)
    if not weight_load["stage_weight_scope_ok"]:
        raise RuntimeError(
            "StageState 加载了非本 stage 的 decoder layer 权重: "
            f"stage_range={weight_load['stage_start_idx']}:{weight_load['stage_end_idx']} "
            f"unexpected_layer_indices={weight_load['unexpected_layer_indices']}"
        )


def assert_text_tp_shard_shapes(stage_state: dict[str, Any]) -> None:
    """Fail fast if TP-sharded projection tensors are not shard-sized."""

    weight_load = summarize_text_weight_load(stage_state)
    if not weight_load["tp_shard_shape_ok"]:
        raise RuntimeError(
            "TP shard-only StageState 的投影权重形状不是本 rank shard 大小: "
            f"rank={weight_load['tp_shard_rank']}/{weight_load['tp_shard_world_size']} "
            f"check_count={weight_load['tp_sharded_projection_check_count']} "
            f"mismatches={weight_load['tp_shard_shape_mismatches']}"
        )


def _strip_text_phase(phase_payload: dict[str, Any]) -> dict[str, Any]:
    compact_payload = dict(phase_payload)
    compact_payload.pop("attention_mask", None)
    compact_payload.pop("cos", None)
    compact_payload.pop("sin", None)
    compact_payload.pop("position_ids", None)
    return compact_payload


def compact_text_scaffold(stage_state: dict[str, Any]) -> dict[str, Any]:
    scaffold = dict(stage_state)
    scaffold["runtime_inputs_local_rebuild"] = True
    scaffold["runtime_prefill_cache_policy"] = "recompute"
    scaffold.pop("cache_by_layer", None)
    scaffold["prefill"] = _strip_text_phase(scaffold["prefill"])
    scaffold["decode_steps"] = [
        _strip_text_phase(step_payload)
        for step_payload in scaffold["decode_steps"]
    ]
    return scaffold


def compact_text_stage_state(stage_state: dict[str, Any]) -> dict[str, Any]:
    scaffold = dict(stage_state)
    scaffold["runtime_only_prompt_local_rebuild"] = True
    scaffold.pop("input_ids", None)
    scaffold.pop("prefill_attention_mask_2d", None)
    scaffold.pop("prefill_seq_len", None)
    scaffold.pop("batch_size", None)
    scaffold.pop("token_id_dtype", None)
    return scaffold


def compact_multimodal_runtime_scaffold(stage_state: dict[str, Any]) -> dict[str, Any]:
    scaffold = dict(stage_state)
    stage_input = scaffold.get("stage_input")
    layer_input = scaffold.get("layer_input")
    if torch.is_tensor(stage_input) and torch.is_tensor(layer_input):
        if _transport_tensor_alias_key(stage_input) == _transport_tensor_alias_key(layer_input):
            scaffold.pop("layer_input", None)
    return scaffold


def _pack_text_transport_value(
    value: Any,
    *,
    path: tuple[str, ...],
    tensor_payload: dict[str, torch.Tensor | None],
    tensor_aliases: dict[_TransportTensorAliasKey, str],
) -> Any:
    if torch.is_tensor(value):
        key = ".".join(path)
        alias_key = _transport_tensor_alias_key(value)
        existing_key = tensor_aliases.get(alias_key)
        if existing_key is not None:
            return {_TEXT_TENSOR_REF: existing_key}
        tensor_payload[key] = _clone_tensor_cpu(value)
        tensor_aliases[alias_key] = key
        return {_TEXT_TENSOR_REF: key}
    if isinstance(value, dict):
        return {
            key: _pack_text_transport_value(
                item,
                path=(*path, str(key)),
                tensor_payload=tensor_payload,
                tensor_aliases=tensor_aliases,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _pack_text_transport_value(
                item,
                path=(*path, str(index)),
                tensor_payload=tensor_payload,
                tensor_aliases=tensor_aliases,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return {
            _TEXT_TUPLE_REF: [
                _pack_text_transport_value(
                    item,
                    path=(*path, str(index)),
                    tensor_payload=tensor_payload,
                    tensor_aliases=tensor_aliases,
                )
                for index, item in enumerate(value)
            ]
        }
    return value


def pack_named_tensor_dict_transport(
    payload: dict[str, Any],
    *,
    root_key: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor | None]]:
    tensor_payload: dict[str, torch.Tensor | None] = {}
    tensor_aliases: dict[_TransportTensorAliasKey, str] = {}
    meta = _pack_text_transport_value(
        payload,
        path=(root_key,),
        tensor_payload=tensor_payload,
        tensor_aliases=tensor_aliases,
    )
    if not isinstance(meta, dict):
        raise RuntimeError(f"{root_key} transport meta 必须是 dict。")
    return (
        {
            "version": 1,
            root_key: meta,
        },
        tensor_payload,
    )


def pack_text_scaffold_transport(
    scaffold: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor | None]]:
    return pack_named_tensor_dict_transport(scaffold, root_key="scaffold")


def _restore_text_transport_value(
    value: Any,
    *,
    tensor_payload: dict[str, torch.Tensor | None],
) -> Any:
    if isinstance(value, dict):
        tensor_ref = value.get(_TEXT_TENSOR_REF)
        if isinstance(tensor_ref, str):
            if tensor_ref not in tensor_payload:
                raise RuntimeError(f"text scaffold transport 缺少 tensor payload: {tensor_ref}")
            return _clone_tensor_cpu(tensor_payload[tensor_ref])
        tuple_ref = value.get(_TEXT_TUPLE_REF)
        if isinstance(tuple_ref, list):
            return tuple(
                _restore_text_transport_value(
                    item,
                    tensor_payload=tensor_payload,
                )
                for item in tuple_ref
            )
        return {
            key: _restore_text_transport_value(
                item,
                tensor_payload=tensor_payload,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _restore_text_transport_value(
                item,
                tensor_payload=tensor_payload,
            )
            for item in value
        ]
    return value


def restore_named_tensor_dict_transport(
    meta: dict[str, Any],
    tensor_payload: dict[str, torch.Tensor | None] | None,
    *,
    root_key: str,
) -> dict[str, Any]:
    if not isinstance(meta, dict):
        raise RuntimeError(f"{root_key} transport meta 缺少 metadata。")
    if tensor_payload is None:
        raise RuntimeError(f"{root_key} transport 缺少 tensor payload。")
    payload_meta = meta.get(root_key)
    if not isinstance(payload_meta, dict):
        raise RuntimeError(f"{root_key} transport meta 缺少 {root_key}。")
    restored = _restore_text_transport_value(
        payload_meta,
        tensor_payload=tensor_payload,
    )
    if not isinstance(restored, dict):
        raise RuntimeError(f"{root_key} transport 恢复结果不是 dict。")
    return restored


def restore_text_scaffold_transport(
    meta: dict[str, Any],
    tensor_payload: dict[str, torch.Tensor | None] | None,
) -> dict[str, Any]:
    return restore_named_tensor_dict_transport(
        meta,
        tensor_payload,
        root_key="scaffold",
    )


def pack_runtime_input_transport(
    runtime_inputs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor | None]]:
    return pack_named_tensor_dict_transport(runtime_inputs, root_key="runtime_inputs")


def restore_runtime_input_transport(
    meta: dict[str, Any],
    tensor_payload: dict[str, torch.Tensor | None] | None,
) -> dict[str, Any]:
    return restore_named_tensor_dict_transport(
        meta,
        tensor_payload,
        root_key="runtime_inputs",
    )


def build_text_stage_state(
    *,
    spec: StageSpec,
    stage_state_device: torch.device,
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
        "prefill_attention_mask_2d": _runtime_tensor(prefill_attention_mask_2d, device=stage_state_device),
        "batch_size": int(batch_size),
        "token_id_dtype": _dtype_name(token_id_dtype),
        "hidden_size": int(hidden_size),
        "layers": layers,
        "tp_weight_sharded": False if text_stage_weights is None else text_stage_weights.tp_weight_sharded,
        "tp_shard_rank": None if text_stage_weights is None else text_stage_weights.tp_shard_rank,
        "tp_shard_world_size": None if text_stage_weights is None else text_stage_weights.tp_shard_world_size,
        "tp_sharded_parameter_names": () if text_stage_weights is None else text_stage_weights.tp_sharded_parameter_names,
        "tp_replicated_parameter_names": () if text_stage_weights is None else text_stage_weights.tp_replicated_parameter_names,
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
    stage_state: dict[str, Any],
    *,
    config_spec: TextModelConfigSpec,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    if not stage_state.get("runtime_inputs_local_rebuild"):
        return stage_state

    rotary_emb = _builder_dep("build_text_rotary_embedding", build_text_rotary_embedding)(
        config_spec,
        device=torch.device("cpu"),
    )
    restored_state = dict(stage_state)
    restored_state["prefill"] = _restore_text_phase(
        restored_state["prefill"],
        config_spec=config_spec,
        compute_dtype=compute_dtype,
        rotary_emb=rotary_emb,
    )
    restored_state["decode_steps"] = [
        _restore_text_phase(
            step_payload,
            config_spec=config_spec,
            compute_dtype=compute_dtype,
            rotary_emb=rotary_emb,
        )
        for step_payload in restored_state["decode_steps"]
    ]
    return restored_state


def _infer_runtime_batch_size(stage_state: dict[str, Any]) -> int | None:
    for key in ("prefill_attention_mask_2d", "stage_input", "layer_input"):
        value = stage_state.get(key)
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
    prefill = stage_state.get("prefill")
    if isinstance(prefill, dict):
        for key in ("attention_mask_2d", "stage_input", "layer_input"):
            value = prefill.get(key)
            if torch.is_tensor(value) and value.ndim >= 1:
                return int(value.shape[0])
    return None


_MM_PREFILL_RUNTIME_REBUILD_FLAG = "mm_prefill_runtime_tensors_local_rebuild"
_MM_PREFILL_RUNTIME_FIELDS = (
    "prefill_attention_mask_2d",
    "prefill_attention_mask",
    "prefill_position_ids",
    "prefill_cos",
    "prefill_sin",
)


def _restore_multimodal_prefill_runtime_tensors(
    stage_state: dict[str, Any],
    *,
    runtime_config: dict[str, Any],
    config_spec: TextModelConfigSpec,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    if not stage_state.get("runtime_only_generate"):
        return stage_state
    if stage_state.get("modality") != "multimodal":
        return stage_state
    if all(stage_state.get(field_name) is not None for field_name in _MM_PREFILL_RUNTIME_FIELDS):
        return stage_state

    shared = runtime_config.get("_mm_startup_shared")
    if not isinstance(shared, dict):
        missing_fields = [
            field_name
            for field_name in _MM_PREFILL_RUNTIME_FIELDS
            if stage_state.get(field_name) is None
        ]
        raise RuntimeError(
            "multimodal runtime-only scaffold 缺少 prefill runtime tensors，"
            f"且 runtime_config 没有 _mm_startup_shared，无法本地重建: {missing_fields}"
        )

    stage_input = stage_state.get("stage_input")
    if not torch.is_tensor(stage_input):
        stage_input = stage_state.get("layer_input")
    if not torch.is_tensor(stage_input):
        raise RuntimeError("multimodal runtime-only scaffold 缺少 stage_input，无法本地重建 prefill tensors。")

    rotary_emb = build_text_rotary_embedding(
        config_spec,
        device=torch.device("cpu"),
    )
    restored_mm_state = build_mm_stage_state(
        shared,
        stage_input=stage_input,
        start_idx=int(stage_state["start_idx"]),
        end_idx=int(stage_state["end_idx"]),
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
        config_spec=config_spec,
        rotary_emb=rotary_emb,
        visual_pos_masks=stage_state.get("visual_pos_masks"),
        deepstack_by_layer=stage_state.get("deepstack_by_layer") or {},
    )

    restored = dict(stage_state)
    restored.pop(_MM_PREFILL_RUNTIME_REBUILD_FLAG, None)
    restored["prefill_attention_mask_2d"] = _runtime_tensor(
        restored_mm_state.attention_mask_2d,
        device=torch.device("cpu"),
    )
    restored["prefill_attention_mask"] = _runtime_tensor(
        restored_mm_state.attention_mask,
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
    )
    restored["prefill_position_ids"] = _runtime_tensor(
        restored_mm_state.position_ids,
        device=torch.device("cpu"),
    )
    restored["prefill_cos"] = _runtime_tensor(
        restored_mm_state.cos,
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
    )
    restored["prefill_sin"] = _runtime_tensor(
        restored_mm_state.sin,
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
    )
    if restored.get("rope_deltas") is None and restored_mm_state.rope_deltas is not None:
        restored["rope_deltas"] = _runtime_tensor(
            restored_mm_state.rope_deltas,
            device=torch.device("cpu"),
        )
    return restored


def materialize_text_stage_state(
    *,
    stage_state_scaffold: StageState,
    runtime_config: dict[str, Any],
    compute_dtype: torch.dtype,
    tp_shard_rank: int | None = None,
    tp_shard_world_size: int | None = None,
) -> StageState:
    stage_state = _restore_text_prompt_stage_state(
        dict(stage_state_scaffold),
        runtime_config=runtime_config,
    )
    model_path = runtime_config["model_path"]
    start_idx = int(stage_state["start_idx"])
    end_idx = int(stage_state["end_idx"])

    weight_index = _builder_dep("load_model_weight_index", load_model_weight_index)(model_path)
    config_spec = _builder_dep("load_text_model_config_spec", load_text_model_config_spec)(model_path)
    stage_state = _restore_multimodal_prefill_runtime_tensors(
        stage_state,
        runtime_config=runtime_config,
        config_spec=config_spec,
        compute_dtype=compute_dtype,
    )
    if stage_state.get("save_dtype") is None:
        stage_state["save_dtype"] = _dtype_name(compute_dtype)
    if stage_state.get("hidden_size") is None:
        stage_state["hidden_size"] = int(config_spec.hidden_size)
    if stage_state.get("batch_size") is None:
        batch_size = _infer_runtime_batch_size(stage_state)
        if batch_size is not None:
            stage_state["batch_size"] = batch_size
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

    stage_state["layers"] = [dict(layer_bundle) for layer_bundle in stage_weights.layer_bundles]
    stage_state["tp_weight_sharded"] = stage_weights.tp_weight_sharded
    stage_state["tp_shard_rank"] = stage_weights.tp_shard_rank
    stage_state["tp_shard_world_size"] = stage_weights.tp_shard_world_size
    stage_state["tp_sharded_parameter_names"] = stage_weights.tp_sharded_parameter_names
    stage_state["tp_replicated_parameter_names"] = stage_weights.tp_replicated_parameter_names
    stage_state.pop("cache_by_layer", None)

    if start_idx == 0 and stage_weights.embed_tokens_weight is not None:
        stage_state["embed_tokens_weight"] = stage_weights.embed_tokens_weight
    if end_idx == config_spec.num_hidden_layers - 1:
        if stage_weights.final_norm_weight is not None:
            stage_state["final_norm_weight"] = stage_weights.final_norm_weight
        if stage_weights.final_norm_eps is not None:
            stage_state["final_norm_eps"] = stage_weights.final_norm_eps
        if stage_weights.lm_head_weight is not None:
            stage_state["lm_head_weight"] = stage_weights.lm_head_weight
        stage_state["lm_head_bias"] = stage_weights.lm_head_bias
    assert_text_weight_scope(stage_state)
    assert_text_tp_shard_shapes(stage_state)
    return restore_text_stage_inputs(
        stage_state,
        config_spec=config_spec,
        compute_dtype=compute_dtype,
    )
