"""Dedicated hybrid-parallel runtime for HexGen-style PP+TP stage execution."""

import torch
import torch.distributed as dist

from ..distributed import (
    broadcast_cpu,
    broadcast_tensor_payload_cpu,
    broadcast_object_cpu,
    recv_tensor_payload_cpu,
    recv_object_cpu,
    send_tensor_payload_cpu,
    send_object_cpu,
    startup_log,
    startup_timer,
)
from ..generate_buffers import (
    build_decode_attention_mask_buffer,
    decode_attention_mask_view,
    fill_decode_input_ids,
)
from ..gen_hetero_groups import (
    build_hybrid_layout,
    build_p2p_lists,
    build_pp_rank_groups,
    parse_tp_degrees,
)
from .pipeline_parallel import (
    prepare_multimodal_decode_pipeline,
    prepare_multimodal_generate_pipeline,
    prepare_multimodal_prefill_pipeline,
    prepare_text_decode_pipeline,
    prepare_text_generate_pipeline,
    prepare_text_prefill_pipeline,
    prepare_text_pipeline,
)
# HYBRID composes the pure TP backend. These helpers are a local backend-level
# reuse surface, but they intentionally stay out of package-level __all__.
from .tensor_parallel import (
    broadcast_token_id,
    build_generate_cache_map,
    build_generate_phase_state,
    build_runtime_only_stage_input_template,
    infer_runtime_tensor_device,
    infer_runtime_tensor_dtype,
    infer_runtime_token_dtype,
    is_runtime_only_generate_state,
    strip_runtime_layer_cache,
    token_tensor_to_list,
    run_stage_state_tp,
)
from ...debug.tp_debug import TpDebugConfig
from ..schema import (
    HYBRID_RUNTIME_INPUT_PROTOCOL,
    HybridRankContext,
    HybridRuntimeInputSchema,
    StageHandoffPayload,
    StageState,
    TextHybridManifest,
)
from ..stage import (
    apply_stage_handoff_payload,
    build_stage_handoff_payload,
    get_stage_input,
    get_stage_output,
    run_stage_tp,
)
from ..transport import StageCommunicator
from ...models.qwen3vl.execution import (
    forward_text_embeddings,
    trace_text_decode_logits_tp_with_runtime_cache,
    trace_text_decode_stage_tp_with_runtime_cache,
)
from ...models.qwen3vl.functional import dtype_from_name, resolve_comm_dtype
from ...models.qwen3vl.runtime_mm_stage import build_mm_decode_state_from_weights
from ...models.qwen3vl.runtime_builder import (
    DirectStageStateBuilder,
    build_direct_stage_state,
    compact_mm_shared_for_transport,
    materialize_text_stage_state,
    pack_model_input_transport,
    pack_text_scaffold_transport,
    prepare_text_prompt_meta,
    restore_model_input_transport,
    restore_text_scaffold_transport,
    restore_mm_startup_transport,
    seed_mm_startup_runtime_config,
)
from ...models.qwen3vl.runtime_text_stage import (
    assert_text_tp_shard_shapes,
    summarize_text_weight_load,
)
from ...models.qwen3vl.capture import load_bundle, move_bundle
from ...models.qwen3vl.weights import (
    build_text_rotary_embedding,
    build_text_runtime_aux_tensors,
    load_model_weight_index,
    load_text_model_config_spec,
    load_tensors_from_index,
)


_RANK_LOCAL_SCAFFOLD_REBUILD_KEY = "rank_local_fields_local_rebuild"
_RANK_LOCAL_SCAFFOLD_FIELDS = (
    "stage_idx",
    "start_idx",
    "end_idx",
    "save_dtype",
    "hidden_size",
    "batch_size",
)
_COMPUTE_DTYPE_REF_NAMES = (
    "model.language_model.layers.0.input_layernorm.weight",
    "model.language_model.norm.weight",
    "model.language_model.embed_tokens.weight",
)
_MM_PREFILL_SCAFFOLD_REBUILD_KEY = "mm_prefill_runtime_tensors_local_rebuild"
_MM_PREFILL_SCAFFOLD_DERIVED_FIELDS = (
    "prefill_attention_mask_2d",
    "prefill_attention_mask",
    "prefill_position_ids",
    "prefill_cos",
    "prefill_sin",
)
_MM_FRONTEND_METADATA_SCAFFOLD_REBUILD_KEY = "mm_frontend_metadata_local_rebuild"
_MM_FRONTEND_METADATA_SCAFFOLD_FIELDS = (
    "num_frames",
    "frame_paths",
)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _strip_rank_local_scaffold_fields(scaffold: StageState) -> StageState:
    compact = dict(scaffold)
    removed_any = False
    for field_name in _RANK_LOCAL_SCAFFOLD_FIELDS:
        if field_name in compact:
            compact.pop(field_name)
            removed_any = True
    if removed_any:
        compact[_RANK_LOCAL_SCAFFOLD_REBUILD_KEY] = True
    return compact


def _strip_multimodal_prefill_scaffold_tensors(scaffold: StageState) -> StageState:
    if not bool(scaffold.get("runtime_only_generate")):
        return scaffold
    if str(scaffold.get("modality", "text")) != "multimodal":
        return scaffold
    compact = dict(scaffold)
    removed_any = False
    for field_name in _MM_PREFILL_SCAFFOLD_DERIVED_FIELDS:
        if field_name in compact:
            compact.pop(field_name)
            removed_any = True
    if removed_any:
        compact[_MM_PREFILL_SCAFFOLD_REBUILD_KEY] = True
    return compact


def _strip_multimodal_frontend_scaffold_metadata(scaffold: StageState) -> StageState:
    if not bool(scaffold.get("runtime_only_generate")):
        return scaffold
    if str(scaffold.get("modality", "text")) != "multimodal":
        return scaffold
    compact = dict(scaffold)
    removed_any = False
    for field_name in _MM_FRONTEND_METADATA_SCAFFOLD_FIELDS:
        if field_name in compact:
            compact.pop(field_name)
            removed_any = True
    if removed_any:
        compact[_MM_FRONTEND_METADATA_SCAFFOLD_REBUILD_KEY] = True
    return compact


def _compact_hybrid_scaffold_broadcast(scaffold: StageState) -> StageState:
    return _strip_rank_local_scaffold_fields(
        _strip_multimodal_frontend_scaffold_metadata(
            _strip_multimodal_prefill_scaffold_tensors(scaffold)
        )
    )


def _use_model_input_broadcast(manifest: TextHybridManifest) -> bool:
    runtime_config = manifest.runtime_config
    return (
        str(runtime_config.get("mode", "")) == "generate"
        and not bool(runtime_config.get("include_runtime_reference", True))
    )


def _build_model_input_broadcast_payload(
    runtime_config: dict[str, object],
    *,
    stage_idx: int,
    runtime_modality: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "protocol": HYBRID_RUNTIME_INPUT_PROTOCOL,
        "modality": runtime_modality,
        "mode": "generate",
        "runtime_only_generate": True,
    }
    if runtime_modality == "text":
        input_ids = runtime_config.get("_runtime_only_input_ids")
        if not torch.is_tensor(input_ids):
            raise RuntimeError("HYBRID text runtime input 缺少 _runtime_only_input_ids。")
        payload["input_ids"] = input_ids
        attention_mask = runtime_config.get("_runtime_only_attention_mask")
        if torch.is_tensor(attention_mask):
            payload["attention_mask_2d"] = attention_mask
        payload["runtime_only_prompt_local_rebuild"] = True
        HybridRuntimeInputSchema.validate(
            payload,
            context=f"build stage_idx={int(stage_idx)}",
        )
        return payload

    if runtime_modality != "multimodal":
        raise RuntimeError(f"不支持的 HYBRID runtime input modality={runtime_modality!r}。")

    shared = runtime_config.get("_mm_startup_shared")
    stage_handoffs = runtime_config.get("_mm_startup_stage_handoffs")
    if not isinstance(shared, dict):
        raise RuntimeError("HYBRID multimodal runtime input 缺少 _mm_startup_shared。")
    if not isinstance(stage_handoffs, dict):
        raise RuntimeError("HYBRID multimodal runtime input 缺少 _mm_startup_stage_handoffs。")

    stage_payload = stage_handoffs.get(int(stage_idx))
    if not isinstance(stage_payload, dict):
        raise RuntimeError(f"HYBRID multimodal runtime input 缺少 stage_idx={stage_idx} 的 handoff。")
    stage_input = stage_payload.get("stage_input")
    if not torch.is_tensor(stage_input):
        raise RuntimeError("HYBRID multimodal runtime input 缺少 stage_input。")
    payload["shared"] = compact_mm_shared_for_transport(
        dict(shared),
        include_derived=False,
    )
    payload["stage_handoff"] = {
        "stage_input": stage_input,
    }

    stage_visuals = runtime_config.get("_mm_startup_stage_visuals")
    stage_visual_payload = None
    if isinstance(stage_visuals, dict):
        maybe_stage_visual_payload = stage_visuals.get(int(stage_idx))
        if isinstance(maybe_stage_visual_payload, dict):
            stage_visual_payload = maybe_stage_visual_payload
    if stage_visual_payload is not None:
        payload["stage_visuals"] = {
            "visual_pos_masks": stage_visual_payload.get("visual_pos_masks"),
            "deepstack_by_layer": {
                int(layer_idx): deepstack
                for layer_idx, deepstack in (stage_visual_payload.get("deepstack_by_layer") or {}).items()
            },
        }
    HybridRuntimeInputSchema.validate(
        payload,
        context=f"build stage_idx={int(stage_idx)}",
    )
    return payload


def _restore_stage_state_from_model_input(
    model_input: dict[str, object],
    *,
    stage_meta,
    runtime_config: dict[str, object],
    compute_dtype: torch.dtype,
) -> StageState:
    HybridRuntimeInputSchema.validate(
        model_input,
        context=f"restore stage_idx={int(stage_meta.stage_idx)}",
    )
    modality = str(model_input.get("modality", runtime_config.get("modality", "text")))
    restored: StageState = {
        "module_name": f"{modality}_generate_stage",
        "stage_type": f"{modality}_generate_runtime_only",
        "runtime_only_generate": True,
        "modality": modality,
        "stage_idx": int(stage_meta.stage_idx),
        "start_idx": int(stage_meta.start_idx),
        "end_idx": int(stage_meta.end_idx),
        "save_dtype": _dtype_name(compute_dtype),
        "max_new_tokens": int(runtime_config.get("max_new_tokens", 4)),
        "layers": [],
        "runtime_inputs_from_broadcast": True,
    }
    if modality == "text":
        input_ids = model_input.get("input_ids")
        if torch.is_tensor(input_ids):
            runtime_config["_runtime_only_input_ids"] = input_ids
            attention_mask_2d = model_input.get("attention_mask_2d")
            if torch.is_tensor(attention_mask_2d):
                runtime_config["_runtime_only_attention_mask"] = attention_mask_2d
            else:
                runtime_config.pop("_runtime_only_attention_mask", None)
            runtime_config["_runtime_only_prompt_metadata_ready"] = True
        restored["runtime_only_prompt_local_rebuild"] = True
        return restored

    if modality != "multimodal":
        raise RuntimeError(f"不支持的 HYBRID runtime input modality={modality!r}。")

    shared = model_input.get("shared")
    stage_handoff = model_input.get("stage_handoff")
    if not isinstance(shared, dict):
        raise RuntimeError("HYBRID multimodal runtime input 恢复时缺少 shared。")
    if not isinstance(stage_handoff, dict):
        raise RuntimeError("HYBRID multimodal runtime input 恢复时缺少 stage_handoff。")
    stage_input = stage_handoff.get("stage_input")
    if not torch.is_tensor(stage_input):
        raise RuntimeError("HYBRID multimodal runtime input 恢复时缺少 stage_input。")
    stage_idx = int(stage_meta.stage_idx)
    runtime_config["_mm_startup_shared"] = dict(shared)
    runtime_config["_mm_startup_stage_handoffs"] = {
        stage_idx: {
            "stage_input": stage_input,
        }
    }
    runtime_config["_mm_startup_contract_ready"] = True
    runtime_config["_mm_frontend_state_ready"] = True
    restored["stage_input"] = stage_input
    if shared.get("rope_deltas") is not None:
        restored["rope_deltas"] = shared["rope_deltas"]
    stage_visuals = model_input.get("stage_visuals")
    if isinstance(stage_visuals, dict):
        runtime_config["_mm_startup_stage_visuals"] = {
            stage_idx: dict(stage_visuals),
        }
        if stage_visuals.get("visual_pos_masks") is not None:
            restored["visual_pos_masks"] = stage_visuals["visual_pos_masks"]
        deepstack_by_layer = stage_visuals.get("deepstack_by_layer")
    else:
        runtime_config.pop("_mm_startup_stage_visuals", None)
        deepstack_by_layer = None
    if isinstance(deepstack_by_layer, dict) and deepstack_by_layer:
        restored["deepstack_by_layer"] = {
            int(layer_idx): deepstack
            for layer_idx, deepstack in deepstack_by_layer.items()
        }
        restored["deepstack_layer_indices"] = sorted(restored["deepstack_by_layer"])
    return restored


def _first_floating_tensor_dtype(value) -> torch.dtype | None:
    if torch.is_tensor(value):
        return value.dtype if value.is_floating_point() else None
    if isinstance(value, dict):
        for item in value.values():
            dtype = _first_floating_tensor_dtype(item)
            if dtype is not None:
                return dtype
    if isinstance(value, (list, tuple)):
        for item in value:
            dtype = _first_floating_tensor_dtype(item)
            if dtype is not None:
                return dtype
    return None


def _infer_model_compute_dtype(model_path: str) -> torch.dtype:
    weight_index = load_model_weight_index(model_path)
    for tensor_name in _COMPUTE_DTYPE_REF_NAMES:
        if not weight_index.has_tensor(tensor_name):
            continue
        tensors = load_tensors_from_index(
            weight_index,
            [tensor_name],
            device=torch.device("cpu"),
            compute_dtype=None,
            strict=True,
        )
        tensor = tensors.get(tensor_name)
        if tensor is not None:
            return tensor.dtype
    raise RuntimeError("无法从本地模型权重推导 HYBRID scaffold compute dtype。")


def _resolve_scaffold_compute_dtype(
    scaffold: StageState,
    *,
    manifest: TextHybridManifest,
    stage_meta,
    compute_dtype_arg: str,
) -> torch.dtype:
    if compute_dtype_arg != "auto":
        return dtype_from_name(compute_dtype_arg)

    scaffold_save_dtype = scaffold.get("save_dtype")
    if isinstance(scaffold_save_dtype, str):
        return dtype_from_name(scaffold_save_dtype)

    for configured in (
        getattr(stage_meta, "save_dtype", None),
        manifest.runtime_config.get("save_dtype"),
        manifest.save_dtype,
    ):
        if isinstance(configured, str) and configured != "auto":
            return dtype_from_name(configured)

    tensor_dtype = _first_floating_tensor_dtype(scaffold)
    if tensor_dtype is not None:
        return tensor_dtype

    return _infer_model_compute_dtype(str(manifest.runtime_config["model_path"]))


def _restore_rank_local_scaffold_fields(
    scaffold: StageState,
    *,
    stage_meta,
    compute_dtype: torch.dtype,
) -> StageState:
    restored = dict(scaffold)
    restored.pop(_RANK_LOCAL_SCAFFOLD_REBUILD_KEY, None)
    restored.setdefault("stage_idx", int(stage_meta.stage_idx))
    restored.setdefault("start_idx", int(stage_meta.start_idx))
    restored.setdefault("end_idx", int(stage_meta.end_idx))
    restored.setdefault("save_dtype", _dtype_name(compute_dtype))
    return restored


def _build_rank_group_index(rank_groups: list[list[int]], world_size: int) -> list[int]:
    group_index_by_rank = [-1] * world_size
    for group_idx, ranks in enumerate(rank_groups):
        for rank in ranks:
            if rank < 0 or rank >= world_size:
                raise ValueError(f"rank={rank} 超出 world_size={world_size}。")
            group_index_by_rank[rank] = group_idx
    if any(group_idx < 0 for group_idx in group_index_by_rank):
        raise ValueError("并不是所有 rank 都能映射到 group。")
    return group_index_by_rank


def _all_hybrid_stages_are_direct(manifest: TextHybridManifest) -> bool:
    manifest_is_direct = getattr(manifest, "is_direct", None)
    if manifest_is_direct is not None:
        return bool(manifest_is_direct)
    return all(
        getattr(stage, "replay_bundle_path", getattr(stage, "bundle_path", None)) is None
        for stage in getattr(manifest, "stages", [])
    )


def _broadcast_stage_state_transport(
    stage_state: StageState | None,
    *,
    src: int,
    group,
    meta_label: str,
    tensor_label: str,
) -> StageState:
    state_meta = None
    state_tensors = None
    if stage_state is not None:
        # Historical helper name is text_scaffold, but the transport itself is a
        # generic dict+tensor codec that also works for multimodal StageState.
        state_meta, state_tensors = pack_text_scaffold_transport(stage_state)
    state_meta = broadcast_object_cpu(
        state_meta,
        src=src,
        group=group,
        label=meta_label,
    )
    state_tensors = broadcast_tensor_payload_cpu(
        state_tensors,
        src=src,
        group=group,
        label=tensor_label,
    )
    restored = restore_text_scaffold_transport(
        state_meta,
        state_tensors,
    )
    if not isinstance(restored, dict):
        raise RuntimeError("StageState transport 恢复结果不是 dict。")
    return restored


def _validate_rank_local_sharded_state(stage_state: StageState, rank_stage: HybridRankContext) -> None:
    if rank_stage.tp_degree <= 1:
        return
    if not bool(stage_state.get("tp_weight_sharded")):
        raise RuntimeError(
            "direct TP stage 必须 materialize 成 rank-local shard StageState，"
            f"stage_idx={rank_stage.stage_idx} rank_local={rank_stage.local_rank}/{rank_stage.tp_degree} "
            "但 StageState 没有 tp_weight_sharded=True。"
        )
    state_rank = stage_state.get("tp_shard_rank")
    state_world_size = stage_state.get("tp_shard_world_size")
    if state_rank != rank_stage.local_rank or state_world_size != rank_stage.tp_degree:
        raise RuntimeError(
            "direct TP stage 的本地 shard 标记和 rank layout 不一致，"
            f"stage_idx={rank_stage.stage_idx} expected={rank_stage.local_rank}/{rank_stage.tp_degree} "
            f"actual={state_rank}/{state_world_size}。"
        )
    assert_text_tp_shard_shapes(stage_state)


def _record_tp_stage_weight_load_consistency(stage_state: StageState, rank_stage: HybridRankContext) -> None:
    if rank_stage.tp_degree <= 1:
        return

    weight_load = summarize_text_weight_load(stage_state)
    local_bytes = int(weight_load["loaded_weight_tensor_bytes"])
    checked = bool(dist.is_available() and dist.is_initialized())
    if checked:
        gathered: list[int | None] = [None for _ in range(rank_stage.tp_degree)]
        dist.all_gather_object(gathered, local_bytes, group=rank_stage.stage_group)
        if any(value is None for value in gathered):
            raise RuntimeError(
                "direct TP stage 权重字节数聚合结果不完整，"
                f"stage_idx={rank_stage.stage_idx} rank_local={rank_stage.local_rank}/{rank_stage.tp_degree} "
                f"bytes={gathered}"
            )
        stage_bytes = [int(value) for value in gathered]
        bytes_equal = len(set(stage_bytes)) == 1
        if not bytes_equal:
            raise RuntimeError(
                "direct TP stage 各 rank 加载的本地权重字节数不一致，"
                f"stage_idx={rank_stage.stage_idx} bytes={stage_bytes}"
            )
    else:
        stage_bytes = [local_bytes]
        bytes_equal = None

    stage_state["_tp_stage_loaded_weight_tensor_bytes"] = stage_bytes
    stage_state["_tp_stage_loaded_weight_tensor_bytes_equal"] = bytes_equal
    stage_state["_tp_stage_loaded_weight_tensor_bytes_checked"] = checked


def _need_text_prompt_meta(manifest: TextHybridManifest) -> bool:
    runtime_config = manifest.runtime_config
    return (
        _all_hybrid_stages_are_direct(manifest)
        and str(runtime_config.get("modality", "multimodal")) == "text"
        and str(runtime_config.get("mode", "")) == "generate"
        and not bool(runtime_config.get("include_runtime_reference", True))
    )


def _seed_text_prompt_meta(manifest: TextHybridManifest, *, rank: int) -> None:
    runtime_config = manifest.runtime_config
    if runtime_config.get("_runtime_only_prompt_metadata_ready") or not _need_text_prompt_meta(manifest):
        return

    prompt_metadata = None
    if rank == 0:
        with startup_timer("hybrid-direct-loader", "prepare runtime-only text prompt metadata"):
            prompt_metadata = prepare_text_prompt_meta(runtime_config)

    startup_log(
        "hybrid-direct-loader",
        f"rank={rank} waiting runtime-only text prompt metadata broadcast from src=0",
    )
    prompt_metadata = broadcast_tensor_payload_cpu(
        prompt_metadata,
        src=0,
        label="runtime_only_text_prompt_metadata",
    )
    if prompt_metadata is None or prompt_metadata.get("input_ids") is None:
        raise RuntimeError("runtime-only text prompt metadata 广播后缺少 input_ids。")
    runtime_config["_runtime_only_input_ids"] = prompt_metadata["input_ids"]
    if prompt_metadata.get("attention_mask") is not None:
        runtime_config["_runtime_only_attention_mask"] = prompt_metadata["attention_mask"]
    else:
        runtime_config.pop("_runtime_only_attention_mask", None)
    runtime_config["_runtime_only_prompt_metadata_ready"] = True


def _need_mm_startup_contract(manifest: TextHybridManifest) -> bool:
    runtime_config = manifest.runtime_config
    return (
        _all_hybrid_stages_are_direct(manifest)
        and str(runtime_config.get("modality", "multimodal")) == "multimodal"
    )


def _seed_mm_startup_contract(
    manifest: TextHybridManifest,
    *,
    rank: int,
    local_stage_idx: int,
    rank_stage: HybridRankContext,
) -> None:
    runtime_config = manifest.runtime_config
    if (
        runtime_config.get("_mm_startup_contract_ready")
        or not _need_mm_startup_contract(manifest)
        or rank_stage.local_rank != 0
    ):
        return

    label = f"multimodal_startup_contract stage_idx={int(local_stage_idx)}"
    startup_contract = None
    if rank == 0:
        with startup_timer("hybrid-direct-loader", "prepare multimodal startup payloads"):
            with DirectStageStateBuilder(
                stage_specs=manifest.stages,
                runtime_config=dict(runtime_config),
                include_text_weights=False,
                mm_activate_frontend=True,
            ) as builder:
                startup_meta, startup_tensors = builder.export_mm_startup_transport(
                    local_stage_indices=[int(local_stage_idx)],
                )
                startup_contract = restore_mm_startup_transport(
                    startup_meta,
                    startup_tensors,
                )
                if manifest.stage_rank_groups:
                    leader_stage_indices = [
                        (int(stage_idx), int(stage_ranks[0]))
                        for stage_idx, stage_ranks in enumerate(manifest.stage_rank_groups)
                        if stage_ranks
                    ]
                else:
                    leader_stage_indices = []
                for stage_idx, leader_rank in leader_stage_indices:
                    if leader_rank == rank:
                        continue
                    dst_meta, dst_tensors = builder.export_mm_startup_transport(
                        local_stage_indices=[stage_idx],
                    )
                    send_object_cpu(
                        dst_meta,
                        dst=leader_rank,
                        label=f"multimodal_startup_contract_meta stage_idx={stage_idx}",
                    )
                    send_tensor_payload_cpu(
                        dst_tensors,
                        dst=leader_rank,
                        label=f"multimodal_startup_contract_tensors stage_idx={stage_idx}",
                    )
    else:
        startup_log(
            "hybrid-direct-loader",
            f"rank={rank} waiting {label} from src=0",
        )
        startup_meta = recv_object_cpu(
            src=0,
            label=f"multimodal_startup_contract_meta stage_idx={int(local_stage_idx)}",
        )
        startup_tensors = recv_tensor_payload_cpu(
            src=0,
            label=f"multimodal_startup_contract_tensors stage_idx={int(local_stage_idx)}",
        )
        startup_contract = restore_mm_startup_transport(
            startup_meta,
            startup_tensors,
        )

    seed_mm_startup_runtime_config(
        runtime_config,
        startup_contract,
        local_stage_indices=[int(local_stage_idx)],
    )


def load_stage_state_for_hybrid_rank(
    manifest: TextHybridManifest,
    *,
    rank: int,
    rank_stage: HybridRankContext,
    device: torch.device,
    compute_dtype_arg: str,
) -> tuple[StageState, torch.dtype]:
    stage_meta = manifest.stages[rank_stage.stage_idx]
    all_direct = _all_hybrid_stages_are_direct(manifest)
    runtime_modality = str(manifest.runtime_config.get("modality", "multimodal"))
    _seed_text_prompt_meta(manifest, rank=rank)
    _seed_mm_startup_contract(
        manifest,
        rank=rank,
        local_stage_idx=rank_stage.stage_idx,
        rank_stage=rank_stage,
    )
    use_rank_local_sharded_state = all_direct and rank_stage.tp_degree > 1
    use_single_rank_direct_state = all_direct and rank_stage.tp_degree == 1

    if use_single_rank_direct_state:
        if rank_stage.local_rank != 0:
            raise ValueError(
                "tp_degree=1 的 direct stage 只能有 local_rank=0，"
                f"当前拿到 local_rank={rank_stage.local_rank}"
            )
        startup_log(
            "hybrid-direct-loader",
            f"rank={rank} stage_idx={rank_stage.stage_idx} leader building local direct stage "
            f"range={stage_meta.start_idx}:{stage_meta.end_idx} without stage-group broadcast",
        )
        stage_state = build_direct_stage_state(
            stage_idx=stage_meta.stage_idx,
            start_idx=stage_meta.start_idx,
            end_idx=stage_meta.end_idx,
            runtime_config=manifest.runtime_config,
            mm_activate_frontend=(
                stage_meta.start_idx == 0 if runtime_modality == "multimodal" else None
            ),
        )
        compute_dtype_name = stage_state["save_dtype"] if compute_dtype_arg == "auto" else compute_dtype_arg
        compute_dtype = dtype_from_name(compute_dtype_name)
        startup_log("hybrid-direct-loader", f"rank={rank} entering post-load barrier")
        with startup_timer(
            "hybrid-direct-loader",
            f"post-load barrier rank={rank} stage_idx={rank_stage.stage_idx}",
        ):
            dist.barrier()
        startup_log(
            "hybrid-direct-loader",
            f"rank={rank} stage_idx={rank_stage.stage_idx} moving local direct StageState "
            f"to device={device} compute_dtype={compute_dtype}",
        )
        return move_bundle(stage_state, device, compute_dtype), compute_dtype

    if use_rank_local_sharded_state:
        use_model_input = _use_model_input_broadcast(manifest)
        scaffold_label_prefix = "text_scaffold" if runtime_modality == "text" else "stage_scaffold"
        transport_label_prefix = "runtime_inputs" if use_model_input else scaffold_label_prefix
        scaffold_kind = (
            "model input"
            if use_model_input
            else ("text scaffold" if runtime_modality == "text" else "stage scaffold")
        )
        if rank_stage.local_rank == 0:
            startup_log(
                "hybrid-direct-loader",
                f"rank={rank} stage_idx={rank_stage.stage_idx} leader building shared {scaffold_kind} "
                f"range={stage_meta.start_idx}:{stage_meta.end_idx}",
            )
            if use_model_input:
                leader_scaffold = None
                leader_model_input = _build_model_input_broadcast_payload(
                    manifest.runtime_config,
                    stage_idx=rank_stage.stage_idx,
                    runtime_modality=runtime_modality,
                )
            else:
                leader_scaffold = build_direct_stage_state(
                    stage_idx=stage_meta.stage_idx,
                    start_idx=stage_meta.start_idx,
                    end_idx=stage_meta.end_idx,
                    runtime_config=manifest.runtime_config,
                    include_text_weights=False,
                    mm_activate_frontend=(
                        stage_meta.start_idx == 0 if runtime_modality == "multimodal" else None
                    ),
                )
                leader_scaffold = _compact_hybrid_scaffold_broadcast(leader_scaffold)
                leader_model_input = None
        else:
            leader_scaffold = None
            leader_model_input = None

        startup_log(
            "hybrid-direct-loader",
            f"rank={rank} stage_idx={rank_stage.stage_idx} waiting {scaffold_kind} broadcast from leader="
            f"{rank_stage.leader_rank}",
        )
        transport_meta = None
        transport_tensors = None
        if use_model_input:
            if leader_model_input is not None:
                transport_meta, transport_tensors = pack_model_input_transport(leader_model_input)
        elif leader_scaffold is not None:
            transport_meta, transport_tensors = pack_text_scaffold_transport(leader_scaffold)
        transport_meta = broadcast_object_cpu(
            transport_meta,
            src=rank_stage.leader_rank,
            group=rank_stage.stage_group,
            label=f"{transport_label_prefix}_meta stage_idx={rank_stage.stage_idx}",
        )
        transport_tensors = broadcast_tensor_payload_cpu(
            transport_tensors,
            src=rank_stage.leader_rank,
            group=rank_stage.stage_group,
            label=f"{transport_label_prefix}_tensors stage_idx={rank_stage.stage_idx}",
        )
        if use_model_input:
            model_input = restore_model_input_transport(
                transport_meta,
                transport_tensors,
            )
            HybridRuntimeInputSchema.validate(
                model_input,
                context=f"broadcast stage_idx={rank_stage.stage_idx}",
            )
            compute_dtype = _resolve_scaffold_compute_dtype(
                model_input,
                manifest=manifest,
                stage_meta=stage_meta,
                compute_dtype_arg=compute_dtype_arg,
            )
            scaffold = _restore_stage_state_from_model_input(
                model_input,
                stage_meta=stage_meta,
                runtime_config=manifest.runtime_config,
                compute_dtype=compute_dtype,
            )
        else:
            scaffold = restore_text_scaffold_transport(
                transport_meta,
                transport_tensors,
            )
            compute_dtype = _resolve_scaffold_compute_dtype(
                scaffold,
                manifest=manifest,
                stage_meta=stage_meta,
                compute_dtype_arg=compute_dtype_arg,
            )
            scaffold = _restore_rank_local_scaffold_fields(
                scaffold,
                stage_meta=stage_meta,
                compute_dtype=compute_dtype,
            )
        startup_log(
            "hybrid-direct-loader",
            f"rank={rank} stage_idx={rank_stage.stage_idx} materializing local direct shard "
            f"tp_local_rank={rank_stage.local_rank}/{rank_stage.tp_degree} "
            f"compute_dtype={compute_dtype}",
        )
        with startup_timer(
            "hybrid-direct-loader",
            f"materialize local direct shard rank={rank} stage_idx={rank_stage.stage_idx} "
            f"tp_local_rank={rank_stage.local_rank}/{rank_stage.tp_degree}",
        ):
            stage_state = materialize_text_stage_state(
                stage_state_scaffold=scaffold,
                runtime_config=manifest.runtime_config,
                compute_dtype=compute_dtype,
                tp_shard_rank=rank_stage.local_rank,
                tp_shard_world_size=rank_stage.tp_degree,
            )
        _validate_rank_local_sharded_state(stage_state, rank_stage)
        _record_tp_stage_weight_load_consistency(stage_state, rank_stage)
        startup_log("hybrid-direct-loader", f"rank={rank} entering post-load barrier")
        with startup_timer(
            "hybrid-direct-loader",
            f"post-load barrier rank={rank} stage_idx={rank_stage.stage_idx}",
        ):
            dist.barrier()
        startup_log(
            "hybrid-direct-loader",
            f"rank={rank} stage_idx={rank_stage.stage_idx} moving materialized local StageState "
            f"to device={device} compute_dtype={compute_dtype}",
        )
        return move_bundle(stage_state, device, compute_dtype), compute_dtype
    elif all_direct:
        if rank_stage.local_rank == 0:
            startup_log(
                "hybrid-direct-loader",
                f"rank={rank} stage_idx={rank_stage.stage_idx} leader building local direct stage "
                f"range={stage_meta.start_idx}:{stage_meta.end_idx}",
            )
            leader_state = build_direct_stage_state(
                stage_idx=stage_meta.stage_idx,
                start_idx=stage_meta.start_idx,
                end_idx=stage_meta.end_idx,
                runtime_config=manifest.runtime_config,
                mm_activate_frontend=(
                    stage_meta.start_idx == 0 if runtime_modality == "multimodal" else None
                ),
            )
        else:
            leader_state = None

        startup_log(
            "hybrid-direct-loader",
            f"rank={rank} stage_idx={rank_stage.stage_idx} waiting stage-group broadcast from leader={rank_stage.leader_rank}",
        )
        stage_state = _broadcast_stage_state_transport(
            leader_state,
            src=rank_stage.leader_rank,
            group=rank_stage.stage_group,
            meta_label=f"stage_state_meta stage_idx={rank_stage.stage_idx}",
            tensor_label=f"stage_state_tensors stage_idx={rank_stage.stage_idx}",
        )
        startup_log("hybrid-direct-loader", f"rank={rank} entering post-load barrier")
        with startup_timer(
            "hybrid-direct-loader",
            f"post-load barrier rank={rank} stage_idx={rank_stage.stage_idx}",
        ):
            dist.barrier()
    else:
        replay_bundle_path = getattr(stage_meta, "replay_bundle_path", None) or getattr(
            stage_meta,
            "bundle_path",
            None,
        )
        if replay_bundle_path is None:
            raise RuntimeError(f"replay manifest 缺少 stage_idx={rank_stage.stage_idx} 的 bundle path。")
        if rank_stage.local_rank == 0:
            startup_log(
                "hybrid-direct-loader",
                f"rank={rank} stage_idx={rank_stage.stage_idx} leader loading bundle file {replay_bundle_path}",
            )
        replay_state = load_bundle(replay_bundle_path) if rank_stage.local_rank == 0 else None
        stage_state = _broadcast_stage_state_transport(
            replay_state,
            src=rank_stage.leader_rank,
            group=rank_stage.stage_group,
            meta_label=f"bundle_file_meta stage_idx={rank_stage.stage_idx}",
            tensor_label=f"bundle_file_tensors stage_idx={rank_stage.stage_idx}",
        )

    compute_dtype_name = stage_state["save_dtype"] if compute_dtype_arg == "auto" else compute_dtype_arg
    compute_dtype = dtype_from_name(compute_dtype_name)
    startup_log(
        "hybrid-direct-loader",
        f"rank={rank} stage_idx={rank_stage.stage_idx} moving StageState to device={device} compute_dtype={compute_dtype}",
    )
    return move_bundle(stage_state, device, compute_dtype), compute_dtype


def prepare_text_hybrid(
    *,
    stage_ranges: list[tuple[int, int]],
    tp_degrees: list[int],
    bundle_dir: str,
    manifest_path: str,
    num_frames: int = 8,
    save_dtype: str = "auto",
) -> TextHybridManifest:
    pipeline_manifest = prepare_text_pipeline(
        stage_ranges=stage_ranges,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        num_frames=num_frames,
        save_dtype=save_dtype,
    )
    tp_degrees = parse_tp_degrees(tp_degrees)
    if len(tp_degrees) != pipeline_manifest.num_stages:
        raise ValueError(
            f"stage 数是 {pipeline_manifest.num_stages}，但 TP 度数拿到 {len(tp_degrees)} 个。"
        )

    layout = build_hybrid_layout(tp_degrees)
    manifest = TextHybridManifest.from_pipeline_manifest(pipeline_manifest, layout)
    torch.save(manifest.to_dict(), manifest_path)
    return manifest


def prepare_text_prefill_hybrid(
    *,
    stage_ranges: list[tuple[int, int]],
    tp_degrees: list[int],
    bundle_dir: str,
    manifest_path: str,
    prompt: str = "请用中文简要介绍一下人工智能。",
    save_dtype: str = "auto",
    model_path: str | None = None,
) -> TextHybridManifest:
    pipeline_manifest = prepare_text_prefill_pipeline(
        stage_ranges=stage_ranges,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        prompt=prompt,
        save_dtype=save_dtype,
        model_path=model_path,
    )
    tp_degrees = parse_tp_degrees(tp_degrees)
    if len(tp_degrees) != pipeline_manifest.num_stages:
        raise ValueError(
            f"stage 数是 {pipeline_manifest.num_stages}，但 TP 度数拿到 {len(tp_degrees)} 个。"
        )

    layout = build_hybrid_layout(tp_degrees)
    manifest = TextHybridManifest.from_pipeline_manifest(
        pipeline_manifest,
        layout,
        runtime="text_prefill_hybrid",
    )
    torch.save(manifest.to_dict(), manifest_path)
    return manifest


def prepare_multimodal_prefill_hybrid(
    *,
    stage_ranges: list[tuple[int, int]],
    tp_degrees: list[int],
    bundle_dir: str,
    manifest_path: str,
    num_frames: int = 8,
    save_dtype: str = "auto",
    model_path: str | None = None,
    frame_dir: str | None = None,
) -> TextHybridManifest:
    pipeline_manifest = prepare_multimodal_prefill_pipeline(
        stage_ranges=stage_ranges,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        num_frames=num_frames,
        save_dtype=save_dtype,
        model_path=model_path,
        frame_dir=frame_dir,
    )
    tp_degrees = parse_tp_degrees(tp_degrees)
    if len(tp_degrees) != pipeline_manifest.num_stages:
        raise ValueError(
            f"stage 数是 {pipeline_manifest.num_stages}，但 TP 度数拿到 {len(tp_degrees)} 个。"
        )

    layout = build_hybrid_layout(tp_degrees)
    manifest = TextHybridManifest.from_pipeline_manifest(
        pipeline_manifest,
        layout,
        runtime="multimodal_prefill_hybrid",
    )
    torch.save(manifest.to_dict(), manifest_path)
    return manifest


def prepare_multimodal_decode_hybrid(
    *,
    stage_ranges: list[tuple[int, int]],
    tp_degrees: list[int],
    bundle_dir: str,
    manifest_path: str,
    num_frames: int = 8,
    decode_token_id: int | None = None,
    save_dtype: str = "auto",
    model_path: str | None = None,
    frame_dir: str | None = None,
) -> TextHybridManifest:
    pipeline_manifest = prepare_multimodal_decode_pipeline(
        stage_ranges=stage_ranges,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        num_frames=num_frames,
        decode_token_id=decode_token_id,
        save_dtype=save_dtype,
        model_path=model_path,
        frame_dir=frame_dir,
    )
    tp_degrees = parse_tp_degrees(tp_degrees)
    if len(tp_degrees) != pipeline_manifest.num_stages:
        raise ValueError(
            f"stage 数是 {pipeline_manifest.num_stages}，但 TP 度数拿到 {len(tp_degrees)} 个。"
        )

    layout = build_hybrid_layout(tp_degrees)
    manifest = TextHybridManifest.from_pipeline_manifest(
        pipeline_manifest,
        layout,
        runtime="multimodal_decode_hybrid",
    )
    torch.save(manifest.to_dict(), manifest_path)
    return manifest


def prepare_multimodal_generate_hybrid(
    *,
    stage_ranges: list[tuple[int, int]],
    tp_degrees: list[int],
    bundle_dir: str,
    manifest_path: str,
    num_frames: int = 8,
    max_new_tokens: int = 4,
    save_dtype: str = "auto",
    model_path: str | None = None,
    frame_dir: str | None = None,
) -> TextHybridManifest:
    pipeline_manifest = prepare_multimodal_generate_pipeline(
        stage_ranges=stage_ranges,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        num_frames=num_frames,
        max_new_tokens=max_new_tokens,
        save_dtype=save_dtype,
        model_path=model_path,
        frame_dir=frame_dir,
    )
    tp_degrees = parse_tp_degrees(tp_degrees)
    if len(tp_degrees) != pipeline_manifest.num_stages:
        raise ValueError(
            f"stage 数是 {pipeline_manifest.num_stages}，但 TP 度数拿到 {len(tp_degrees)} 个。"
        )

    layout = build_hybrid_layout(tp_degrees)
    manifest = TextHybridManifest.from_pipeline_manifest(
        pipeline_manifest,
        layout,
        runtime="multimodal_generate_hybrid",
    )
    torch.save(manifest.to_dict(), manifest_path)
    return manifest


def prepare_text_decode_hybrid(
    *,
    stage_ranges: list[tuple[int, int]],
    tp_degrees: list[int],
    bundle_dir: str,
    manifest_path: str,
    prompt: str = "请用中文简要介绍一下人工智能。",
    decode_token_id: int | None = None,
    save_dtype: str = "auto",
    model_path: str | None = None,
) -> TextHybridManifest:
    pipeline_manifest = prepare_text_decode_pipeline(
        stage_ranges=stage_ranges,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        prompt=prompt,
        decode_token_id=decode_token_id,
        save_dtype=save_dtype,
        model_path=model_path,
    )
    tp_degrees = parse_tp_degrees(tp_degrees)
    if len(tp_degrees) != pipeline_manifest.num_stages:
        raise ValueError(
            f"stage 数是 {pipeline_manifest.num_stages}，但 TP 度数拿到 {len(tp_degrees)} 个。"
        )

    layout = build_hybrid_layout(tp_degrees)
    manifest = TextHybridManifest.from_pipeline_manifest(
        pipeline_manifest,
        layout,
        runtime="text_decode_hybrid",
    )
    torch.save(manifest.to_dict(), manifest_path)
    return manifest


def prepare_text_generate_hybrid(
    *,
    stage_ranges: list[tuple[int, int]],
    tp_degrees: list[int],
    bundle_dir: str,
    manifest_path: str,
    prompt: str = "请用中文简要介绍一下人工智能。",
    max_new_tokens: int = 4,
    save_dtype: str = "auto",
    model_path: str | None = None,
) -> TextHybridManifest:
    pipeline_manifest = prepare_text_generate_pipeline(
        stage_ranges=stage_ranges,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        save_dtype=save_dtype,
        model_path=model_path,
    )
    tp_degrees = parse_tp_degrees(tp_degrees)
    if len(tp_degrees) != pipeline_manifest.num_stages:
        raise ValueError(
            f"stage 数是 {pipeline_manifest.num_stages}，但 TP 度数拿到 {len(tp_degrees)} 个。"
        )

    layout = build_hybrid_layout(tp_degrees)
    manifest = TextHybridManifest.from_pipeline_manifest(
        pipeline_manifest,
        layout,
        runtime="text_generate_hybrid",
    )
    torch.save(manifest.to_dict(), manifest_path)
    return manifest


def load_hybrid_manifest(manifest_path: str) -> TextHybridManifest:
    manifest = torch.load(manifest_path, map_location="cpu")
    if isinstance(manifest, TextHybridManifest):
        return manifest

    manifest_dict = manifest.to_dict() if hasattr(manifest, "to_dict") else manifest
    if "tp_degrees" not in manifest_dict:
        raise ValueError("manifest 里没有 tp_degrees，不能按 hybrid 运行。")

    layout = build_hybrid_layout(parse_tp_degrees(manifest_dict["tp_degrees"]))
    if (
        "stage_rank_groups" in manifest_dict
        and manifest_dict["stage_rank_groups"] != layout.stage_rank_groups
    ):
        raise ValueError("manifest 里的 stage_rank_groups 和 tp_degrees 不一致。")

    for key in (
        "stage_rank_groups",
        "pp_rank_groups",
        "send_list",
        "recv_list",
        "send_empty_list",
        "recv_empty_list",
        "world_size",
        "num_stages",
    ):
        if key not in manifest_dict:
            manifest_dict[key] = getattr(layout, key)
    manifest_dict.setdefault("runtime", "text_hybrid")
    return TextHybridManifest.from_dict(manifest_dict)


def init_stage_groups(stage_rank_groups: list[list[int]]) -> list:
    # 所有 rank 按相同顺序建组，避免 group 初始化乱序。
    return [dist.new_group(ranks=ranks) for ranks in stage_rank_groups]


def resolve_rank_stage(
    rank: int,
    stage_rank_groups: list[list[int]],
    stage_groups: list,
    *,
    pp_rank_groups: list[list[int]] | None = None,
    send_list: list[list[int]] | None = None,
    recv_list: list[list[int]] | None = None,
    send_empty_list: list[list[bool]] | None = None,
    recv_empty_list: list[list[bool]] | None = None,
) -> HybridRankContext:
    world_size = sum(len(ranks) for ranks in stage_rank_groups)
    if pp_rank_groups is None:
        pp_rank_groups = build_pp_rank_groups(stage_rank_groups)
    if any(
        value is None for value in (send_list, recv_list, send_empty_list, recv_empty_list)
    ):
        p2p_lists = build_p2p_lists(stage_rank_groups, pp_rank_groups)
        if send_list is None:
            send_list = p2p_lists["send_list"]
        if recv_list is None:
            recv_list = p2p_lists["recv_list"]
        if send_empty_list is None:
            send_empty_list = p2p_lists["send_empty_list"]
        if recv_empty_list is None:
            recv_empty_list = p2p_lists["recv_empty_list"]

    pp_group_index_by_rank = _build_rank_group_index(pp_rank_groups, world_size)
    current_pp_group = pp_rank_groups[pp_group_index_by_rank[rank]]

    for stage_idx, ranks in enumerate(stage_rank_groups):
        if rank in ranks:
            return HybridRankContext(
                stage_idx=stage_idx,
                stage_ranks=ranks,
                tp_degree=len(ranks),
                local_rank=ranks.index(rank),
                leader_rank=ranks[0],
                prev_leader_rank=None if stage_idx == 0 else stage_rank_groups[stage_idx - 1][0],
                next_leader_rank=None if stage_idx + 1 >= len(stage_rank_groups) else stage_rank_groups[stage_idx + 1][0],
                stage_group=stage_groups[stage_idx],
                pp_group_idx=pp_group_index_by_rank[rank],
                current_pp_group=current_pp_group,
                send_list=send_list[rank],
                recv_list=recv_list[rank],
                send_empty_list=send_empty_list[rank],
                recv_empty_list=recv_empty_list[rank],
            )
    raise ValueError(f"rank={rank} 不在任何 stage rank group 里。")


def _recv_stage_handoff_for_rank(
    rank: int,
    rank_stage: HybridRankContext,
    handoff_transport: StageCommunicator,
    stage_state: StageState,
) -> tuple[StageHandoffPayload | None, list[str]]:
    if len(rank_stage.recv_list) != len(rank_stage.recv_empty_list):
        raise ValueError("recv_list 和 recv_empty_list 的长度不一致。")

    incoming_handoff = None
    payload_keys: list[str] = []

    for src, expect_empty in zip(rank_stage.recv_list, rank_stage.recv_empty_list):
        received_message = handoff_transport.recv(src=src, stage_state=stage_state)
        handoff = received_message.handoff

        if expect_empty:
            if handoff is not None:
                raise ValueError(
                    f"rank={rank} 期望从 src={src} 收到空 payload，但拿到了非空 payload。"
                )
            continue

        if handoff is None or handoff.hidden_states is None:
            raise ValueError(f"rank={rank} 没有从 src={src} 收到有效的 hidden_states payload。")
        if rank_stage.local_rank != 0:
            raise ValueError(
                f"rank={rank} 不是 stage leader，但从 src={src} 收到了非空 stage payload。"
            )
        if incoming_handoff is not None:
            raise ValueError(f"rank={rank} 收到了多个非空 stage payload，当前实现只支持一个。")

        incoming_handoff = handoff
        payload_keys = received_message.summary.payload_keys

    return incoming_handoff, payload_keys


def _send_stage_handoff_for_rank(
    rank: int,
    rank_stage: HybridRankContext,
    handoff_transport: StageCommunicator,
    handoff: StageHandoffPayload | None,
) -> tuple[tuple[int, ...] | None, list[str], dict[str, tuple[int, ...] | None]]:
    if len(rank_stage.send_list) != len(rank_stage.send_empty_list):
        raise ValueError("send_list 和 send_empty_list 的长度不一致。")

    non_empty_summary = None
    for dst, is_empty in zip(rank_stage.send_list, rank_stage.send_empty_list):
        if is_empty:
            handoff_transport.send_empty(dst=dst)
            continue

        if handoff is None or handoff.hidden_states is None:
            raise ValueError(f"rank={rank} 需要向 dst={dst} 发送非空 payload，但 handoff 为空。")
        if rank_stage.local_rank != 0:
            raise ValueError(
                f"rank={rank} 不是 stage leader，但被要求向 dst={dst} 发送非空 stage payload。"
            )
        if non_empty_summary is not None:
            raise ValueError(f"rank={rank} 被要求发送多个非空 stage payload，当前实现只支持一个。")

        non_empty_summary = handoff_transport.send(handoff, dst=dst)

    if non_empty_summary is None:
        return None, [], {}

    return (
        non_empty_summary.tensor_shapes.get(StageHandoffPayload.HIDDEN_STATES_KEY),
        non_empty_summary.payload_keys,
        non_empty_summary.tensor_shapes,
    )


def _build_runtime_only_text_generate_phase_state(
    stage_state: StageState,
    *,
    phase_kind: str,
    attention_mask_2d: torch.Tensor,
    config_spec,
    rotary_emb,
    decode_input_ids_buffer: torch.Tensor | None = None,
    mm_dummy_embed_tokens_weight: torch.Tensor | None = None,
) -> StageState:
    query_len = int(stage_state["prefill_seq_len"]) if phase_kind == "prefill" else 1
    runtime_state = dict(stage_state)
    if phase_kind == "decode":
        # The runtime-only StageState keeps the prefill handoff as the local
        # startup seed. Decode must receive/build a fresh one-token input;
        # otherwise TP followers allocate a prefill-sized broadcast buffer.
        for key in (
            "stage_input",
            "layer_input",
            "stage_output",
            "layer_output",
            "hidden_stage_output",
            "norm_output",
            "output_token_id",
        ):
            runtime_state.pop(key, None)
    is_multimodal = str(stage_state.get("modality", "text")) == "multimodal"
    runtime_state["stage_type"] = (
        "text_prefill_last"
        if phase_kind == "prefill" and "final_norm_weight" in stage_state
        else "text"
        if phase_kind == "prefill"
        else "text_decode_last"
        if "final_norm_weight" in stage_state
        else "text_decode"
    )
    runtime_state["attention_mask_2d"] = attention_mask_2d

    if is_multimodal and phase_kind == "prefill":
        runtime_state["attention_mask"] = stage_state["prefill_attention_mask"]
        runtime_state["position_ids"] = stage_state.get("prefill_position_ids")
        runtime_state["cos"] = stage_state["prefill_cos"]
        runtime_state["sin"] = stage_state["prefill_sin"]
    elif is_multimodal and phase_kind == "decode":
        decode_input_ids = (
            torch.zeros(
                (int(stage_state["batch_size"]), 1),
                device=infer_runtime_tensor_device(stage_state),
                dtype=infer_runtime_token_dtype(stage_state),
            )
            if decode_input_ids_buffer is None
            else decode_input_ids_buffer.zero_()
        )
        dummy_embed_tokens_weight = (
            torch.zeros(
                (1, int(config_spec.hidden_size)),
                device=infer_runtime_tensor_device(stage_state),
                dtype=infer_runtime_tensor_dtype(stage_state),
            )
            if mm_dummy_embed_tokens_weight is None
            else mm_dummy_embed_tokens_weight
        )
        decode_state = build_mm_decode_state_from_weights(
            decode_input_ids=decode_input_ids,
            attention_mask_2d=attention_mask_2d,
            past_length=int(attention_mask_2d.shape[-1]) - query_len,
            rope_deltas=stage_state["rope_deltas"],
            embed_tokens_weight=dummy_embed_tokens_weight,
            config_spec=config_spec,
            device=infer_runtime_tensor_device(stage_state),
            compute_dtype=infer_runtime_tensor_dtype(stage_state),
            rotary_emb=rotary_emb,
        )
        runtime_state["attention_mask"] = decode_state.attention_mask
        runtime_state["position_ids"] = decode_state.position_ids
        runtime_state["cos"] = decode_state.cos
        runtime_state["sin"] = decode_state.sin
        runtime_state["visual_pos_masks"] = None
        runtime_state["deepstack_by_layer"] = {}
        runtime_state["deepstack_layer_indices"] = []
    else:
        runtime_aux = build_text_runtime_aux_tensors(
            attention_mask_2d=attention_mask_2d,
            batch_size=int(stage_state["batch_size"]),
            seq_len=query_len,
            past_length=int(attention_mask_2d.shape[-1]) - query_len,
            config_spec=config_spec,
            device=infer_runtime_tensor_device(stage_state),
            compute_dtype=infer_runtime_tensor_dtype(stage_state),
            rotary_emb=rotary_emb,
        )
        runtime_state["attention_mask"] = runtime_aux["attention_mask"]
        runtime_state["position_ids"] = runtime_aux["position_ids"]
        runtime_state["cos"] = runtime_aux["cos"]
        runtime_state["sin"] = runtime_aux["sin"]
    if phase_kind == "prefill" and stage_state.get("input_ids") is not None:
        runtime_state["input_ids"] = stage_state["input_ids"]
    if phase_kind == "decode":
        runtime_state["decode_input_ids"] = (
            torch.zeros(
                (int(stage_state["batch_size"]), 1),
                device=infer_runtime_tensor_device(stage_state),
                dtype=infer_runtime_token_dtype(stage_state),
            )
            if decode_input_ids_buffer is None
            else decode_input_ids_buffer
        )
    return runtime_state


def _run_text_generate_hybrid_phase_impl(
    *,
    rank: int,
    rank_stage: HybridRankContext,
    manifest: TextHybridManifest,
    runtime_state: StageState,
    handoff_transport: StageCommunicator,
    phase_kind: str,
    current_token_id: int | None,
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None,
    comm_dtype: torch.dtype,
    tp_attn_math_mode: str,
    tp_mlp_math_mode: str,
    return_tensor: bool,
) -> tuple[dict, dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None]:
    is_first_stage = rank_stage.stage_idx == 0
    is_last_stage = rank_stage.stage_idx == manifest.num_stages - 1

    reference_input = runtime_state.get("stage_input")
    if reference_input is None:
        reference_input = runtime_state.get("layer_input")
    incoming_handoff, received_payload_keys = _recv_stage_handoff_for_rank(
        rank,
        rank_stage,
        handoff_transport,
        runtime_state,
    )
    if incoming_handoff is not None:
        runtime_state = apply_stage_handoff_payload(runtime_state, incoming_handoff)

    if is_first_stage:
        if rank_stage.local_rank == 0:
            if phase_kind == "prefill":
                if runtime_state.get("embed_tokens_weight") is not None and "input_ids" in runtime_state:
                    leader_input = forward_text_embeddings(runtime_state["input_ids"], runtime_state)
                else:
                    leader_input = reference_input
            elif phase_kind == "decode":
                if current_token_id is None:
                    raise ValueError("decode phase 需要 current_token_id，但当前拿到 None。")
                decode_input_ids = fill_decode_input_ids(
                    runtime_state["decode_input_ids"],
                    current_token_id,
                )
                leader_input = forward_text_embeddings(decode_input_ids, runtime_state)
                runtime_state["decode_input_ids_runtime"] = decode_input_ids
            else:
                raise ValueError(f"不支持的 phase_kind={phase_kind!r}")
        else:
            leader_input = None
        boundary_max = None
        boundary_mean = None
    else:
        if rank_stage.local_rank == 0:
            if incoming_handoff is None or incoming_handoff.hidden_states is None:
                raise ValueError(f"rank={rank} 是非首 stage leader，但没有收到非空 stage payload。")
            leader_input = get_stage_input(runtime_state)
        else:
            leader_input = None
        boundary_max, boundary_mean = None, None

    stage_input = broadcast_cpu(
        reference_tensor=(
            reference_input
            if reference_input is not None
            else build_runtime_only_stage_input_template(
                runtime_state,
                query_len=(int(runtime_state["prefill_seq_len"]) if phase_kind == "prefill" else 1),
            )
        ),
        tensor=leader_input,
        src=rank_stage.leader_rank,
        comm_dtype=comm_dtype,
        group=rank_stage.stage_group,
        profile_context={
            "phase": phase_kind,
            "module": "runtime_input",
            "reason": "stage_input_broadcast",
            "stage_idx": rank_stage.stage_idx,
        },
    )
    if not is_first_stage and reference_input is not None:
        boundary_max, boundary_mean = (
            (stage_input - reference_input).abs().max().item(),
            (stage_input - reference_input).abs().mean().item(),
        )

    embedding_max = None
    embedding_mean = None
    if is_first_stage and reference_input is not None:
        embedding_diff = (stage_input - reference_input).abs()
        embedding_max = embedding_diff.max().item()
        embedding_mean = embedding_diff.mean().item()

    hidden_stage_max = None
    hidden_stage_mean = None
    norm_max = None
    norm_mean = None
    predicted_token_id = None
    reference_token_id = None
    updated_cache = cache_by_layer

    if phase_kind == "prefill":
        if is_last_stage:
            prefill_runtime_state = strip_runtime_layer_cache(runtime_state)
            trace = trace_text_decode_logits_tp_with_runtime_cache(
                stage_input,
                prefill_runtime_state,
                rank_stage.local_rank,
                rank_stage.tp_degree,
                comm_dtype,
                tp_group=rank_stage.stage_group,
                tp_src_rank=rank_stage.leader_rank,
                attn_math_mode=tp_attn_math_mode,
                mlp_math_mode=tp_mlp_math_mode,
                cache_by_layer={},
                profile_phase=phase_kind,
            )
            stage_output = trace["logits"]
            if runtime_state.get("hidden_stage_output") is not None:
                hidden_stage_diff = (trace["stage_output"] - runtime_state["hidden_stage_output"]).abs()
                hidden_stage_max = hidden_stage_diff.max().item()
                hidden_stage_mean = hidden_stage_diff.mean().item()
            if runtime_state.get("norm_output") is not None:
                norm_diff = (trace["norm_output"] - runtime_state["norm_output"]).abs()
                norm_max = norm_diff.max().item()
                norm_mean = norm_diff.mean().item()
            updated_cache = trace["cache_by_layer"]
        else:
            prefill_runtime_state = strip_runtime_layer_cache(runtime_state)
            trace = trace_text_decode_stage_tp_with_runtime_cache(
                stage_input,
                prefill_runtime_state,
                rank_stage.local_rank,
                rank_stage.tp_degree,
                comm_dtype,
                tp_group=rank_stage.stage_group,
                tp_src_rank=rank_stage.leader_rank,
                attn_math_mode=tp_attn_math_mode,
                mlp_math_mode=tp_mlp_math_mode,
                cache_by_layer={},
                profile_phase=phase_kind,
            )
            stage_output = trace["stage_output"]
            updated_cache = trace["cache_by_layer"]
    elif phase_kind == "decode":
        if is_last_stage:
            trace = trace_text_decode_logits_tp_with_runtime_cache(
                stage_input,
                runtime_state,
                rank_stage.local_rank,
                rank_stage.tp_degree,
                comm_dtype,
                tp_group=rank_stage.stage_group,
                tp_src_rank=rank_stage.leader_rank,
                attn_math_mode=tp_attn_math_mode,
                mlp_math_mode=tp_mlp_math_mode,
                cache_by_layer=cache_by_layer,
                profile_phase=phase_kind,
            )
            stage_output = trace["logits"]
            updated_cache = trace["cache_by_layer"]
            if runtime_state.get("hidden_stage_output") is not None:
                hidden_stage_diff = (trace["stage_output"] - runtime_state["hidden_stage_output"]).abs()
                hidden_stage_max = hidden_stage_diff.max().item()
                hidden_stage_mean = hidden_stage_diff.mean().item()
            if runtime_state.get("norm_output") is not None:
                norm_diff = (trace["norm_output"] - runtime_state["norm_output"]).abs()
                norm_max = norm_diff.max().item()
                norm_mean = norm_diff.mean().item()
        else:
            trace = trace_text_decode_stage_tp_with_runtime_cache(
                stage_input,
                runtime_state,
                rank_stage.local_rank,
                rank_stage.tp_degree,
                comm_dtype,
                tp_group=rank_stage.stage_group,
                tp_src_rank=rank_stage.leader_rank,
                attn_math_mode=tp_attn_math_mode,
                mlp_math_mode=tp_mlp_math_mode,
                cache_by_layer=cache_by_layer,
                profile_phase=phase_kind,
            )
            stage_output = trace["stage_output"]
            updated_cache = trace["cache_by_layer"]
    else:
        raise ValueError(f"不支持的 phase_kind={phase_kind!r}")

    reference_output = runtime_state.get("stage_output")
    if reference_output is None:
        reference_output = runtime_state.get("layer_output")
    if reference_output is None:
        stage_max, stage_mean = None, None
    else:
        stage_diff = (stage_output - reference_output).abs()
        stage_max = stage_diff.max().item()
        stage_mean = stage_diff.mean().item()

    sent_shape = None
    sent_payload_keys: list[str] = []
    sent_tensor_shapes: dict[str, tuple[int, ...] | None] = {}
    if any(not is_empty for is_empty in rank_stage.send_empty_list):
        next_stage_range = None
        if rank_stage.next_leader_rank is not None:
            next_stage_meta = manifest.stages[rank_stage.stage_idx + 1]
            next_stage_range = (next_stage_meta.start_idx, next_stage_meta.end_idx)
        outgoing_handoff = build_stage_handoff_payload(
            stage_output,
            runtime_state,
            target_stage_range=next_stage_range,
        )
    else:
        outgoing_handoff = None
    sent_shape, sent_payload_keys, sent_tensor_shapes = _send_stage_handoff_for_rank(
        rank,
        rank_stage,
        handoff_transport,
        outgoing_handoff,
    )

    if is_last_stage and rank_stage.local_rank == 0:
        predicted_token_id = int(stage_output[0, -1].argmax().item())
        if runtime_state.get("output_token_id") is not None:
            reference_token_id = int(runtime_state["output_token_id"])

    stats = {
        "input_shape": tuple(stage_input.shape),
        "output_shape": tuple(stage_output.shape),
        "boundary_max_diff": boundary_max,
        "boundary_mean_diff": boundary_mean,
        "embedding_max_diff": embedding_max,
        "embedding_mean_diff": embedding_mean,
        "hidden_stage_max_diff": hidden_stage_max,
        "hidden_stage_mean_diff": hidden_stage_mean,
        "norm_max_diff": norm_max,
        "norm_mean_diff": norm_mean,
        "stage_max_diff": stage_max,
        "stage_mean_diff": stage_mean,
        "sent_shape": sent_shape,
        "received_payload_keys": received_payload_keys,
        "sent_payload_keys": sent_payload_keys,
        "sent_tensor_shapes": sent_tensor_shapes,
        "predicted_token_id": predicted_token_id,
        "reference_token_id": reference_token_id,
    }
    if return_tensor and is_last_stage and rank_stage.local_rank == 0:
        stats["stage_output_tensor"] = stage_output
    return stats, updated_cache


def _run_text_generate_hybrid_phase(
    *,
    rank: int,
    rank_stage: HybridRankContext,
    manifest: TextHybridManifest,
    runtime_state: StageState,
    handoff_transport: StageCommunicator,
    phase_kind: str,
    current_token_id: int | None,
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None,
    comm_dtype: torch.dtype,
    tp_attn_math_mode: str,
    tp_mlp_math_mode: str,
    return_tensor: bool,
) -> tuple[dict, dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None]:
    return _run_text_generate_hybrid_phase_impl(
        rank=rank,
        rank_stage=rank_stage,
        manifest=manifest,
        runtime_state=runtime_state,
        handoff_transport=handoff_transport,
        phase_kind=phase_kind,
        current_token_id=current_token_id,
        cache_by_layer=cache_by_layer,
        comm_dtype=comm_dtype,
        tp_attn_math_mode=tp_attn_math_mode,
        tp_mlp_math_mode=tp_mlp_math_mode,
        return_tensor=return_tensor,
    )


def _run_text_generate_hybrid_rank(
    *,
    rank: int,
    world_size: int,
    manifest: TextHybridManifest,
    device: torch.device,
    compute_dtype_arg: str,
    comm_dtype_arg: str,
    tp_attn_math_mode: str,
    tp_mlp_math_mode: str,
    return_tensors: bool,
) -> dict:
    if world_size != manifest.world_size:
        raise ValueError(f"WORLD_SIZE={world_size}，但 hybrid manifest 需要 {manifest.world_size}。")

    stage_groups = init_stage_groups(manifest.stage_rank_groups)
    rank_stage = resolve_rank_stage(
        rank,
        manifest.stage_rank_groups,
        stage_groups,
        pp_rank_groups=manifest.pp_rank_groups,
        send_list=manifest.send_list,
        recv_list=manifest.recv_list,
        send_empty_list=manifest.send_empty_list,
        recv_empty_list=manifest.recv_empty_list,
    )

    stage_state, compute_dtype = load_stage_state_for_hybrid_rank(
        manifest,
        rank=rank,
        rank_stage=rank_stage,
        device=device,
        compute_dtype_arg=compute_dtype_arg,
    )
    comm_dtype = resolve_comm_dtype(comm_dtype_arg, compute_dtype)
    handoff_transport = StageCommunicator(device=device, comm_dtype=comm_dtype)
    runtime_only_generate = is_runtime_only_generate_state(stage_state)
    runtime_only_context = None
    if runtime_only_generate:
        config_spec = load_text_model_config_spec(manifest.runtime_config["model_path"])
        runtime_only_context = {
            "config_spec": config_spec,
            "rotary_emb": build_text_rotary_embedding(config_spec, device=device),
        }

    if runtime_only_generate:
        prefill_state = _build_runtime_only_text_generate_phase_state(
            stage_state,
            phase_kind="prefill",
            attention_mask_2d=stage_state["prefill_attention_mask_2d"],
            config_spec=runtime_only_context["config_spec"],
            rotary_emb=runtime_only_context["rotary_emb"],
        )
    else:
        prefill_state = build_generate_phase_state(
            stage_state,
            stage_state["prefill"],
            stage_type=("text_prefill_last" if rank_stage.stage_idx == manifest.num_stages - 1 else "text"),
        )
    prefill_stats, prefill_cache = _run_text_generate_hybrid_phase(
        rank=rank,
        rank_stage=rank_stage,
        manifest=manifest,
        runtime_state=prefill_state,
        handoff_transport=handoff_transport,
        phase_kind="prefill",
        current_token_id=None,
        cache_by_layer=None,
        comm_dtype=comm_dtype,
        tp_attn_math_mode=tp_attn_math_mode,
        tp_mlp_math_mode=tp_mlp_math_mode,
        return_tensor=return_tensors,
    )

    current_token_id = broadcast_token_id(
        prefill_stats["predicted_token_id"] if rank_stage.stage_idx == manifest.num_stages - 1 and rank_stage.local_rank == 0 else None,
        src=manifest.stage_rank_groups[-1][0],
    )
    generated_token_ids = [current_token_id]
    cache_by_layer = prefill_cache if prefill_cache is not None else build_generate_cache_map(stage_state)
    step_stats = []
    step_output_tensors = []
    attention_mask_buffer = None
    decode_input_ids_buffer = None
    mm_dummy_embed_tokens_weight = None
    if runtime_only_generate:
        attention_mask_buffer = build_decode_attention_mask_buffer(
            stage_state["prefill_attention_mask_2d"],
            max_new_tokens=int(stage_state["max_new_tokens"]),
        )
        decode_input_ids_buffer = torch.empty(
            (int(stage_state["batch_size"]), 1),
            device=infer_runtime_tensor_device(stage_state),
            dtype=infer_runtime_token_dtype(stage_state),
        )
        if str(stage_state.get("modality", "text")) == "multimodal":
            mm_dummy_embed_tokens_weight = torch.zeros(
                (1, int(runtime_only_context["config_spec"].hidden_size)),
                device=infer_runtime_tensor_device(stage_state),
                dtype=infer_runtime_tensor_dtype(stage_state),
            )

    decode_iterable = (
        range(int(stage_state["max_new_tokens"]) - 1)
        if runtime_only_generate
        else stage_state["decode_steps"]
    )
    for step_payload in decode_iterable:
        if runtime_only_generate:
            current_attention_mask_2d = decode_attention_mask_view(
                attention_mask_buffer,
                prefill_seq_len=int(stage_state["prefill_seq_len"]),
                step_idx=int(step_payload),
            )
            decode_state = _build_runtime_only_text_generate_phase_state(
                stage_state,
                phase_kind="decode",
                attention_mask_2d=current_attention_mask_2d,
                config_spec=runtime_only_context["config_spec"],
                rotary_emb=runtime_only_context["rotary_emb"],
                decode_input_ids_buffer=decode_input_ids_buffer,
                mm_dummy_embed_tokens_weight=mm_dummy_embed_tokens_weight,
            )
        else:
            decode_state = build_generate_phase_state(
                stage_state,
                step_payload,
                stage_type=("text_decode_last" if rank_stage.stage_idx == manifest.num_stages - 1 else "text_decode"),
            )
        current_step_stats, cache_by_layer = _run_text_generate_hybrid_phase(
            rank=rank,
            rank_stage=rank_stage,
            manifest=manifest,
            runtime_state=decode_state,
            handoff_transport=handoff_transport,
            phase_kind="decode",
            current_token_id=current_token_id,
            cache_by_layer=cache_by_layer,
            comm_dtype=comm_dtype,
            tp_attn_math_mode=tp_attn_math_mode,
            tp_mlp_math_mode=tp_mlp_math_mode,
            return_tensor=return_tensors,
        )
        current_token_id = broadcast_token_id(
            current_step_stats["predicted_token_id"] if rank_stage.stage_idx == manifest.num_stages - 1 and rank_stage.local_rank == 0 else None,
            src=manifest.stage_rank_groups[-1][0],
        )
        generated_token_ids.append(current_token_id)
        if "stage_output_tensor" in current_step_stats:
            step_output_tensors.append(current_step_stats.pop("stage_output_tensor"))
        step_stats.append(current_step_stats)

    stats = {
        "rank": rank,
        "stage_idx": rank_stage.stage_idx,
        "stage_ranks": rank_stage.stage_ranks,
        "local_rank": rank_stage.local_rank,
        "tp_degree": rank_stage.tp_degree,
        "leader_rank": rank_stage.leader_rank,
        "pp_group_idx": rank_stage.pp_group_idx,
        "current_pp_group": rank_stage.current_pp_group,
        "num_stages": manifest.num_stages,
        "start_idx": stage_state["start_idx"],
        "end_idx": stage_state["end_idx"],
        "num_layers": len(stage_state["layers"]),
        "weight_load": summarize_text_weight_load(stage_state),
        "device": str(device),
        "comm_dtype": str(comm_dtype),
        "tp_attn_math_mode": tp_attn_math_mode,
        "tp_mlp_math_mode": tp_mlp_math_mode,
        "prefill_seq_len": int(stage_state["prefill_seq_len"]),
        "max_new_tokens": int(stage_state["max_new_tokens"]),
        "prefill": prefill_stats,
        "steps": step_stats,
        "generated_token_ids": generated_token_ids,
        "reference_generated_token_ids": (
            None
            if runtime_only_generate or stage_state.get("generated_token_ids") is None
            else token_tensor_to_list(stage_state["generated_token_ids"])
        ),
    }
    if return_tensors and rank_stage.stage_idx == manifest.num_stages - 1 and rank_stage.local_rank == 0:
        stats["prefill_output_tensor"] = prefill_stats.pop("stage_output_tensor")
        stats["step_output_tensors"] = step_output_tensors
    return stats


class TextHybridRunner:
    """Stateful rank runner for HexGen-style hybrid PP+TP execution."""

    def __init__(
        self,
        manifest: TextHybridManifest,
        device: torch.device,
        compute_dtype_arg: str,
        comm_dtype_arg: str,
        tp_attn_math_mode: str = "orig",
        tp_mlp_math_mode: str = "orig",
        debug_config: TpDebugConfig | None = None,
        return_tensors: bool = False,
    ) -> None:
        self.manifest = manifest
        self.device = device
        self.compute_dtype_arg = compute_dtype_arg
        self.comm_dtype_arg = comm_dtype_arg
        self.tp_attn_math_mode = tp_attn_math_mode
        self.tp_mlp_math_mode = tp_mlp_math_mode
        self.debug_config = debug_config or TpDebugConfig()
        self.return_tensors = return_tensors

    def run_rank(self, rank: int, world_size: int) -> dict:
        return run_text_hybrid_rank(
            rank=rank,
            world_size=world_size,
            manifest=self.manifest,
            device=self.device,
            compute_dtype_arg=self.compute_dtype_arg,
            comm_dtype_arg=self.comm_dtype_arg,
            tp_attn_math_mode=self.tp_attn_math_mode,
            tp_mlp_math_mode=self.tp_mlp_math_mode,
            debug_config=self.debug_config,
            return_tensors=self.return_tensors,
        )


def run_text_hybrid_rank(
    *,
    rank: int,
    world_size: int,
    manifest: TextHybridManifest,
    device: torch.device,
    compute_dtype_arg: str,
    comm_dtype_arg: str,
    tp_attn_math_mode: str = "orig",
    tp_mlp_math_mode: str = "orig",
    debug_config: TpDebugConfig | None = None,
    return_tensors: bool = False,
) -> dict:
    debug_config = debug_config or TpDebugConfig()
    if manifest.pipeline_type in {"text_generate", "multimodal_generate"}:
        return _run_text_generate_hybrid_rank(
            rank=rank,
            world_size=world_size,
            manifest=manifest,
            device=device,
            compute_dtype_arg=compute_dtype_arg,
            comm_dtype_arg=comm_dtype_arg,
            tp_attn_math_mode=tp_attn_math_mode,
            tp_mlp_math_mode=tp_mlp_math_mode,
            return_tensors=return_tensors,
        )

    if world_size != manifest.world_size:
        raise ValueError(f"WORLD_SIZE={world_size}，但 hybrid manifest 需要 {manifest.world_size}。")

    stage_groups = init_stage_groups(manifest.stage_rank_groups)
    rank_stage = resolve_rank_stage(
        rank,
        manifest.stage_rank_groups,
        stage_groups,
        pp_rank_groups=manifest.pp_rank_groups,
        send_list=manifest.send_list,
        recv_list=manifest.recv_list,
        send_empty_list=manifest.send_empty_list,
        recv_empty_list=manifest.recv_empty_list,
    )

    stage_state, compute_dtype = load_stage_state_for_hybrid_rank(
        manifest,
        rank=rank,
        rank_stage=rank_stage,
        device=device,
        compute_dtype_arg=compute_dtype_arg,
    )
    comm_dtype = resolve_comm_dtype(comm_dtype_arg, compute_dtype)
    handoff_transport = StageCommunicator(device=device, comm_dtype=comm_dtype)

    reference_input = get_stage_input(stage_state)
    incoming_handoff, received_payload_keys = _recv_stage_handoff_for_rank(
        rank,
        rank_stage,
        handoff_transport,
        stage_state,
    )
    if incoming_handoff is not None:
        stage_state = apply_stage_handoff_payload(stage_state, incoming_handoff)

    if rank_stage.stage_idx == 0:
        leader_input = reference_input if rank_stage.local_rank == 0 else None
    elif rank_stage.local_rank == 0:
        if incoming_handoff is None or incoming_handoff.hidden_states is None:
            raise ValueError(f"rank={rank} 是非首 stage leader，但没有收到非空 stage payload。")
        leader_input = get_stage_input(stage_state)
    else:
        leader_input = None

    stage_input = broadcast_cpu(
        reference_tensor=reference_input,
        tensor=leader_input,
        src=rank_stage.leader_rank,
        comm_dtype=comm_dtype,
        group=rank_stage.stage_group,
        profile_context={
            "phase": "prefill",
            "module": "runtime_input",
            "reason": "stage_input_broadcast",
            "stage_idx": rank_stage.stage_idx,
        },
    )

    reference_output = stage_state.get("stage_output")
    tp_stage_stats = run_stage_state_tp(
        stage_input=stage_input,
        stage_state=stage_state,
        reference_input_override=reference_input,
        local_rank=rank_stage.local_rank,
        tp_degree=rank_stage.tp_degree,
        comm_dtype=comm_dtype,
        tp_group=rank_stage.stage_group,
        leader_rank=rank_stage.leader_rank,
        tp_attn_math_mode=tp_attn_math_mode,
        tp_mlp_math_mode=tp_mlp_math_mode,
        debug_config=debug_config,
    )
    stage_output = tp_stage_stats.pop("stage_output")

    if any(not is_empty for is_empty in rank_stage.send_empty_list):
        next_stage_range = None
        if rank_stage.next_leader_rank is not None:
            next_stage_meta = manifest.stages[rank_stage.stage_idx + 1]
            next_stage_range = (next_stage_meta.start_idx, next_stage_meta.end_idx)
        outgoing_handoff = build_stage_handoff_payload(
            stage_output,
            stage_state,
            target_stage_range=next_stage_range,
        )
    else:
        outgoing_handoff = None
    sent_shape, sent_payload_keys, sent_tensor_shapes = _send_stage_handoff_for_rank(
        rank,
        rank_stage,
        handoff_transport,
        outgoing_handoff,
    )

    stats = {
        "rank": rank,
        "stage_idx": rank_stage.stage_idx,
        "stage_ranks": rank_stage.stage_ranks,
        "local_rank": rank_stage.local_rank,
        "tp_degree": rank_stage.tp_degree,
        "leader_rank": rank_stage.leader_rank,
        "pp_group_idx": rank_stage.pp_group_idx,
        "current_pp_group": rank_stage.current_pp_group,
        "num_stages": manifest.num_stages,
        "start_idx": stage_state["start_idx"],
        "end_idx": stage_state["end_idx"],
        "num_layers": len(stage_state["layers"]),
        "weight_load": summarize_text_weight_load(stage_state),
        "device": str(device),
        "comm_dtype": str(comm_dtype),
        "tp_attn_math_mode": tp_attn_math_mode,
        "tp_mlp_math_mode": tp_mlp_math_mode,
        "input_shape": tuple(stage_input.shape),
        "output_shape": tuple(stage_output.shape),
        "sent_shape": sent_shape,
        "received_payload_keys": received_payload_keys,
        "sent_payload_keys": sent_payload_keys,
        "sent_tensor_shapes": sent_tensor_shapes,
        "boundary_max_diff": tp_stage_stats["boundary_max_diff"],
        "boundary_mean_diff": tp_stage_stats["boundary_mean_diff"],
        "direct_max_diff": tp_stage_stats["direct_max_diff"],
        "direct_mean_diff": tp_stage_stats["direct_mean_diff"],
        "stage_max_diff": tp_stage_stats["stage_max_diff"],
        "stage_mean_diff": tp_stage_stats["stage_mean_diff"],
        "tp_direct_max_diff": tp_stage_stats["tp_direct_max_diff"],
        "tp_direct_mean_diff": tp_stage_stats["tp_direct_mean_diff"],
        "trace_summary": tp_stage_stats["trace_summary"],
        "next_leader_rank": rank_stage.next_leader_rank,
        "send_list": rank_stage.send_list,
        "recv_list": rank_stage.recv_list,
        "send_empty_list": rank_stage.send_empty_list,
        "recv_empty_list": rank_stage.recv_empty_list,
        "traces": tp_stage_stats["traces"],
        "outlier_dump": tp_stage_stats["outlier_dump"],
    }
    if return_tensors and rank_stage.local_rank == 0:
        stats["stage_output"] = stage_output
        stats["reference_output"] = reference_output
    return stats


DIRECT_RUNTIME_EXPORTS = [
    "TextHybridManifest",
    "HybridRankContext",
    "init_stage_groups",
    "resolve_rank_stage",
    "TextHybridRunner",
    "load_stage_state_for_hybrid_rank",
    "run_text_hybrid_rank",
]

LEGACY_REPLAY_EXPORTS = [
    "prepare_text_hybrid",
    "prepare_multimodal_decode_hybrid",
    "prepare_multimodal_generate_hybrid",
    "prepare_multimodal_prefill_hybrid",
    "prepare_text_decode_hybrid",
    "prepare_text_generate_hybrid",
    "prepare_text_prefill_hybrid",
    "load_hybrid_manifest",
]

__all__ = [*DIRECT_RUNTIME_EXPORTS]
