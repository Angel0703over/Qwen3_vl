"""Dedicated hybrid-parallel runtime for HexGen-style PP+TP stage execution."""

import torch
import torch.distributed as dist

from qwen3vl_tp_runtime.hexgen_core.distributed import (
    broadcast_cpu,
    broadcast_object_cpu,
    startup_log,
    startup_timer,
)
from qwen3vl_tp_runtime.hexgen_core.gen_hetero_groups import (
    build_hybrid_layout,
    build_p2p_lists,
    build_pp_rank_groups,
    parse_tp_degrees,
)
from qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel import (
    prepare_multimodal_decode_pipeline,
    prepare_multimodal_generate_pipeline,
    prepare_multimodal_prefill_pipeline,
    prepare_text_decode_pipeline,
    prepare_text_generate_pipeline,
    prepare_text_prefill_pipeline,
    prepare_text_pipeline,
)
from qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel import (
    build_stage_traces,
    run_text_tensor_parallel_stage,
)
from qwen3vl_tp_runtime.hexgen_core.schema import HybridRankContext, StageHandoffPayload, TextHybridManifest
from qwen3vl_tp_runtime.hexgen_core.stage import (
    apply_stage_handoff_payload,
    build_stage_handoff_payload,
    get_stage_input,
    get_stage_output,
    run_stage_tp,
)
from qwen3vl_tp_runtime.hexgen_core.transport import StageHandoffTransport
from qwen3vl_tp_runtime.models.qwen3vl.execution import (
    forward_text_embeddings,
    trace_text_decode_logits_tp_with_runtime_cache,
    trace_text_decode_stage_tp_with_runtime_cache,
)
from qwen3vl_tp_runtime.models.qwen3vl.functional import dtype_from_name, resolve_comm_dtype
from qwen3vl_tp_runtime.models.qwen3vl.runtime_builder import (
    build_direct_stage_bundle,
    compact_runtime_only_text_prompt_metadata_for_broadcast,
    materialize_direct_text_stage_bundle_from_scaffold,
    prepare_runtime_only_text_generate_prompt_metadata,
    restore_runtime_only_text_prompt_metadata_from_broadcast,
)
from qwen3vl_tp_runtime.models.qwen3vl.capture import load_bundle, move_bundle
from qwen3vl_tp_runtime.models.qwen3vl.weights import (
    build_text_rotary_embedding,
    build_text_runtime_aux_tensors,
    load_text_model_config_spec,
)


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
    return all(stage.bundle_path is None for stage in manifest.stages)


def _should_seed_runtime_only_text_prompt_metadata(manifest: TextHybridManifest) -> bool:
    runtime_config = manifest.runtime_config
    return (
        _all_hybrid_stages_are_direct(manifest)
        and str(runtime_config.get("modality", "multimodal")) == "text"
        and str(runtime_config.get("mode", "")) == "generate"
        and not bool(runtime_config.get("include_runtime_reference", True))
    )


def _seed_runtime_only_text_prompt_metadata(manifest: TextHybridManifest, *, rank: int) -> None:
    runtime_config = manifest.runtime_config
    if runtime_config.get("_runtime_only_prompt_metadata_ready") or not _should_seed_runtime_only_text_prompt_metadata(
        manifest
    ):
        return

    prompt_metadata = None
    if rank == 0:
        with startup_timer("hybrid-direct-loader", "prepare runtime-only text prompt metadata"):
            prompt_metadata = prepare_runtime_only_text_generate_prompt_metadata(runtime_config)
        prompt_metadata = compact_runtime_only_text_prompt_metadata_for_broadcast(prompt_metadata)

    startup_log(
        "hybrid-direct-loader",
        f"rank={rank} waiting runtime-only text prompt metadata broadcast from src=0",
    )
    prompt_metadata = broadcast_object_cpu(
        prompt_metadata,
        src=0,
        label="runtime_only_text_prompt_metadata",
    )
    prompt_metadata = restore_runtime_only_text_prompt_metadata_from_broadcast(prompt_metadata)
    runtime_config["_runtime_only_input_ids"] = prompt_metadata["input_ids"]
    if prompt_metadata.get("attention_mask") is not None:
        runtime_config["_runtime_only_attention_mask"] = prompt_metadata["attention_mask"]
    else:
        runtime_config.pop("_runtime_only_attention_mask", None)
    runtime_config["_runtime_only_prompt_metadata_ready"] = True


def load_stage_bundle_for_hybrid_rank(
    manifest: TextHybridManifest,
    *,
    rank: int,
    rank_stage: HybridRankContext,
    device: torch.device,
    compute_dtype_arg: str,
    ) -> tuple[dict, torch.dtype]:
    stage_meta = manifest.stages[rank_stage.stage_idx]
    all_direct = _all_hybrid_stages_are_direct(manifest)
    runtime_modality = str(manifest.runtime_config.get("modality", "multimodal"))
    _seed_runtime_only_text_prompt_metadata(manifest, rank=rank)
    use_rank_local_text_tp_bundle = (
        all_direct
        and runtime_modality == "text"
        and rank_stage.tp_degree > 1
    )
    use_single_rank_direct_bundle = all_direct and rank_stage.tp_degree == 1

    if use_single_rank_direct_bundle:
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
        bundle = build_direct_stage_bundle(
            stage_idx=stage_meta.stage_idx,
            start_idx=stage_meta.start_idx,
            end_idx=stage_meta.end_idx,
            runtime_config=manifest.runtime_config,
        )
        compute_dtype_name = bundle["save_dtype"] if compute_dtype_arg == "auto" else compute_dtype_arg
        compute_dtype = dtype_from_name(compute_dtype_name)
        startup_log("hybrid-direct-loader", f"rank={rank} entering post-load barrier")
        dist.barrier()
        startup_log(
            "hybrid-direct-loader",
            f"rank={rank} stage_idx={rank_stage.stage_idx} moving local direct bundle "
            f"to device={device} compute_dtype={compute_dtype}",
        )
        return move_bundle(bundle, device, compute_dtype), compute_dtype

    if use_rank_local_text_tp_bundle:
        if rank_stage.local_rank == 0:
            startup_log(
                "hybrid-direct-loader",
                f"rank={rank} stage_idx={rank_stage.stage_idx} leader building shared text scaffold "
                f"range={stage_meta.start_idx}:{stage_meta.end_idx}",
            )
            leader_scaffold = build_direct_stage_bundle(
                stage_idx=stage_meta.stage_idx,
                start_idx=stage_meta.start_idx,
                end_idx=stage_meta.end_idx,
                runtime_config=manifest.runtime_config,
                include_text_weights=False,
            )
        else:
            leader_scaffold = None

        startup_log(
            "hybrid-direct-loader",
            f"rank={rank} stage_idx={rank_stage.stage_idx} waiting text scaffold broadcast from leader="
            f"{rank_stage.leader_rank}",
        )
        scaffold = broadcast_object_cpu(
            leader_scaffold,
            src=rank_stage.leader_rank,
            group=rank_stage.stage_group,
            label=f"text_scaffold stage_idx={rank_stage.stage_idx}",
        )
        compute_dtype_name = scaffold["save_dtype"] if compute_dtype_arg == "auto" else compute_dtype_arg
        compute_dtype = dtype_from_name(compute_dtype_name)
        startup_log(
            "hybrid-direct-loader",
            f"rank={rank} stage_idx={rank_stage.stage_idx} materializing local text shard "
            f"tp_local_rank={rank_stage.local_rank}/{rank_stage.tp_degree} "
            f"compute_dtype={compute_dtype}",
        )
        bundle = materialize_direct_text_stage_bundle_from_scaffold(
            stage_bundle_scaffold=scaffold,
            runtime_config=manifest.runtime_config,
            compute_dtype=compute_dtype,
            tp_shard_rank=rank_stage.local_rank,
            tp_shard_world_size=rank_stage.tp_degree,
        )
        startup_log("hybrid-direct-loader", f"rank={rank} entering post-load barrier")
        dist.barrier()
        startup_log(
            "hybrid-direct-loader",
            f"rank={rank} stage_idx={rank_stage.stage_idx} moving materialized local bundle "
            f"to device={device} compute_dtype={compute_dtype}",
        )
        return move_bundle(bundle, device, compute_dtype), compute_dtype
    elif all_direct:
        if rank_stage.local_rank == 0:
            startup_log(
                "hybrid-direct-loader",
                f"rank={rank} stage_idx={rank_stage.stage_idx} leader building local direct stage "
                f"range={stage_meta.start_idx}:{stage_meta.end_idx}",
            )
            leader_bundle = build_direct_stage_bundle(
                stage_idx=stage_meta.stage_idx,
                start_idx=stage_meta.start_idx,
                end_idx=stage_meta.end_idx,
                runtime_config=manifest.runtime_config,
            )
        else:
            leader_bundle = None

        startup_log(
            "hybrid-direct-loader",
            f"rank={rank} stage_idx={rank_stage.stage_idx} waiting stage-group broadcast from leader={rank_stage.leader_rank}",
        )
        bundle = broadcast_object_cpu(
            leader_bundle,
            src=rank_stage.leader_rank,
            group=rank_stage.stage_group,
            label=f"stage_bundle stage_idx={rank_stage.stage_idx}",
        )
        startup_log("hybrid-direct-loader", f"rank={rank} entering post-load barrier")
        dist.barrier()
    else:
        if rank_stage.local_rank == 0:
            startup_log(
                "hybrid-direct-loader",
                f"rank={rank} stage_idx={rank_stage.stage_idx} leader loading bundle file {stage_meta.bundle_path}",
            )
        leader_bundle = load_bundle(stage_meta.bundle_path) if rank_stage.local_rank == 0 else None
        bundle = broadcast_object_cpu(
            leader_bundle,
            src=rank_stage.leader_rank,
            group=rank_stage.stage_group,
            label=f"bundle-file stage_idx={rank_stage.stage_idx}",
        )

    compute_dtype_name = bundle["save_dtype"] if compute_dtype_arg == "auto" else compute_dtype_arg
    compute_dtype = dtype_from_name(compute_dtype_name)
    startup_log(
        "hybrid-direct-loader",
        f"rank={rank} stage_idx={rank_stage.stage_idx} moving stage bundle to device={device} compute_dtype={compute_dtype}",
    )
    return move_bundle(bundle, device, compute_dtype), compute_dtype


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
    handoff_transport: StageHandoffTransport,
    bundle: dict,
) -> tuple[StageHandoffPayload | None, list[str]]:
    if len(rank_stage.recv_list) != len(rank_stage.recv_empty_list):
        raise ValueError("recv_list 和 recv_empty_list 的长度不一致。")

    incoming_handoff = None
    payload_keys: list[str] = []

    for src, expect_empty in zip(rank_stage.recv_list, rank_stage.recv_empty_list):
        received_message = handoff_transport.recv(src=src, stage_bundle=bundle)
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
    handoff_transport: StageHandoffTransport,
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


def _build_generate_phase_bundle(
    stage_bundle: dict,
    phase_payload: dict,
    *,
    stage_type: str,
) -> dict:
    runtime_bundle = {
        key: value
        for key, value in stage_bundle.items()
        if key not in {"prefill", "decode_steps", "generated_token_ids"}
    }
    runtime_bundle.update(phase_payload)
    runtime_bundle["stage_type"] = stage_type
    if stage_type in {"text_decode", "text_decode_last"}:
        runtime_bundle["visual_pos_masks"] = phase_payload.get("visual_pos_masks")
        runtime_bundle["deepstack_by_layer"] = dict(phase_payload.get("deepstack_by_layer", {}))
        runtime_bundle["deepstack_layer_indices"] = list(phase_payload.get("deepstack_layer_indices", []))
    if "layer_input" not in runtime_bundle and "stage_input" in runtime_bundle:
        runtime_bundle["layer_input"] = runtime_bundle["stage_input"]
    return runtime_bundle


def _strip_runtime_layer_cache(stage_bundle: dict) -> dict:
    stripped_bundle = dict(stage_bundle)
    stripped_bundle["layers"] = [
        {
            key: value
            for key, value in layer_bundle.items()
            if key not in {"past_key", "past_value"}
        }
        for layer_bundle in stage_bundle["layers"]
    ]
    return stripped_bundle


def _build_generate_cache_map(stage_bundle: dict) -> dict[int, tuple[torch.Tensor | None, torch.Tensor | None]]:
    return {
        int(layer_bundle["layer_idx"]): (
            layer_bundle.get("past_key"),
            layer_bundle.get("past_value"),
        )
        for layer_bundle in stage_bundle["layers"]
    }


def _is_runtime_only_generate_bundle(stage_bundle: dict) -> bool:
    return bool(stage_bundle.get("runtime_only_generate"))


def _infer_runtime_tensor_device(stage_bundle: dict) -> torch.device:
    if "embed_tokens_weight" in stage_bundle and stage_bundle["embed_tokens_weight"] is not None:
        return stage_bundle["embed_tokens_weight"].device
    if stage_bundle.get("layers"):
        return stage_bundle["layers"][0]["q_weight"].device
    if "final_norm_weight" in stage_bundle and stage_bundle["final_norm_weight"] is not None:
        return stage_bundle["final_norm_weight"].device
    if stage_bundle.get("input_ids") is not None:
        return stage_bundle["input_ids"].device
    return stage_bundle["prefill_attention_mask_2d"].device


def _infer_runtime_tensor_dtype(stage_bundle: dict) -> torch.dtype:
    if "embed_tokens_weight" in stage_bundle and stage_bundle["embed_tokens_weight"] is not None:
        return stage_bundle["embed_tokens_weight"].dtype
    if stage_bundle.get("layers"):
        return stage_bundle["layers"][0]["q_weight"].dtype
    if "final_norm_weight" in stage_bundle and stage_bundle["final_norm_weight"] is not None:
        return stage_bundle["final_norm_weight"].dtype
    return torch.float32


def _infer_runtime_token_dtype(stage_bundle: dict) -> torch.dtype:
    if stage_bundle.get("input_ids") is not None:
        return stage_bundle["input_ids"].dtype
    token_id_dtype = stage_bundle.get("token_id_dtype")
    if isinstance(token_id_dtype, str):
        return dtype_from_name(token_id_dtype)
    return torch.int64


def _build_runtime_only_stage_input_template(stage_bundle: dict, *, query_len: int) -> torch.Tensor:
    batch_size = int(stage_bundle["batch_size"])
    hidden_size = int(stage_bundle["hidden_size"])
    return torch.empty(
        (batch_size, query_len, hidden_size),
        device=_infer_runtime_tensor_device(stage_bundle),
        dtype=_infer_runtime_tensor_dtype(stage_bundle),
    )


def _build_runtime_only_text_generate_phase_bundle(
    stage_bundle: dict,
    *,
    phase_kind: str,
    attention_mask_2d: torch.Tensor,
    config_spec,
    rotary_emb,
) -> dict:
    query_len = int(stage_bundle["prefill_seq_len"]) if phase_kind == "prefill" else 1
    runtime_bundle = dict(stage_bundle)
    runtime_bundle["stage_type"] = (
        "text_prefill_last"
        if phase_kind == "prefill" and "final_norm_weight" in stage_bundle
        else "text"
        if phase_kind == "prefill"
        else "text_decode_last"
        if "final_norm_weight" in stage_bundle
        else "text_decode"
    )
    runtime_bundle["attention_mask_2d"] = attention_mask_2d
    runtime_aux = build_text_runtime_aux_tensors(
        attention_mask_2d=attention_mask_2d,
        batch_size=int(stage_bundle["batch_size"]),
        seq_len=query_len,
        past_length=int(attention_mask_2d.shape[-1]) - query_len,
        config_spec=config_spec,
        device=_infer_runtime_tensor_device(stage_bundle),
        compute_dtype=_infer_runtime_tensor_dtype(stage_bundle),
        rotary_emb=rotary_emb,
    )
    runtime_bundle["attention_mask"] = runtime_aux["attention_mask"]
    runtime_bundle["position_ids"] = runtime_aux["position_ids"]
    runtime_bundle["cos"] = runtime_aux["cos"]
    runtime_bundle["sin"] = runtime_aux["sin"]
    if phase_kind == "prefill" and stage_bundle.get("input_ids") is not None:
        runtime_bundle["input_ids"] = stage_bundle["input_ids"]
    if phase_kind == "decode":
        runtime_bundle["decode_input_ids"] = torch.zeros(
            (int(stage_bundle["batch_size"]), 1),
            device=_infer_runtime_tensor_device(stage_bundle),
            dtype=_infer_runtime_token_dtype(stage_bundle),
        )
    return runtime_bundle


def _token_tensor_to_list(token_tensor: torch.Tensor) -> list[int]:
    if token_tensor.dim() == 2:
        token_tensor = token_tensor[0]
    return [int(token_id) for token_id in token_tensor.tolist()]


def _broadcast_token_id(token_id: int | None, *, src: int) -> int:
    token_tensor = torch.tensor(
        [-1 if token_id is None else token_id],
        dtype=torch.int64,
    )
    dist.broadcast(token_tensor, src=src)
    return int(token_tensor.item())


def _run_text_generate_hybrid_phase(
    *,
    rank: int,
    rank_stage: HybridRankContext,
    manifest: TextHybridManifest,
    runtime_bundle: dict,
    handoff_transport: StageHandoffTransport,
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

    reference_input = runtime_bundle.get("stage_input")
    if reference_input is None:
        reference_input = runtime_bundle.get("layer_input")
    incoming_handoff, received_payload_keys = _recv_stage_handoff_for_rank(
        rank,
        rank_stage,
        handoff_transport,
        runtime_bundle,
    )
    if incoming_handoff is not None:
        runtime_bundle = apply_stage_handoff_payload(runtime_bundle, incoming_handoff)

    if is_first_stage:
        if rank_stage.local_rank == 0:
            if phase_kind == "prefill":
                if runtime_bundle.get("embed_tokens_weight") is not None and "input_ids" in runtime_bundle:
                    leader_input = forward_text_embeddings(runtime_bundle["input_ids"], runtime_bundle)
                else:
                    leader_input = reference_input
            elif phase_kind == "decode":
                if current_token_id is None:
                    raise ValueError("decode phase 需要 current_token_id，但当前拿到 None。")
                decode_input_ids = torch.tensor(
                    [[current_token_id]],
                    device=_infer_runtime_tensor_device(runtime_bundle),
                    dtype=runtime_bundle["decode_input_ids"].dtype,
                )
                leader_input = forward_text_embeddings(decode_input_ids, runtime_bundle)
                runtime_bundle["decode_input_ids_runtime"] = decode_input_ids
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
            leader_input = get_stage_input(runtime_bundle)
        else:
            leader_input = None
        boundary_max, boundary_mean = None, None

    stage_input = broadcast_cpu(
        reference_tensor=(
            reference_input
            if reference_input is not None
            else _build_runtime_only_stage_input_template(
                runtime_bundle,
                query_len=(int(runtime_bundle["prefill_seq_len"]) if phase_kind == "prefill" else 1),
            )
        ),
        tensor=leader_input,
        src=rank_stage.leader_rank,
        comm_dtype=comm_dtype,
        group=rank_stage.stage_group,
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
            prefill_runtime_bundle = _strip_runtime_layer_cache(runtime_bundle)
            trace = trace_text_decode_logits_tp_with_runtime_cache(
                stage_input,
                prefill_runtime_bundle,
                rank_stage.local_rank,
                rank_stage.tp_degree,
                comm_dtype,
                tp_group=rank_stage.stage_group,
                tp_src_rank=rank_stage.leader_rank,
                attn_math_mode=tp_attn_math_mode,
                mlp_math_mode=tp_mlp_math_mode,
                cache_by_layer={},
            )
            stage_output = trace["logits"]
            if runtime_bundle.get("hidden_stage_output") is not None:
                hidden_stage_diff = (trace["stage_output"] - runtime_bundle["hidden_stage_output"]).abs()
                hidden_stage_max = hidden_stage_diff.max().item()
                hidden_stage_mean = hidden_stage_diff.mean().item()
            if runtime_bundle.get("norm_output") is not None:
                norm_diff = (trace["norm_output"] - runtime_bundle["norm_output"]).abs()
                norm_max = norm_diff.max().item()
                norm_mean = norm_diff.mean().item()
            updated_cache = trace["cache_by_layer"]
        else:
            prefill_runtime_bundle = _strip_runtime_layer_cache(runtime_bundle)
            trace = trace_text_decode_stage_tp_with_runtime_cache(
                stage_input,
                prefill_runtime_bundle,
                rank_stage.local_rank,
                rank_stage.tp_degree,
                comm_dtype,
                tp_group=rank_stage.stage_group,
                tp_src_rank=rank_stage.leader_rank,
                attn_math_mode=tp_attn_math_mode,
                mlp_math_mode=tp_mlp_math_mode,
                cache_by_layer={},
            )
            stage_output = trace["stage_output"]
            updated_cache = trace["cache_by_layer"]
    elif phase_kind == "decode":
        if is_last_stage:
            trace = trace_text_decode_logits_tp_with_runtime_cache(
                stage_input,
                runtime_bundle,
                rank_stage.local_rank,
                rank_stage.tp_degree,
                comm_dtype,
                tp_group=rank_stage.stage_group,
                tp_src_rank=rank_stage.leader_rank,
                attn_math_mode=tp_attn_math_mode,
                mlp_math_mode=tp_mlp_math_mode,
                cache_by_layer=cache_by_layer,
            )
            stage_output = trace["logits"]
            updated_cache = trace["cache_by_layer"]
            if runtime_bundle.get("hidden_stage_output") is not None:
                hidden_stage_diff = (trace["stage_output"] - runtime_bundle["hidden_stage_output"]).abs()
                hidden_stage_max = hidden_stage_diff.max().item()
                hidden_stage_mean = hidden_stage_diff.mean().item()
            if runtime_bundle.get("norm_output") is not None:
                norm_diff = (trace["norm_output"] - runtime_bundle["norm_output"]).abs()
                norm_max = norm_diff.max().item()
                norm_mean = norm_diff.mean().item()
        else:
            trace = trace_text_decode_stage_tp_with_runtime_cache(
                stage_input,
                runtime_bundle,
                rank_stage.local_rank,
                rank_stage.tp_degree,
                comm_dtype,
                tp_group=rank_stage.stage_group,
                tp_src_rank=rank_stage.leader_rank,
                attn_math_mode=tp_attn_math_mode,
                mlp_math_mode=tp_mlp_math_mode,
                cache_by_layer=cache_by_layer,
            )
            stage_output = trace["stage_output"]
            updated_cache = trace["cache_by_layer"]
    else:
        raise ValueError(f"不支持的 phase_kind={phase_kind!r}")

    reference_output = runtime_bundle.get("stage_output")
    if reference_output is None:
        reference_output = runtime_bundle.get("layer_output")
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
            runtime_bundle,
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
        if runtime_bundle.get("output_token_id") is not None:
            reference_token_id = int(runtime_bundle["output_token_id"])

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

    stage_bundle, compute_dtype = load_stage_bundle_for_hybrid_rank(
        manifest,
        rank=rank,
        rank_stage=rank_stage,
        device=device,
        compute_dtype_arg=compute_dtype_arg,
    )
    comm_dtype = resolve_comm_dtype(comm_dtype_arg, compute_dtype)
    handoff_transport = StageHandoffTransport(device=device, comm_dtype=comm_dtype)
    runtime_only_generate = _is_runtime_only_generate_bundle(stage_bundle)
    runtime_only_context = None
    if runtime_only_generate:
        config_spec = load_text_model_config_spec(manifest.runtime_config["model_path"])
        runtime_only_context = {
            "config_spec": config_spec,
            "rotary_emb": build_text_rotary_embedding(config_spec, device=device),
        }

    if runtime_only_generate:
        prefill_bundle = _build_runtime_only_text_generate_phase_bundle(
            stage_bundle,
            phase_kind="prefill",
            attention_mask_2d=stage_bundle["prefill_attention_mask_2d"],
            config_spec=runtime_only_context["config_spec"],
            rotary_emb=runtime_only_context["rotary_emb"],
        )
    else:
        prefill_bundle = _build_generate_phase_bundle(
            stage_bundle,
            stage_bundle["prefill"],
            stage_type=("text_prefill_last" if rank_stage.stage_idx == manifest.num_stages - 1 else "text"),
        )
    prefill_stats, prefill_cache = _run_text_generate_hybrid_phase(
        rank=rank,
        rank_stage=rank_stage,
        manifest=manifest,
        runtime_bundle=prefill_bundle,
        handoff_transport=handoff_transport,
        phase_kind="prefill",
        current_token_id=None,
        cache_by_layer=None,
        comm_dtype=comm_dtype,
        tp_attn_math_mode=tp_attn_math_mode,
        tp_mlp_math_mode=tp_mlp_math_mode,
        return_tensor=return_tensors,
    )

    current_token_id = _broadcast_token_id(
        prefill_stats["predicted_token_id"] if rank_stage.stage_idx == manifest.num_stages - 1 and rank_stage.local_rank == 0 else None,
        src=manifest.stage_rank_groups[-1][0],
    )
    generated_token_ids = [current_token_id]
    cache_by_layer = prefill_cache if prefill_cache is not None else _build_generate_cache_map(stage_bundle)
    step_stats = []
    step_output_tensors = []
    current_attention_mask_2d = stage_bundle["prefill_attention_mask_2d"]

    decode_iterable = (
        range(int(stage_bundle["max_new_tokens"]) - 1)
        if runtime_only_generate
        else stage_bundle["decode_steps"]
    )
    for step_payload in decode_iterable:
        if runtime_only_generate:
            current_attention_mask_2d = torch.cat(
                [
                    current_attention_mask_2d,
                    torch.ones(
                        (current_attention_mask_2d.shape[0], 1),
                        device=current_attention_mask_2d.device,
                        dtype=current_attention_mask_2d.dtype,
                    ),
                ],
                dim=-1,
            )
            decode_bundle = _build_runtime_only_text_generate_phase_bundle(
                stage_bundle,
                phase_kind="decode",
                attention_mask_2d=current_attention_mask_2d,
                config_spec=runtime_only_context["config_spec"],
                rotary_emb=runtime_only_context["rotary_emb"],
            )
        else:
            decode_bundle = _build_generate_phase_bundle(
                stage_bundle,
                step_payload,
                stage_type=("text_decode_last" if rank_stage.stage_idx == manifest.num_stages - 1 else "text_decode"),
            )
        current_step_stats, cache_by_layer = _run_text_generate_hybrid_phase(
            rank=rank,
            rank_stage=rank_stage,
            manifest=manifest,
            runtime_bundle=decode_bundle,
            handoff_transport=handoff_transport,
            phase_kind="decode",
            current_token_id=current_token_id,
            cache_by_layer=cache_by_layer,
            comm_dtype=comm_dtype,
            tp_attn_math_mode=tp_attn_math_mode,
            tp_mlp_math_mode=tp_mlp_math_mode,
            return_tensor=return_tensors,
        )
        current_token_id = _broadcast_token_id(
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
        "start_idx": stage_bundle["start_idx"],
        "end_idx": stage_bundle["end_idx"],
        "num_layers": len(stage_bundle["layers"]),
        "device": str(device),
        "comm_dtype": str(comm_dtype),
        "tp_attn_math_mode": tp_attn_math_mode,
        "tp_mlp_math_mode": tp_mlp_math_mode,
        "prefill_seq_len": int(stage_bundle["prefill_seq_len"]),
        "max_new_tokens": int(stage_bundle["max_new_tokens"]),
        "prefill": prefill_stats,
        "steps": step_stats,
        "generated_token_ids": generated_token_ids,
        "reference_generated_token_ids": (
            None
            if runtime_only_generate or stage_bundle.get("generated_token_ids") is None
            else _token_tensor_to_list(stage_bundle["generated_token_ids"])
        ),
    }
    if return_tensors and rank_stage.stage_idx == manifest.num_stages - 1 and rank_stage.local_rank == 0:
        stats["prefill_output_tensor"] = prefill_stats.pop("stage_output_tensor")
        stats["step_output_tensors"] = step_output_tensors
        if not runtime_only_generate:
            stats["reference_prefill_output_tensor"] = stage_bundle["prefill"]["logits"]
            stats["reference_step_output_tensors"] = [
                step_payload["logits"] for step_payload in stage_bundle["decode_steps"]
            ]
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
        compare_direct: bool = False,
        trace_layers: bool = False,
        dump_layer: int | None = None,
        dump_topk: int = 5,
        return_tensors: bool = False,
    ) -> None:
        self.manifest = manifest
        self.device = device
        self.compute_dtype_arg = compute_dtype_arg
        self.comm_dtype_arg = comm_dtype_arg
        self.tp_attn_math_mode = tp_attn_math_mode
        self.tp_mlp_math_mode = tp_mlp_math_mode
        self.compare_direct = compare_direct
        self.trace_layers = trace_layers
        self.dump_layer = dump_layer
        self.dump_topk = dump_topk
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
            compare_direct=self.compare_direct,
            trace_layers=self.trace_layers,
            dump_layer=self.dump_layer,
            dump_topk=self.dump_topk,
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
    compare_direct: bool = False,
    trace_layers: bool = False,
    dump_layer: int | None = None,
    dump_topk: int = 5,
    return_tensors: bool = False,
) -> dict:
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

    bundle, compute_dtype = load_stage_bundle_for_hybrid_rank(
        manifest,
        rank=rank,
        rank_stage=rank_stage,
        device=device,
        compute_dtype_arg=compute_dtype_arg,
    )
    comm_dtype = resolve_comm_dtype(comm_dtype_arg, compute_dtype)
    handoff_transport = StageHandoffTransport(device=device, comm_dtype=comm_dtype)

    reference_input = get_stage_input(bundle)
    incoming_handoff, received_payload_keys = _recv_stage_handoff_for_rank(
        rank,
        rank_stage,
        handoff_transport,
        bundle,
    )
    if incoming_handoff is not None:
        bundle = apply_stage_handoff_payload(bundle, incoming_handoff)

    if rank_stage.stage_idx == 0:
        leader_input = reference_input if rank_stage.local_rank == 0 else None
    elif rank_stage.local_rank == 0:
        if incoming_handoff is None or incoming_handoff.hidden_states is None:
            raise ValueError(f"rank={rank} 是非首 stage leader，但没有收到非空 stage payload。")
        leader_input = get_stage_input(bundle)
    else:
        leader_input = None

    stage_input = broadcast_cpu(
        reference_tensor=reference_input,
        tensor=leader_input,
        src=rank_stage.leader_rank,
        comm_dtype=comm_dtype,
        group=rank_stage.stage_group,
    )

    reference_output = bundle.get("stage_output")
    tp_stage_stats = run_text_tensor_parallel_stage(
        stage_input=stage_input,
        bundle=bundle,
        reference_input_override=reference_input,
        local_rank=rank_stage.local_rank,
        tp_degree=rank_stage.tp_degree,
        comm_dtype=comm_dtype,
        tp_group=rank_stage.stage_group,
        leader_rank=rank_stage.leader_rank,
        tp_attn_math_mode=tp_attn_math_mode,
        tp_mlp_math_mode=tp_mlp_math_mode,
        compare_direct=compare_direct,
        trace_layers=trace_layers,
        dump_layer=dump_layer,
        dump_topk=dump_topk,
    )
    stage_output = tp_stage_stats.pop("stage_output")

    if any(not is_empty for is_empty in rank_stage.send_empty_list):
        next_stage_range = None
        if rank_stage.next_leader_rank is not None:
            next_stage_meta = manifest.stages[rank_stage.stage_idx + 1]
            next_stage_range = (next_stage_meta.start_idx, next_stage_meta.end_idx)
        outgoing_handoff = build_stage_handoff_payload(
            stage_output,
            bundle,
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
        "start_idx": bundle["start_idx"],
        "end_idx": bundle["end_idx"],
        "num_layers": len(bundle["layers"]),
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


__all__ = [
    "TextHybridManifest",
    "HybridRankContext",
    "prepare_text_hybrid",
    "prepare_multimodal_decode_hybrid",
    "prepare_multimodal_generate_hybrid",
    "prepare_multimodal_prefill_hybrid",
    "prepare_text_decode_hybrid",
    "prepare_text_generate_hybrid",
    "prepare_text_prefill_hybrid",
    "load_hybrid_manifest",
    "init_stage_groups",
    "resolve_rank_stage",
    "build_stage_traces",
    "TextHybridRunner",
    "run_text_hybrid_rank",
]
