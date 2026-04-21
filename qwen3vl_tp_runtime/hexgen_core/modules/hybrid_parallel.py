"""Dedicated hybrid-parallel runtime for HexGen-style PP+TP stage execution."""

import torch
import torch.distributed as dist

from qwen3vl_tp_runtime.hexgen_core.distributed import broadcast_cpu
from qwen3vl_tp_runtime.hexgen_core.gen_hetero_groups import (
    build_hybrid_layout,
    build_p2p_lists,
    build_pp_rank_groups,
    parse_tp_degrees,
)
from qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel import (
    load_stage_bundle_by_index,
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
from qwen3vl_tp_runtime.models.qwen3vl.forward import (
    forward_text_embeddings,
    trace_text_decode_logits_tp_with_runtime_cache,
    trace_text_decode_stage_tp_with_runtime_cache,
    trace_text_prefill_stage_logits_tp,
)
from qwen3vl_tp_runtime.models.qwen3vl.ops import resolve_comm_dtype


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
    if "layer_input" not in runtime_bundle and "stage_input" in runtime_bundle:
        runtime_bundle["layer_input"] = runtime_bundle["stage_input"]
    return runtime_bundle


def _build_generate_cache_map(stage_bundle: dict) -> dict[int, tuple[torch.Tensor | None, torch.Tensor | None]]:
    return {
        int(layer_bundle["layer_idx"]): (
            layer_bundle.get("past_key"),
            layer_bundle.get("past_value"),
        )
        for layer_bundle in stage_bundle["layers"]
    }


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

    reference_input = get_stage_input(runtime_bundle)
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
                leader_input = forward_text_embeddings(runtime_bundle["input_ids"], runtime_bundle)
            elif phase_kind == "decode":
                if current_token_id is None:
                    raise ValueError("decode phase 需要 current_token_id，但当前拿到 None。")
                decode_input_ids = torch.tensor(
                    [[current_token_id]],
                    device=reference_input.device,
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
        reference_tensor=reference_input,
        tensor=leader_input,
        src=rank_stage.leader_rank,
        comm_dtype=comm_dtype,
        group=rank_stage.stage_group,
    )
    if not is_first_stage:
        boundary_max, boundary_mean = (
            (stage_input - reference_input).abs().max().item(),
            (stage_input - reference_input).abs().mean().item(),
        )

    embedding_max = None
    embedding_mean = None
    if is_first_stage:
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
            trace = trace_text_prefill_stage_logits_tp(
                stage_input,
                runtime_bundle,
                rank_stage.local_rank,
                rank_stage.tp_degree,
                comm_dtype,
                tp_group=rank_stage.stage_group,
                tp_src_rank=rank_stage.leader_rank,
                attn_math_mode=tp_attn_math_mode,
                mlp_math_mode=tp_mlp_math_mode,
            )
            stage_output = trace["logits"]
            hidden_stage_diff = (trace["hidden_stage_output"] - runtime_bundle["hidden_stage_output"]).abs()
            hidden_stage_max = hidden_stage_diff.max().item()
            hidden_stage_mean = hidden_stage_diff.mean().item()
            norm_diff = (trace["norm_output"] - runtime_bundle["norm_output"]).abs()
            norm_max = norm_diff.max().item()
            norm_mean = norm_diff.mean().item()
        else:
            stage_output = run_stage_tp(
                stage_input,
                runtime_bundle,
                rank_stage.local_rank,
                rank_stage.tp_degree,
                comm_dtype,
                tp_group=rank_stage.stage_group,
                tp_src_rank=rank_stage.leader_rank,
                tp_attn_math_mode=tp_attn_math_mode,
                tp_mlp_math_mode=tp_mlp_math_mode,
            )
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
            hidden_stage_diff = (trace["stage_output"] - runtime_bundle["hidden_stage_output"]).abs()
            hidden_stage_max = hidden_stage_diff.max().item()
            hidden_stage_mean = hidden_stage_diff.mean().item()
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

    reference_output = get_stage_output(runtime_bundle)
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

    stage_bundle, compute_dtype = load_stage_bundle_by_index(
        manifest,
        rank_stage.stage_idx,
        device,
        compute_dtype_arg,
    )
    comm_dtype = resolve_comm_dtype(comm_dtype_arg, compute_dtype)
    handoff_transport = StageHandoffTransport(device=device, comm_dtype=comm_dtype)

    prefill_bundle = _build_generate_phase_bundle(
        stage_bundle,
        stage_bundle["prefill"],
        stage_type=("text_prefill_last" if rank_stage.stage_idx == manifest.num_stages - 1 else "text"),
    )
    prefill_stats, _ = _run_text_generate_hybrid_phase(
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
    cache_by_layer = _build_generate_cache_map(stage_bundle)
    step_stats = []
    step_output_tensors = []

    for step_payload in stage_bundle["decode_steps"]:
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
        "reference_generated_token_ids": _token_tensor_to_list(stage_bundle["generated_token_ids"]),
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
    if manifest.pipeline_type == "text_generate":
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

    bundle, compute_dtype = load_stage_bundle_by_index(
        manifest,
        rank_stage.stage_idx,
        device,
        compute_dtype_arg,
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
