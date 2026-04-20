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
)
from qwen3vl_tp_runtime.hexgen_core.transport import StageHandoffTransport
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


class TextHybridRunner:
    """Stateful rank runner for HexGen-style hybrid PP+TP execution."""

    def __init__(
        self,
        manifest: TextHybridManifest,
        device: torch.device,
        compute_dtype_arg: str,
        comm_dtype_arg: str,
        tp_attn_math_mode: str = "orig",
        tp_mlp_math_mode: str = "float32",
        compare_direct: bool = False,
        trace_layers: bool = False,
        dump_layer: int | None = None,
        dump_topk: int = 5,
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
    tp_mlp_math_mode: str = "float32",
    compare_direct: bool = False,
    trace_layers: bool = False,
    dump_layer: int | None = None,
    dump_topk: int = 5,
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

    return {
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


__all__ = [
    "TextHybridManifest",
    "HybridRankContext",
    "prepare_text_hybrid",
    "load_hybrid_manifest",
    "init_stage_groups",
    "resolve_rank_stage",
    "build_stage_traces",
    "TextHybridRunner",
    "run_text_hybrid_rank",
]
