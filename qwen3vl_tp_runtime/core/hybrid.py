import torch
import torch.distributed as dist

from qwen3vl_tp_runtime.hexgen_core.gen_hetero_groups import (
    build_hybrid_layout,
    build_p2p_lists,
    build_pp_rank_groups,
    build_stage_rank_groups,
    parse_tp_degrees,
)
from qwen3vl_tp_runtime.core.dist import broadcast_cpu
from qwen3vl_tp_runtime.core.pipeline import (
    load_pipeline_manifest,
    load_stage_bundle_by_index,
    prepare_text_pipeline,
    tensor_diff_stats,
)
from qwen3vl_tp_runtime.core.ops import resolve_comm_dtype
from qwen3vl_tp_runtime.core.stage import (
    get_stage_input,
    get_stage_output,
    run_stage,
    run_stage_tp,
    trace_stage,
    trace_stage_tp,
)
from qwen3vl_tp_runtime.core.transport import recv_hidden_states, send_hidden_states


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
) -> dict:
    pipeline_manifest = prepare_text_pipeline(
        stage_ranges=stage_ranges,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        num_frames=num_frames,
        save_dtype=save_dtype,
    )
    tp_degrees = parse_tp_degrees(tp_degrees)
    if len(tp_degrees) != pipeline_manifest["num_stages"]:
        raise ValueError(
            f"stage 数是 {pipeline_manifest['num_stages']}，但 TP 度数拿到 {len(tp_degrees)} 个。"
        )

    layout = build_hybrid_layout(tp_degrees)
    manifest = dict(pipeline_manifest)
    manifest["runtime"] = "text_hybrid"
    manifest["tp_degrees"] = layout["tp_degrees"]
    manifest["stage_rank_groups"] = layout["stage_rank_groups"]
    manifest["pp_rank_groups"] = layout["pp_rank_groups"]
    manifest["send_list"] = layout["send_list"]
    manifest["recv_list"] = layout["recv_list"]
    manifest["send_empty_list"] = layout["send_empty_list"]
    manifest["recv_empty_list"] = layout["recv_empty_list"]
    manifest["world_size"] = layout["world_size"]
    torch.save(manifest, manifest_path)
    return manifest


def load_hybrid_manifest(manifest_path: str) -> dict:
    manifest = load_pipeline_manifest(manifest_path)
    if "tp_degrees" not in manifest:
        raise ValueError("manifest 里没有 tp_degrees，不能按 hybrid 运行。")

    layout = build_hybrid_layout(parse_tp_degrees(manifest["tp_degrees"]))
    if "stage_rank_groups" in manifest and manifest["stage_rank_groups"] != layout["stage_rank_groups"]:
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
        manifest.setdefault(key, layout[key])
    return manifest


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
) -> dict:
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
            return {
                "stage_idx": stage_idx,
                "stage_ranks": ranks,
                "tp_degree": len(ranks),
                "local_rank": ranks.index(rank),
                "leader_rank": ranks[0],
                "prev_leader_rank": None if stage_idx == 0 else stage_rank_groups[stage_idx - 1][0],
                "next_leader_rank": None if stage_idx + 1 >= len(stage_rank_groups) else stage_rank_groups[stage_idx + 1][0],
                "stage_group": stage_groups[stage_idx],
                "pp_group_idx": pp_group_index_by_rank[rank],
                "current_pp_group": current_pp_group,
                "send_list": send_list[rank],
                "recv_list": recv_list[rank],
                "send_empty_list": send_empty_list[rank],
                "recv_empty_list": recv_empty_list[rank],
            }
    raise ValueError(f"rank={rank} 不在任何 stage rank group 里。")


def _flat_index_to_tuple(flat_idx: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    indices = []
    for dim in reversed(shape):
        indices.append(flat_idx % dim)
        flat_idx //= dim
    return tuple(reversed(indices))


def _build_tensor_outliers(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    topk: int,
    last_dim_offset: int = 0,
) -> list[dict]:
    diff = (lhs - rhs).abs().reshape(-1).to(torch.float32)
    if diff.numel() == 0:
        return []

    k = min(topk, diff.numel())
    values, flat_indices = torch.topk(diff, k=k)
    lhs_flat = lhs.reshape(-1)
    rhs_flat = rhs.reshape(-1)
    shape = tuple(lhs.shape)

    outliers = []
    for value, flat_idx in zip(values.tolist(), flat_indices.tolist()):
        index = _flat_index_to_tuple(flat_idx, shape)
        if last_dim_offset and index:
            index = (*index[:-1], index[-1] + last_dim_offset)
        outliers.append(
            {
                "index": index,
                "lhs": lhs_flat[flat_idx].item(),
                "rhs": rhs_flat[flat_idx].item(),
                "abs_diff": value,
            }
        )
    return outliers


def _align_tp_trace_pair(
    tp_tensor: torch.Tensor,
    direct_tensor: torch.Tensor,
    *,
    local_rank: int,
    tp_degree: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if tp_tensor.shape == direct_tensor.shape:
        return tp_tensor, direct_tensor, 0

    if tp_tensor.dim() != direct_tensor.dim():
        raise ValueError(
            "TP trace tensor 维度和 direct trace tensor 不一致，"
            f"tp_shape={tuple(tp_tensor.shape)} direct_shape={tuple(direct_tensor.shape)}"
        )
    if tp_tensor.shape[:-1] != direct_tensor.shape[:-1]:
        raise ValueError(
            "TP trace tensor 和 direct trace tensor 的前置维度不一致，"
            f"tp_shape={tuple(tp_tensor.shape)} direct_shape={tuple(direct_tensor.shape)}"
        )
    if tp_degree <= 0:
        raise ValueError(f"tp_degree 必须是正整数，当前拿到 {tp_degree}。")
    if direct_tensor.shape[-1] % tp_degree != 0:
        raise ValueError(
            "direct trace tensor 的最后一维不能按 TP 度数整除，"
            f"direct_shape={tuple(direct_tensor.shape)} tp_degree={tp_degree}"
        )

    shard = direct_tensor.shape[-1] // tp_degree
    if tp_tensor.shape[-1] != shard:
        raise ValueError(
            "TP trace tensor 的最后一维和按 TP 切分后的 shard 大小不一致，"
            f"tp_shape={tuple(tp_tensor.shape)} direct_shape={tuple(direct_tensor.shape)} "
            f"tp_degree={tp_degree}"
        )

    start = local_rank * shard
    end = start + shard
    return tp_tensor, direct_tensor[..., start:end], start


def _tp_vs_direct_stats(
    tp_tensor: torch.Tensor,
    direct_tensor: torch.Tensor,
    *,
    local_rank: int,
    tp_degree: int,
) -> tuple[float, float]:
    aligned_tp, aligned_direct, _ = _align_tp_trace_pair(
        tp_tensor,
        direct_tensor,
        local_rank=local_rank,
        tp_degree=tp_degree,
    )
    return tensor_diff_stats(aligned_tp, aligned_direct)


def _tp_vs_direct_outliers(
    tp_tensor: torch.Tensor,
    direct_tensor: torch.Tensor,
    *,
    local_rank: int,
    tp_degree: int,
    topk: int,
) -> list[dict]:
    aligned_tp, aligned_direct, last_dim_offset = _align_tp_trace_pair(
        tp_tensor,
        direct_tensor,
        local_rank=local_rank,
        tp_degree=tp_degree,
    )
    return _build_tensor_outliers(
        aligned_tp,
        aligned_direct,
        topk,
        last_dim_offset=last_dim_offset,
    )


def _build_layer_outlier_dump(
    reference_trace: dict,
    direct_trace: dict,
    tp_trace: dict,
    local_rank: int,
    tp_degree: int,
    topk: int,
) -> dict:
    return {
        "layer_idx": reference_trace["layer_idx"],
        "deepstack_applied": reference_trace["deepstack_applied"],
        "direct_vs_ref": {
            "layer_input": _build_tensor_outliers(direct_trace["layer_input"], reference_trace["layer_input"], topk),
            "attn_output": _build_tensor_outliers(direct_trace["attn_output"], reference_trace["attn_output"], topk),
            "gate_out": _build_tensor_outliers(direct_trace["gate_out"], reference_trace["gate_out"], topk),
            "up_out": _build_tensor_outliers(direct_trace["up_out"], reference_trace["up_out"], topk),
            "fused": _build_tensor_outliers(direct_trace["fused"], reference_trace["fused"], topk),
            "mlp_output": _build_tensor_outliers(direct_trace["mlp_output"], reference_trace["mlp_output"], topk),
            "layer_output": _build_tensor_outliers(direct_trace["layer_output"], reference_trace["layer_output"], topk),
        },
        "tp_vs_direct": {
            "layer_input": _tp_vs_direct_outliers(
                tp_trace["layer_input"],
                direct_trace["layer_input"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "attn_output": _tp_vs_direct_outliers(
                tp_trace["attn_output"],
                direct_trace["attn_output"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "gate_out": _tp_vs_direct_outliers(
                tp_trace["gate_out"],
                direct_trace["gate_out"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "up_out": _tp_vs_direct_outliers(
                tp_trace["up_out"],
                direct_trace["up_out"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "fused": _tp_vs_direct_outliers(
                tp_trace["fused"],
                direct_trace["fused"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "mlp_output": _tp_vs_direct_outliers(
                tp_trace["mlp_output"],
                direct_trace["mlp_output"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "layer_output": _tp_vs_direct_outliers(
                tp_trace["layer_output"],
                direct_trace["layer_output"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
        },
    }


def build_stage_traces(
    reference_input: torch.Tensor,
    stage_input: torch.Tensor,
    bundle: dict,
    local_rank: int,
    tp_degree: int,
    comm_dtype: torch.dtype,
    tp_group,
    leader_rank: int,
    tp_attn_math_mode: str = "orig",
    tp_mlp_math_mode: str = "float32",
    dump_layer: int | None = None,
    dump_topk: int = 5,
) -> tuple[list[dict], dict | None]:
    reference_traces = trace_stage(reference_input, bundle)
    direct_traces = trace_stage(stage_input, bundle)
    tp_traces = trace_stage_tp(
        stage_input,
        bundle,
        rank=local_rank,
        world_size=tp_degree,
        comm_dtype=comm_dtype,
        tp_group=tp_group,
        tp_src_rank=leader_rank,
        tp_attn_math_mode=tp_attn_math_mode,
        tp_mlp_math_mode=tp_mlp_math_mode,
    )

    traces = []
    outlier_dump = None
    for reference_trace, direct_trace, tp_trace in zip(reference_traces, direct_traces, tp_traces):
        layer_idx = reference_trace["layer_idx"]
        traces.append(
            {
                "layer_idx": layer_idx,
                "deepstack_applied": reference_trace["deepstack_applied"],
                "direct_vs_ref": {
                    "layer_input": tensor_diff_stats(direct_trace["layer_input"], reference_trace["layer_input"]),
                    "attn_output": tensor_diff_stats(direct_trace["attn_output"], reference_trace["attn_output"]),
                    "gate_out": tensor_diff_stats(direct_trace["gate_out"], reference_trace["gate_out"]),
                    "up_out": tensor_diff_stats(direct_trace["up_out"], reference_trace["up_out"]),
                    "fused": tensor_diff_stats(direct_trace["fused"], reference_trace["fused"]),
                    "mlp_output": tensor_diff_stats(direct_trace["mlp_output"], reference_trace["mlp_output"]),
                    "layer_output": tensor_diff_stats(direct_trace["layer_output"], reference_trace["layer_output"]),
                    "post_deepstack": tensor_diff_stats(
                        direct_trace["post_deepstack"],
                        reference_trace["post_deepstack"],
                    ),
                },
                "tp_vs_direct": {
                    "layer_input": _tp_vs_direct_stats(
                        tp_trace["layer_input"],
                        direct_trace["layer_input"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "attn_output": _tp_vs_direct_stats(
                        tp_trace["attn_output"],
                        direct_trace["attn_output"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "gate_out": _tp_vs_direct_stats(
                        tp_trace["gate_out"],
                        direct_trace["gate_out"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "up_out": _tp_vs_direct_stats(
                        tp_trace["up_out"],
                        direct_trace["up_out"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "fused": _tp_vs_direct_stats(
                        tp_trace["fused"],
                        direct_trace["fused"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "mlp_output": _tp_vs_direct_stats(
                        tp_trace["mlp_output"],
                        direct_trace["mlp_output"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "layer_output": _tp_vs_direct_stats(
                        tp_trace["layer_output"],
                        direct_trace["layer_output"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "post_deepstack": _tp_vs_direct_stats(
                        tp_trace["post_deepstack"],
                        direct_trace["post_deepstack"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                },
            }
        )
        if dump_layer is not None and layer_idx == dump_layer:
            outlier_dump = _build_layer_outlier_dump(
                reference_trace=reference_trace,
                direct_trace=direct_trace,
                tp_trace=tp_trace,
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=dump_topk,
            )
    return traces, outlier_dump


def run_text_hybrid_rank(
    *,
    rank: int,
    world_size: int,
    manifest: dict,
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
    if world_size != manifest["world_size"]:
        raise ValueError(f"WORLD_SIZE={world_size}，但 hybrid manifest 需要 {manifest['world_size']}。")

    stage_groups = init_stage_groups(manifest["stage_rank_groups"])
    rank_stage = resolve_rank_stage(
        rank,
        manifest["stage_rank_groups"],
        stage_groups,
        pp_rank_groups=manifest.get("pp_rank_groups"),
        send_list=manifest.get("send_list"),
        recv_list=manifest.get("recv_list"),
        send_empty_list=manifest.get("send_empty_list"),
        recv_empty_list=manifest.get("recv_empty_list"),
    )

    bundle, compute_dtype = load_stage_bundle_by_index(
        manifest,
        rank_stage["stage_idx"],
        device,
        compute_dtype_arg,
    )
    comm_dtype = resolve_comm_dtype(comm_dtype_arg, compute_dtype)

    reference_input = get_stage_input(bundle)
    if rank_stage["stage_idx"] == 0 and rank_stage["local_rank"] == 0:
        leader_input = reference_input
    elif rank_stage["local_rank"] == 0:
        leader_input = recv_hidden_states(
            src=rank_stage["prev_leader_rank"],
            device=device,
            hidden_dtype=reference_input.dtype,
            comm_dtype=comm_dtype,
        )
    else:
        leader_input = None

    stage_input = broadcast_cpu(
        reference_tensor=reference_input,
        tensor=leader_input,
        src=rank_stage["leader_rank"],
        comm_dtype=comm_dtype,
        group=rank_stage["stage_group"],
    )
    boundary_max, boundary_mean = tensor_diff_stats(stage_input, reference_input)

    reference_output = get_stage_output(bundle)
    direct_output = None
    direct_max = None
    direct_mean = None
    tp_direct_max = None
    tp_direct_mean = None
    if compare_direct or trace_layers:
        direct_output = run_stage(stage_input, bundle)
        direct_max, direct_mean = tensor_diff_stats(direct_output, reference_output)

    stage_output = run_stage_tp(
        stage_input,
        bundle,
        rank=rank_stage["local_rank"],
        world_size=rank_stage["tp_degree"],
        comm_dtype=comm_dtype,
        tp_group=rank_stage["stage_group"],
        tp_src_rank=rank_stage["leader_rank"],
        tp_attn_math_mode=tp_attn_math_mode,
        tp_mlp_math_mode=tp_mlp_math_mode,
    )
    stage_max, stage_mean = tensor_diff_stats(stage_output, reference_output)
    if direct_output is not None:
        tp_direct_max, tp_direct_mean = tensor_diff_stats(stage_output, direct_output)

    sent_shape = None
    if rank_stage["local_rank"] == 0 and rank_stage["next_leader_rank"] is not None:
        sent_shape = send_hidden_states(stage_output, dst=rank_stage["next_leader_rank"], comm_dtype=comm_dtype)

    traces = None
    outlier_dump = None
    if trace_layers or dump_layer is not None:
        traces, outlier_dump = build_stage_traces(
            reference_input=reference_input,
            stage_input=stage_input,
            bundle=bundle,
            local_rank=rank_stage["local_rank"],
            tp_degree=rank_stage["tp_degree"],
            comm_dtype=comm_dtype,
            tp_group=rank_stage["stage_group"],
            leader_rank=rank_stage["leader_rank"],
            tp_attn_math_mode=tp_attn_math_mode,
            tp_mlp_math_mode=tp_mlp_math_mode,
            dump_layer=dump_layer,
            dump_topk=dump_topk,
        )

    return {
        "rank": rank,
        "stage_idx": rank_stage["stage_idx"],
        "stage_ranks": rank_stage["stage_ranks"],
        "local_rank": rank_stage["local_rank"],
        "tp_degree": rank_stage["tp_degree"],
        "leader_rank": rank_stage["leader_rank"],
        "pp_group_idx": rank_stage["pp_group_idx"],
        "current_pp_group": rank_stage["current_pp_group"],
        "num_stages": manifest["num_stages"],
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
        "boundary_max_diff": boundary_max,
        "boundary_mean_diff": boundary_mean,
        "direct_max_diff": direct_max,
        "direct_mean_diff": direct_mean,
        "stage_max_diff": stage_max,
        "stage_mean_diff": stage_mean,
        "tp_direct_max_diff": tp_direct_max,
        "tp_direct_mean_diff": tp_direct_mean,
        "next_leader_rank": rank_stage["next_leader_rank"],
        "send_list": rank_stage["send_list"],
        "recv_list": rank_stage["recv_list"],
        "send_empty_list": rank_stage["send_empty_list"],
        "recv_empty_list": rank_stage["recv_empty_list"],
        "traces": traces,
        "outlier_dump": outlier_dump,
    }
