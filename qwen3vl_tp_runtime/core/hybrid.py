import torch
import torch.distributed as dist

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


def parse_tp_degrees(values: list[int]) -> list[int]:
    if not values:
        raise ValueError("至少要提供一个 TP 度数。")
    tp_degrees = [int(value) for value in values]
    if any(value <= 0 for value in tp_degrees):
        raise ValueError(f"TP 度数必须是正整数，当前拿到 {tp_degrees!r}。")
    return tp_degrees


def build_stage_rank_groups(tp_degrees: list[int]) -> list[list[int]]:
    stage_rank_groups = []
    rank_cursor = 0
    for tp_degree in tp_degrees:
        ranks = list(range(rank_cursor, rank_cursor + tp_degree))
        stage_rank_groups.append(ranks)
        rank_cursor += tp_degree
    return stage_rank_groups


def build_hybrid_layout(tp_degrees: list[int]) -> dict:
    stage_rank_groups = build_stage_rank_groups(tp_degrees)
    return {
        "tp_degrees": tp_degrees,
        "stage_rank_groups": stage_rank_groups,
        "world_size": sum(tp_degrees),
        "num_stages": len(tp_degrees),
    }


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
    manifest["world_size"] = layout["world_size"]
    torch.save(manifest, manifest_path)
    return manifest


def load_hybrid_manifest(manifest_path: str) -> dict:
    manifest = load_pipeline_manifest(manifest_path)
    if "tp_degrees" not in manifest or "stage_rank_groups" not in manifest:
        raise ValueError("manifest 里没有 tp_degrees / stage_rank_groups，不能按 hybrid 运行。")
    return manifest


def init_stage_groups(stage_rank_groups: list[list[int]]) -> list:
    # 所有 rank 按相同顺序建组，避免 group 初始化乱序。
    return [dist.new_group(ranks=ranks) for ranks in stage_rank_groups]


def resolve_rank_stage(rank: int, stage_rank_groups: list[list[int]], stage_groups: list) -> dict:
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
        outliers.append(
            {
                "index": _flat_index_to_tuple(flat_idx, shape),
                "lhs": lhs_flat[flat_idx].item(),
                "rhs": rhs_flat[flat_idx].item(),
                "abs_diff": value,
            }
        )
    return outliers


def _build_layer_outlier_dump(
    reference_trace: dict,
    direct_trace: dict,
    tp_trace: dict,
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
            "layer_input": _build_tensor_outliers(tp_trace["layer_input"], direct_trace["layer_input"], topk),
            "attn_output": _build_tensor_outliers(tp_trace["attn_output"], direct_trace["attn_output"], topk),
            "gate_out": _build_tensor_outliers(tp_trace["gate_out"], direct_trace["gate_out"], topk),
            "up_out": _build_tensor_outliers(tp_trace["up_out"], direct_trace["up_out"], topk),
            "fused": _build_tensor_outliers(tp_trace["fused"], direct_trace["fused"], topk),
            "mlp_output": _build_tensor_outliers(tp_trace["mlp_output"], direct_trace["mlp_output"], topk),
            "layer_output": _build_tensor_outliers(tp_trace["layer_output"], direct_trace["layer_output"], topk),
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
                    "layer_input": tensor_diff_stats(tp_trace["layer_input"], direct_trace["layer_input"]),
                    "attn_output": tensor_diff_stats(tp_trace["attn_output"], direct_trace["attn_output"]),
                    "gate_out": tensor_diff_stats(tp_trace["gate_out"], direct_trace["gate_out"]),
                    "up_out": tensor_diff_stats(tp_trace["up_out"], direct_trace["up_out"]),
                    "fused": tensor_diff_stats(tp_trace["fused"], direct_trace["fused"]),
                    "mlp_output": tensor_diff_stats(tp_trace["mlp_output"], direct_trace["mlp_output"]),
                    "layer_output": tensor_diff_stats(tp_trace["layer_output"], direct_trace["layer_output"]),
                    "post_deepstack": tensor_diff_stats(
                        tp_trace["post_deepstack"],
                        direct_trace["post_deepstack"],
                    ),
                },
            }
        )
        if dump_layer is not None and layer_idx == dump_layer:
            outlier_dump = _build_layer_outlier_dump(
                reference_trace=reference_trace,
                direct_trace=direct_trace,
                tp_trace=tp_trace,
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
    compare_direct: bool = False,
    trace_layers: bool = False,
    dump_layer: int | None = None,
    dump_topk: int = 5,
) -> dict:
    if world_size != manifest["world_size"]:
        raise ValueError(f"WORLD_SIZE={world_size}，但 hybrid manifest 需要 {manifest['world_size']}。")

    stage_groups = init_stage_groups(manifest["stage_rank_groups"])
    rank_stage = resolve_rank_stage(rank, manifest["stage_rank_groups"], stage_groups)

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
        "num_stages": manifest["num_stages"],
        "start_idx": bundle["start_idx"],
        "end_idx": bundle["end_idx"],
        "num_layers": len(bundle["layers"]),
        "device": str(device),
        "comm_dtype": str(comm_dtype),
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
        "traces": traces,
        "outlier_dump": outlier_dump,
    }
