from pathlib import Path

import torch

from qwen3vl_tp_runtime.core.capture import capture_text_stage_bundle, load_bundle, move_bundle
from qwen3vl_tp_runtime.core.ops import dtype_from_name, resolve_comm_dtype
from qwen3vl_tp_runtime.core.stage import get_stage_input, get_stage_output, run_stage
from qwen3vl_tp_runtime.core.transport import recv_hidden_states, send_hidden_states


def tensor_diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs - rhs).abs()
    return diff.max().item(), diff.mean().item()


def parse_stage_range(spec: str) -> tuple[int, int]:
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"stage range 必须是 start:end 形式，当前拿到 {spec!r}。")

    start_idx = int(parts[0])
    end_idx = int(parts[1])
    if start_idx > end_idx:
        raise ValueError(f"stage range 要满足 start <= end，当前拿到 {spec!r}。")
    return start_idx, end_idx


def parse_stage_ranges(specs: list[str]) -> list[tuple[int, int]]:
    if not specs:
        raise ValueError("至少要提供一个 stage range。")

    ranges = [parse_stage_range(spec) for spec in specs]
    for idx in range(1, len(ranges)):
        prev_start, prev_end = ranges[idx - 1]
        cur_start, cur_end = ranges[idx]
        if cur_start != prev_end + 1:
            raise ValueError(
                "text pipeline 当前要求 stage 连续切分，"
                f"但拿到前一段 {prev_start}:{prev_end}，后一段 {cur_start}:{cur_end}。"
            )
    return ranges


def build_stage_bundle_path(bundle_dir: str, stage_idx: int, start_idx: int, end_idx: int) -> str:
    bundle_name = f"stage_{stage_idx:02d}_{start_idx:03d}_{end_idx:03d}.pt"
    return str(Path(bundle_dir) / bundle_name)


def prepare_text_pipeline(
    *,
    stage_ranges: list[tuple[int, int]],
    bundle_dir: str,
    manifest_path: str,
    num_frames: int = 8,
    save_dtype: str = "auto",
) -> dict:
    # 当前 prepare 复用已有单 stage 抓取逻辑，先把多段组织和边界校验搭起来。
    bundle_dir_path = Path(bundle_dir)
    bundle_dir_path.mkdir(parents=True, exist_ok=True)

    stages = []
    stage_bundles = []
    for stage_idx, (start_idx, end_idx) in enumerate(stage_ranges):
        bundle_path = build_stage_bundle_path(bundle_dir, stage_idx, start_idx, end_idx)
        stage_bundle = capture_text_stage_bundle(
            start_idx=start_idx,
            end_idx=end_idx,
            num_frames=num_frames,
            bundle_path=bundle_path,
            save_dtype=save_dtype,
        )
        stage_bundles.append(stage_bundle)
        stages.append(
            {
                "stage_idx": stage_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "bundle_path": bundle_path,
                "num_layers": len(stage_bundle["layers"]),
                "save_dtype": stage_bundle["save_dtype"],
            }
        )

    boundaries = []
    for stage_idx in range(len(stage_bundles) - 1):
        lhs = stage_bundles[stage_idx]["stage_output"]
        rhs = stage_bundles[stage_idx + 1]["stage_input"]
        max_diff, mean_diff = tensor_diff_stats(lhs, rhs)
        boundaries.append(
            {
                "src_stage_idx": stage_idx,
                "dst_stage_idx": stage_idx + 1,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
            }
        )

    manifest = {
        "pipeline_type": "text",
        "num_stages": len(stages),
        "stage_ranges": stage_ranges,
        "bundle_dir": str(bundle_dir_path),
        "stages": stages,
        "boundaries": boundaries,
        "num_frames": num_frames,
        "save_dtype": stage_bundles[0]["save_dtype"],
    }

    save_path = Path(manifest_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, save_path)
    return manifest


def load_pipeline_manifest(manifest_path: str) -> dict:
    return torch.load(manifest_path, map_location="cpu")


def load_stage_bundle_for_rank(
    manifest: dict,
    rank: int,
    device: torch.device,
    compute_dtype_arg: str,
) -> tuple[dict, torch.dtype]:
    return load_stage_bundle_by_index(manifest, rank, device, compute_dtype_arg)


def load_stage_bundle_by_index(
    manifest: dict,
    stage_idx: int,
    device: torch.device,
    compute_dtype_arg: str,
) -> tuple[dict, torch.dtype]:
    stage_meta = manifest["stages"][stage_idx]
    bundle = load_bundle(stage_meta["bundle_path"])
    compute_dtype_name = bundle["save_dtype"] if compute_dtype_arg == "auto" else compute_dtype_arg
    compute_dtype = dtype_from_name(compute_dtype_name)
    return move_bundle(bundle, device, compute_dtype), compute_dtype


def run_text_pipeline_rank(
    *,
    rank: int,
    world_size: int,
    manifest: dict,
    device: torch.device,
    compute_dtype_arg: str,
    comm_dtype_arg: str,
) -> dict:
    if world_size != manifest["num_stages"]:
        raise ValueError(
            f"WORLD_SIZE={world_size}，但 manifest 里有 {manifest['num_stages']} 个 stage。"
        )

    bundle, compute_dtype = load_stage_bundle_for_rank(manifest, rank, device, compute_dtype_arg)
    comm_dtype = resolve_comm_dtype(comm_dtype_arg, compute_dtype)

    if rank == 0:
        stage_input = get_stage_input(bundle)
        boundary_max = None
        boundary_mean = None
    else:
        reference_input = get_stage_input(bundle)
        stage_input = recv_hidden_states(
            src=rank - 1,
            device=device,
            hidden_dtype=reference_input.dtype,
            comm_dtype=comm_dtype,
        )
        boundary_max, boundary_mean = tensor_diff_stats(stage_input, reference_input)

    reference_output = get_stage_output(bundle)
    stage_output = run_stage(stage_input, bundle)
    stage_max, stage_mean = tensor_diff_stats(stage_output, reference_output)

    sent_shape = None
    if rank < world_size - 1:
        sent_shape = send_hidden_states(stage_output, dst=rank + 1, comm_dtype=comm_dtype)

    return {
        "rank": rank,
        "stage_idx": rank,
        "num_stages": world_size,
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
        "stage_max_diff": stage_max,
        "stage_mean_diff": stage_mean,
    }
