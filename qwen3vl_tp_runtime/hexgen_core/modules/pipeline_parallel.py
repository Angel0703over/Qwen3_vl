"""Dedicated pipeline-parallel runtime for captured multimodal text-stage bundles."""

from pathlib import Path

import torch

from qwen3vl_tp_runtime.hexgen_core.schema import (
    BoundaryStats,
    StageHandoffPayload,
    StageSpec,
    TextPipelineManifest,
)
from qwen3vl_tp_runtime.hexgen_core.stage import (
    apply_stage_handoff_payload,
    build_stage_handoff_payload,
    get_stage_input,
    get_stage_output,
    run_stage,
)
from qwen3vl_tp_runtime.hexgen_core.transport import StageHandoffTransport
from qwen3vl_tp_runtime.models.qwen3vl.capture import (
    capture_text_prefill_stage_bundle,
    capture_text_stage_bundle,
    load_bundle,
    move_bundle,
)
from qwen3vl_tp_runtime.models.qwen3vl.ops import dtype_from_name, resolve_comm_dtype


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
) -> TextPipelineManifest:
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
            StageSpec(
                stage_idx=stage_idx,
                start_idx=start_idx,
                end_idx=end_idx,
                bundle_path=bundle_path,
                num_layers=len(stage_bundle["layers"]),
                save_dtype=stage_bundle["save_dtype"],
            )
        )

    boundaries = []
    for stage_idx in range(len(stage_bundles) - 1):
        lhs = stage_bundles[stage_idx]["stage_output"]
        rhs = stage_bundles[stage_idx + 1]["stage_input"]
        max_diff, mean_diff = tensor_diff_stats(lhs, rhs)
        boundaries.append(
            BoundaryStats(
                src_stage_idx=stage_idx,
                dst_stage_idx=stage_idx + 1,
                max_diff=max_diff,
                mean_diff=mean_diff,
            )
        )

    manifest = TextPipelineManifest(
        pipeline_type="text",
        num_stages=len(stages),
        stage_ranges=stage_ranges,
        bundle_dir=str(bundle_dir_path),
        stages=stages,
        boundaries=boundaries,
        num_frames=num_frames,
        save_dtype=stage_bundles[0]["save_dtype"],
    )

    save_path = Path(manifest_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest.to_dict(), save_path)
    return manifest


def prepare_text_prefill_pipeline(
    *,
    stage_ranges: list[tuple[int, int]],
    bundle_dir: str,
    manifest_path: str,
    prompt: str = "请用中文简要介绍一下人工智能。",
    save_dtype: str = "auto",
    model_path: str | None = None,
) -> TextPipelineManifest:
    bundle_dir_path = Path(bundle_dir)
    bundle_dir_path.mkdir(parents=True, exist_ok=True)

    stages = []
    stage_bundles = []
    for stage_idx, (start_idx, end_idx) in enumerate(stage_ranges):
        bundle_path = build_stage_bundle_path(bundle_dir, stage_idx, start_idx, end_idx)
        stage_bundle = capture_text_prefill_stage_bundle(
            start_idx=start_idx,
            end_idx=end_idx,
            prompt=prompt,
            bundle_path=bundle_path,
            save_dtype=save_dtype,
            **({"model_path": model_path} if model_path is not None else {}),
        )
        stage_bundles.append(stage_bundle)
        stages.append(
            StageSpec(
                stage_idx=stage_idx,
                start_idx=start_idx,
                end_idx=end_idx,
                bundle_path=bundle_path,
                num_layers=len(stage_bundle["layers"]),
                save_dtype=stage_bundle["save_dtype"],
            )
        )

    if stage_bundles[-1]["stage_type"] != "text_prefill_last":
        raise ValueError("text prefill pipeline 的最后一个 stage 必须覆盖到最后一层，才能输出 logits。")

    boundaries = []
    for stage_idx in range(len(stage_bundles) - 1):
        lhs = stage_bundles[stage_idx]["stage_output"]
        rhs = stage_bundles[stage_idx + 1]["stage_input"]
        max_diff, mean_diff = tensor_diff_stats(lhs, rhs)
        boundaries.append(
            BoundaryStats(
                src_stage_idx=stage_idx,
                dst_stage_idx=stage_idx + 1,
                max_diff=max_diff,
                mean_diff=mean_diff,
            )
        )

    manifest = TextPipelineManifest(
        pipeline_type="text_prefill",
        num_stages=len(stages),
        stage_ranges=stage_ranges,
        bundle_dir=str(bundle_dir_path),
        stages=stages,
        boundaries=boundaries,
        num_frames=0,
        save_dtype=stage_bundles[0]["save_dtype"],
    )

    save_path = Path(manifest_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest.to_dict(), save_path)
    return manifest


def load_pipeline_manifest(manifest_path: str) -> TextPipelineManifest:
    payload = torch.load(manifest_path, map_location="cpu")
    if isinstance(payload, TextPipelineManifest):
        return payload
    return TextPipelineManifest.from_dict(payload)


def load_stage_bundle_by_index(
    manifest: TextPipelineManifest,
    stage_idx: int,
    device: torch.device,
    compute_dtype_arg: str,
) -> tuple[dict, torch.dtype]:
    stage_meta = manifest.stages[stage_idx]
    bundle = load_bundle(stage_meta.bundle_path)
    compute_dtype_name = bundle["save_dtype"] if compute_dtype_arg == "auto" else compute_dtype_arg
    compute_dtype = dtype_from_name(compute_dtype_name)
    return move_bundle(bundle, device, compute_dtype), compute_dtype


def load_stage_bundle_for_rank(
    manifest: TextPipelineManifest,
    rank: int,
    device: torch.device,
    compute_dtype_arg: str,
) -> tuple[dict, torch.dtype]:
    return load_stage_bundle_by_index(manifest, rank, device, compute_dtype_arg)


class TextPipelineRunner:
    """Stateful rank runner for a simple sequential text pipeline."""

    def __init__(
        self,
        manifest: TextPipelineManifest,
        device: torch.device,
        compute_dtype_arg: str,
        comm_dtype_arg: str,
        return_tensors: bool = False,
    ) -> None:
        self.manifest = manifest
        self.device = device
        self.compute_dtype_arg = compute_dtype_arg
        self.comm_dtype_arg = comm_dtype_arg
        self.return_tensors = return_tensors

    def run_rank(self, rank: int, world_size: int) -> dict:
        if world_size != self.manifest.num_stages:
            raise ValueError(
                f"WORLD_SIZE={world_size}，但 manifest 里有 {self.manifest.num_stages} 个 stage。"
            )

        bundle, compute_dtype = load_stage_bundle_for_rank(
            self.manifest,
            rank,
            self.device,
            self.compute_dtype_arg,
        )
        comm_dtype = resolve_comm_dtype(self.comm_dtype_arg, compute_dtype)
        handoff_transport = StageHandoffTransport(device=self.device, comm_dtype=comm_dtype)

        if rank == 0:
            stage_input = get_stage_input(bundle)
            boundary_max = None
            boundary_mean = None
            received_payload_keys = []
        else:
            reference_input = get_stage_input(bundle)
            received_message = handoff_transport.recv(src=rank - 1, stage_bundle=bundle)
            handoff = received_message.handoff
            if handoff is None or handoff.hidden_states is None:
                raise ValueError(f"stage {rank} 没有收到有效的 hidden_states payload。")

            bundle = apply_stage_handoff_payload(bundle, handoff)
            stage_input = get_stage_input(bundle)
            boundary_max, boundary_mean = tensor_diff_stats(stage_input, reference_input)
            received_payload_keys = received_message.summary.payload_keys

        reference_output = get_stage_output(bundle)
        stage_output = run_stage(stage_input, bundle)
        stage_max, stage_mean = tensor_diff_stats(stage_output, reference_output)

        sent_shape = None
        sent_payload_keys = []
        sent_tensor_shapes = {}
        if rank < world_size - 1:
            next_stage_meta = self.manifest.stages[rank + 1]
            handoff = build_stage_handoff_payload(
                stage_output,
                bundle,
                target_stage_range=(next_stage_meta.start_idx, next_stage_meta.end_idx),
            )
            summary = handoff_transport.send(handoff, dst=rank + 1)
            sent_shape = summary.tensor_shapes.get(StageHandoffPayload.HIDDEN_STATES_KEY)
            sent_payload_keys = summary.payload_keys
            sent_tensor_shapes = summary.tensor_shapes

        stats = {
            "rank": rank,
            "stage_idx": rank,
            "num_stages": world_size,
            "start_idx": bundle["start_idx"],
            "end_idx": bundle["end_idx"],
            "num_layers": len(bundle["layers"]),
            "device": str(self.device),
            "comm_dtype": str(comm_dtype),
            "input_shape": tuple(stage_input.shape),
            "output_shape": tuple(stage_output.shape),
            "sent_shape": sent_shape,
            "received_payload_keys": received_payload_keys,
            "sent_payload_keys": sent_payload_keys,
            "sent_tensor_shapes": sent_tensor_shapes,
            "boundary_max_diff": boundary_max,
            "boundary_mean_diff": boundary_mean,
            "stage_max_diff": stage_max,
            "stage_mean_diff": stage_mean,
        }
        if self.return_tensors:
            stats["stage_output"] = stage_output
            stats["reference_output"] = reference_output
        return stats


def run_text_pipeline_rank(
    *,
    rank: int,
    world_size: int,
    manifest: TextPipelineManifest,
    device: torch.device,
    compute_dtype_arg: str,
    comm_dtype_arg: str,
    return_tensors: bool = False,
) -> dict:
    runner = TextPipelineRunner(
        manifest=manifest,
        device=device,
        compute_dtype_arg=compute_dtype_arg,
        comm_dtype_arg=comm_dtype_arg,
        return_tensors=return_tensors,
    )
    return runner.run_rank(rank, world_size)


__all__ = [
    "StageSpec",
    "BoundaryStats",
    "TextPipelineManifest",
    "TextPipelineRunner",
    "tensor_diff_stats",
    "parse_stage_range",
    "parse_stage_ranges",
    "build_stage_bundle_path",
    "prepare_text_pipeline",
    "prepare_text_prefill_pipeline",
    "load_pipeline_manifest",
    "load_stage_bundle_by_index",
    "load_stage_bundle_for_rank",
    "run_text_pipeline_rank",
]
