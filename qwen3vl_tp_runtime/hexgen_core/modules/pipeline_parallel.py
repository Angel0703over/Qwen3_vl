"""Dedicated pipeline-parallel runtime for captured multimodal text-stage bundles."""

import gc
from pathlib import Path

import torch
import torch.distributed as dist

from qwen3vl_tp_runtime.hexgen_core.distributed import broadcast_object_cpu, startup_log, startup_timer
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
    capture_multimodal_decode_stage_bundle,
    capture_multimodal_generate_stage_bundle,
    capture_multimodal_prefill_stage_bundle,
    capture_text_decode_stage_bundle,
    capture_text_generate_stage_bundle,
    capture_text_prefill_stage_bundle,
    capture_text_stage_bundle,
    load_bundle,
    move_bundle,
)
from qwen3vl_tp_runtime.models.qwen3vl.runtime_builder import (
    build_direct_stage_bundle,
    compact_text_prompt_meta,
    prepare_text_prompt_meta,
    restore_text_prompt_meta,
)
from qwen3vl_tp_runtime.models.qwen3vl.execution import (
    forward_text_embeddings,
    trace_text_decode_logits_with_runtime_cache,
    trace_text_decode_stage_with_runtime_cache,
)
from qwen3vl_tp_runtime.models.qwen3vl.functional import dtype_from_name, resolve_comm_dtype
from qwen3vl_tp_runtime.models.qwen3vl.weights import (
    build_text_rotary_embedding,
    build_text_runtime_aux_tensors,
    load_text_model_config_spec,
)


def tensor_diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs - rhs).abs()
    return diff.max().item(), diff.mean().item()


def _release_unused_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def prepare_multimodal_prefill_pipeline(
    *,
    stage_ranges: list[tuple[int, int]],
    bundle_dir: str,
    manifest_path: str,
    num_frames: int = 8,
    save_dtype: str = "auto",
    model_path: str | None = None,
    frame_dir: str | None = None,
) -> TextPipelineManifest:
    bundle_dir_path = Path(bundle_dir)
    bundle_dir_path.mkdir(parents=True, exist_ok=True)

    stages = []
    boundaries = []
    prev_stage_output = None
    prev_stage_idx = None
    last_stage_type = None
    manifest_save_dtype = None
    for stage_idx, (start_idx, end_idx) in enumerate(stage_ranges):
        print(
            f"[prepare-mm] capturing stage {stage_idx}/{len(stage_ranges) - 1} "
            f"range={start_idx}:{end_idx} num_frames={num_frames}",
            flush=True,
        )
        bundle_path = build_stage_bundle_path(bundle_dir, stage_idx, start_idx, end_idx)
        stage_bundle = capture_multimodal_prefill_stage_bundle(
            start_idx=start_idx,
            end_idx=end_idx,
            num_frames=num_frames,
            bundle_path=bundle_path,
            save_dtype=save_dtype,
            **({"model_path": model_path} if model_path is not None else {}),
            **({"frame_dir": frame_dir} if frame_dir is not None else {}),
        )
        if prev_stage_output is not None and prev_stage_idx is not None:
            max_diff, mean_diff = tensor_diff_stats(prev_stage_output, stage_bundle["stage_input"])
            boundaries.append(
                BoundaryStats(
                    src_stage_idx=prev_stage_idx,
                    dst_stage_idx=stage_idx,
                    max_diff=max_diff,
                    mean_diff=mean_diff,
                )
            )
            del prev_stage_output

        prev_stage_output = stage_bundle["stage_output"]
        prev_stage_idx = stage_idx
        last_stage_type = stage_bundle["stage_type"]
        if manifest_save_dtype is None:
            manifest_save_dtype = stage_bundle["save_dtype"]
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
        del stage_bundle
        _release_unused_memory()
        print(
            f"[prepare-mm] finished stage {stage_idx}/{len(stage_ranges) - 1} "
            f"range={start_idx}:{end_idx}",
            flush=True,
        )

    if last_stage_type != "text_prefill_last":
        raise ValueError("multimodal prefill pipeline 的最后一个 stage 必须覆盖到最后一层，才能输出 logits。")

    if prev_stage_output is not None:
        del prev_stage_output
        _release_unused_memory()

    manifest = TextPipelineManifest(
        pipeline_type="multimodal_prefill",
        num_stages=len(stages),
        stage_ranges=stage_ranges,
        bundle_dir=str(bundle_dir_path),
        stages=stages,
        boundaries=boundaries,
        num_frames=num_frames,
        save_dtype=manifest_save_dtype,
    )

    save_path = Path(manifest_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest.to_dict(), save_path)
    return manifest


def prepare_multimodal_decode_pipeline(
    *,
    stage_ranges: list[tuple[int, int]],
    bundle_dir: str,
    manifest_path: str,
    num_frames: int = 8,
    decode_token_id: int | None = None,
    save_dtype: str = "auto",
    model_path: str | None = None,
    frame_dir: str | None = None,
) -> TextPipelineManifest:
    bundle_dir_path = Path(bundle_dir)
    bundle_dir_path.mkdir(parents=True, exist_ok=True)

    stages = []
    boundaries = []
    prev_stage_output = None
    prev_stage_idx = None
    last_stage_type = None
    manifest_save_dtype = None
    resolved_decode_token_id = decode_token_id
    for stage_idx, (start_idx, end_idx) in enumerate(stage_ranges):
        print(
            f"[prepare-mm-decode] capturing stage {stage_idx}/{len(stage_ranges) - 1} "
            f"range={start_idx}:{end_idx} num_frames={num_frames}",
            flush=True,
        )
        bundle_path = build_stage_bundle_path(bundle_dir, stage_idx, start_idx, end_idx)
        stage_bundle = capture_multimodal_decode_stage_bundle(
            start_idx=start_idx,
            end_idx=end_idx,
            num_frames=num_frames,
            decode_token_id=resolved_decode_token_id,
            bundle_path=bundle_path,
            save_dtype=save_dtype,
            **({"model_path": model_path} if model_path is not None else {}),
            **({"frame_dir": frame_dir} if frame_dir is not None else {}),
        )
        if resolved_decode_token_id is None:
            resolved_decode_token_id = int(stage_bundle["decode_token_id"])
        if prev_stage_output is not None and prev_stage_idx is not None:
            max_diff, mean_diff = tensor_diff_stats(prev_stage_output, stage_bundle["stage_input"])
            boundaries.append(
                BoundaryStats(
                    src_stage_idx=prev_stage_idx,
                    dst_stage_idx=stage_idx,
                    max_diff=max_diff,
                    mean_diff=mean_diff,
                )
            )
            del prev_stage_output

        prev_stage_output = stage_bundle["stage_output"]
        prev_stage_idx = stage_idx
        last_stage_type = stage_bundle["stage_type"]
        if manifest_save_dtype is None:
            manifest_save_dtype = stage_bundle["save_dtype"]
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
        del stage_bundle
        _release_unused_memory()
        print(
            f"[prepare-mm-decode] finished stage {stage_idx}/{len(stage_ranges) - 1} "
            f"range={start_idx}:{end_idx}",
            flush=True,
        )

    if last_stage_type != "text_decode_last":
        raise ValueError("multimodal decode pipeline 的最后一个 stage 必须覆盖到最后一层，才能输出 logits。")

    if prev_stage_output is not None:
        del prev_stage_output
        _release_unused_memory()

    manifest = TextPipelineManifest(
        pipeline_type="multimodal_decode",
        num_stages=len(stages),
        stage_ranges=stage_ranges,
        bundle_dir=str(bundle_dir_path),
        stages=stages,
        boundaries=boundaries,
        num_frames=num_frames,
        save_dtype=manifest_save_dtype,
    )

    save_path = Path(manifest_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest.to_dict(), save_path)
    return manifest


def prepare_multimodal_generate_pipeline(
    *,
    stage_ranges: list[tuple[int, int]],
    bundle_dir: str,
    manifest_path: str,
    num_frames: int = 8,
    max_new_tokens: int = 4,
    save_dtype: str = "auto",
    model_path: str | None = None,
    frame_dir: str | None = None,
) -> TextPipelineManifest:
    bundle_dir_path = Path(bundle_dir)
    bundle_dir_path.mkdir(parents=True, exist_ok=True)

    stages = []
    stage_bundles = []
    for stage_idx, (start_idx, end_idx) in enumerate(stage_ranges):
        bundle_path = build_stage_bundle_path(bundle_dir, stage_idx, start_idx, end_idx)
        stage_bundle = capture_multimodal_generate_stage_bundle(
            start_idx=start_idx,
            end_idx=end_idx,
            num_frames=num_frames,
            max_new_tokens=max_new_tokens,
            bundle_path=bundle_path,
            save_dtype=save_dtype,
            **({"model_path": model_path} if model_path is not None else {}),
            **({"frame_dir": frame_dir} if frame_dir is not None else {}),
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
        lhs = stage_bundles[stage_idx]["prefill"]["stage_output"]
        rhs = stage_bundles[stage_idx + 1]["prefill"]["stage_input"]
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
        pipeline_type="multimodal_generate",
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


def prepare_text_decode_pipeline(
    *,
    stage_ranges: list[tuple[int, int]],
    bundle_dir: str,
    manifest_path: str,
    prompt: str = "请用中文简要介绍一下人工智能。",
    decode_token_id: int | None = None,
    save_dtype: str = "auto",
    model_path: str | None = None,
) -> TextPipelineManifest:
    bundle_dir_path = Path(bundle_dir)
    bundle_dir_path.mkdir(parents=True, exist_ok=True)

    stages = []
    stage_bundles = []
    resolved_decode_token_id = decode_token_id
    for stage_idx, (start_idx, end_idx) in enumerate(stage_ranges):
        bundle_path = build_stage_bundle_path(bundle_dir, stage_idx, start_idx, end_idx)
        stage_bundle = capture_text_decode_stage_bundle(
            start_idx=start_idx,
            end_idx=end_idx,
            prompt=prompt,
            decode_token_id=resolved_decode_token_id,
            bundle_path=bundle_path,
            save_dtype=save_dtype,
            **({"model_path": model_path} if model_path is not None else {}),
        )
        if resolved_decode_token_id is None:
            resolved_decode_token_id = int(stage_bundle["decode_token_id"])
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

    if stage_bundles[-1]["stage_type"] != "text_decode_last":
        raise ValueError("text decode pipeline 的最后一个 stage 必须覆盖到最后一层，才能输出 logits。")

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
        pipeline_type="text_decode",
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


def prepare_text_generate_pipeline(
    *,
    stage_ranges: list[tuple[int, int]],
    bundle_dir: str,
    manifest_path: str,
    prompt: str = "请用中文简要介绍一下人工智能。",
    max_new_tokens: int = 4,
    save_dtype: str = "auto",
    model_path: str | None = None,
) -> TextPipelineManifest:
    bundle_dir_path = Path(bundle_dir)
    bundle_dir_path.mkdir(parents=True, exist_ok=True)

    stages = []
    stage_bundles = []
    for stage_idx, (start_idx, end_idx) in enumerate(stage_ranges):
        bundle_path = build_stage_bundle_path(bundle_dir, stage_idx, start_idx, end_idx)
        stage_bundle = capture_text_generate_stage_bundle(
            start_idx=start_idx,
            end_idx=end_idx,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
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

    boundaries = []
    for stage_idx in range(len(stage_bundles) - 1):
        lhs = stage_bundles[stage_idx]["prefill"]["stage_output"]
        rhs = stage_bundles[stage_idx + 1]["prefill"]["stage_input"]
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
        pipeline_type="text_generate",
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
    if stage_meta.bundle_path:
        bundle = load_bundle(stage_meta.bundle_path)
    else:
        bundle = build_direct_stage_bundle(
            stage_idx=stage_idx,
            start_idx=stage_meta.start_idx,
            end_idx=stage_meta.end_idx,
            runtime_config=manifest.runtime_config,
        )
    compute_dtype_name = bundle["save_dtype"] if compute_dtype_arg == "auto" else compute_dtype_arg
    compute_dtype = dtype_from_name(compute_dtype_name)
    return move_bundle(bundle, device, compute_dtype), compute_dtype


def _all_stages_are_direct(manifest: TextPipelineManifest) -> bool:
    return all(stage.bundle_path is None for stage in manifest.stages)


def _need_text_prompt_meta(manifest: TextPipelineManifest) -> bool:
    runtime_config = manifest.runtime_config
    return (
        _all_stages_are_direct(manifest)
        and str(runtime_config.get("modality", "multimodal")) == "text"
        and str(runtime_config.get("mode", "")) == "generate"
        and not bool(runtime_config.get("include_runtime_reference", True))
    )


def _seed_text_prompt_meta(manifest: TextPipelineManifest, *, rank: int) -> None:
    runtime_config = manifest.runtime_config
    if runtime_config.get("_runtime_only_prompt_metadata_ready") or not _need_text_prompt_meta(manifest):
        return

    prompt_metadata = None
    if rank == 0:
        with startup_timer("pp-direct-loader", "prepare runtime-only text prompt metadata"):
            prompt_metadata = prepare_text_prompt_meta(runtime_config)
        prompt_metadata = compact_text_prompt_meta(prompt_metadata)

    startup_log(
        "pp-direct-loader",
        f"rank={rank} waiting runtime-only text prompt metadata broadcast from src=0",
    )
    prompt_metadata = broadcast_object_cpu(
        prompt_metadata,
        src=0,
        label="runtime_only_text_prompt_metadata",
    )
    prompt_metadata = restore_text_prompt_meta(prompt_metadata)
    runtime_config["_runtime_only_input_ids"] = prompt_metadata["input_ids"]
    if prompt_metadata.get("attention_mask") is not None:
        runtime_config["_runtime_only_attention_mask"] = prompt_metadata["attention_mask"]
    else:
        runtime_config.pop("_runtime_only_attention_mask", None)
    runtime_config["_runtime_only_prompt_metadata_ready"] = True


def load_stage_bundle_for_rank(
    manifest: TextPipelineManifest,
    rank: int,
    device: torch.device,
    compute_dtype_arg: str,
) -> tuple[dict, torch.dtype]:
    if _all_stages_are_direct(manifest) and dist.is_available() and dist.is_initialized():
        _seed_text_prompt_meta(manifest, rank=rank)
        stage_meta = manifest.stages[rank]
        startup_log(
            "pp-direct-loader",
            f"rank={rank} locally building stage_idx={stage_meta.stage_idx} "
            f"range={stage_meta.start_idx}:{stage_meta.end_idx}",
        )
        bundle, compute_dtype = load_stage_bundle_by_index(
            manifest,
            rank,
            device,
            compute_dtype_arg,
        )
        startup_log("pp-direct-loader", f"rank={rank} entering post-load barrier")
        dist.barrier()
        startup_log("pp-direct-loader", f"rank={rank} local stage ready compute_dtype={compute_dtype}")
        return bundle, compute_dtype

    return load_stage_bundle_by_index(manifest, rank, device, compute_dtype_arg)


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


def _run_text_generate_phase(
    *,
    rank: int,
    world_size: int,
    manifest: TextPipelineManifest,
    runtime_bundle: dict,
    handoff_transport: StageHandoffTransport,
    phase_kind: str,
    current_token_id: int | None,
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None,
    return_tensor: bool,
) -> tuple[dict, dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None]:
    is_last_stage = rank == world_size - 1
    embedding_max = None
    embedding_mean = None
    reference_input = runtime_bundle.get("stage_input")
    if reference_input is None:
        reference_input = runtime_bundle.get("layer_input")
    query_len = int(runtime_bundle["prefill_seq_len"]) if phase_kind == "prefill" else 1

    if rank == 0:
        stage_input = reference_input
        if runtime_bundle.get("embed_tokens_weight") is not None:
            if phase_kind == "prefill" and "input_ids" in runtime_bundle:
                stage_input = forward_text_embeddings(runtime_bundle["input_ids"], runtime_bundle)
                if reference_input is not None:
                    embedding_max, embedding_mean = tensor_diff_stats(stage_input, reference_input)
            elif phase_kind == "decode" and current_token_id is not None and "decode_input_ids" in runtime_bundle:
                decode_input_ids = torch.tensor(
                    [[current_token_id]],
                    device=_infer_runtime_tensor_device(runtime_bundle),
                    dtype=runtime_bundle["decode_input_ids"].dtype,
                )
                stage_input = forward_text_embeddings(decode_input_ids, runtime_bundle)
                if reference_input is not None:
                    embedding_max, embedding_mean = tensor_diff_stats(stage_input, reference_input)
                runtime_bundle["decode_input_ids_runtime"] = decode_input_ids
        if stage_input is None:
            raise RuntimeError("首 stage 缺少可用的 stage_input。")
        runtime_bundle["stage_input"] = stage_input
        runtime_bundle["layer_input"] = stage_input
        boundary_max = None
        boundary_mean = None
        received_payload_keys: list[str] = []
    else:
        received_message = handoff_transport.recv(src=rank - 1, stage_bundle=runtime_bundle)
        handoff = received_message.handoff
        if handoff is None or handoff.hidden_states is None:
            raise ValueError(f"stage {rank} 没有收到有效的 hidden_states payload。")

        runtime_bundle = apply_stage_handoff_payload(runtime_bundle, handoff)
        stage_input = get_stage_input(runtime_bundle)
        if reference_input is None:
            boundary_max, boundary_mean = None, None
        else:
            boundary_max, boundary_mean = tensor_diff_stats(stage_input, reference_input)
        received_payload_keys = received_message.summary.payload_keys

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
            trace = trace_text_decode_logits_with_runtime_cache(
                stage_input,
                prefill_runtime_bundle,
                cache_by_layer={},
            )
            stage_output = trace["logits"]
            if runtime_bundle.get("hidden_stage_output") is not None:
                hidden_stage_output = trace["stage_output"]
                hidden_stage_max, hidden_stage_mean = tensor_diff_stats(
                    hidden_stage_output,
                    runtime_bundle["hidden_stage_output"],
                )
            if runtime_bundle.get("norm_output") is not None:
                norm_max, norm_mean = tensor_diff_stats(trace["norm_output"], runtime_bundle["norm_output"])
            updated_cache = trace["cache_by_layer"]
        else:
            prefill_runtime_bundle = _strip_runtime_layer_cache(runtime_bundle)
            trace = trace_text_decode_stage_with_runtime_cache(
                stage_input,
                prefill_runtime_bundle,
                cache_by_layer={},
            )
            stage_output = trace["stage_output"]
            updated_cache = trace["cache_by_layer"]
    elif phase_kind == "decode":
        if is_last_stage:
            trace = trace_text_decode_logits_with_runtime_cache(
                stage_input,
                runtime_bundle,
                cache_by_layer=cache_by_layer,
            )
            stage_output = trace["logits"]
            if runtime_bundle.get("hidden_stage_output") is not None:
                hidden_stage_output = trace["stage_output"]
                hidden_stage_max, hidden_stage_mean = tensor_diff_stats(
                    hidden_stage_output,
                    runtime_bundle["hidden_stage_output"],
                )
            if runtime_bundle.get("norm_output") is not None:
                norm_max, norm_mean = tensor_diff_stats(trace["norm_output"], runtime_bundle["norm_output"])
            updated_cache = trace["cache_by_layer"]
        else:
            trace = trace_text_decode_stage_with_runtime_cache(
                stage_input,
                runtime_bundle,
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
        stage_max, stage_mean = tensor_diff_stats(stage_output, reference_output)

    sent_shape = None
    sent_payload_keys: list[str] = []
    sent_tensor_shapes: dict[str, tuple[int, ...] | None] = {}
    if rank < world_size - 1:
        next_stage_meta = manifest.stages[rank + 1]
        handoff = build_stage_handoff_payload(
            stage_output,
            runtime_bundle,
            target_stage_range=(next_stage_meta.start_idx, next_stage_meta.end_idx),
        )
        summary = handoff_transport.send(handoff, dst=rank + 1)
        sent_shape = summary.tensor_shapes.get(StageHandoffPayload.HIDDEN_STATES_KEY)
        sent_payload_keys = summary.payload_keys
        sent_tensor_shapes = summary.tensor_shapes

    if is_last_stage:
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
    if return_tensor:
        stats["stage_output_tensor"] = stage_output
    return stats, updated_cache


class TextGeneratePipelineRunner:
    """Stateful rank runner for a text-only greedy PP generation replay."""

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

        stage_bundle, compute_dtype = load_stage_bundle_for_rank(
            self.manifest,
            rank,
            self.device,
            self.compute_dtype_arg,
        )
        comm_dtype = resolve_comm_dtype(self.comm_dtype_arg, compute_dtype)
        handoff_transport = StageHandoffTransport(device=self.device, comm_dtype=comm_dtype)
        runtime_only_generate = _is_runtime_only_generate_bundle(stage_bundle)
        runtime_only_context = None
        if runtime_only_generate:
            config_spec = load_text_model_config_spec(self.manifest.runtime_config["model_path"])
            runtime_only_context = {
                "config_spec": config_spec,
                "rotary_emb": build_text_rotary_embedding(config_spec, device=self.device),
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
                stage_type=("text_prefill_last" if rank == world_size - 1 else "text"),
            )
        prefill_stats, prefill_cache = _run_text_generate_phase(
            rank=rank,
            world_size=world_size,
            manifest=self.manifest,
            runtime_bundle=prefill_bundle,
            handoff_transport=handoff_transport,
            phase_kind="prefill",
            current_token_id=None,
            cache_by_layer=None,
            return_tensor=self.return_tensors and rank == world_size - 1,
        )

        current_token_id = _broadcast_token_id(
            prefill_stats["predicted_token_id"] if rank == world_size - 1 else None,
            src=world_size - 1,
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
                    stage_type=("text_decode_last" if rank == world_size - 1 else "text_decode"),
                )
            current_step_stats, cache_by_layer = _run_text_generate_phase(
                rank=rank,
                world_size=world_size,
                manifest=self.manifest,
                runtime_bundle=decode_bundle,
                handoff_transport=handoff_transport,
                phase_kind="decode",
                current_token_id=current_token_id,
                cache_by_layer=cache_by_layer,
                return_tensor=self.return_tensors and rank == world_size - 1,
            )
            current_token_id = _broadcast_token_id(
                current_step_stats["predicted_token_id"] if rank == world_size - 1 else None,
                src=world_size - 1,
            )
            generated_token_ids.append(current_token_id)
            if "stage_output_tensor" in current_step_stats:
                step_output_tensors.append(current_step_stats.pop("stage_output_tensor"))
            step_stats.append(current_step_stats)

        stats = {
            "rank": rank,
            "stage_idx": rank,
            "num_stages": world_size,
            "start_idx": stage_bundle["start_idx"],
            "end_idx": stage_bundle["end_idx"],
            "num_layers": len(stage_bundle["layers"]),
            "device": str(self.device),
            "comm_dtype": str(comm_dtype),
            "max_new_tokens": int(stage_bundle["max_new_tokens"]),
            "prefill_seq_len": int(stage_bundle["prefill_seq_len"]),
            "prefill": prefill_stats,
            "steps": step_stats,
            "generated_token_ids": generated_token_ids,
            "reference_generated_token_ids": (
                None
                if runtime_only_generate or stage_bundle.get("generated_token_ids") is None
                else _token_tensor_to_list(stage_bundle["generated_token_ids"])
            ),
        }
        if self.return_tensors and rank == world_size - 1:
            stats["prefill_output_tensor"] = prefill_stats.pop("stage_output_tensor")
            stats["step_output_tensors"] = step_output_tensors
        return stats


def run_text_generate_pipeline_rank(
    *,
    rank: int,
    world_size: int,
    manifest: TextPipelineManifest,
    device: torch.device,
    compute_dtype_arg: str,
    comm_dtype_arg: str,
    return_tensors: bool = False,
) -> dict:
    runner = TextGeneratePipelineRunner(
        manifest=manifest,
        device=device,
        compute_dtype_arg=compute_dtype_arg,
        comm_dtype_arg=comm_dtype_arg,
        return_tensors=return_tensors,
    )
    return runner.run_rank(rank, world_size)


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
    "TextGeneratePipelineRunner",
    "TextPipelineRunner",
    "tensor_diff_stats",
    "parse_stage_range",
    "parse_stage_ranges",
    "build_stage_bundle_path",
    "prepare_multimodal_decode_pipeline",
    "prepare_multimodal_generate_pipeline",
    "prepare_multimodal_prefill_pipeline",
    "prepare_text_decode_pipeline",
    "prepare_text_generate_pipeline",
    "prepare_text_pipeline",
    "prepare_text_prefill_pipeline",
    "load_pipeline_manifest",
    "load_stage_bundle_by_index",
    "load_stage_bundle_for_rank",
    "run_text_generate_pipeline_rank",
    "run_text_pipeline_rank",
]
