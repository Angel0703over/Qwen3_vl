"""Direct runtime builders that construct stage bundles from model_path without disk artifacts."""

from __future__ import annotations

from typing import Any

from qwen3vl_tp_runtime.hexgen_core.gen_hetero_groups import build_hybrid_layout, parse_tp_degrees
from qwen3vl_tp_runtime.hexgen_core.schema import StageSpec, TextHybridManifest, TextPipelineManifest
from qwen3vl_tp_runtime.models.qwen3vl.capture import (
    capture_multimodal_decode_stage_bundle,
    capture_multimodal_generate_stage_bundle,
    capture_multimodal_prefill_stage_bundle,
    capture_text_decode_stage_bundle,
    capture_text_generate_stage_bundle,
    capture_text_prefill_stage_bundle,
)


def _pipeline_type(modality: str, mode: str) -> str:
    if modality == "text":
        return f"text_{mode}"
    return f"multimodal_{mode}"


def _runtime_name(modality: str, mode: str, backend: str) -> str:
    return f"{_pipeline_type(modality, mode)}_{backend}"


def _build_runtime_config(
    *,
    modality: str,
    mode: str,
    model_path: str,
    save_dtype: str,
    prompt: str | None = None,
    decode_token_id: int | None = None,
    max_new_tokens: int | None = None,
    num_frames: int | None = None,
    frame_dir: str | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "modality": modality,
        "mode": mode,
        "model_path": model_path,
        "save_dtype": save_dtype,
    }
    if prompt is not None:
        config["prompt"] = prompt
    if decode_token_id is not None:
        config["decode_token_id"] = int(decode_token_id)
    if max_new_tokens is not None:
        config["max_new_tokens"] = int(max_new_tokens)
    if num_frames is not None:
        config["num_frames"] = int(num_frames)
    if frame_dir is not None:
        config["frame_dir"] = frame_dir
    return config


def build_direct_stage_bundle(
    *,
    stage_idx: int,
    start_idx: int,
    end_idx: int,
    runtime_config: dict[str, Any],
) -> dict:
    modality = runtime_config["modality"]
    mode = runtime_config["mode"]
    save_dtype = runtime_config.get("save_dtype", "auto")
    model_path = runtime_config["model_path"]

    common_kwargs = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "bundle_path": None,
        "save_dtype": save_dtype,
        "model_path": model_path,
    }

    if modality == "text":
        if mode == "prefill":
            return capture_text_prefill_stage_bundle(
                prompt=runtime_config.get("prompt", "请用中文简要介绍一下人工智能。"),
                **common_kwargs,
            )
        if mode == "decode":
            return capture_text_decode_stage_bundle(
                prompt=runtime_config.get("prompt", "请用中文简要介绍一下人工智能。"),
                decode_token_id=runtime_config.get("decode_token_id"),
                **common_kwargs,
            )
        if mode == "generate":
            return capture_text_generate_stage_bundle(
                prompt=runtime_config.get("prompt", "请用中文简要介绍一下人工智能。"),
                max_new_tokens=int(runtime_config.get("max_new_tokens", 4)),
                **common_kwargs,
            )
    elif modality == "multimodal":
        multimodal_kwargs = {
            "num_frames": int(runtime_config.get("num_frames", 8)),
        }
        if runtime_config.get("frame_dir") is not None:
            multimodal_kwargs["frame_dir"] = runtime_config["frame_dir"]
        if mode == "prefill":
            return capture_multimodal_prefill_stage_bundle(
                **common_kwargs,
                **multimodal_kwargs,
            )
        if mode == "decode":
            return capture_multimodal_decode_stage_bundle(
                decode_token_id=runtime_config.get("decode_token_id"),
                **common_kwargs,
                **multimodal_kwargs,
            )
        if mode == "generate":
            return capture_multimodal_generate_stage_bundle(
                max_new_tokens=int(runtime_config.get("max_new_tokens", 4)),
                **common_kwargs,
                **multimodal_kwargs,
            )

    raise ValueError(
        f"不支持的 direct stage 构造组合: modality={modality!r} mode={mode!r} stage_idx={stage_idx}"
    )


def build_direct_pipeline_manifest(
    *,
    modality: str,
    mode: str,
    stage_ranges: list[tuple[int, int]],
    model_path: str,
    save_dtype: str,
    prompt: str | None = None,
    decode_token_id: int | None = None,
    max_new_tokens: int | None = None,
    num_frames: int | None = None,
    frame_dir: str | None = None,
) -> TextPipelineManifest:
    runtime_config = _build_runtime_config(
        modality=modality,
        mode=mode,
        model_path=model_path,
        save_dtype=save_dtype,
        prompt=prompt,
        decode_token_id=decode_token_id,
        max_new_tokens=max_new_tokens,
        num_frames=num_frames,
        frame_dir=frame_dir,
    )
    stages = [
        StageSpec(
            stage_idx=stage_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            num_layers=end_idx - start_idx + 1,
            save_dtype=save_dtype,
            bundle_path=None,
        )
        for stage_idx, (start_idx, end_idx) in enumerate(stage_ranges)
    ]
    return TextPipelineManifest(
        pipeline_type=_pipeline_type(modality, mode),
        num_stages=len(stages),
        stage_ranges=stage_ranges,
        bundle_dir="<direct>",
        stages=stages,
        boundaries=[],
        num_frames=0 if modality == "text" else int(num_frames or 8),
        save_dtype=save_dtype,
        runtime_config=runtime_config,
    )


def build_direct_hybrid_manifest(
    *,
    modality: str,
    mode: str,
    stage_ranges: list[tuple[int, int]],
    tp_degrees: list[int],
    model_path: str,
    save_dtype: str,
    prompt: str | None = None,
    decode_token_id: int | None = None,
    max_new_tokens: int | None = None,
    num_frames: int | None = None,
    frame_dir: str | None = None,
    backend: str = "hybrid",
) -> TextHybridManifest:
    pipeline_manifest = build_direct_pipeline_manifest(
        modality=modality,
        mode=mode,
        stage_ranges=stage_ranges,
        model_path=model_path,
        save_dtype=save_dtype,
        prompt=prompt,
        decode_token_id=decode_token_id,
        max_new_tokens=max_new_tokens,
        num_frames=num_frames,
        frame_dir=frame_dir,
    )
    parsed_tp_degrees = parse_tp_degrees(tp_degrees)
    if len(parsed_tp_degrees) != pipeline_manifest.num_stages:
        raise ValueError(
            f"stage 数是 {pipeline_manifest.num_stages}，但 TP 度数拿到 {len(parsed_tp_degrees)} 个。"
        )
    layout = build_hybrid_layout(parsed_tp_degrees)
    return TextHybridManifest.from_pipeline_manifest(
        pipeline_manifest,
        layout,
        runtime=_runtime_name(modality, mode, backend),
    )


__all__ = [
    "build_direct_stage_bundle",
    "build_direct_pipeline_manifest",
    "build_direct_hybrid_manifest",
]
