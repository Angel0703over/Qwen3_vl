"""Compatibility exports for PP/TP/Hybrid runtime modules.

Prefer importing concrete modules such as `pipeline_parallel`, `hybrid_parallel`,
or `tensor_parallel` directly.

`__all__` intentionally describes the direct-runtime surface. Legacy
prepare/replay helpers remain available as lazy compatibility attributes, but
are listed separately under `LEGACY_REPLAY_EXPORTS`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

DIRECT_RUNTIME_EXPORTS = [
    "StageSpec",
    "BoundaryStats",
    "TextPipelineManifest",
    "TextGeneratePipelineRunner",
    "TextPipelineRunner",
    "load_stage_bundle_by_index",
    "load_stage_bundle_for_rank",
    "parse_stage_range",
    "parse_stage_ranges",
    "run_text_generate_pipeline_rank",
    "run_text_pipeline_rank",
    "tensor_diff_stats",
    "TextHybridRunner",
    "init_stage_groups",
    "resolve_rank_stage",
    "run_text_hybrid_rank",
]

LEGACY_REPLAY_EXPORTS = [
    "build_stage_bundle_path",
    "load_pipeline_manifest",
    "prepare_multimodal_decode_pipeline",
    "prepare_multimodal_generate_pipeline",
    "prepare_multimodal_prefill_pipeline",
    "prepare_text_decode_pipeline",
    "prepare_text_generate_pipeline",
    "prepare_text_prefill_pipeline",
    "prepare_text_pipeline",
    "TextTensorParallelRunner",
    "load_text_stage_bundle",
    "run_text_tensor_parallel_rank",
    "build_stage_traces",
    "load_hybrid_manifest",
    "prepare_multimodal_decode_hybrid",
    "prepare_multimodal_generate_hybrid",
    "prepare_multimodal_prefill_hybrid",
    "prepare_text_decode_hybrid",
    "prepare_text_generate_hybrid",
    "prepare_text_prefill_hybrid",
    "prepare_text_hybrid",
]

__all__ = [*DIRECT_RUNTIME_EXPORTS]

_NAME_TO_MODULE = {
    "StageSpec": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "BoundaryStats": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "TextPipelineManifest": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "TextGeneratePipelineRunner": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "TextPipelineRunner": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "load_stage_bundle_by_index": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "load_stage_bundle_for_rank": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "parse_stage_range": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "parse_stage_ranges": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "run_text_generate_pipeline_rank": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "run_text_pipeline_rank": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "tensor_diff_stats": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "build_stage_bundle_path": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "load_pipeline_manifest": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "prepare_multimodal_decode_pipeline": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "prepare_multimodal_generate_pipeline": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "prepare_multimodal_prefill_pipeline": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "prepare_text_decode_pipeline": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "prepare_text_generate_pipeline": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "prepare_text_prefill_pipeline": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "prepare_text_pipeline": "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel",
    "TextTensorParallelRunner": "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel",
    "load_text_stage_bundle": "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel",
    "run_text_tensor_parallel_rank": "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel",
    "build_stage_traces": "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel",
    "TextHybridRunner": "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel",
    "init_stage_groups": "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel",
    "resolve_rank_stage": "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel",
    "run_text_hybrid_rank": "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel",
    "load_hybrid_manifest": "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel",
    "prepare_multimodal_decode_hybrid": "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel",
    "prepare_multimodal_generate_hybrid": "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel",
    "prepare_multimodal_prefill_hybrid": "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel",
    "prepare_text_decode_hybrid": "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel",
    "prepare_text_generate_hybrid": "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel",
    "prepare_text_prefill_hybrid": "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel",
    "prepare_text_hybrid": "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel",
}

_EXPORT_GROUP_BY_NAME = {
    **{name: "direct" for name in DIRECT_RUNTIME_EXPORTS},
    **{name: "legacy_replay" for name in LEGACY_REPLAY_EXPORTS},
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_GROUP_BY_NAME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(DIRECT_RUNTIME_EXPORTS) | set(LEGACY_REPLAY_EXPORTS))
