"""Dedicated PP/TP/Hybrid runtime module surfaces modeled after multi-file engine layouts."""

from qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel import (
    TextHybridRunner,
    build_stage_traces,
    init_stage_groups,
    load_hybrid_manifest,
    prepare_text_hybrid,
    resolve_rank_stage,
    run_text_hybrid_rank,
)
from qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel import (
    BoundaryStats,
    StageSpec,
    TextPipelineManifest,
    TextPipelineRunner,
    build_stage_bundle_path,
    load_pipeline_manifest,
    load_stage_bundle_by_index,
    load_stage_bundle_for_rank,
    parse_stage_range,
    parse_stage_ranges,
    prepare_text_pipeline,
    run_text_pipeline_rank,
    tensor_diff_stats,
)
from qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel import (
    TextTensorParallelRunner,
    load_text_stage_bundle,
    run_text_tensor_parallel_rank,
)

__all__ = [
    "StageSpec",
    "BoundaryStats",
    "TextPipelineManifest",
    "TextPipelineRunner",
    "build_stage_bundle_path",
    "load_pipeline_manifest",
    "load_stage_bundle_by_index",
    "load_stage_bundle_for_rank",
    "parse_stage_range",
    "parse_stage_ranges",
    "prepare_text_pipeline",
    "run_text_pipeline_rank",
    "tensor_diff_stats",
    "TextTensorParallelRunner",
    "load_text_stage_bundle",
    "run_text_tensor_parallel_rank",
    "TextHybridRunner",
    "build_stage_traces",
    "init_stage_groups",
    "load_hybrid_manifest",
    "prepare_text_hybrid",
    "resolve_rank_stage",
    "run_text_hybrid_rank",
]
