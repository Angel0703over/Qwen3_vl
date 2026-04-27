"""Compatibility exports for concrete model-family packages.

Prefer importing from the concrete family module such as
`qwen3vl_tp_runtime.models.qwen3vl.*`.

`__all__` describes only the direct-runtime surface. The historical flat
`qwen3vl_tp_runtime.models.*` compatibility attributes for capture/replay remain
lazy and are listed separately under `LEGACY_CAPTURE_EXPORTS`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

DIRECT_RUNTIME_EXPORTS = [
    "StageState",
    "DirectStageStateBuilder",
    "StageStateLoader",
    "extract_decoder_layer_params_live",
    "inspect_model_weights",
    "load_model_weight_index",
    "build_inputs",
    "build_cache_by_layer_from_past_key_values",
    "build_text_causal_mask",
    "build_text_decoder_stage_parameter_names",
    "build_text_decoder_stage_weight_plan",
    "build_direct_stage_state",
    "build_direct_pipeline_manifest",
    "build_direct_tp_manifest",
    "build_direct_hybrid_manifest",
    "materialize_text_stage_state",
    "build_text_hf_config",
    "build_text_rotary_embedding",
    "build_text_inputs",
    "list_frames",
    "load_model",
    "load_processor",
    "load_tensors_by_name",
    "load_tensors_from_index",
    "load_text_decoder_stage_weight_bundle",
    "load_text_model_config_spec",
    "encode_image_features",
    "encode_video_features",
    "materialize_visual_features",
    "ModelWeightIndex",
    "MultimodalRuntimeInputs",
    "TextModelConfigSpec",
    "TextStageWeightBundle",
    "prepare_text_prefill_runtime_inputs",
    "prepare_text_decode_runtime_inputs",
    "prepare_text_prefill_runtime_inputs_from_weights",
    "prepare_text_decode_runtime_inputs_from_weights",
    "prepare_multimodal_prefill_runtime_inputs",
    "prepare_multimodal_decode_runtime_inputs",
    "compose_layer_state",
    "apply_deepstack",
    "apply_rope",
    "attn_eager",
    "forward_attention_cached",
    "forward_attention_cached_tp",
    "get_deepstack_embeds",
    "forward_attention",
    "forward_attention_tp",
    "forward_mlp",
    "forward_mlp_tp",
    "forward_decoder_layer_cached",
    "forward_decoder_layer_cached_tp",
    "forward_decoder_layer",
    "forward_decoder_layer_tp",
    "forward_layer_range",
    "forward_layer_range_tp",
    "forward_text_decode_logits",
    "forward_text_decode_logits_tp",
    "trace_text_decode_logits_with_runtime_cache",
    "forward_text_decode_stage",
    "forward_text_decode_stage_tp",
    "forward_text_embeddings",
    "forward_text_prefill_logits",
    "forward_text_prefill_stage_logits",
    "forward_text_prefill_stage_logits_tp",
    "forward_text_stage",
    "forward_text_stage_tp",
    "trace_attention_cached",
    "trace_attention_cached_tp",
    "trace_text_prefill_logits",
    "trace_text_prefill_stage_logits",
    "trace_decoder_layer_cached",
    "trace_decoder_layer_cached_tp",
    "trace_decoder_layer",
    "trace_decoder_layer_tp",
    "trace_text_decode_logits",
    "trace_text_decode_stage",
    "trace_text_decode_stage_tp",
    "trace_text_stage",
    "trace_text_stage_tp",
    "TextDecoderStageWeightPlan",
    "dtype_from_name",
    "resolve_save_dtype",
    "resolve_comm_dtype",
    "cast_cpu",
    "build_causal_mask",
    "rms_norm",
    "rotate_half",
    "repeat_kv",
]

LEGACY_CAPTURE_EXPORTS = [
    "capture_decoder_layer_params",
    "capture_multimodal_decode_bundle",
    "capture_multimodal_decode_stage_bundle",
    "capture_multimodal_generate_stage_bundle",
    "capture_multimodal_prefill_bundle",
    "capture_multimodal_prefill_stage_bundle",
    "capture_text_decode_bundle",
    "capture_text_decode_stage_bundle",
    "capture_text_generate_bundle",
    "capture_text_generate_stage_bundle",
    "capture_full_layer_bundle",
    "capture_layer_range_bundle",
    "capture_text_prefill_bundle",
    "capture_text_prefill_stage_bundle",
    "capture_text_stage_bundle",
    "extract_past_key_values",
    "load_bundle",
    "move_bundle",
    "resolve_runtime_tensors",
    "run_forward_with_runtime_hook",
]

LEGACY_STAGE_BUNDLE_EXPORTS = [
    "DirectStageBundleBuilder",
    "build_direct_stage_bundle",
    "build_live_multimodal_stage_bundle",
    "compose_layer_bundle",
]

__all__ = [*DIRECT_RUNTIME_EXPORTS]


def __getattr__(name: str) -> Any:
    if name in DIRECT_RUNTIME_EXPORTS or name in LEGACY_CAPTURE_EXPORTS or name in LEGACY_STAGE_BUNDLE_EXPORTS:
        model_mod = import_module("qwen3vl_tp_runtime.models.qwen3vl")
        value = getattr(model_mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | set(DIRECT_RUNTIME_EXPORTS)
        | set(LEGACY_CAPTURE_EXPORTS)
        | set(LEGACY_STAGE_BUNDLE_EXPORTS)
    )
