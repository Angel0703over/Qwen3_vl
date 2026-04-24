"""Weight-index and selective-loader helpers for stage-only/shard-only runtime work."""

from qwen3vl_tp_runtime.models.qwen3vl.weights.index import (
    ModelWeightIndex,
    load_model_weight_index,
)
from qwen3vl_tp_runtime.models.qwen3vl.weights.loader import (
    load_tensors_by_name,
    load_tensors_from_index,
)
from qwen3vl_tp_runtime.models.qwen3vl.weights.planner import (
    TensorSliceSpec,
    TextDecoderStageWeightPlan,
    TextTensorParallelShardPlan,
    build_text_decoder_stage_parameter_names,
    build_text_decoder_stage_tp_shard_plan,
    build_text_decoder_stage_weight_plan,
)
from qwen3vl_tp_runtime.models.qwen3vl.weights.text import (
    TextModelConfigSpec,
    TextStageWeightBundle,
    build_text_causal_mask,
    build_text_hf_config,
    build_text_rotary_embedding,
    build_text_runtime_aux_tensors,
    load_text_decoder_stage_weight_bundle,
    load_text_model_config_spec,
    prepare_text_decode_runtime_inputs_from_weights,
    prepare_text_prefill_runtime_inputs_from_weights,
)

__all__ = [
    "ModelWeightIndex",
    "TensorSliceSpec",
    "TextDecoderStageWeightPlan",
    "TextModelConfigSpec",
    "TextStageWeightBundle",
    "TextTensorParallelShardPlan",
    "build_text_causal_mask",
    "build_text_decoder_stage_parameter_names",
    "build_text_decoder_stage_tp_shard_plan",
    "build_text_decoder_stage_weight_plan",
    "build_text_hf_config",
    "build_text_rotary_embedding",
    "build_text_runtime_aux_tensors",
    "load_model_weight_index",
    "load_text_decoder_stage_weight_bundle",
    "load_text_model_config_spec",
    "load_tensors_by_name",
    "load_tensors_from_index",
    "prepare_text_decode_runtime_inputs_from_weights",
    "prepare_text_prefill_runtime_inputs_from_weights",
]
