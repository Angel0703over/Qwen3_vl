"""Stage bundle views, multimodal handoff helpers, and stage execution dispatch."""

from dataclasses import dataclass
from typing import Any

import torch

from qwen3vl_tp_runtime.hexgen_core.schema import StageHandoffPayload
from qwen3vl_tp_runtime.models.qwen3vl.execution import (
    forward_text_decode_logits,
    forward_text_decode_logits_tp,
    forward_text_decode_stage,
    forward_text_decode_stage_tp,
    forward_text_prefill_stage_logits,
    forward_text_prefill_stage_logits_tp,
    forward_text_stage,
    forward_text_stage_tp,
    trace_text_decode_stage,
    trace_text_decode_stage_tp,
    trace_text_stage,
    trace_text_stage_tp,
)
from qwen3vl_tp_runtime.models.qwen3vl.functional import dtype_from_name


@dataclass(slots=True)
class StageBundleView:
    """A lightweight object wrapper around one captured stage bundle."""

    payload: dict[str, Any]

    @property
    def stage_type(self) -> str:
        stage_type = self.payload.get("stage_type")
        if stage_type is not None:
            return stage_type
        module_name = self.payload.get("module_name")
        if module_name == "text_stage":
            return "text"
        raise ValueError(f"无法从 stage bundle 中识别 stage_type，module_name={module_name!r}")

    @property
    def stage_input(self) -> torch.Tensor:
        return self.payload["stage_input"] if "stage_input" in self.payload else self.payload["layer_input"]

    @property
    def stage_output(self) -> torch.Tensor:
        return self.payload["stage_output"] if "stage_output" in self.payload else self.payload["layer_output"]

    def with_stage_type(self, stage_type: str) -> "StageBundleView":
        bundle = dict(self.payload)
        bundle["stage_type"] = stage_type
        if "stage_input" not in bundle and "layer_input" in bundle:
            bundle["stage_input"] = bundle["layer_input"]
        if "stage_output" not in bundle and "layer_output" in bundle:
            bundle["stage_output"] = bundle["layer_output"]
        return StageBundleView(bundle)


def as_stage_bundle_view(stage_bundle: dict[str, Any] | StageBundleView) -> StageBundleView:
    if isinstance(stage_bundle, StageBundleView):
        return stage_bundle
    return StageBundleView(stage_bundle)


def get_stage_type(stage_bundle: dict[str, Any] | StageBundleView) -> str:
    return as_stage_bundle_view(stage_bundle).stage_type


def get_stage_input(stage_bundle: dict[str, Any] | StageBundleView) -> torch.Tensor:
    return as_stage_bundle_view(stage_bundle).stage_input


def get_stage_output(stage_bundle: dict[str, Any] | StageBundleView) -> torch.Tensor:
    return as_stage_bundle_view(stage_bundle).stage_output


def build_stage_bundle(stage_type: str, bundle: dict[str, Any]) -> dict[str, Any]:
    return as_stage_bundle_view(bundle).with_stage_type(stage_type).payload


def _infer_hidden_states_dtype(bundle: dict[str, Any]) -> torch.dtype:
    stage_input = bundle.get("stage_input")
    if torch.is_tensor(stage_input):
        return stage_input.dtype

    layer_input = bundle.get("layer_input")
    if torch.is_tensor(layer_input):
        return layer_input.dtype

    for key in ("embed_tokens_weight", "final_norm_weight", "lm_head_weight"):
        tensor = bundle.get(key)
        if torch.is_tensor(tensor):
            return tensor.dtype

    layers = bundle.get("layers")
    if isinstance(layers, list):
        for layer_bundle in layers:
            if not isinstance(layer_bundle, dict):
                continue
            for key in (
                "input_layernorm_weight",
                "q_weight",
                "k_weight",
                "v_weight",
                "o_weight",
                "gate_weight",
                "up_weight",
                "down_weight",
                "post_attention_layernorm_weight",
            ):
                tensor = layer_bundle.get(key)
                if torch.is_tensor(tensor):
                    return tensor.dtype

    save_dtype = bundle.get("save_dtype")
    if isinstance(save_dtype, str) and save_dtype and save_dtype != "auto":
        return dtype_from_name(save_dtype)

    raise KeyError("stage bundle 缺少可用于推断 hidden_states dtype 的输入或权重。")


def build_stage_handoff_target_dtypes(
    stage_bundle: dict[str, Any] | StageBundleView,
) -> dict[str, torch.dtype]:
    bundle_view = as_stage_bundle_view(stage_bundle)
    bundle = bundle_view.payload
    target_dtypes = {
        StageHandoffPayload.HIDDEN_STATES_KEY: _infer_hidden_states_dtype(bundle),
    }

    visual_pos_masks = bundle.get("visual_pos_masks")
    if isinstance(visual_pos_masks, torch.Tensor):
        target_dtypes[StageHandoffPayload.VISUAL_POS_MASKS_KEY] = visual_pos_masks.dtype

    deepstack_by_layer = bundle.get("deepstack_by_layer")
    if isinstance(deepstack_by_layer, dict):
        for layer_idx, tensor in deepstack_by_layer.items():
            if tensor is not None:
                target_dtypes[StageHandoffPayload.deepstack_key(int(layer_idx))] = tensor.dtype

    multimodal_meta = bundle.get("multimodal_meta")
    if isinstance(multimodal_meta, dict):
        for name, tensor in multimodal_meta.items():
            if tensor is not None:
                target_dtypes[StageHandoffPayload.multimodal_meta_key(str(name))] = tensor.dtype

    return target_dtypes


def _filter_deepstack_for_stage_range(
    deepstack_by_layer: dict[Any, torch.Tensor | None] | None,
    target_stage_range: tuple[int, int] | None,
) -> dict[int, torch.Tensor | None]:
    if not isinstance(deepstack_by_layer, dict):
        return {}

    filtered: dict[int, torch.Tensor | None] = {}
    for layer_idx, tensor in deepstack_by_layer.items():
        layer_idx_int = int(layer_idx)
        if target_stage_range is not None:
            start_idx, end_idx = target_stage_range
            if not start_idx <= layer_idx_int <= end_idx:
                continue
        filtered[layer_idx_int] = tensor
    return filtered


def build_stage_handoff_payload(
    hidden_states: torch.Tensor | None,
    stage_bundle: dict[str, Any] | StageBundleView,
    multimodal_meta: dict[str, torch.Tensor | None] | None = None,
    target_stage_range: tuple[int, int] | None = None,
) -> StageHandoffPayload:
    bundle = as_stage_bundle_view(stage_bundle).payload
    deepstack_by_layer = bundle.get("deepstack_by_layer")
    bundle_multimodal_meta = bundle.get("multimodal_meta")
    filtered_deepstack = _filter_deepstack_for_stage_range(deepstack_by_layer, target_stage_range)

    merged_meta: dict[str, torch.Tensor | None] = {}
    if isinstance(bundle_multimodal_meta, dict):
        merged_meta.update(bundle_multimodal_meta)
    if multimodal_meta:
        merged_meta.update(multimodal_meta)

    return StageHandoffPayload(
        hidden_states=hidden_states,
        visual_pos_masks=(
            bundle.get("visual_pos_masks")
            if target_stage_range is None or filtered_deepstack
            else None
        ),
        deepstack_feature_pack=filtered_deepstack,
        multimodal_meta=merged_meta,
    )


def apply_stage_handoff_payload(
    stage_bundle: dict[str, Any] | StageBundleView,
    handoff_payload: StageHandoffPayload | None,
    *,
    prefer_local_bundle: bool = True,
) -> dict[str, Any]:
    bundle = dict(as_stage_bundle_view(stage_bundle).payload)
    if handoff_payload is None:
        return bundle

    if handoff_payload.hidden_states is not None:
        bundle["stage_input"] = handoff_payload.hidden_states
        bundle["layer_input"] = handoff_payload.hidden_states

    if handoff_payload.visual_pos_masks is not None:
        current_visual_pos_masks = bundle.get("visual_pos_masks")
        if not prefer_local_bundle or current_visual_pos_masks is None:
            bundle["visual_pos_masks"] = handoff_payload.visual_pos_masks

    if handoff_payload.deepstack_feature_pack:
        filtered_deepstack = _filter_deepstack_for_stage_range(
            handoff_payload.deepstack_feature_pack,
            (
                (int(bundle["start_idx"]), int(bundle["end_idx"]))
                if "start_idx" in bundle and "end_idx" in bundle
                else None
            ),
        )
        current_deepstack = bundle.get("deepstack_by_layer")
        has_local_deepstack = isinstance(current_deepstack, dict) and bool(current_deepstack)
        if filtered_deepstack and (not prefer_local_bundle or not has_local_deepstack):
            bundle["deepstack_by_layer"] = dict(sorted(filtered_deepstack.items()))
            bundle["deepstack_layer_indices"] = sorted(filtered_deepstack)

    if handoff_payload.multimodal_meta:
        incoming_meta = dict(handoff_payload.multimodal_meta)
        current_meta = bundle.get("multimodal_meta")
        if isinstance(current_meta, dict):
            merged_meta = dict(incoming_meta)
            if prefer_local_bundle:
                merged_meta.update(current_meta)
            else:
                current_meta = dict(current_meta)
                current_meta.update(incoming_meta)
                merged_meta = current_meta
        else:
            merged_meta = incoming_meta

        bundle["multimodal_meta"] = merged_meta
        for name, tensor in merged_meta.items():
            if not prefer_local_bundle or bundle.get(name) is None:
                bundle[name] = tensor

    return bundle


def run_stage(stage_input: torch.Tensor, stage_bundle: dict[str, Any] | StageBundleView) -> torch.Tensor:
    bundle_view = as_stage_bundle_view(stage_bundle)
    if bundle_view.stage_type == "text":
        return forward_text_stage(stage_input, bundle_view.payload)
    if bundle_view.stage_type == "text_decode":
        return forward_text_decode_stage(stage_input, bundle_view.payload)
    if bundle_view.stage_type == "text_decode_last":
        return forward_text_decode_logits(stage_input, bundle_view.payload)
    if bundle_view.stage_type == "text_prefill_last":
        return forward_text_prefill_stage_logits(stage_input, bundle_view.payload)
    raise NotImplementedError(f"暂不支持的 stage_type: {bundle_view.stage_type}")


def run_stage_tp(
    stage_input: torch.Tensor,
    stage_bundle: dict[str, Any] | StageBundleView,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    tp_attn_math_mode: str = "orig",
    tp_mlp_math_mode: str = "orig",
) -> torch.Tensor:
    bundle_view = as_stage_bundle_view(stage_bundle)
    if bundle_view.stage_type == "text":
        return forward_text_stage_tp(
            stage_input,
            bundle_view.payload,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
        )
    if bundle_view.stage_type == "text_decode":
        return forward_text_decode_stage_tp(
            stage_input,
            bundle_view.payload,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
        )
    if bundle_view.stage_type == "text_decode_last":
        return forward_text_decode_logits_tp(
            stage_input,
            bundle_view.payload,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
        )
    if bundle_view.stage_type == "text_prefill_last":
        return forward_text_prefill_stage_logits_tp(
            stage_input,
            bundle_view.payload,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
        )
    raise NotImplementedError(f"暂不支持的 stage_type: {bundle_view.stage_type}")


def trace_stage(stage_input: torch.Tensor, stage_bundle: dict[str, Any] | StageBundleView):
    bundle_view = as_stage_bundle_view(stage_bundle)
    if bundle_view.stage_type in {"text", "text_prefill_last"}:
        return trace_text_stage(stage_input, bundle_view.payload)
    if bundle_view.stage_type in {"text_decode", "text_decode_last"}:
        return trace_text_decode_stage(stage_input, bundle_view.payload)
    raise NotImplementedError(f"暂不支持的 stage_type: {bundle_view.stage_type}")


def trace_stage_tp(
    stage_input: torch.Tensor,
    stage_bundle: dict[str, Any] | StageBundleView,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    tp_attn_math_mode: str = "orig",
    tp_mlp_math_mode: str = "orig",
):
    bundle_view = as_stage_bundle_view(stage_bundle)
    if bundle_view.stage_type in {"text", "text_prefill_last"}:
        return trace_text_stage_tp(
            stage_input,
            bundle_view.payload,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
        )
    if bundle_view.stage_type in {"text_decode", "text_decode_last"}:
        return trace_text_decode_stage_tp(
            stage_input,
            bundle_view.payload,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
        )
    raise NotImplementedError(f"暂不支持的 stage_type: {bundle_view.stage_type}")


__all__ = [
    "StageBundleView",
    "as_stage_bundle_view",
    "get_stage_type",
    "get_stage_input",
    "get_stage_output",
    "build_stage_bundle",
    "build_stage_handoff_target_dtypes",
    "build_stage_handoff_payload",
    "apply_stage_handoff_payload",
    "run_stage",
    "run_stage_tp",
    "trace_stage",
    "trace_stage_tp",
]
