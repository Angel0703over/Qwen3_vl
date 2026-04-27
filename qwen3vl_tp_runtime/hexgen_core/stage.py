"""StageState views, multimodal handoff helpers, and stage execution dispatch."""

from dataclasses import dataclass
from typing import Any

import torch

from .schema import StageHandoffPayload, StageState
from ..models.qwen3vl.execution import (
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
from ..models.qwen3vl.functional import dtype_from_name


@dataclass(slots=True)
class StageStateView:
    """A lightweight object wrapper around one direct runtime StageState."""

    payload: dict[str, Any]

    @property
    def stage_type(self) -> str:
        stage_type = self.payload.get("stage_type")
        if stage_type is not None:
            return stage_type
        module_name = self.payload.get("module_name")
        if module_name == "text_stage":
            return "text"
        raise ValueError(f"无法从 StageState 中识别 stage_type，module_name={module_name!r}")

    @property
    def stage_input(self) -> torch.Tensor:
        return self.payload["stage_input"] if "stage_input" in self.payload else self.payload["layer_input"]

    @property
    def stage_output(self) -> torch.Tensor:
        return self.payload["stage_output"] if "stage_output" in self.payload else self.payload["layer_output"]

    def with_stage_type(self, stage_type: str) -> "StageStateView":
        stage_state = dict(self.payload)
        stage_state["stage_type"] = stage_type
        if "stage_input" not in stage_state and "layer_input" in stage_state:
            stage_state["stage_input"] = stage_state["layer_input"]
        if "stage_output" not in stage_state and "layer_output" in stage_state:
            stage_state["stage_output"] = stage_state["layer_output"]
        return StageStateView(stage_state)


StageBundleView = StageStateView


def as_stage_state_view(stage_state: StageState | StageStateView) -> StageStateView:
    if isinstance(stage_state, StageStateView):
        return stage_state
    return StageStateView(stage_state)


def as_stage_bundle_view(stage_bundle: dict[str, Any] | StageBundleView) -> StageBundleView:
    return as_stage_state_view(stage_bundle)


def get_stage_type(stage_state: StageState | StageStateView) -> str:
    return as_stage_state_view(stage_state).stage_type


def get_stage_input(stage_state: StageState | StageStateView) -> torch.Tensor:
    return as_stage_state_view(stage_state).stage_input


def get_stage_output(stage_state: StageState | StageStateView) -> torch.Tensor:
    return as_stage_state_view(stage_state).stage_output


def build_stage_state(stage_type: str, stage_state: StageState) -> StageState:
    return as_stage_state_view(stage_state).with_stage_type(stage_type).payload


def build_stage_bundle(stage_type: str, bundle: dict[str, Any]) -> dict[str, Any]:
    return build_stage_state(stage_type, bundle)


def _infer_hidden_states_dtype(stage_state: StageState) -> torch.dtype:
    stage_input = stage_state.get("stage_input")
    if torch.is_tensor(stage_input):
        return stage_input.dtype

    layer_input = stage_state.get("layer_input")
    if torch.is_tensor(layer_input):
        return layer_input.dtype

    for key in ("embed_tokens_weight", "final_norm_weight", "lm_head_weight"):
        tensor = stage_state.get(key)
        if torch.is_tensor(tensor):
            return tensor.dtype

    layers = stage_state.get("layers")
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

    save_dtype = stage_state.get("save_dtype")
    if isinstance(save_dtype, str) and save_dtype and save_dtype != "auto":
        return dtype_from_name(save_dtype)

    raise KeyError("StageState 缺少可用于推断 hidden_states dtype 的输入或权重。")


def build_stage_handoff_target_dtypes(
    stage_state: StageState | StageStateView,
) -> dict[str, torch.dtype]:
    state_view = as_stage_state_view(stage_state)
    state_payload = state_view.payload
    target_dtypes = {
        StageHandoffPayload.HIDDEN_STATES_KEY: _infer_hidden_states_dtype(state_payload),
    }

    visual_pos_masks = state_payload.get("visual_pos_masks")
    if isinstance(visual_pos_masks, torch.Tensor):
        target_dtypes[StageHandoffPayload.VISUAL_POS_MASKS_KEY] = visual_pos_masks.dtype

    deepstack_by_layer = state_payload.get("deepstack_by_layer")
    if isinstance(deepstack_by_layer, dict):
        for layer_idx, tensor in deepstack_by_layer.items():
            if tensor is not None:
                target_dtypes[StageHandoffPayload.deepstack_key(int(layer_idx))] = tensor.dtype

    multimodal_meta = state_payload.get("multimodal_meta")
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
    stage_state: StageState | StageStateView,
    multimodal_meta: dict[str, torch.Tensor | None] | None = None,
    target_stage_range: tuple[int, int] | None = None,
) -> StageHandoffPayload:
    state_payload = as_stage_state_view(stage_state).payload
    deepstack_by_layer = state_payload.get("deepstack_by_layer")
    state_multimodal_meta = state_payload.get("multimodal_meta")
    filtered_deepstack = _filter_deepstack_for_stage_range(deepstack_by_layer, target_stage_range)

    merged_meta: dict[str, torch.Tensor | None] = {}
    if isinstance(state_multimodal_meta, dict):
        merged_meta.update(state_multimodal_meta)
    if multimodal_meta:
        merged_meta.update(multimodal_meta)

    return StageHandoffPayload(
        hidden_states=hidden_states,
        visual_pos_masks=(
            state_payload.get("visual_pos_masks")
            if target_stage_range is None or filtered_deepstack
            else None
        ),
        deepstack_feature_pack=filtered_deepstack,
        multimodal_meta=merged_meta,
    )


def apply_stage_handoff_payload(
    stage_state: StageState | StageStateView,
    handoff_payload: StageHandoffPayload | None,
    *,
    prefer_local_state: bool = True,
) -> dict[str, Any]:
    state_payload = dict(as_stage_state_view(stage_state).payload)
    if handoff_payload is None:
        return state_payload

    if handoff_payload.hidden_states is not None:
        state_payload["stage_input"] = handoff_payload.hidden_states
        state_payload["layer_input"] = handoff_payload.hidden_states

    if handoff_payload.visual_pos_masks is not None:
        current_visual_pos_masks = state_payload.get("visual_pos_masks")
        if not prefer_local_state or current_visual_pos_masks is None:
            state_payload["visual_pos_masks"] = handoff_payload.visual_pos_masks

    if handoff_payload.deepstack_feature_pack:
        filtered_deepstack = _filter_deepstack_for_stage_range(
            handoff_payload.deepstack_feature_pack,
            (
                (int(state_payload["start_idx"]), int(state_payload["end_idx"]))
                if "start_idx" in state_payload and "end_idx" in state_payload
                else None
            ),
        )
        current_deepstack = state_payload.get("deepstack_by_layer")
        has_local_deepstack = isinstance(current_deepstack, dict) and bool(current_deepstack)
        if filtered_deepstack and (not prefer_local_state or not has_local_deepstack):
            state_payload["deepstack_by_layer"] = dict(sorted(filtered_deepstack.items()))
            state_payload["deepstack_layer_indices"] = sorted(filtered_deepstack)

    if handoff_payload.multimodal_meta:
        incoming_meta = dict(handoff_payload.multimodal_meta)
        current_meta = state_payload.get("multimodal_meta")
        if isinstance(current_meta, dict):
            merged_meta = dict(incoming_meta)
            if prefer_local_state:
                merged_meta.update(current_meta)
            else:
                current_meta = dict(current_meta)
                current_meta.update(incoming_meta)
                merged_meta = current_meta
        else:
            merged_meta = incoming_meta

        state_payload["multimodal_meta"] = merged_meta
        for name, tensor in merged_meta.items():
            if not prefer_local_state or state_payload.get(name) is None:
                state_payload[name] = tensor

    return state_payload


def run_stage(stage_input: torch.Tensor, stage_state: StageState | StageStateView) -> torch.Tensor:
    state_view = as_stage_state_view(stage_state)
    if state_view.stage_type == "text":
        return forward_text_stage(stage_input, state_view.payload)
    if state_view.stage_type == "text_decode":
        return forward_text_decode_stage(stage_input, state_view.payload)
    if state_view.stage_type == "text_decode_last":
        return forward_text_decode_logits(stage_input, state_view.payload)
    if state_view.stage_type == "text_prefill_last":
        return forward_text_prefill_stage_logits(stage_input, state_view.payload)
    raise NotImplementedError(f"暂不支持的 stage_type: {state_view.stage_type}")


def run_stage_tp(
    stage_input: torch.Tensor,
    stage_state: StageState | StageStateView,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    tp_attn_math_mode: str = "orig",
    tp_mlp_math_mode: str = "orig",
) -> torch.Tensor:
    state_view = as_stage_state_view(stage_state)
    if state_view.stage_type == "text":
        return forward_text_stage_tp(
            stage_input,
            state_view.payload,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
        )
    if state_view.stage_type == "text_decode":
        return forward_text_decode_stage_tp(
            stage_input,
            state_view.payload,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
        )
    if state_view.stage_type == "text_decode_last":
        return forward_text_decode_logits_tp(
            stage_input,
            state_view.payload,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
        )
    if state_view.stage_type == "text_prefill_last":
        return forward_text_prefill_stage_logits_tp(
            stage_input,
            state_view.payload,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
        )
    raise NotImplementedError(f"暂不支持的 stage_type: {state_view.stage_type}")


def trace_stage(stage_input: torch.Tensor, stage_state: StageState | StageStateView):
    state_view = as_stage_state_view(stage_state)
    if state_view.stage_type in {"text", "text_prefill_last"}:
        return trace_text_stage(stage_input, state_view.payload)
    if state_view.stage_type in {"text_decode", "text_decode_last"}:
        return trace_text_decode_stage(stage_input, state_view.payload)
    raise NotImplementedError(f"暂不支持的 stage_type: {state_view.stage_type}")


def trace_stage_tp(
    stage_input: torch.Tensor,
    stage_state: StageState | StageStateView,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    tp_attn_math_mode: str = "orig",
    tp_mlp_math_mode: str = "orig",
):
    state_view = as_stage_state_view(stage_state)
    if state_view.stage_type in {"text", "text_prefill_last"}:
        return trace_text_stage_tp(
            stage_input,
            state_view.payload,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
        )
    if state_view.stage_type in {"text_decode", "text_decode_last"}:
        return trace_text_decode_stage_tp(
            stage_input,
            state_view.payload,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
        )
    raise NotImplementedError(f"暂不支持的 stage_type: {state_view.stage_type}")


__all__ = [
    "StageStateView",
    "as_stage_state_view",
    "get_stage_type",
    "get_stage_input",
    "get_stage_output",
    "build_stage_state",
    "build_stage_handoff_target_dtypes",
    "build_stage_handoff_payload",
    "apply_stage_handoff_payload",
    "run_stage",
    "run_stage_tp",
    "trace_stage",
    "trace_stage_tp",
]
