"""Multimodal live-session helpers for direct runtime building."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from qwen3vl_tp_runtime.models.qwen3vl.execution import apply_deepstack
from qwen3vl_tp_runtime.models.qwen3vl.live.common import (
    MultimodalRuntimeInputs,
)
from qwen3vl_tp_runtime.models.qwen3vl.processing import (
    load_model,
)
from qwen3vl_tp_runtime.models.qwen3vl.runtime_mm_stage import (
    MmRuntimeState,
    MmStateLike,
    build_mm_stage_visual_payload,
    compact_mm_frontend_meta,
    compact_mm_frontend_tensors,
    clone_mm_frontend_seed,
    mm_deepstack_embeds,
    mm_runtime_inputs_from_state,
    mm_text_position_ids,
    move_mm_state,
    restore_mm_frontend_seed_tensors,
)
from qwen3vl_tp_runtime.models.qwen3vl.vision.runtime import (
    prepare_mm_frontend_parts,
    prepare_mm_frontend_seed_parts,
)
from qwen3vl_tp_runtime.models.qwen3vl.weights import (
    load_mm_frontend_config,
    build_text_rotary_embedding,
    load_text_model_config_spec,
)


@dataclass(slots=True)
class MmPrefillSession:
    """Multimodal prefill session state used by runtime/debug helpers."""

    model: torch.nn.Module | None
    raw_inputs: dict[str, Any]
    frontend_state: MmRuntimeState | None
    extra: dict[str, Any]

    @property
    def runtime_inputs(self) -> MultimodalRuntimeInputs:
        if self.frontend_state is not None:
            return mm_runtime_inputs_from_state(self.frontend_state)
        raise RuntimeError("multimodal prefill session 缺少 frontend state。")


@dataclass(slots=True)
class MmFrontendSource:
    """Resolved multimodal frontend tensors plus minimal runtime metadata."""

    frontend_state: MmStateLike
    num_frames: int
    frame_paths: list[str]
    frontend_activation: str
    video_input_metadata: dict[str, Any] | None = None
    frontend_seed: MmStateLike | None = None


def _decoder_device(model: torch.nn.Module) -> torch.device:
    if isinstance(model, torch.nn.Module):
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device):
        return model_device
    raise RuntimeError(f"无法解析 multimodal decoder device: {type(model)!r}")


def _state_device(frontend_state: MmStateLike) -> torch.device:
    for tensor in (
        frontend_state.inputs_embeds,
        frontend_state.attention_mask,
        frontend_state.cos,
        frontend_state.sin,
        frontend_state.input_ids,
        frontend_state.attention_mask_2d,
        frontend_state.position_ids,
        frontend_state.visual_pos_masks,
        frontend_state.rope_deltas,
    ):
        if torch.is_tensor(tensor):
            return tensor.device
    for deepstack in frontend_state.deepstack_by_layer.values():
        if torch.is_tensor(deepstack):
            return deepstack.device
    return torch.device("cpu")


def _build_mm_raw_inputs_from_state(frontend_state: MmStateLike) -> dict[str, Any]:
    raw_inputs: dict[str, Any] = {
        "input_ids": frontend_state.input_ids,
    }
    if frontend_state.attention_mask_2d is not None:
        raw_inputs["attention_mask"] = frontend_state.attention_mask_2d
    return raw_inputs


def text_pos_ids(runtime_inputs: MmStateLike) -> torch.Tensor | None:
    return mm_text_position_ids(runtime_inputs)


def deepstack_embeds(
    runtime_inputs: MmStateLike,
) -> list[torch.Tensor | None] | None:
    return mm_deepstack_embeds(runtime_inputs)


def _build_seeded_mm_session(
    runtime_config: dict[str, Any],
    *,
    frontend_source: MmFrontendSource,
    decoder_model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    load_decoder_model: bool = True,
) -> MmPrefillSession:
    model = decoder_model
    target_device = device
    if model is None and load_decoder_model:
        model = load_model(runtime_config["model_path"], attn_implementation="eager")
    if target_device is None:
        target_device = _decoder_device(model) if model is not None else _state_device(frontend_source.frontend_state)
    frontend_state = move_mm_state(frontend_source.frontend_state, device=target_device)
    raw_inputs = _build_mm_raw_inputs_from_state(frontend_state)
    return MmPrefillSession(
        model=model,
        raw_inputs=raw_inputs,
        frontend_state=frontend_state,
        extra={
            "num_frames": frontend_source.num_frames,
            "frame_paths": list(frontend_source.frame_paths),
            "frontend_activation": frontend_source.frontend_activation,
            "video_input_metadata": dict(frontend_source.video_input_metadata or {}),
        },
    )


def prepare_mm_frontend(
    runtime_config: dict[str, Any],
    *,
    model: torch.nn.Module | None = None,
) -> MmPrefillSession:
    """Run the live multimodal frontend once and return explicit runtime state."""

    frontend_model, frontend_batch, frontend_state = prepare_mm_frontend_parts(
        runtime_config,
        model=model,
    )
    return MmPrefillSession(
        model=frontend_model,
        raw_inputs=frontend_batch.raw_inputs,
        frontend_state=frontend_state,
        extra={
            "num_frames": frontend_batch.num_frames,
            "frame_paths": list(frontend_batch.frame_paths),
            "frontend_activation": "active",
            "video_input_metadata": dict(frontend_batch.video_input_metadata or {}),
        },
    )


def prepare_mm_session(
    runtime_config: dict[str, Any],
    *,
    activate_frontend: bool = True,
    decoder_model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    load_decoder_model: bool = True,
) -> MmPrefillSession:
    """Prepare one multimodal session from frontend seed/state.

    `activate_frontend=False` matches later PP stages in vLLM-style execution:
    the stage may still instantiate the multimodal model structure, but it must
    consume an already prepared frontend seed/state instead of running the visual
    frontend locally.

    This helper is now seed/state-first. Callers can keep `load_decoder_model=True`
    for legacy live-session behavior, or set it to `False` to stay on a thinner
    frontend-only/session-only path while the project moves toward decoder-stage-only
    runtime construction.
    """

    frontend_source = resolve_mm_frontend(
        runtime_config,
        activate_frontend=activate_frontend,
    )
    return _build_seeded_mm_session(
        runtime_config,
        frontend_source=frontend_source,
        decoder_model=decoder_model,
        device=device,
        load_decoder_model=load_decoder_model,
    )


def prepare_mm_frontend_seed(runtime_config: dict[str, Any]) -> dict[str, Any]:
    """Prepare one CPU payload that can seed multimodal direct loaders."""

    _frontend_model, frontend_batch, frontend_seed = prepare_mm_frontend_seed_parts(
        runtime_config,
        device=torch.device("cpu"),
    )
    return {
        "frontend_tensors": compact_mm_frontend_tensors(frontend_seed),
        "frontend_meta": {
            "runtime": compact_mm_frontend_meta(frontend_seed),
            "num_frames": frontend_batch.num_frames,
            "frame_paths": list(frontend_batch.frame_paths),
            "video_input_metadata": dict(frontend_batch.video_input_metadata or {}),
        },
    }


def restore_mm_frontend_seed(payload: dict[str, Any]) -> dict[str, Any]:
    """Clone a broadcast multimodal frontend seed back into local runtime payload."""
    return _restore_mm_frontend_seed_payload(payload)


def _restore_mm_frontend_seed_payload(
    payload: dict[str, Any],
    *,
    runtime_config: dict[str, Any] | None = None,
    config_spec=None,
    mm_config=None,
    device: torch.device | None = None,
    compute_dtype: torch.dtype | None = None,
    rotary_emb=None,
) -> dict[str, Any]:
    """Restore one multimodal frontend payload into local seed tensors."""

    legacy_frontend_plan = payload.get("frontend_plan")
    if legacy_frontend_plan is not None:
        raise RuntimeError(
            "legacy frontend_plan payload 已下线，请改用 frontend_tensors/frontend_state。"
        )

    frontend_tensors = payload.get("frontend_tensors")
    frontend_meta = payload.get("frontend_meta")
    if frontend_tensors is not None:
        meta = frontend_meta if isinstance(frontend_meta, dict) else {}
        runtime_meta = meta.get("runtime")
        restore_payload: dict[str, Any]
        if isinstance(runtime_meta, Mapping):
            restore_payload = {
                key: value
                for key, value in runtime_meta.items()
            }
            restore_payload.update(frontend_tensors if isinstance(frontend_tensors, Mapping) else {})
        elif isinstance(frontend_tensors, Mapping):
            restore_payload = dict(frontend_tensors)
        else:
            restore_payload = {}
        resolved_config_spec = config_spec
        resolved_mm_config = mm_config
        resolved_device = device
        resolved_compute_dtype = compute_dtype
        if restore_payload and (
            "attention_mask" not in restore_payload
            or "cos" not in restore_payload
            or "sin" not in restore_payload
        ):
            if resolved_device is None:
                inputs_embeds = restore_payload.get("inputs_embeds")
                if torch.is_tensor(inputs_embeds):
                    resolved_device = inputs_embeds.device
                    if resolved_compute_dtype is None and inputs_embeds.is_floating_point():
                        resolved_compute_dtype = inputs_embeds.dtype
            if resolved_config_spec is None and runtime_config is not None:
                resolved_config_spec = load_text_model_config_spec(runtime_config["model_path"])
            if resolved_device is not None and rotary_emb is None and resolved_config_spec is not None:
                rotary_emb = build_text_rotary_embedding(
                    resolved_config_spec,
                    device=resolved_device,
                )
        if restore_payload and (
            "position_ids" not in restore_payload
            or "rope_deltas" not in restore_payload
        ):
            if resolved_mm_config is None and runtime_config is not None:
                resolved_mm_config = runtime_config.get("_mm_frontend_config")
                if resolved_mm_config is None:
                    resolved_mm_config = load_mm_frontend_config(runtime_config["model_path"])
                    runtime_config["_mm_frontend_config"] = resolved_mm_config
        restored_frontend_tensors = (
            restore_mm_frontend_seed_tensors(
                restore_payload,
                config_spec=resolved_config_spec,
                mm_config=resolved_mm_config,
                device=resolved_device,
                compute_dtype=resolved_compute_dtype,
                rotary_emb=rotary_emb,
            )
            if isinstance(frontend_tensors, Mapping)
            else clone_mm_frontend_seed(frontend_tensors)
        )
        return {
            "frontend_tensors": restored_frontend_tensors,
            "num_frames": int(meta.get("num_frames", payload.get("num_frames", 0))),
            "frame_paths": list(meta.get("frame_paths") or payload.get("frame_paths") or []),
            "video_input_metadata": dict(
                meta.get("video_input_metadata") or payload.get("video_input_metadata") or {}
            ),
        }

    frontend_state = payload.get("frontend_state")
    if frontend_state is None:
        raise RuntimeError(
            "multimodal frontend seed 缺少 frontend_tensors/frontend_state。"
        )
    return {
        "frontend_tensors": clone_mm_frontend_seed(frontend_state),
        "num_frames": int(payload.get("num_frames", 0)),
        "frame_paths": list(payload.get("frame_paths") or []),
        "video_input_metadata": dict(payload.get("video_input_metadata") or {}),
    }


def seed_mm_runtime_config(runtime_config: dict[str, Any], payload: dict[str, Any]) -> None:
    """Inject one broadcast multimodal frontend seed into runtime_config."""

    frontend_tensors = payload.get("frontend_tensors")
    frontend_meta = payload.get("frontend_meta")
    if frontend_tensors is not None:
        runtime_config["_mm_frontend_seed"] = compact_mm_frontend_tensors(frontend_tensors)
        meta = frontend_meta if isinstance(frontend_meta, dict) else {}
        runtime_meta = meta.get("runtime")
        if isinstance(runtime_meta, Mapping):
            runtime_config["_mm_frontend_meta"] = compact_mm_frontend_meta(runtime_meta)
        else:
            runtime_config["_mm_frontend_meta"] = compact_mm_frontend_meta(frontend_tensors)
        runtime_config["_mm_num_frames"] = int(meta.get("num_frames", payload.get("num_frames", 0)))
        runtime_config["_mm_frame_paths"] = list(meta.get("frame_paths") or payload.get("frame_paths") or [])
        runtime_config["_mm_video_input_metadata"] = dict(
            meta.get("video_input_metadata") or payload.get("video_input_metadata") or {}
        )
    else:
        restored = restore_mm_frontend_seed(payload)
        runtime_config["_mm_frontend_seed"] = compact_mm_frontend_tensors(restored["frontend_tensors"])
        runtime_config["_mm_frontend_meta"] = compact_mm_frontend_meta(restored["frontend_tensors"])
        runtime_config["_mm_num_frames"] = restored["num_frames"]
        runtime_config["_mm_frame_paths"] = restored["frame_paths"]
        runtime_config["_mm_video_input_metadata"] = dict(restored.get("video_input_metadata") or {})
    runtime_config.pop("_mm_frontend_plan", None)
    runtime_config.pop("_mm_frontend_state", None)
    runtime_config["_mm_frontend_state_ready"] = True


def resolve_mm_frontend(
    runtime_config: dict[str, Any],
    *,
    activate_frontend: bool,
    config_spec=None,
    device: torch.device | None = None,
    compute_dtype: torch.dtype | None = None,
    rotary_emb=None,
) -> MmFrontendSource:
    """Resolve one multimodal frontend source into the seed/state mainline."""

    if runtime_config.get("_mm_frontend_plan") is not None:
        raise RuntimeError(
            "legacy `_mm_frontend_plan` 已下线，请改用 `_mm_frontend_seed` 或 `_mm_frontend_state`。"
        )

    seeded_frontend_seed = runtime_config.get("_mm_frontend_seed")
    seeded_frontend_state = runtime_config.get("_mm_frontend_state")
    frontend_activation = "consume-only"

    if (
        seeded_frontend_seed is None
        and seeded_frontend_state is None
    ):
        if not activate_frontend:
            raise RuntimeError(
                "multimodal consume-only stage 需要预先广播的 frontend seed/state，"
                "当前 runtime_config 里没有 `_mm_frontend_seed` 或 `_mm_frontend_state`。"
            )

        frontend_seed = prepare_mm_frontend_seed(dict(runtime_config))
        seed_mm_runtime_config(runtime_config, frontend_seed)
        seeded_frontend_seed = runtime_config.get("_mm_frontend_seed")
        frontend_activation = "active"
    elif activate_frontend:
        frontend_activation = "seeded"

    frontend_state = seeded_frontend_seed
    if frontend_state is None:
        frontend_state = seeded_frontend_state
    if frontend_state is None:
        raise RuntimeError(
            "multimodal frontend seed/state 准备失败。"
        )
    if isinstance(frontend_state, Mapping):
        frontend_state = _restore_mm_frontend_seed_payload(
            {
                "frontend_tensors": frontend_state,
                "frontend_meta": {
                    "runtime": runtime_config.get("_mm_frontend_meta"),
                    "num_frames": runtime_config.get("_mm_num_frames", runtime_config.get("num_frames", 0)),
                    "frame_paths": list(runtime_config.get("_mm_frame_paths") or []),
                    "video_input_metadata": dict(runtime_config.get("_mm_video_input_metadata") or {}),
                },
            },
            runtime_config=runtime_config,
            config_spec=config_spec,
            device=device,
            compute_dtype=compute_dtype,
            rotary_emb=rotary_emb,
        )["frontend_tensors"]

    frame_paths = list(runtime_config.get("_mm_frame_paths") or [])
    num_frames = int(runtime_config.get("_mm_num_frames", runtime_config.get("num_frames", len(frame_paths))))
    video_input_metadata = dict(runtime_config.get("_mm_video_input_metadata") or {})
    runtime_config["_mm_frontend_state_ready"] = True
    return MmFrontendSource(
        frontend_state=frontend_state,
        num_frames=num_frames,
        frame_paths=frame_paths,
        frontend_activation=frontend_activation,
        video_input_metadata=video_input_metadata,
        frontend_seed=seeded_frontend_seed,
    )


def run_mm_prefill_ref(
    model,
    *,
    runtime_inputs: MmStateLike,
    start_idx: int,
    end_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run multimodal prefill up to one stage boundary and capture stage tensors."""

    text_model = model.model.language_model
    hidden_states = runtime_inputs.inputs_embeds
    stage_input = hidden_states.detach().clone() if start_idx == 0 else None
    position_embeddings = (runtime_inputs.cos, runtime_inputs.sin)
    stage_text_pos_ids = text_pos_ids(runtime_inputs)

    with torch.inference_mode():
        for layer_idx in range(end_idx + 1):
            hidden_states = text_model.layers[layer_idx](
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=runtime_inputs.attention_mask,
                position_ids=stage_text_pos_ids,
                past_key_values=None,
                use_cache=False,
            )
            hidden_states = apply_deepstack(
                hidden_states,
                runtime_inputs.visual_pos_masks,
                runtime_inputs.deepstack_by_layer.get(layer_idx),
            )
            if layer_idx + 1 == start_idx:
                stage_input = hidden_states.detach().clone()

    if stage_input is None:
        raise RuntimeError(f"没有拿到 multimodal prefill stage_input，start_idx={start_idx} end_idx={end_idx}")
    return stage_input, hidden_states.detach().clone()


def run_mm_prefill(model, runtime_inputs: MmStateLike):
    """Run the full multimodal prefill forward on the live decoder."""

    text_model = model.model.language_model
    with torch.inference_mode():
        return text_model(
            inputs_embeds=runtime_inputs.inputs_embeds,
            attention_mask=runtime_inputs.attention_mask_2d,
            position_ids=runtime_inputs.position_ids,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
            visual_pos_masks=runtime_inputs.visual_pos_masks,
            deepstack_visual_embeds=deepstack_embeds(runtime_inputs),
        )


def run_mm_decode(
    model,
    *,
    runtime_inputs: MmStateLike,
    past_key_values,
    is_last_stage: bool,
):
    """Run one multimodal decode step on the live decoder."""

    text_model = model.model.language_model
    captured: dict[str, torch.Tensor] = {}

    handle = None
    if is_last_stage:
        def final_norm_pre_hook(_module, module_inputs):
            captured["hidden_stage_output"] = module_inputs[0].detach().clone()

        handle = text_model.norm.register_forward_pre_hook(final_norm_pre_hook)

    try:
        with torch.inference_mode():
            outputs = text_model(
                inputs_embeds=runtime_inputs.inputs_embeds,
                attention_mask=runtime_inputs.attention_mask_2d,
                position_ids=runtime_inputs.position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                visual_pos_masks=runtime_inputs.visual_pos_masks,
                deepstack_visual_embeds=deepstack_embeds(runtime_inputs),
            )
    finally:
        if handle is not None:
            handle.remove()

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("decode full forward 请求了 output_hidden_states=True，但没有拿到 hidden_states。")
    return outputs, hidden_states, captured.get("hidden_stage_output")


def build_mm_stage_visuals(
    runtime_inputs: MmStateLike,
    *,
    start_idx: int,
    end_idx: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor | None, dict[int, torch.Tensor]]:
    """Select the multimodal visual payload needed by one decoder stage."""

    return build_mm_stage_visual_payload(
        runtime_inputs,
        start_idx=start_idx,
        end_idx=end_idx,
        device=device,
        compute_dtype=compute_dtype,
    )
