"""Direct runtime builders that construct StageState objects from a live model_path session."""

from __future__ import annotations

import gc
from typing import Any

import torch
import torch.nn.functional as F

from ...hexgen_core.distributed import startup_log, startup_timer
from ...hexgen_core.gen_hetero_groups import build_hybrid_layout, parse_tp_degrees
from ...hexgen_core.schema import (
    StageSpec,
    StageState,
    TextHybridManifest,
    TextPipelineManifest,
    TensorParallelManifest,
)
from .execution import (
    apply_deepstack,
    trace_text_decode_stage_with_runtime_cache,
    trace_text_prefill_stage_logits,
)
from .functional import rms_norm
from .live import (
    build_cache_by_layer_from_past_key_values,
    extract_decoder_layer_params_live,
    prepare_multimodal_decode_runtime_state,
    prepare_text_decode_runtime_inputs,
)
from .live.common import _resolve_compute_dtype, _runtime_tensor
from .processing import (
    build_text_inputs,
    load_processor,
    load_text_tokenizer,
    load_text_tokenizer_backend,
)
from .runtime_mm import (
    build_mm_stage_visuals as _build_stage_deepstack_payload,
    resolve_mm_frontend,
    run_mm_decode as _run_live_decode_full,
    run_mm_prefill as _run_live_prefill_full,
    run_mm_prefill_ref as _run_live_prefill_stage_reference,
    text_pos_ids as _extract_text_position_ids,
)
from .runtime_mm_stage import (
    MmStateLike,
    build_mm_stage_state,
    build_mm_decode_state_from_weights,
    clone_mm_state,
    compact_mm_runtime_shared,
    mm_state_from_decode_state,
    move_mm_state,
)
from .runtime_text import (
    _prep_rt_text_session,
    compact_text_prompt_meta,
    prepare_text_prompt_meta,
    restore_text_prompt_meta,
)
from .runtime_text_stage import (
    assert_text_tp_shard_shapes,
    assert_text_weight_scope,
    build_text_stage_state,
    compact_text_stage_state,
    compact_text_scaffold,
    materialize_text_stage_state as _materialize_text_stage_state,
    pack_text_scaffold_transport,
    restore_text_scaffold_transport,
)
from .weights import (
    TextModelConfigSpec,
    TextStageWeightBundle,
    build_text_rotary_embedding,
    load_model_weight_index,
    load_tensors_from_index,
    load_text_decoder_stage_weight_bundle,
    load_text_model_config_spec,
    prepare_text_decode_runtime_inputs_from_weights,
    prepare_text_prefill_runtime_inputs_from_weights,
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
    include_runtime_reference: bool | None = None,
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
    if include_runtime_reference is not None:
        config["include_runtime_reference"] = bool(include_runtime_reference)
    return config


def _save_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _clone_tensor_to_cpu(
    tensor: torch.Tensor | None,
    *,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    return _runtime_tensor(
        tensor,
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
    )


def _clone_mm_shared_to_cpu(
    shared: dict[str, Any],
    *,
    compute_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    return {
        str(name): _clone_tensor_to_cpu(
            value,
            compute_dtype=compute_dtype,
        )
        if torch.is_tensor(value) or value is None
        else value
        for name, value in shared.items()
    }


def _clone_mm_deepstack_to_cpu(
    deepstack_by_layer: dict[int, torch.Tensor],
    *,
    compute_dtype: torch.dtype | None = None,
) -> dict[int, torch.Tensor | None]:
    return {
        int(layer_idx): _clone_tensor_to_cpu(
            deepstack,
            compute_dtype=compute_dtype,
        )
        for layer_idx, deepstack in deepstack_by_layer.items()
    }


def _clone_mm_stage_handoffs_to_cpu(
    stage_handoffs: dict[int, dict[str, torch.Tensor | None]],
    *,
    compute_dtype: torch.dtype | None = None,
) -> dict[int, dict[str, Any]]:
    payload: dict[int, dict[str, Any]] = {}
    for stage_idx, stage_payload in stage_handoffs.items():
        payload[int(stage_idx)] = {
            "stage_input": _clone_tensor_to_cpu(
                stage_payload.get("stage_input"),
                compute_dtype=compute_dtype,
            ),
            "stage_output": _clone_tensor_to_cpu(
                stage_payload.get("stage_output"),
                compute_dtype=compute_dtype,
            ),
        }
    return payload


def _clone_mm_stage_visuals_to_cpu(
    stage_visuals: dict[int, Any],
    *,
    compute_dtype: torch.dtype | None = None,
) -> dict[int, dict[str, Any]]:
    payload: dict[int, dict[str, Any]] = {}
    for stage_idx, stage_payload in stage_visuals.items():
        if isinstance(stage_payload, tuple):
            stage_visual_pos_masks, stage_deepstack = stage_payload
        else:
            stage_visual_pos_masks = stage_payload.get("visual_pos_masks")
            stage_deepstack = stage_payload.get("deepstack_by_layer") or {}
        payload[int(stage_idx)] = {
            "visual_pos_masks": _clone_tensor_to_cpu(stage_visual_pos_masks),
            "deepstack_by_layer": _clone_mm_deepstack_to_cpu(
                {
                    int(layer_idx): deepstack
                    for layer_idx, deepstack in stage_deepstack.items()
                },
                compute_dtype=compute_dtype,
            ),
        }
    return payload


def _merge_mm_stage_visuals(
    stage_visuals: dict[int, Any],
) -> tuple[torch.Tensor | None, dict[int, torch.Tensor]]:
    visual_pos_masks = None
    merged_deepstack: dict[int, torch.Tensor] = {}
    for stage_idx in sorted(stage_visuals, key=int):
        payload = stage_visuals[stage_idx]
        if isinstance(payload, tuple):
            stage_visual_pos_masks, stage_deepstack = payload
        else:
            stage_visual_pos_masks = payload.get("visual_pos_masks")
            stage_deepstack = payload.get("deepstack_by_layer") or {}
        if visual_pos_masks is None and stage_visual_pos_masks is not None:
            visual_pos_masks = stage_visual_pos_masks
        for layer_idx, deepstack in stage_deepstack.items():
            merged_deepstack[int(layer_idx)] = deepstack
    return visual_pos_masks, merged_deepstack


_MM_STARTUP_FORBIDDEN_KEYS = (
    "root_input",
    "boundaries",
    "hidden_states",
    "stage_bundle",
    "bundle",
    "replay_bundle",
    "replay_bundle_path",
)
_MM_STARTUP_ALLOWED_TOP_LEVEL_KEYS = frozenset(
    (
        "shared",
        "stage_handoffs",
        "stage_visuals",
        "visual_pos_masks",
        "deepstack_by_layer",
        "num_frames",
        "frame_paths",
    )
)
_MM_STARTUP_ALLOWED_SHARED_KEYS = frozenset(
    (
        "input_ids",
        "attention_mask_2d",
        "position_ids",
        "attention_mask",
        "cos",
        "sin",
        "rope_deltas",
        "mm_token_type_ids",
        "image_grid_thw",
        "video_grid_thw",
    )
)
_MM_STARTUP_ALLOWED_STAGE_HANDOFF_KEYS = frozenset(("stage_input", "stage_output"))
_MM_STARTUP_ALLOWED_STAGE_VISUAL_KEYS = frozenset(("visual_pos_masks", "deepstack_by_layer"))


def _assert_thin_mm_startup_payload(payload: dict[str, Any], *, context: str) -> None:
    forbidden = [key for key in _MM_STARTUP_FORBIDDEN_KEYS if key in payload]
    if forbidden:
        raise RuntimeError(
            "multimodal startup contract 必须保持 thin，不能携带 full/root/replay payload，"
            f"context={context} forbidden_keys={forbidden}"
        )
    unknown_top_level = [
        key for key in payload if key not in _MM_STARTUP_ALLOWED_TOP_LEVEL_KEYS
    ]
    if unknown_top_level:
        raise RuntimeError(
            "multimodal startup contract 只能携带 thin startup 字段，"
            f"context={context} unknown_keys={unknown_top_level}"
        )

    shared = payload.get("shared")
    if isinstance(shared, dict):
        invalid_shared = [
            key for key in shared if key not in _MM_STARTUP_ALLOWED_SHARED_KEYS
        ]
        if invalid_shared:
            raise RuntimeError(
                "multimodal startup shared 只能携带 decoder runtime metadata/tensor，"
                f"context={context} invalid_shared_keys={invalid_shared}"
            )

    stage_handoffs = payload.get("stage_handoffs")
    if isinstance(stage_handoffs, dict):
        for stage_idx, stage_payload in stage_handoffs.items():
            if not isinstance(stage_payload, dict):
                continue
            invalid_stage_keys = [
                key
                for key in stage_payload
                if key not in _MM_STARTUP_ALLOWED_STAGE_HANDOFF_KEYS
            ]
            if invalid_stage_keys:
                raise RuntimeError(
                    "multimodal startup stage_handoffs 只能携带 stage_input/stage_output，"
                    f"context={context} stage_idx={stage_idx} invalid_keys={invalid_stage_keys}"
                )

    stage_visuals = payload.get("stage_visuals")
    if isinstance(stage_visuals, dict):
        for stage_idx, stage_payload in stage_visuals.items():
            if isinstance(stage_payload, tuple):
                continue
            if not isinstance(stage_payload, dict):
                continue
            invalid_visual_keys = [
                key
                for key in stage_payload
                if key not in _MM_STARTUP_ALLOWED_STAGE_VISUAL_KEYS
            ]
            if invalid_visual_keys:
                raise RuntimeError(
                    "multimodal startup stage_visuals 只能携带 local visual/deepstack payload，"
                    f"context={context} stage_idx={stage_idx} invalid_keys={invalid_visual_keys}"
                )


def _normalize_mm_startup_contract(payload: dict[str, Any]) -> dict[str, Any]:
    _assert_thin_mm_startup_payload(payload, context="normalize")
    shared = payload.get("shared")
    if not isinstance(shared, dict):
        raise RuntimeError("multimodal startup contract payload 缺少 shared。")

    stage_handoffs = payload.get("stage_handoffs")
    if not isinstance(stage_handoffs, dict):
        raise RuntimeError("multimodal startup contract payload 缺少 stage_handoffs。")

    stage_visuals_payload = payload.get("stage_visuals")
    stage_visuals = {}
    if isinstance(stage_visuals_payload, dict):
        for stage_idx, stage_payload in stage_visuals_payload.items():
            if isinstance(stage_payload, tuple):
                stage_visual_pos_masks, stage_deepstack = stage_payload
                stage_visuals[int(stage_idx)] = {
                    "visual_pos_masks": stage_visual_pos_masks,
                    "deepstack_by_layer": stage_deepstack,
                }
            elif isinstance(stage_payload, dict):
                stage_visuals[int(stage_idx)] = dict(stage_payload)
    visual_pos_masks = payload.get("visual_pos_masks")
    deepstack_by_layer = payload.get("deepstack_by_layer")
    if deepstack_by_layer is None and stage_visuals:
        visual_pos_masks, deepstack_by_layer = _merge_mm_stage_visuals(stage_visuals)

    return {
        "shared": shared,
        "stage_handoffs": {
            int(stage_idx): dict(stage_payload)
            for stage_idx, stage_payload in stage_handoffs.items()
        },
        "stage_visuals": stage_visuals,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_by_layer": {
            int(layer_idx): deepstack
            for layer_idx, deepstack in (deepstack_by_layer or {}).items()
        },
        "num_frames": int(payload.get("num_frames", 0)),
        "frame_paths": list(payload.get("frame_paths") or []),
    }


def _build_mm_startup_transport_payload(
    *,
    shared: dict[str, Any],
    stage_handoffs: dict[int, dict[str, torch.Tensor | None]],
    stage_visuals: dict[int, Any],
    num_frames: int,
    frame_paths: list[str],
    compute_dtype: torch.dtype | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor | None]]:
    tensor_payload: dict[str, torch.Tensor | None] = {}
    for name, value in shared.items():
        tensor_payload[f"shared.{name}"] = _clone_tensor_to_cpu(
            value,
            compute_dtype=compute_dtype,
        )
    for stage_idx, stage_payload in stage_handoffs.items():
        tensor_payload[f"stage_handoffs.{int(stage_idx)}.stage_input"] = _clone_tensor_to_cpu(
            stage_payload.get("stage_input"),
            compute_dtype=compute_dtype,
        )
        tensor_payload[f"stage_handoffs.{int(stage_idx)}.stage_output"] = _clone_tensor_to_cpu(
            stage_payload.get("stage_output"),
            compute_dtype=compute_dtype,
        )
    for stage_idx, stage_payload in stage_visuals.items():
        if isinstance(stage_payload, tuple):
            stage_visual_pos_masks, stage_deepstack = stage_payload
        else:
            stage_visual_pos_masks = stage_payload.get("visual_pos_masks")
            stage_deepstack = stage_payload.get("deepstack_by_layer") or {}
        tensor_payload[f"stage_visuals.{int(stage_idx)}.visual_pos_masks"] = _clone_tensor_to_cpu(
            stage_visual_pos_masks,
        )
        for layer_idx, deepstack in stage_deepstack.items():
            tensor_payload[f"stage_visuals.{int(stage_idx)}.deepstack_by_layer.{int(layer_idx)}"] = (
                _clone_tensor_to_cpu(
                    deepstack,
                    compute_dtype=compute_dtype,
                )
            )
    return (
        {
            "num_frames": int(num_frames),
            "frame_paths": list(frame_paths),
        },
        tensor_payload,
    )


def pack_mm_startup_transport(
    payload: dict[str, Any],
    *,
    compute_dtype: torch.dtype | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor | None]]:
    normalized = _normalize_mm_startup_contract(payload)
    return _build_mm_startup_transport_payload(
        shared=normalized["shared"],
        stage_handoffs=normalized["stage_handoffs"],
        stage_visuals=normalized["stage_visuals"],
        num_frames=normalized["num_frames"],
        frame_paths=normalized["frame_paths"],
        compute_dtype=compute_dtype,
    )


def restore_mm_startup_transport(
    meta: dict[str, Any],
    tensor_payload: dict[str, torch.Tensor | None] | None,
) -> dict[str, Any]:
    if not isinstance(meta, dict):
        raise RuntimeError("multimodal startup transport meta 缺少 metadata。")
    _assert_thin_mm_startup_payload(meta, context="restore_meta")
    if tensor_payload is None:
        raise RuntimeError("multimodal startup transport 缺少 tensor payload。")

    restored: dict[str, Any] = {
        "shared": {},
        "stage_handoffs": {},
        "stage_visuals": {},
        "num_frames": int(meta.get("num_frames", 0)),
        "frame_paths": list(meta.get("frame_paths") or []),
    }
    for name, tensor in tensor_payload.items():
        parts = name.split(".")
        if not parts:
            continue
        if parts[0] in _MM_STARTUP_FORBIDDEN_KEYS:
            raise RuntimeError(
                "multimodal startup transport 不能携带 full/root/replay tensor，"
                f"forbidden_key={parts[0]}"
            )
        if parts[0] == "shared" and len(parts) == 2:
            if parts[1] not in _MM_STARTUP_ALLOWED_SHARED_KEYS:
                raise RuntimeError(
                    "multimodal startup transport shared tensor 字段不在 thin schema 内，"
                    f"key={parts[1]}"
                )
            restored["shared"][parts[1]] = _clone_tensor_to_cpu(tensor)
            continue
        if parts[0] == "stage_handoffs":
            if len(parts) != 3 or parts[2] not in _MM_STARTUP_ALLOWED_STAGE_HANDOFF_KEYS:
                raise RuntimeError(
                    "multimodal startup transport stage_handoffs 只能携带 stage_input/stage_output，"
                    f"tensor_name={name}"
                )
            stage_idx = int(parts[1])
            restored["stage_handoffs"].setdefault(stage_idx, {})[parts[2]] = _clone_tensor_to_cpu(
                tensor,
            )
            continue
        if parts[0] == "stage_visuals" and len(parts) >= 3:
            stage_idx = int(parts[1])
            stage_visuals = restored["stage_visuals"].setdefault(
                stage_idx,
                {
                    "visual_pos_masks": None,
                    "deepstack_by_layer": {},
                },
            )
            if parts[2] == "visual_pos_masks" and len(parts) == 3:
                stage_visuals["visual_pos_masks"] = _clone_tensor_to_cpu(tensor)
                continue
            if parts[2] == "deepstack_by_layer" and len(parts) == 4:
                stage_visuals["deepstack_by_layer"][int(parts[3])] = _clone_tensor_to_cpu(tensor)
                continue
            raise RuntimeError(
                "multimodal startup transport stage_visuals tensor 字段不在 thin schema 内，"
                f"tensor_name={name}"
            )
        raise RuntimeError(
            "multimodal startup transport tensor 字段不在 thin schema 内，"
            f"tensor_name={name}"
        )

    if not restored["shared"]:
        raise RuntimeError("multimodal startup transport 缺少 shared tensor payload。")
    if not restored["stage_handoffs"]:
        raise RuntimeError("multimodal startup transport 缺少 stage_handoffs tensor payload。")
    if not restored["stage_visuals"]:
        restored.pop("stage_visuals", None)
    return restored


def _has_mm_startup_contract(runtime_config: dict[str, Any]) -> bool:
    return (
        runtime_config.get("_mm_startup_contract_ready", False)
        and runtime_config.get("_mm_startup_shared") is not None
        and runtime_config.get("_mm_startup_stage_handoffs") is not None
    )


def _default_attention_mask_2d(
    input_ids: torch.Tensor | None,
    attention_mask_2d: torch.Tensor | None,
) -> torch.Tensor | None:
    if attention_mask_2d is not None:
        return attention_mask_2d
    if input_ids is None:
        return None
    return torch.ones_like(input_ids)


def _default_runtime_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _compute_logits(model, norm_output: torch.Tensor) -> torch.Tensor:
    return F.linear(norm_output, model.lm_head.weight, model.lm_head.bias)


def _build_stage_layer_bundles(
    model,
    *,
    start_idx: int,
    end_idx: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
) -> list[dict[str, Any]]:
    text_model = model.model.language_model
    bundles = []
    for layer_idx in range(start_idx, end_idx + 1):
        layer_bundle = extract_decoder_layer_params_live(
            text_model.layers[layer_idx],
            layer_idx,
            device=device,
            compute_dtype=compute_dtype,
        )
        if cache_by_layer is not None and layer_idx in cache_by_layer:
            past_key, past_value = cache_by_layer[layer_idx]
            layer_bundle["past_key"] = _runtime_tensor(past_key, device=device, compute_dtype=compute_dtype)
            layer_bundle["past_value"] = _runtime_tensor(past_value, device=device, compute_dtype=compute_dtype)
        bundles.append(layer_bundle)
    return bundles


class DirectStageStateBuilder:
    """Reuse one builder session to materialize multiple direct StageState objects."""

    def __init__(
        self,
        *,
        stage_specs: list[StageSpec],
        runtime_config: dict[str, Any],
        tp_shard_rank: int | None = None,
        tp_shard_world_size: int | None = None,
        include_text_weights: bool = True,
        mm_activate_frontend: bool | None = None,
    ) -> None:
        if not stage_specs:
            raise ValueError("DirectStageStateBuilder 至少需要一个 stage spec。")

        self.stage_specs = sorted(stage_specs, key=lambda spec: spec.stage_idx)
        self.stage_specs_by_idx = {spec.stage_idx: spec for spec in self.stage_specs}
        if len(self.stage_specs_by_idx) != len(self.stage_specs):
            raise ValueError("stage_idx 必须唯一。")

        self.runtime_config = runtime_config
        self.tp_shard_rank = tp_shard_rank
        self.tp_shard_world_size = tp_shard_world_size
        self.include_text_weights = include_text_weights
        self.include_runtime_reference = bool(runtime_config.get("include_runtime_reference", True))
        self.modality = runtime_config["modality"]
        self.mode = runtime_config["mode"]
        self.runtime_only_text_generate = (
            self.modality == "text" and self.mode == "generate" and not self.include_runtime_reference
        )
        inferred_mm_frontend = self.modality == "multimodal" and any(
            spec.start_idx == 0 for spec in self.stage_specs
        )
        if self.modality != "multimodal":
            self.mm_activate_frontend = False
        elif mm_activate_frontend is None:
            self.mm_activate_frontend = inferred_mm_frontend
        else:
            self.mm_activate_frontend = bool(mm_activate_frontend)
            if self.mm_activate_frontend and not inferred_mm_frontend:
                raise ValueError(
                    "multimodal frontend-active builder 需要包含 start_idx=0 的 stage。"
                )
        self.weight_backed_multimodal = self.modality == "multimodal"
        self.stage_summary = ",".join(
            f"{spec.stage_idx}:{spec.start_idx}-{spec.end_idx}" for spec in self.stage_specs
        )
        self.log_component = f"direct-builder:{self.modality}_{self.mode}"
        self.stage_state_device = torch.device("cpu")
        if self.runtime_only_text_generate:
            self.session_kind = "runtime-only-text"
        elif self.modality == "text":
            self.session_kind = "file-backed-text"
        elif self.weight_backed_multimodal:
            self.session_kind = "mm-direct-active" if self.mm_activate_frontend else "mm-direct"
        else:
            self.session_kind = "mm-consume-only"
        self._text_weight_index = None
        self._text_model_config: TextModelConfigSpec | None = None
        self._text_stage_static_weights: dict[int, TextStageWeightBundle] = {}
        self._text_reference_layer_static_bundles: dict[int, dict[str, Any]] = {}
        self._text_last_stage_static_weights: TextStageWeightBundle | None = None
        self._text_embed_tokens_weight: torch.Tensor | None = None
        self._text_rotary_emb = None
        self._prefill_stage_inputs_by_stage: dict[int, torch.Tensor] = {}
        self._prefill_stage_outputs_by_stage: dict[int, torch.Tensor] = {}
        self._prefill_full_state: dict[str, Any] | None = None
        self._decode_state: dict[str, Any] | None = None
        self._generate_state: dict[str, Any] | None = None
        self._mm_prefill_state = None
        self._mm_prefill_shared: dict[str, Any] | None = None
        self._mm_prefill_visuals_by_stage: dict[int, tuple[torch.Tensor | None, dict[int, torch.Tensor]]] = {}
        self._mm_prefill_root_input: torch.Tensor | None = None
        self._mm_prefill_full_visual_pos_masks: torch.Tensor | None = None
        self._mm_prefill_full_deepstack_by_layer: dict[int, torch.Tensor] = {}
        self._mm_prefill_full_runtime: MmStateLike | None = None
        self.prefill_runtime_inputs = None

        if self.modality == "text":
            prepare_message = (
                f"prepare runtime-only text session stages=[{self.stage_summary}]"
                if self.runtime_only_text_generate
                else f"prepare file-backed text session stages=[{self.stage_summary}]"
            )
            with startup_timer(self.log_component, prepare_message):
                self._text_weight_index = load_model_weight_index(self.runtime_config["model_path"])
                self._text_model_config = load_text_model_config_spec(self.runtime_config["model_path"])
                if self.runtime_only_text_generate:
                    self.raw_inputs, self.compute_dtype, self.extra, self.device = _prep_rt_text_session(
                        self._text_weight_index,
                        runtime_config
                    )
                else:
                    processor = load_processor(self.runtime_config["model_path"])
                    prompt = self.runtime_config.get("prompt", "请用中文简要介绍一下人工智能。")
                    self.raw_inputs = build_text_inputs(processor, prompt)
                    self.extra = {"prompt": prompt}
                    self.device = _default_runtime_device()

                    embed_tokens_weight = load_tensors_from_index(
                        self._text_weight_index,
                        ["model.language_model.embed_tokens.weight"],
                        device=self.stage_state_device,
                        compute_dtype=None,
                        strict=True,
                    )["model.language_model.embed_tokens.weight"]
                    self.compute_dtype = _resolve_compute_dtype(
                        embed_tokens_weight,
                        runtime_config.get("save_dtype", "auto"),
                    )
                    self._text_embed_tokens_weight = _runtime_tensor(
                        embed_tokens_weight,
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    )
                    self._text_rotary_emb = build_text_rotary_embedding(
                        self._text_model_config,
                        device=self.device,
                    )
                    self.prefill_runtime_inputs = prepare_text_prefill_runtime_inputs_from_weights(
                        input_ids=self.raw_inputs["input_ids"],
                        attention_mask_2d=self.raw_inputs.get("attention_mask"),
                        embed_tokens_weight=self._text_embed_tokens_weight,
                        config_spec=self._text_model_config,
                        device=self.device,
                        compute_dtype=self.compute_dtype,
                        rotary_emb=self._text_rotary_emb,
                    )
                self.num_layers = self._text_model_config.num_hidden_layers
        elif self.weight_backed_multimodal:
            prepare_message = (
                f"prepare multimodal frontend+direct session stages=[{self.stage_summary}]"
                if self.mm_activate_frontend
                else f"prepare multimodal direct session stages=[{self.stage_summary}]"
            )
            with startup_timer(
                self.log_component,
                prepare_message,
            ):
                self._text_weight_index = load_model_weight_index(self.runtime_config["model_path"])
                self._text_model_config = load_text_model_config_spec(self.runtime_config["model_path"])
                self.device = _default_runtime_device()
                self.runtime_config.setdefault("_mm_weight_index", self._text_weight_index)
                if _has_mm_startup_contract(self.runtime_config):
                    self._load_mm_startup_contract()
                    if not self._prefill_stage_inputs_by_stage:
                        raise RuntimeError("multimodal startup contract 缺少 stage boundary hidden states。")
                    first_stage_input = self._prefill_stage_inputs_by_stage[self.stage_specs[0].stage_idx]
                    self.raw_inputs = {
                        "input_ids": self._mm_prefill_shared.get("input_ids"),
                    }
                    if self._mm_prefill_shared.get("attention_mask_2d") is not None:
                        self.raw_inputs["attention_mask"] = self._mm_prefill_shared["attention_mask_2d"]
                    self.extra = {
                        "num_frames": int(self.runtime_config.get("_mm_num_frames", 0)),
                        "frame_paths": list(self.runtime_config.get("_mm_frame_paths") or []),
                        "frontend_activation": "startup-contract",
                    }
                    self.compute_dtype = _resolve_compute_dtype(
                        first_stage_input,
                        runtime_config.get("save_dtype", "auto"),
                    )
                else:
                    if not self.mm_activate_frontend:
                        raise RuntimeError(
                            "multimodal non-frontend direct stage 必须消费 startup contract，"
                            "不能本地解析/构建视觉 frontend。请先由 stage0 广播 "
                            "multimodal startup contract，再构建 non-stage0。"
                        )
                    resolved_frontend = resolve_mm_frontend(
                        self.runtime_config,
                        activate_frontend=self.mm_activate_frontend,
                        config_spec=self._text_model_config,
                        device=self.device,
                    )
                    prefill_source = move_mm_state(
                        resolved_frontend.frontend_state,
                        device=self.device,
                    )
                    if prefill_source is None:
                        raise RuntimeError(
                            "multimodal frontend seed/state 缺失，无法构建 prefill runtime inputs。"
                            "（legacy frontend plan 兼容输入仍可接受。）"
                        )
                    self.compute_dtype = _resolve_compute_dtype(
                        prefill_source.inputs_embeds,
                        runtime_config.get("save_dtype", "auto"),
                    )
                    self._seed_mm_prefill_shared_from_state(prefill_source)
                    self.raw_inputs = {
                        "input_ids": prefill_source.input_ids,
                    }
                    if prefill_source.attention_mask_2d is not None:
                        self.raw_inputs["attention_mask"] = prefill_source.attention_mask_2d
                    self.extra = {
                        "num_frames": resolved_frontend.num_frames,
                        "frame_paths": list(resolved_frontend.frame_paths),
                        "frontend_activation": resolved_frontend.frontend_activation,
                    }
                self._text_rotary_emb = build_text_rotary_embedding(
                    self._text_model_config,
                    device=self.device,
                )
                self.num_layers = self._text_model_config.num_hidden_layers
        else:
            raise ValueError(f"不支持的 builder modality={self.modality!r}")

        self.has_last_stage = any(spec.end_idx == self.num_layers - 1 for spec in self.stage_specs)
        if self.runtime_only_text_generate:
            self.prefill_input_ids = self.raw_inputs.get("input_ids")
            self.prefill_attention_mask_2d_raw = self.raw_inputs.get("attention_mask")
        elif self.modality == "multimodal" and self._mm_prefill_shared is not None and self.prefill_runtime_inputs is None:
            self.prefill_input_ids = self._mm_prefill_shared.get("input_ids")
            self.prefill_attention_mask_2d_raw = self._mm_prefill_shared.get("attention_mask_2d")
        else:
            self.prefill_input_ids = self.prefill_runtime_inputs.input_ids
            self.prefill_attention_mask_2d_raw = self.prefill_runtime_inputs.attention_mask_2d
        self.prefill_attention_mask_2d = _default_attention_mask_2d(
            self.prefill_input_ids,
            self.prefill_attention_mask_2d_raw,
        )
        if self.mode in {"decode", "generate"} and (
            self.prefill_input_ids is None or self.prefill_attention_mask_2d is None
        ):
            raise RuntimeError(f"{self.mode} stage builder 需要 prefill input_ids 和 attention_mask_2d。")

        startup_log(
            self.log_component,
            f"builder session ready session_kind={self.session_kind} "
                f"device={self.device} state_device={self.stage_state_device} "
            f"compute_dtype={self.compute_dtype} num_layers={self.num_layers} "
            f"tp_shard_rank={self.tp_shard_rank} tp_shard_world_size={self.tp_shard_world_size} "
            f"include_text_weights={self.include_text_weights} "
            f"include_runtime_reference={self.include_runtime_reference}",
        )
        if self.modality == "multimodal":
            startup_log(
                self.log_component,
                f"multimodal_frontend_mode={'active' if self.mm_activate_frontend else 'consume-only'} "
                f"seed_ready={bool(self.runtime_config.get('_mm_frontend_state_ready'))} "
                f"startup_contract_ready={bool(self.runtime_config.get('_mm_startup_contract_ready'))}",
            )
        if self.runtime_only_text_generate:
            startup_log(
                self.log_component,
                f"skip prefill boundary capture for runtime-only text generate stages=[{self.stage_summary}]",
            )
        elif self.modality == "multimodal" and self._mm_prefill_shared is not None and self._prefill_stage_inputs_by_stage:
            startup_log(
                self.log_component,
                f"reuse multimodal startup contract stages=[{self.stage_summary}]",
            )
        else:
            with startup_timer(
                self.log_component,
                f"capture prefill stage boundaries stages=[{self.stage_summary}]",
            ):
                self._capture_prefill_stage_boundaries()
            if self.modality == "multimodal" and not hasattr(self, "model"):
                self._compact_mm_prefill_runtime()

    def __enter__(self) -> "DirectStageStateBuilder":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def close(self) -> None:
        startup_log(self.log_component, f"releasing builder session session_kind={self.session_kind}")
        self._prefill_full_state = None
        self._decode_state = None
        self._generate_state = None
        self._prefill_stage_inputs_by_stage.clear()
        self._prefill_stage_outputs_by_stage.clear()
        self._text_stage_static_weights.clear()
        self._text_reference_layer_static_bundles.clear()
        self._text_last_stage_static_weights = None
        self._text_embed_tokens_weight = None
        self._text_rotary_emb = None
        self._mm_prefill_state = None
        self._mm_prefill_shared = None
        self._mm_prefill_visuals_by_stage.clear()
        self._mm_prefill_root_input = None
        self._mm_prefill_full_visual_pos_masks = None
        self._mm_prefill_full_deepstack_by_layer.clear()
        self._mm_prefill_full_runtime = None

        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "raw_inputs"):
            del self.raw_inputs
        if hasattr(self, "prefill_runtime_inputs"):
            del self.prefill_runtime_inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prefill_mm_inputs(self) -> MmStateLike:
        if self.modality == "multimodal" and self._mm_prefill_state is not None:
            return self._mm_prefill_state
        if self.modality == "multimodal" and self._mm_prefill_shared is not None:
            return self._ensure_mm_full_prefill_runtime()
        if self.prefill_runtime_inputs is None:
            raise RuntimeError("prefill runtime inputs 已被收缩；请改用 stage-scoped multimodal state。")
        return self.prefill_runtime_inputs

    def _mm_prefill_rope_deltas(self) -> torch.Tensor | None:
        if self._mm_prefill_shared is not None:
            return self._mm_prefill_shared.get("rope_deltas")
        runtime_inputs = self._prefill_mm_inputs()
        return getattr(runtime_inputs, "rope_deltas", None)

    def _compact_mm_prefill_runtime(self) -> None:
        if self.modality != "multimodal" or self._mm_prefill_state is None:
            return
        self._mm_prefill_shared = compact_mm_runtime_shared(self._mm_prefill_state)
        self._mm_prefill_visuals_by_stage = {
            spec.stage_idx: self._build_stage_visual_payload(spec, self._mm_prefill_state)
            for spec in self.stage_specs
        }
        self._mm_prefill_root_input = (
            self._prefill_stage_inputs_by_stage[0].detach().clone()
            if 0 in self._prefill_stage_inputs_by_stage
            else None
        )
        (
            self._mm_prefill_full_visual_pos_masks,
            self._mm_prefill_full_deepstack_by_layer,
        ) = self._merge_mm_prefill_visuals(device=torch.device("cpu"))
        self._mm_prefill_full_runtime = None
        self._mm_prefill_state = None
        self.prefill_runtime_inputs = None

    def _seed_mm_prefill_shared_from_state(
        self,
        runtime_state: MmStateLike,
    ) -> None:
        self._mm_prefill_shared = compact_mm_runtime_shared(runtime_state)
        self._mm_prefill_visuals_by_stage = {
            spec.stage_idx: self._build_stage_visual_payload(spec, runtime_state)
            for spec in self.stage_specs
        }
        self._mm_prefill_root_input = runtime_state.inputs_embeds.detach().clone()
        self._mm_prefill_full_visual_pos_masks = None
        self._mm_prefill_full_deepstack_by_layer = {}
        self._mm_prefill_full_runtime = None
        self._mm_prefill_state = None
        self.prefill_runtime_inputs = None

    def _resolve_mm_startup_stage_indices(
        self,
        local_stage_indices: list[int] | None,
    ) -> list[int]:
        if local_stage_indices is None:
            return [int(spec.stage_idx) for spec in self.stage_specs]
        selected_stage_indices = [int(stage_idx) for stage_idx in local_stage_indices]
        missing_stage_indices = [
            stage_idx
            for stage_idx in selected_stage_indices
            if stage_idx not in self.stage_specs_by_idx
        ]
        if missing_stage_indices:
            raise RuntimeError(
                f"multimodal startup payload 缺少 local stage spec: {missing_stage_indices}"
            )
        return selected_stage_indices

    def _collect_mm_startup_payload_parts(
        self,
        *,
        local_stage_indices: list[int] | None = None,
    ) -> tuple[
        dict[str, Any],
        dict[int, dict[str, torch.Tensor | None]],
        dict[int, tuple[torch.Tensor | None, dict[int, torch.Tensor]]],
    ]:
        if self.modality != "multimodal":
            raise RuntimeError("只有 multimodal builder 可以导出 startup payload。")
        if self._mm_prefill_shared is None:
            self._compact_mm_prefill_runtime()
        if self._mm_prefill_shared is None:
            raise RuntimeError("multimodal startup payload 尚未准备完成。")

        selected_stage_indices = self._resolve_mm_startup_stage_indices(local_stage_indices)
        missing_stage_handoffs = [
            stage_idx
            for stage_idx in selected_stage_indices
            if stage_idx not in self._prefill_stage_inputs_by_stage
            or stage_idx not in self._prefill_stage_outputs_by_stage
        ]
        if missing_stage_handoffs:
            raise RuntimeError(
                f"multimodal startup payload 缺少 local stage handoff: {missing_stage_handoffs}"
            )

        selected_stage_handoffs = {
            int(stage_idx): {
                "stage_input": self._prefill_stage_inputs_by_stage[stage_idx],
                "stage_output": self._prefill_stage_outputs_by_stage[stage_idx],
            }
            for stage_idx in selected_stage_indices
        }
        selected_stage_visuals = {
            int(stage_idx): self._mm_prefill_visuals_by_stage[stage_idx]
            for stage_idx in selected_stage_indices
            if stage_idx in self._mm_prefill_visuals_by_stage
        }
        return self._mm_prefill_shared, selected_stage_handoffs, selected_stage_visuals

    def export_mm_startup_payload(
        self,
        *,
        local_stage_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        shared, selected_stage_handoffs, selected_stage_visuals = self._collect_mm_startup_payload_parts(
            local_stage_indices=local_stage_indices,
        )

        payload: dict[str, Any] = {
            "shared": _clone_mm_shared_to_cpu(
                shared,
                compute_dtype=self.compute_dtype,
            ),
            "stage_handoffs": _clone_mm_stage_handoffs_to_cpu(
                selected_stage_handoffs,
                compute_dtype=self.compute_dtype,
            ),
            "num_frames": int(self.extra["num_frames"]),
            "frame_paths": list(self.extra["frame_paths"]),
        }
        if selected_stage_visuals:
            payload["stage_visuals"] = _clone_mm_stage_visuals_to_cpu(
                selected_stage_visuals,
                compute_dtype=self.compute_dtype,
            )
        return payload

    def export_mm_startup_transport(
        self,
        *,
        local_stage_indices: list[int] | None = None,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor | None]]:
        shared, selected_stage_handoffs, selected_stage_visuals = self._collect_mm_startup_payload_parts(
            local_stage_indices=local_stage_indices,
        )
        return _build_mm_startup_transport_payload(
            shared=shared,
            stage_handoffs=selected_stage_handoffs,
            stage_visuals=selected_stage_visuals,
            num_frames=int(self.extra["num_frames"]),
            frame_paths=list(self.extra["frame_paths"]),
            compute_dtype=self.compute_dtype,
        )

    def _build_mm_stage_visual_payload_from_full(
        self,
        spec: StageSpec,
    ) -> tuple[torch.Tensor | None, dict[int, torch.Tensor]]:
        stage_deepstack = {
            int(layer_idx): deepstack.detach().clone()
            for layer_idx, deepstack in self._mm_prefill_full_deepstack_by_layer.items()
            if spec.start_idx <= int(layer_idx) <= spec.end_idx
        }
        visual_pos_masks = self._mm_prefill_full_visual_pos_masks
        if not stage_deepstack:
            visual_pos_masks = None
        return (
            None if visual_pos_masks is None else visual_pos_masks.detach().clone(),
            stage_deepstack,
        )

    def _load_mm_startup_contract(self) -> None:
        shared = self.runtime_config.get("_mm_startup_shared")
        stage_handoffs = self.runtime_config.get("_mm_startup_stage_handoffs") or {}
        stage_visuals = self.runtime_config.get("_mm_startup_stage_visuals") or {}
        if not isinstance(shared, dict) or not isinstance(stage_handoffs, dict):
            raise RuntimeError("multimodal startup contract 缺少 shared/stage_handoffs。")

        self._mm_prefill_shared = _clone_mm_shared_to_cpu(shared)
        self._mm_prefill_root_input = None
        self._mm_prefill_visuals_by_stage = {}
        if isinstance(stage_visuals, dict):
            self._mm_prefill_visuals_by_stage.update(
                {
                    int(stage_idx): (
                        _clone_tensor_to_cpu(stage_payload.get("visual_pos_masks")),
                        _clone_mm_deepstack_to_cpu(stage_payload.get("deepstack_by_layer") or {}),
                    )
                    for stage_idx, stage_payload in stage_visuals.items()
                    if isinstance(stage_payload, dict)
                }
            )
        self._mm_prefill_full_visual_pos_masks = _clone_tensor_to_cpu(
            self.runtime_config.get("_mm_startup_visual_pos_masks"),
        )
        self._mm_prefill_full_deepstack_by_layer = _clone_mm_deepstack_to_cpu(
            self.runtime_config.get("_mm_startup_deepstack_by_layer") or {},
        )
        for spec in self.stage_specs:
            stage_payload = stage_handoffs.get(spec.stage_idx)
            if not isinstance(stage_payload, dict):
                raise RuntimeError(
                    f"multimodal startup contract 缺少 stage_idx={spec.stage_idx} 的 stage handoff。"
                )
            self._prefill_stage_inputs_by_stage[spec.stage_idx] = (
                _clone_tensor_to_cpu(
                    stage_payload.get("stage_input"),
                )
            )
            self._prefill_stage_outputs_by_stage[spec.stage_idx] = (
                _clone_tensor_to_cpu(
                    stage_payload.get("stage_output"),
                )
            )
            if spec.stage_idx not in self._mm_prefill_visuals_by_stage:
                self._mm_prefill_visuals_by_stage[spec.stage_idx] = (
                    self._build_mm_stage_visual_payload_from_full(spec)
                )
        self._mm_prefill_full_runtime = None

    def _merge_mm_prefill_visuals(
        self,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, dict[int, torch.Tensor]]:
        if self._mm_prefill_full_visual_pos_masks is not None or self._mm_prefill_full_deepstack_by_layer:
            visual_pos_masks = _runtime_tensor(
                self._mm_prefill_full_visual_pos_masks,
                device=device,
            )
            merged_deepstack = {
                int(layer_idx): _runtime_tensor(
                    deepstack,
                    device=device,
                    compute_dtype=self.compute_dtype,
                )
                for layer_idx, deepstack in self._mm_prefill_full_deepstack_by_layer.items()
            }
            return visual_pos_masks, merged_deepstack
        visual_pos_masks = None
        merged_deepstack: dict[int, torch.Tensor] = {}
        for stage_idx in sorted(self._mm_prefill_visuals_by_stage):
            stage_visual_pos_masks, stage_deepstack = self._mm_prefill_visuals_by_stage[stage_idx]
            if visual_pos_masks is None and stage_visual_pos_masks is not None:
                visual_pos_masks = _runtime_tensor(stage_visual_pos_masks, device=device)
            for layer_idx, deepstack in stage_deepstack.items():
                merged_deepstack[int(layer_idx)] = _runtime_tensor(
                    deepstack,
                    device=device,
                    compute_dtype=self.compute_dtype,
                )
        return visual_pos_masks, merged_deepstack

    def _ensure_mm_full_prefill_runtime(self) -> MmStateLike:
        if self._mm_prefill_full_runtime is not None:
            return self._mm_prefill_full_runtime
        if self._mm_prefill_shared is None:
            raise RuntimeError("multimodal shared prefill state 尚未初始化。")
        stage0_input = self._prefill_stage_inputs_by_stage.get(0)
        if stage0_input is None:
            stage0_input = self._mm_prefill_root_input
        if stage0_input is None:
            raise RuntimeError("multimodal startup contract 缺少 stage0 handoff activation。")
        visual_pos_masks, deepstack_by_layer = self._merge_mm_prefill_visuals(device=self.device)
        self._mm_prefill_full_runtime = build_mm_stage_state(
            self._mm_prefill_shared,
            stage_input=stage0_input,
            start_idx=0,
            end_idx=self.num_layers - 1,
            device=self.device,
            compute_dtype=self.compute_dtype,
            visual_pos_masks=visual_pos_masks,
            deepstack_by_layer=deepstack_by_layer,
        )
        return self._mm_prefill_full_runtime

    def _build_mm_stage_state(
        self,
        spec: StageSpec,
        runtime_state: MmStateLike | dict[str, Any],
        *,
        stage_input: torch.Tensor,
        visual_payload: tuple[torch.Tensor | None, dict[int, torch.Tensor]] | None = None,
        device: torch.device | None = None,
    ):
        target_device = self.stage_state_device if device is None else device
        if visual_payload is None:
            if runtime_state is self._mm_prefill_shared and spec.stage_idx in self._mm_prefill_visuals_by_stage:
                visual_payload = self._mm_prefill_visuals_by_stage[spec.stage_idx]
            else:
                visual_payload = self._build_stage_visual_payload(spec, runtime_state)
        visual_pos_masks, deepstack_by_layer = visual_payload
        return build_mm_stage_state(
            runtime_state,
            stage_input=stage_input,
            start_idx=spec.start_idx,
            end_idx=spec.end_idx,
            device=target_device,
            compute_dtype=self.compute_dtype,
            visual_pos_masks=visual_pos_masks,
            deepstack_by_layer=deepstack_by_layer,
        )

    def export_mm_startup_contract(self) -> dict[str, Any]:
        return self.export_mm_startup_payload()

    def _capture_prefill_stage_boundaries(self) -> None:
        if not hasattr(self, "model"):
            prefill_state = self._ensure_prefill_full_state()
            for spec in self.stage_specs:
                stage_input = self._lookup_stage_boundary_tensor(
                    prefill_state,
                    spec,
                    key="stage_input",
                )
                stage_output = self._lookup_stage_boundary_tensor(
                    prefill_state,
                    spec,
                    key="stage_output",
                )
                if stage_input is None or stage_output is None:
                    raise RuntimeError(
                        "file-backed prefill state 缺少完整 stage handoff，"
                        f"stage_idx={spec.stage_idx} range={spec.start_idx}:{spec.end_idx}"
                    )
                self._prefill_stage_inputs_by_stage[spec.stage_idx] = stage_input.detach().clone()
                self._prefill_stage_outputs_by_stage[spec.stage_idx] = stage_output.detach().clone()
            return

        start_stage_ids: dict[int, list[int]] = {}
        end_stage_ids: dict[int, list[int]] = {}
        for spec in self.stage_specs:
            start_stage_ids.setdefault(spec.start_idx, []).append(spec.stage_idx)
            end_stage_ids.setdefault(spec.end_idx, []).append(spec.stage_idx)

        runtime_inputs = self._prefill_mm_inputs()
        text_model = self.model.model.language_model
        text_position_ids = _extract_text_position_ids(runtime_inputs)
        position_embeddings = (runtime_inputs.cos, runtime_inputs.sin)

        hidden_states = runtime_inputs.inputs_embeds
        for stage_idx in start_stage_ids.get(0, []):
            self._prefill_stage_inputs_by_stage[stage_idx] = hidden_states.detach().clone()

        with torch.inference_mode():
            for layer_idx in range(self.num_layers):
                hidden_states = text_model.layers[layer_idx](
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=runtime_inputs.attention_mask,
                    position_ids=text_position_ids,
                    past_key_values=None,
                    use_cache=False,
                )
                hidden_states = apply_deepstack(
                    hidden_states,
                    runtime_inputs.visual_pos_masks,
                    runtime_inputs.deepstack_by_layer.get(layer_idx),
                )
                for stage_idx in end_stage_ids.get(layer_idx, []):
                    self._prefill_stage_outputs_by_stage[stage_idx] = hidden_states.detach().clone()
                for stage_idx in start_stage_ids.get(layer_idx + 1, []):
                    self._prefill_stage_inputs_by_stage[stage_idx] = hidden_states.detach().clone()

        missing_inputs = set(self.stage_specs_by_idx) - set(self._prefill_stage_inputs_by_stage)
        if missing_inputs:
            raise RuntimeError(f"prefill stage input 捕获不完整，缺少 stage_idx={sorted(missing_inputs)}")
        missing_outputs = set(self.stage_specs_by_idx) - set(self._prefill_stage_outputs_by_stage)
        if missing_outputs:
            raise RuntimeError(f"prefill stage output 捕获不完整，缺少 stage_idx={sorted(missing_outputs)}")

    def _get_text_last_stage_static_weights(self) -> TextStageWeightBundle:
        if self._text_last_stage_static_weights is not None:
            return self._text_last_stage_static_weights
        if self._text_weight_index is None or self._text_model_config is None:
            raise RuntimeError("text weight index/config 尚未初始化。")

        last_layer_idx = self.num_layers - 1
        self._text_last_stage_static_weights = load_text_decoder_stage_weight_bundle(
            model_path=self.runtime_config["model_path"],
            start_idx=last_layer_idx,
            end_idx=last_layer_idx,
            is_first_stage=False,
            is_last_stage=True,
            device=self.stage_state_device,
            compute_dtype=self.compute_dtype,
            weight_index=self._text_weight_index,
            config_spec=self._text_model_config,
        )
        return self._text_last_stage_static_weights

    def _get_text_embed_tokens_weight(self) -> torch.Tensor:
        if self._text_embed_tokens_weight is not None:
            return self._text_embed_tokens_weight
        if self._text_weight_index is None:
            raise RuntimeError("text weight index 尚未初始化。")
        embed_tokens_weight = load_tensors_from_index(
            self._text_weight_index,
            ["model.language_model.embed_tokens.weight"],
            device=self.stage_state_device,
            compute_dtype=None,
            strict=True,
        )["model.language_model.embed_tokens.weight"]
        self._text_embed_tokens_weight = _runtime_tensor(
            embed_tokens_weight,
            device=self.stage_state_device,
            compute_dtype=self.compute_dtype,
        )
        if self._text_embed_tokens_weight is None:
            raise RuntimeError("embed_tokens_weight 不能为空。")
        return self._text_embed_tokens_weight

    def _get_text_reference_layer_static_bundle(self, layer_idx: int) -> dict[str, Any]:
        if layer_idx in self._text_reference_layer_static_bundles:
            return self._text_reference_layer_static_bundles[layer_idx]
        if self._text_weight_index is None or self._text_model_config is None:
            raise RuntimeError("text weight index/config 尚未初始化。")

        stage_weights = load_text_decoder_stage_weight_bundle(
            model_path=self.runtime_config["model_path"],
            start_idx=layer_idx,
            end_idx=layer_idx,
            is_first_stage=False,
            is_last_stage=False,
            device=self.stage_state_device,
            compute_dtype=self.compute_dtype,
            weight_index=self._text_weight_index,
            config_spec=self._text_model_config,
        )
        layer_bundle = dict(stage_weights.layer_bundles[0])
        self._text_reference_layer_static_bundles[layer_idx] = layer_bundle
        return layer_bundle

    def _build_text_reference_single_layer_stage_bundle(
        self,
        layer_idx: int,
        runtime_inputs: MmStateLike,
    ) -> dict[str, Any]:
        static_layer_bundle = self._get_text_reference_layer_static_bundle(layer_idx)
        runtime_layer_bundle = {
            key: (
                _runtime_tensor(value, device=self.device, compute_dtype=self.compute_dtype)
                if torch.is_tensor(value)
                else value
            )
            for key, value in static_layer_bundle.items()
        }
        visual_pos_masks, deepstack_by_layer = _build_stage_deepstack_payload(
            runtime_inputs,
            start_idx=layer_idx,
            end_idx=layer_idx,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )
        return {
            "module_name": "text_reference_layer",
            "stage_type": "text_decode",
            "start_idx": layer_idx,
            "end_idx": layer_idx,
            "attention_mask": _runtime_tensor(runtime_inputs.attention_mask, device=self.device),
            "cos": _runtime_tensor(runtime_inputs.cos, device=self.device, compute_dtype=self.compute_dtype),
            "sin": _runtime_tensor(runtime_inputs.sin, device=self.device, compute_dtype=self.compute_dtype),
            "visual_pos_masks": visual_pos_masks,
            "deepstack_by_layer": deepstack_by_layer,
            "deepstack_layer_indices": sorted(deepstack_by_layer),
            "layers": [runtime_layer_bundle],
        }

    def _get_text_final_runtime_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
        last_stage_weights = self._get_text_last_stage_static_weights()
        final_norm_weight = _runtime_tensor(
            last_stage_weights.final_norm_weight,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )
        lm_head_weight = _runtime_tensor(
            last_stage_weights.lm_head_weight,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )
        lm_head_bias = _runtime_tensor(
            last_stage_weights.lm_head_bias,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )
        if final_norm_weight is None or lm_head_weight is None or last_stage_weights.final_norm_eps is None:
            raise RuntimeError("text last stage 缺少 final norm / lm_head 权重，无法做 file-backed reference。")
        return final_norm_weight, lm_head_weight, lm_head_bias, float(last_stage_weights.final_norm_eps)

    def _run_local_last_stage_prefill_reference(
        self,
        *,
        hidden_stage_output: torch.Tensor | None,
        stage_input: torch.Tensor,
        prefill_runtime_state: MmStateLike,
        layer_bundles: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        final_norm_weight, lm_head_weight, lm_head_bias, final_norm_eps = self._get_text_final_runtime_weights()

        runtime_hidden_stage_output = _runtime_tensor(
            hidden_stage_output,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )
        if runtime_hidden_stage_output is None:
            stage_bundle = {
                "module_name": f"{self.modality}_prefill_stage_reference",
                "stage_type": "text_prefill_last",
                "start_idx": 0,
                "end_idx": len(layer_bundles) - 1,
                "attention_mask": _runtime_tensor(prefill_runtime_state.attention_mask, device=self.device),
                "cos": _runtime_tensor(
                    prefill_runtime_state.cos,
                    device=self.device,
                    compute_dtype=self.compute_dtype,
                ),
                "sin": _runtime_tensor(
                    prefill_runtime_state.sin,
                    device=self.device,
                    compute_dtype=self.compute_dtype,
                ),
                "visual_pos_masks": _runtime_tensor(
                    getattr(prefill_runtime_state, "visual_pos_masks", None),
                    device=self.device,
                ),
                "deepstack_by_layer": {
                    int(layer_idx): _runtime_tensor(
                        deepstack,
                        device=self.device,
                        compute_dtype=self.compute_dtype,
                    )
                    for layer_idx, deepstack in getattr(prefill_runtime_state, "deepstack_by_layer", {}).items()
                },
                "deepstack_layer_indices": sorted(getattr(prefill_runtime_state, "deepstack_by_layer", {})),
                "layers": [
                    {
                        key: (
                            _runtime_tensor(value, device=self.device, compute_dtype=self.compute_dtype)
                            if torch.is_tensor(value)
                            else value
                        )
                        for key, value in layer_bundle.items()
                    }
                    for layer_bundle in layer_bundles
                ],
                "final_norm_weight": final_norm_weight,
                "final_norm_eps": final_norm_eps,
                "lm_head_weight": lm_head_weight,
                "lm_head_bias": lm_head_bias,
            }
            trace = trace_text_prefill_stage_logits(
                _runtime_tensor(
                    stage_input,
                    device=self.device,
                    compute_dtype=self.compute_dtype,
                ),
                stage_bundle,
            )
            runtime_hidden_stage_output = trace["hidden_stage_output"].detach().clone()
            norm_output = trace["norm_output"].detach().clone()
            logits = trace["logits"].detach().clone()
        else:
            runtime_hidden_stage_output = runtime_hidden_stage_output.detach().clone()
            norm_output = rms_norm(runtime_hidden_stage_output, final_norm_weight, final_norm_eps)
            logits = F.linear(norm_output, lm_head_weight, lm_head_bias)

        return {
            "hidden_stage_output": runtime_hidden_stage_output,
            "norm_output": norm_output,
            "logits": logits,
        }

    def _single_full_coverage_stage_spec(self) -> StageSpec | None:
        if len(self.stage_specs) != 1:
            return None
        spec = self.stage_specs[0]
        if spec.start_idx != 0 or spec.end_idx != self.num_layers - 1:
            return None
        return spec

    def _build_local_stage_trace_bundle(
        self,
        *,
        module_name: str,
        stage_type: str,
        start_idx: int,
        end_idx: int,
        runtime_state: MmStateLike,
        layer_bundles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        deepstack_by_layer = {
            int(layer_idx): _runtime_tensor(
                deepstack,
                device=self.device,
                compute_dtype=self.compute_dtype,
            )
            for layer_idx, deepstack in getattr(runtime_state, "deepstack_by_layer", {}).items()
        }
        return {
            "module_name": module_name,
            "stage_type": stage_type,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "attention_mask": _runtime_tensor(runtime_state.attention_mask, device=self.device),
            "cos": _runtime_tensor(
                runtime_state.cos,
                device=self.device,
                compute_dtype=self.compute_dtype,
            ),
            "sin": _runtime_tensor(
                runtime_state.sin,
                device=self.device,
                compute_dtype=self.compute_dtype,
            ),
            "visual_pos_masks": _runtime_tensor(
                getattr(runtime_state, "visual_pos_masks", None),
                device=self.device,
            ),
            "deepstack_by_layer": deepstack_by_layer,
            "deepstack_layer_indices": sorted(deepstack_by_layer),
            "layers": [
                {
                    key: (
                        _runtime_tensor(value, device=self.device, compute_dtype=self.compute_dtype)
                        if torch.is_tensor(value)
                        else value
                    )
                    for key, value in layer_bundle.items()
                }
                for layer_bundle in layer_bundles
            ],
        }

    def _can_use_local_full_stage_prefill_reference(self) -> bool:
        spec = self._single_full_coverage_stage_spec()
        return (
            spec is not None
            and self.modality == "multimodal"
            and self._mm_prefill_shared is not None
            and (
                spec.stage_idx in self._prefill_stage_inputs_by_stage
                or self._mm_prefill_root_input is not None
            )
        )

    def _run_local_full_stage_prefill_reference(self) -> dict[str, Any]:
        spec = self._single_full_coverage_stage_spec()
        if spec is None:
            raise RuntimeError("local full-stage prefill reference 需要单个 full-coverage stage。")

        stage_input = self._prefill_stage_inputs_by_stage.get(spec.stage_idx)
        if stage_input is None:
            stage_input = self._mm_prefill_root_input
        if stage_input is None:
            raise RuntimeError("local full-stage prefill reference 缺少本地 stage0/root handoff activation。")
        prefill_runtime_state = self._build_mm_stage_state(
            spec,
            self._mm_prefill_shared if self._mm_prefill_shared is not None else self._prefill_mm_inputs(),
            stage_input=stage_input,
            device=self.device,
        )
        layer_bundles = self._build_layers_for_stage(spec)
        stage_bundle = self._build_local_stage_trace_bundle(
            module_name=f"{self.modality}_prefill_stage_reference",
            stage_type="text_prefill_last",
            start_idx=spec.start_idx,
            end_idx=spec.end_idx,
            runtime_state=prefill_runtime_state,
            layer_bundles=layer_bundles,
        )
        trace = trace_text_decode_stage_with_runtime_cache(
            _runtime_tensor(
                stage_input,
                device=self.device,
                compute_dtype=self.compute_dtype,
            ),
            stage_bundle,
            cache_by_layer={},
        )

        hidden_state_list = [
            _runtime_tensor(
                stage_input,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            )
        ]
        hidden_state_list.extend(
            _runtime_tensor(
                layer_trace["post_deepstack"].detach().clone(),
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            )
            for layer_trace in trace["layer_traces"]
        )
        hidden_stage_output = trace["stage_output"].detach().clone()
        final_norm_weight, lm_head_weight, lm_head_bias, final_norm_eps = self._get_text_final_runtime_weights()
        norm_output = rms_norm(hidden_stage_output, final_norm_weight, final_norm_eps)
        logits = F.linear(norm_output, lm_head_weight, lm_head_bias)
        stage_handoffs = self._build_stage_handoffs_from_hidden_states(
            hidden_state_list,
            device=self.stage_state_device,
        )
        return self._build_file_backed_prefill_state(
            runtime_state=prefill_runtime_state,
            stage_handoffs=stage_handoffs,
            cache_by_layer=trace["cache_by_layer"],
            norm_output=norm_output,
            logits=logits,
            hidden_states=hidden_state_list,
        )

    def _build_stage_handoffs_from_hidden_states(
        self,
        hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
        *,
        device: torch.device | None = None,
    ) -> dict[int, dict[str, torch.Tensor]]:
        hidden_state_list = list(hidden_states)
        if not hidden_state_list:
            return {}
        target_device = hidden_state_list[0].device if device is None else device
        return {
            int(spec.stage_idx): {
                "stage_input": _runtime_tensor(
                    hidden_state_list[spec.start_idx],
                    device=target_device,
                    compute_dtype=self.compute_dtype,
                ),
                "stage_output": _runtime_tensor(
                    hidden_state_list[spec.end_idx + 1],
                    device=target_device,
                    compute_dtype=self.compute_dtype,
                ),
            }
            for spec in self.stage_specs
        }

    def _keep_full_hidden_states_in_state(self) -> bool:
        return self.modality == "text"

    def _build_stage_boundary_maps(self) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        start_stage_ids: dict[int, list[int]] = {}
        end_stage_ids: dict[int, list[int]] = {}
        for spec in self.stage_specs:
            start_stage_ids.setdefault(spec.start_idx, []).append(spec.stage_idx)
            end_stage_ids.setdefault(spec.end_idx, []).append(spec.stage_idx)
        return start_stage_ids, end_stage_ids

    def _move_stage_handoffs(
        self,
        stage_handoffs: dict[int, dict[str, torch.Tensor | None]],
        *,
        device: torch.device,
    ) -> dict[int, dict[str, torch.Tensor | None]]:
        return {
            int(stage_idx): {
                "stage_input": _runtime_tensor(
                    stage_payload.get("stage_input"),
                    device=device,
                    compute_dtype=self.compute_dtype,
                ),
                "stage_output": _runtime_tensor(
                    stage_payload.get("stage_output"),
                    device=device,
                    compute_dtype=self.compute_dtype,
                ),
            }
            for stage_idx, stage_payload in stage_handoffs.items()
        }

    def _lookup_stage_handoff(
        self,
        payload: dict[str, Any],
        spec: StageSpec,
    ) -> dict[str, torch.Tensor] | None:
        stage_handoffs = payload.get("stage_handoffs")
        if not isinstance(stage_handoffs, dict):
            return None
        stage_handoff = stage_handoffs.get(spec.stage_idx)
        if not isinstance(stage_handoff, dict):
            return None
        return stage_handoff

    def _lookup_stage_boundary_tensor(
        self,
        payload: dict[str, Any],
        spec: StageSpec,
        *,
        key: str,
    ) -> torch.Tensor | None:
        stage_handoff = self._lookup_stage_handoff(payload, spec)
        if stage_handoff is not None and stage_handoff.get(key) is not None:
            return stage_handoff[key]
        hidden_states = payload.get("hidden_states")
        if isinstance(hidden_states, (tuple, list)):
            hidden_state_idx = spec.start_idx if key == "stage_input" else spec.end_idx + 1
            if 0 <= hidden_state_idx < len(hidden_states):
                return hidden_states[hidden_state_idx]
        return None

    def _normalize_cache_by_layer(
        self,
        cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]],
        *,
        device: torch.device,
    ) -> dict[int, tuple[torch.Tensor | None, torch.Tensor | None]]:
        return {
            int(layer_idx): (
                _runtime_tensor(past_key, device=device, compute_dtype=self.compute_dtype),
                _runtime_tensor(past_value, device=device, compute_dtype=self.compute_dtype),
            )
            for layer_idx, (past_key, past_value) in cache_by_layer.items()
        }

    def _infer_cache_seq_len(
        self,
        cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]],
    ) -> int:
        lengths = {
            int(past_key.shape[-2])
            for past_key, _past_value in cache_by_layer.values()
            if past_key is not None
        }
        if not lengths:
            return 0
        if len(lengths) != 1:
            raise RuntimeError(f"cache_by_layer 的 past length 不一致: {sorted(lengths)}")
        return lengths.pop()

    def _build_file_backed_prefill_state(
        self,
        *,
        runtime_state: MmStateLike | None,
        stage_handoffs: dict[int, dict[str, torch.Tensor | None]],
        cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]],
        norm_output: torch.Tensor,
        logits: torch.Tensor,
        hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "norm_output": _runtime_tensor(norm_output, device=self.stage_state_device, compute_dtype=self.compute_dtype),
            "logits": _runtime_tensor(logits, device=self.stage_state_device, compute_dtype=self.compute_dtype),
            "prefill_input_ids": _runtime_tensor(self.prefill_input_ids, device=self.stage_state_device),
            "prefill_attention_mask_2d": _runtime_tensor(self.prefill_attention_mask_2d, device=self.stage_state_device),
            "cache_by_layer": self._normalize_cache_by_layer(cache_by_layer, device=self.stage_state_device),
            "stage_handoffs": self._move_stage_handoffs(
                stage_handoffs,
                device=self.stage_state_device,
            ),
        }
        if self.modality == "multimodal" and runtime_state is not None:
            payload["mm_runtime_shared"] = compact_mm_runtime_shared(
                move_mm_state(runtime_state, device=self.stage_state_device)
            )
        if hidden_states is not None and self._keep_full_hidden_states_in_state():
            payload["hidden_states"] = tuple(
                _runtime_tensor(hidden, device=self.stage_state_device, compute_dtype=self.compute_dtype)
                for hidden in hidden_states
            )
        return payload

    def _file_backed_mm_shared(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        mm_shared = payload.get("mm_runtime_shared")
        if isinstance(mm_shared, dict):
            return mm_shared
        if self._mm_prefill_shared is not None:
            return self._mm_prefill_shared
        return None

    def _build_file_backed_decode_state(
        self,
        *,
        decode_source: str,
        decode_token_id: int,
        decode_result: dict[str, Any],
        cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]],
    ) -> dict[str, Any]:
        state = {
            "decode_source": decode_source,
            "decode_token_id": decode_token_id,
            "decode_input_ids": decode_result["decode_input_ids"],
            "attention_mask_2d": decode_result["attention_mask_2d"],
            "attention_mask": decode_result["attention_mask"],
            "cos": decode_result["cos"],
            "sin": decode_result["sin"],
            "position_ids": decode_result["position_ids"],
            "stage_handoffs": decode_result["stage_handoffs"],
            "hidden_stage_output": decode_result["hidden_stage_output"],
            "norm_output": decode_result["norm_output"],
            "logits": decode_result["logits"],
            "cache_by_layer": cache_by_layer,
        }
        if self.modality == "text":
            state["visual_pos_masks"] = None
            state["deepstack_by_layer"] = {}
            state["mm_runtime_state"] = None
        else:
            decode_mm_state = decode_result["mm_runtime_state"]
            state["visual_pos_masks"] = decode_mm_state.visual_pos_masks
            state["deepstack_by_layer"] = {
                int(layer_idx): deepstack.detach().clone()
                for layer_idx, deepstack in decode_mm_state.deepstack_by_layer.items()
            }
            state["mm_runtime_state"] = decode_mm_state
        if "hidden_states" in decode_result and self._keep_full_hidden_states_in_state():
            state["hidden_states"] = decode_result["hidden_states"]
        return state

    def _build_file_backed_generate_step(
        self,
        *,
        step_idx: int,
        decode_result: dict[str, Any],
        output_token_id: int,
    ) -> dict[str, Any]:
        step_state = {
            "step_idx": step_idx,
            "decode_input_ids": decode_result["decode_input_ids"],
            "attention_mask_2d": decode_result["attention_mask_2d"],
            "attention_mask": decode_result["attention_mask"],
            "cos": decode_result["cos"],
            "sin": decode_result["sin"],
            "position_ids": decode_result["position_ids"],
            "stage_handoffs": decode_result["stage_handoffs"],
            "hidden_stage_output": decode_result["hidden_stage_output"],
            "norm_output": decode_result["norm_output"],
            "logits": decode_result["logits"],
            "output_token_id": output_token_id,
        }
        if self.modality == "text":
            step_state["visual_pos_masks"] = None
            step_state["deepstack_by_layer"] = {}
            step_state["mm_runtime_state"] = None
        else:
            decode_mm_state = decode_result["mm_runtime_state"]
            step_state["visual_pos_masks"] = decode_mm_state.visual_pos_masks
            step_state["deepstack_by_layer"] = {
                int(layer_idx): deepstack.detach().clone()
                for layer_idx, deepstack in decode_mm_state.deepstack_by_layer.items()
            }
            step_state["mm_runtime_state"] = decode_mm_state
        if "hidden_states" in decode_result and self._keep_full_hidden_states_in_state():
            step_state["hidden_states"] = decode_result["hidden_states"]
        return step_state

    def _run_text_file_backed_prefill(self) -> dict[str, Any]:
        runtime_inputs = self._prefill_mm_inputs()
        hidden_states = _runtime_tensor(
            runtime_inputs.inputs_embeds,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )
        if hidden_states is None:
            raise RuntimeError(f"{self.modality} prefill 缺少 inputs_embeds。")

        keep_hidden_states = self._keep_full_hidden_states_in_state()
        hidden_state_list = [hidden_states.detach().clone()] if keep_hidden_states else None
        cache_by_layer_runtime: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] = {}
        start_stage_ids, end_stage_ids = self._build_stage_boundary_maps()
        stage_handoffs: dict[int, dict[str, torch.Tensor | None]] = {
            int(spec.stage_idx): {}
            for spec in self.stage_specs
        }

        for stage_idx in start_stage_ids.get(0, []):
            stage_handoffs[stage_idx]["stage_input"] = hidden_states.detach().clone()

        for layer_idx in range(self.num_layers):
            layer_stage_bundle = self._build_text_reference_single_layer_stage_bundle(
                layer_idx,
                runtime_inputs,
            )
            trace = trace_text_decode_stage_with_runtime_cache(
                hidden_states,
                layer_stage_bundle,
                cache_by_layer=cache_by_layer_runtime,
            )
            hidden_states = trace["stage_output"].detach().clone()
            if hidden_state_list is not None:
                hidden_state_list.append(hidden_states)
            cache_by_layer_runtime[layer_idx] = trace["cache_by_layer"][layer_idx]
            for stage_idx in end_stage_ids.get(layer_idx, []):
                stage_handoffs[stage_idx]["stage_output"] = hidden_states.detach().clone()
            for stage_idx in start_stage_ids.get(layer_idx + 1, []):
                stage_handoffs[stage_idx]["stage_input"] = hidden_states.detach().clone()

        final_norm_weight, lm_head_weight, lm_head_bias, final_norm_eps = self._get_text_final_runtime_weights()
        norm_output = rms_norm(hidden_states, final_norm_weight, final_norm_eps)
        logits = F.linear(norm_output, lm_head_weight, lm_head_bias)

        return self._build_file_backed_prefill_state(
            runtime_state=runtime_inputs,
            stage_handoffs=stage_handoffs,
            cache_by_layer=cache_by_layer_runtime,
            norm_output=norm_output,
            logits=logits,
            hidden_states=hidden_state_list,
        )

    def _run_text_file_backed_decode(
        self,
        *,
        decode_input_ids: torch.Tensor,
        attention_mask_2d: torch.Tensor,
        cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]],
        rope_deltas: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self._text_model_config is None:
            raise RuntimeError("text config 尚未初始化。")

        past_length = self._infer_cache_seq_len(cache_by_layer)
        if self.modality == "text":
            decode_runtime_inputs = prepare_text_decode_runtime_inputs_from_weights(
                decode_input_ids=decode_input_ids,
                attention_mask_2d=attention_mask_2d,
                past_length=past_length,
                embed_tokens_weight=self._get_text_embed_tokens_weight(),
                config_spec=self._text_model_config,
                device=self.device,
                compute_dtype=self.compute_dtype,
                rotary_emb=self._text_rotary_emb,
            )
        else:
            if rope_deltas is None:
                raise RuntimeError("multimodal file-backed decode 需要显式 rope_deltas。")
            decode_runtime_inputs = build_mm_decode_state_from_weights(
                decode_input_ids=decode_input_ids,
                attention_mask_2d=attention_mask_2d,
                past_length=past_length,
                rope_deltas=rope_deltas,
                embed_tokens_weight=self._get_text_embed_tokens_weight(),
                config_spec=self._text_model_config,
                device=self.device,
                compute_dtype=self.compute_dtype,
                rotary_emb=self._text_rotary_emb,
            )
        hidden_states = _runtime_tensor(
            decode_runtime_inputs.inputs_embeds,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )
        if hidden_states is None:
            raise RuntimeError("text decode 缺少 inputs_embeds。")

        keep_hidden_states = self._keep_full_hidden_states_in_state()
        hidden_state_list = [hidden_states.detach().clone()] if keep_hidden_states else None
        cache_by_layer_runtime = self._normalize_cache_by_layer(cache_by_layer, device=self.device)
        start_stage_ids, end_stage_ids = self._build_stage_boundary_maps()
        stage_handoffs: dict[int, dict[str, torch.Tensor | None]] = {
            int(spec.stage_idx): {}
            for spec in self.stage_specs
        }

        for stage_idx in start_stage_ids.get(0, []):
            stage_handoffs[stage_idx]["stage_input"] = hidden_states.detach().clone()

        for layer_idx in range(self.num_layers):
            layer_stage_bundle = self._build_text_reference_single_layer_stage_bundle(
                layer_idx,
                decode_runtime_inputs,
            )
            trace = trace_text_decode_stage_with_runtime_cache(
                hidden_states,
                layer_stage_bundle,
                cache_by_layer=cache_by_layer_runtime,
            )
            hidden_states = trace["stage_output"].detach().clone()
            if hidden_state_list is not None:
                hidden_state_list.append(hidden_states)
            cache_by_layer_runtime[layer_idx] = trace["cache_by_layer"][layer_idx]
            for stage_idx in end_stage_ids.get(layer_idx, []):
                stage_handoffs[stage_idx]["stage_output"] = hidden_states.detach().clone()
            for stage_idx in start_stage_ids.get(layer_idx + 1, []):
                stage_handoffs[stage_idx]["stage_input"] = hidden_states.detach().clone()

        final_norm_weight, lm_head_weight, lm_head_bias, final_norm_eps = self._get_text_final_runtime_weights()
        norm_output = rms_norm(hidden_states, final_norm_weight, final_norm_eps)
        logits = F.linear(norm_output, lm_head_weight, lm_head_bias)

        payload = {
            "decode_input_ids": _runtime_tensor(decode_input_ids, device=self.stage_state_device),
            "attention_mask_2d": _runtime_tensor(attention_mask_2d, device=self.stage_state_device),
            "attention_mask": _runtime_tensor(decode_runtime_inputs.attention_mask, device=self.stage_state_device),
            "cos": _runtime_tensor(decode_runtime_inputs.cos, device=self.stage_state_device, compute_dtype=self.compute_dtype),
            "sin": _runtime_tensor(decode_runtime_inputs.sin, device=self.stage_state_device, compute_dtype=self.compute_dtype),
            "position_ids": _runtime_tensor(decode_runtime_inputs.position_ids, device=self.stage_state_device),
            "mm_runtime_state": None
            if self.modality == "text"
            else clone_mm_state(decode_runtime_inputs),
            "stage_handoffs": self._move_stage_handoffs(
                stage_handoffs,
                device=self.stage_state_device,
            ),
            "hidden_stage_output": _runtime_tensor(
                hidden_states,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            ),
            "norm_output": _runtime_tensor(norm_output, device=self.stage_state_device, compute_dtype=self.compute_dtype),
            "logits": _runtime_tensor(logits, device=self.stage_state_device, compute_dtype=self.compute_dtype),
            "cache_by_layer": self._normalize_cache_by_layer(cache_by_layer_runtime, device=self.stage_state_device),
        }
        if hidden_state_list is not None:
            payload["hidden_states"] = tuple(
                _runtime_tensor(hidden, device=self.stage_state_device, compute_dtype=self.compute_dtype)
                for hidden in hidden_state_list
            )
        return payload

    def _ensure_prefill_full_state(self) -> dict[str, Any]:
        if self._prefill_full_state is not None:
            return self._prefill_full_state

        if not hasattr(self, "model"):
            with startup_timer(self.log_component, f"run file-backed {self.modality} prefill reference"):
                if self._can_use_local_full_stage_prefill_reference():
                    self._prefill_full_state = self._run_local_full_stage_prefill_reference()
                else:
                    self._prefill_full_state = self._run_text_file_backed_prefill()
            return self._prefill_full_state

        with startup_timer(self.log_component, "run full prefill reference"):
            outputs = _run_live_prefill_full(self.model, self._prefill_mm_inputs())
            norm_output = outputs.last_hidden_state.detach().clone()
            logits = _compute_logits(self.model, norm_output).detach().clone()
            prefill_cache_by_layer = build_cache_by_layer_from_past_key_values(
                outputs.past_key_values,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            )
            self._prefill_full_state = {
                "past_key_values": outputs.past_key_values,
                "norm_output": norm_output,
                "logits": logits,
                "prefill_input_ids": self.prefill_input_ids,
                "prefill_attention_mask_2d": self.prefill_attention_mask_2d,
                "cache_by_layer": prefill_cache_by_layer,
            }
        return self._prefill_full_state

    def _ensure_decode_state(self) -> dict[str, Any]:
        if self._decode_state is not None:
            return self._decode_state

        if not hasattr(self, "model"):
            with startup_timer(self.log_component, f"prepare file-backed {self.modality} decode reference state"):
                prefill_state = self._ensure_prefill_full_state()
                prefill_logits = prefill_state["logits"]
                greedy_decode_token_id = int(prefill_logits[0, -1].argmax().item())
                decode_token_id_value = self.runtime_config.get("decode_token_id")
                if decode_token_id_value is None:
                    decode_token_id_value = greedy_decode_token_id
                    decode_source = "greedy_from_prefill"
                else:
                    decode_token_id_value = int(decode_token_id_value)
                    decode_source = "provided"

                decode_input_ids = torch.tensor(
                    [[decode_token_id_value]],
                    device=self.stage_state_device,
                    dtype=self.prefill_input_ids.dtype,
                )
                decode_attention_mask_2d = torch.cat(
                    [
                        prefill_state["prefill_attention_mask_2d"],
                        torch.ones(
                            (prefill_state["prefill_attention_mask_2d"].shape[0], 1),
                            device=self.stage_state_device,
                            dtype=prefill_state["prefill_attention_mask_2d"].dtype,
                        ),
                    ],
                    dim=-1,
                )
                prefill_mm_shared = self._file_backed_mm_shared(prefill_state)
                decode_result = self._run_text_file_backed_decode(
                    decode_input_ids=decode_input_ids,
                    attention_mask_2d=decode_attention_mask_2d,
                    cache_by_layer=prefill_state["cache_by_layer"],
                    rope_deltas=None if prefill_mm_shared is None else prefill_mm_shared.get("rope_deltas"),
                )
                self._decode_state = self._build_file_backed_decode_state(
                    decode_source=decode_source,
                    decode_token_id=decode_token_id_value,
                    decode_result=decode_result,
                    cache_by_layer=prefill_state["cache_by_layer"],
                )
            return self._decode_state

        with startup_timer(self.log_component, "prepare decode reference state"):
            prefill_state = self._ensure_prefill_full_state()
            prefill_logits = prefill_state["logits"]
            greedy_decode_token_id = int(prefill_logits[0, -1].argmax().item())
            decode_token_id_value = self.runtime_config.get("decode_token_id")
            if decode_token_id_value is None:
                decode_token_id_value = greedy_decode_token_id
                decode_source = "greedy_from_prefill"
            else:
                decode_token_id_value = int(decode_token_id_value)
                decode_source = "provided"

            decode_input_ids = torch.tensor(
                [[decode_token_id_value]],
                device=self.device,
                dtype=self.prefill_input_ids.dtype,
            )
            decode_attention_mask_2d = torch.cat(
                [
                    self.prefill_attention_mask_2d,
                    torch.ones(
                        (self.prefill_attention_mask_2d.shape[0], 1),
                        device=self.device,
                        dtype=self.prefill_attention_mask_2d.dtype,
                    ),
                ],
                dim=-1,
            )

            if self.modality == "text":
                decode_runtime_inputs = prepare_text_decode_runtime_inputs(
                    self.model,
                    decode_input_ids=decode_input_ids,
                    attention_mask_2d=decode_attention_mask_2d,
                    past_key_values=prefill_state["past_key_values"],
                )
                decode_runtime_state = decode_runtime_inputs
            else:
                rope_deltas = self._mm_prefill_rope_deltas()
                if rope_deltas is None:
                    raise RuntimeError("multimodal decode 需要显式 rope_deltas，但 prefill frontend state 里没有。")
                decode_runtime_state = prepare_multimodal_decode_runtime_state(
                    self.model,
                    decode_input_ids=decode_input_ids,
                    attention_mask_2d=decode_attention_mask_2d,
                    past_key_values=prefill_state["past_key_values"],
                    rope_deltas=rope_deltas,
                )

            decode_outputs, hidden_states, hidden_stage_output = _run_live_decode_full(
                self.model,
                runtime_inputs=decode_runtime_state,
                past_key_values=prefill_state["past_key_values"],
                is_last_stage=self.has_last_stage,
            )
            self._decode_state = {
                "decode_source": decode_source,
                "decode_token_id": decode_token_id_value,
                "decode_input_ids": decode_input_ids.detach().clone(),
                "attention_mask_2d": decode_attention_mask_2d.detach().clone(),
                "attention_mask": decode_runtime_state.attention_mask.detach().clone(),
                "cos": decode_runtime_state.cos.detach().clone(),
                "sin": decode_runtime_state.sin.detach().clone(),
                "position_ids": None
                if decode_runtime_state.position_ids is None
                else decode_runtime_state.position_ids.detach().clone(),
                "visual_pos_masks": None
                if decode_runtime_state.visual_pos_masks is None
                else decode_runtime_state.visual_pos_masks.detach().clone(),
                "deepstack_by_layer": {
                    int(layer_idx): deepstack_embeds.detach().clone()
                    for layer_idx, deepstack_embeds in decode_runtime_state.deepstack_by_layer.items()
                },
                "mm_runtime_state": None if self.modality == "text" else clone_mm_state(decode_runtime_state),
                "hidden_states": tuple(hidden.detach().clone() for hidden in hidden_states),
                "stage_handoffs": self._build_stage_handoffs_from_hidden_states(hidden_states),
                "hidden_stage_output": None
                if hidden_stage_output is None
                else hidden_stage_output.detach().clone(),
                "norm_output": decode_outputs.last_hidden_state.detach().clone(),
                "logits": _compute_logits(self.model, decode_outputs.last_hidden_state).detach().clone(),
                "cache_by_layer": prefill_state["cache_by_layer"],
            }
        return self._decode_state

    def _ensure_generate_state(self) -> dict[str, Any]:
        if self._generate_state is not None:
            return self._generate_state

        if not hasattr(self, "model"):
            with startup_timer(self.log_component, f"prepare file-backed {self.modality} generate reference state"):
                prefill_state = self._ensure_prefill_full_state()
                current_attention_mask_2d = prefill_state["prefill_attention_mask_2d"]
                current_cache_by_layer = prefill_state["cache_by_layer"]
                prefill_mm_shared = self._file_backed_mm_shared(prefill_state)
                generated_token_ids = [int(prefill_state["logits"][0, -1].argmax().item())]
                step_results: list[dict[str, Any]] = []
                max_new_tokens = int(self.runtime_config.get("max_new_tokens", 4))
                if max_new_tokens <= 0:
                    raise ValueError(f"max_new_tokens 必须大于 0，当前拿到 {max_new_tokens}。")

                startup_log(
                    self.log_component,
                    f"file-backed generate decode planning max_new_tokens={max_new_tokens} "
                    f"decode_steps={max_new_tokens - 1}",
                )
                for step_idx in range(max_new_tokens - 1):
                    decode_input_ids = torch.tensor(
                        [[generated_token_ids[-1]]],
                        device=self.stage_state_device,
                        dtype=self.prefill_input_ids.dtype,
                    )
                    current_attention_mask_2d = torch.cat(
                        [
                            current_attention_mask_2d,
                            torch.ones(
                                (current_attention_mask_2d.shape[0], 1),
                                device=self.stage_state_device,
                                dtype=current_attention_mask_2d.dtype,
                            ),
                        ],
                        dim=-1,
                    )
                    decode_result = self._run_text_file_backed_decode(
                        decode_input_ids=decode_input_ids,
                        attention_mask_2d=current_attention_mask_2d,
                        cache_by_layer=current_cache_by_layer,
                        rope_deltas=None if prefill_mm_shared is None else prefill_mm_shared.get("rope_deltas"),
                    )
                    next_token_id = int(decode_result["logits"][0, -1].argmax().item())
                    generated_token_ids.append(next_token_id)
                    step_results.append(
                        self._build_file_backed_generate_step(
                            step_idx=step_idx,
                            decode_result=decode_result,
                            output_token_id=next_token_id,
                        )
                    )
                    current_cache_by_layer = decode_result["cache_by_layer"]

                self._generate_state = {
                    "max_new_tokens": max_new_tokens,
                    "generated_token_ids": generated_token_ids,
                    "prefill_norm_output": prefill_state["norm_output"],
                    "prefill_logits": prefill_state["logits"],
                    "cache_by_layer": prefill_state["cache_by_layer"],
                    "step_results": step_results,
                }
                if self.modality == "multimodal" and prefill_mm_shared is not None:
                    self._generate_state["mm_runtime_shared"] = _clone_mm_shared_to_cpu(
                        prefill_mm_shared,
                        compute_dtype=self.compute_dtype,
                    )
            return self._generate_state

        with startup_timer(self.log_component, "prepare generate reference state"):
            prefill_state = self._ensure_prefill_full_state()
            current_attention_mask_2d = self.prefill_attention_mask_2d
            current_past_key_values = prefill_state["past_key_values"]
            generated_token_ids = [int(prefill_state["logits"][0, -1].argmax().item())]
            step_results: list[dict[str, Any]] = []
            max_new_tokens = int(self.runtime_config.get("max_new_tokens", 4))
            if max_new_tokens <= 0:
                raise ValueError(f"max_new_tokens 必须大于 0，当前拿到 {max_new_tokens}。")

            startup_log(
                self.log_component,
                f"generate decode planning max_new_tokens={max_new_tokens} decode_steps={max_new_tokens - 1}",
            )
            for step_idx in range(max_new_tokens - 1):
                startup_log(
                    self.log_component,
                    f"generate decode step {step_idx + 1}/{max_new_tokens - 1}",
                )
                decode_input_ids = torch.tensor(
                    [[generated_token_ids[-1]]],
                    device=self.device,
                    dtype=self.prefill_input_ids.dtype,
                )
                current_attention_mask_2d = torch.cat(
                    [
                        current_attention_mask_2d,
                        torch.ones(
                            (current_attention_mask_2d.shape[0], 1),
                            device=self.device,
                            dtype=current_attention_mask_2d.dtype,
                        ),
                    ],
                    dim=-1,
                )

                if self.modality == "text":
                    decode_runtime_inputs = prepare_text_decode_runtime_inputs(
                        self.model,
                        decode_input_ids=decode_input_ids,
                        attention_mask_2d=current_attention_mask_2d,
                        past_key_values=current_past_key_values,
                    )
                    decode_runtime_state = decode_runtime_inputs
                else:
                    rope_deltas = self._mm_prefill_rope_deltas()
                    if rope_deltas is None:
                        raise RuntimeError("multimodal decode 需要显式 rope_deltas，但 prefill frontend state 里没有。")
                    decode_runtime_state = prepare_multimodal_decode_runtime_state(
                        self.model,
                        decode_input_ids=decode_input_ids,
                        attention_mask_2d=current_attention_mask_2d,
                        past_key_values=current_past_key_values,
                        rope_deltas=rope_deltas,
                    )

                decode_outputs, hidden_states, hidden_stage_output = _run_live_decode_full(
                    self.model,
                    runtime_inputs=decode_runtime_state,
                    past_key_values=current_past_key_values,
                    is_last_stage=self.has_last_stage,
                )
                norm_output = decode_outputs.last_hidden_state.detach().clone()
                logits = _compute_logits(self.model, norm_output).detach().clone()
                next_token_id = int(logits[0, -1].argmax().item())
                generated_token_ids.append(next_token_id)

                step_results.append(
                    {
                        "step_idx": step_idx,
                        "decode_input_ids": decode_input_ids.detach().clone(),
                        "attention_mask_2d": current_attention_mask_2d.detach().clone(),
                        "attention_mask": decode_runtime_state.attention_mask.detach().clone(),
                        "cos": decode_runtime_state.cos.detach().clone(),
                        "sin": decode_runtime_state.sin.detach().clone(),
                        "position_ids": None
                        if decode_runtime_state.position_ids is None
                        else decode_runtime_state.position_ids.detach().clone(),
                        "mm_runtime_state": None if self.modality == "text" else clone_mm_state(decode_runtime_state),
                        "hidden_states": tuple(hidden.detach().clone() for hidden in hidden_states),
                        "stage_handoffs": self._build_stage_handoffs_from_hidden_states(hidden_states),
                        "hidden_stage_output": None
                        if hidden_stage_output is None
                        else hidden_stage_output.detach().clone(),
                        "norm_output": norm_output,
                        "logits": logits,
                        "output_token_id": next_token_id,
                    }
                )
                current_past_key_values = decode_outputs.past_key_values

            self._generate_state = {
                "max_new_tokens": max_new_tokens,
                "generated_token_ids": generated_token_ids,
                "prefill_norm_output": prefill_state["norm_output"],
                "prefill_logits": prefill_state["logits"],
                "cache_by_layer": prefill_state["cache_by_layer"],
                "step_results": step_results,
            }
        return self._generate_state

    def _build_layers_for_stage(
        self,
        spec: StageSpec,
        *,
        cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.include_text_weights:
            return []
        if not hasattr(self, "model"):
            static_weights = self._get_text_stage_static_weights(spec)
            layer_bundles = [dict(layer_bundle) for layer_bundle in static_weights.layer_bundles]
            if cache_by_layer is not None:
                for layer_bundle in layer_bundles:
                    layer_idx = int(layer_bundle["layer_idx"])
                    if layer_idx not in cache_by_layer:
                        continue
                    past_key, past_value = cache_by_layer[layer_idx]
                    layer_bundle["past_key"] = _runtime_tensor(
                        past_key,
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    )
                    layer_bundle["past_value"] = _runtime_tensor(
                        past_value,
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    )
            return layer_bundles
        return _build_stage_layer_bundles(
            self.model,
            start_idx=spec.start_idx,
            end_idx=spec.end_idx,
            device=self.stage_state_device,
            compute_dtype=self.compute_dtype,
            cache_by_layer=cache_by_layer,
        )

    def _get_text_stage_static_weights(self, spec: StageSpec) -> TextStageWeightBundle:
        if spec.stage_idx in self._text_stage_static_weights:
            return self._text_stage_static_weights[spec.stage_idx]
        if self._text_weight_index is None or self._text_model_config is None:
            raise RuntimeError("text weight index/config 尚未初始化。")

        with startup_timer(
            self.log_component,
            f"load text stage weights stage_idx={spec.stage_idx} range={spec.start_idx}:{spec.end_idx}",
        ):
            stage_weights = load_text_decoder_stage_weight_bundle(
                model_path=self.runtime_config["model_path"],
                start_idx=spec.start_idx,
                end_idx=spec.end_idx,
                is_first_stage=spec.start_idx == 0,
                is_last_stage=spec.end_idx == self.num_layers - 1,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
                weight_index=self._text_weight_index,
                config_spec=self._text_model_config,
                tp_shard_rank=self.tp_shard_rank,
                tp_shard_world_size=self.tp_shard_world_size,
            )
        self._text_stage_static_weights[spec.stage_idx] = stage_weights
        return stage_weights

    def _get_stage_state_static_weights(self, spec: StageSpec) -> TextStageWeightBundle | None:
        if self.modality == "text":
            if not self.include_text_weights:
                return None
            return self._get_text_stage_static_weights(spec)
        if hasattr(self, "model"):
            return None
        return self._get_text_stage_static_weights(spec)

    def _get_stage_state_embed_tokens_weight(
        self,
        stage_static_weights: TextStageWeightBundle | None,
    ) -> torch.Tensor:
        if stage_static_weights is not None and stage_static_weights.embed_tokens_weight is not None:
            return stage_static_weights.embed_tokens_weight
        if hasattr(self, "model"):
            return _runtime_tensor(
                self.model.model.language_model.embed_tokens.weight,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            )
        return _runtime_tensor(
            self._get_text_embed_tokens_weight(),
            device=self.stage_state_device,
            compute_dtype=self.compute_dtype,
        )

    def _get_stage_state_final_output_weights(
        self,
        stage_static_weights: TextStageWeightBundle | None,
    ) -> dict[str, Any]:
        if stage_static_weights is not None:
            return {
                "final_norm_weight": stage_static_weights.final_norm_weight,
                "final_norm_eps": stage_static_weights.final_norm_eps,
                "lm_head_weight": stage_static_weights.lm_head_weight,
                "lm_head_bias": stage_static_weights.lm_head_bias,
            }
        if hasattr(self, "model"):
            return {
                "final_norm_weight": _runtime_tensor(
                    self.model.model.language_model.norm.weight,
                    device=self.stage_state_device,
                    compute_dtype=self.compute_dtype,
                ),
                "final_norm_eps": self.model.model.language_model.norm.variance_epsilon,
                "lm_head_weight": _runtime_tensor(
                    self.model.lm_head.weight,
                    device=self.stage_state_device,
                    compute_dtype=self.compute_dtype,
                ),
                "lm_head_bias": _runtime_tensor(
                    self.model.lm_head.bias,
                    device=self.stage_state_device,
                    compute_dtype=self.compute_dtype,
                ),
            }
        final_norm_weight, lm_head_weight, lm_head_bias, final_norm_eps = self._get_text_final_runtime_weights()
        return {
            "final_norm_weight": _runtime_tensor(
                final_norm_weight,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            ),
            "final_norm_eps": final_norm_eps,
            "lm_head_weight": _runtime_tensor(
                lm_head_weight,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            ),
            "lm_head_bias": _runtime_tensor(
                lm_head_bias,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            ),
        }

    def _build_stage_visual_payload(
        self,
        spec: StageSpec,
        runtime_inputs: MmStateLike,
    ) -> tuple[torch.Tensor | None, dict[int, torch.Tensor]]:
        return _build_stage_deepstack_payload(
            runtime_inputs,
            start_idx=spec.start_idx,
            end_idx=spec.end_idx,
            device=self.stage_state_device,
            compute_dtype=self.compute_dtype,
        )

    def _build_prefill_stage_state(self, spec: StageSpec) -> StageState:
        stage_input = self._prefill_stage_inputs_by_stage[spec.stage_idx]
        hidden_stage_output = self._prefill_stage_outputs_by_stage[spec.stage_idx]
        is_last_stage = spec.end_idx == self.num_layers - 1
        layer_bundles = self._build_layers_for_stage(spec)
        if self.modality == "multimodal":
            prefill_runtime_state = self._build_mm_stage_state(
                spec,
                self._mm_prefill_shared if self._mm_prefill_shared is not None else self._prefill_mm_inputs(),
                stage_input=stage_input,
            )
        else:
            prefill_runtime_state = self._prefill_mm_inputs()
        visual_pos_masks = _runtime_tensor(
            getattr(prefill_runtime_state, "visual_pos_masks", None),
            device=self.stage_state_device,
        )
        deepstack_by_layer = {
            int(layer_idx): _runtime_tensor(
                deepstack,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            )
            for layer_idx, deepstack in getattr(prefill_runtime_state, "deepstack_by_layer", {}).items()
        }
        stage_static_weights = self._get_stage_state_static_weights(spec)

        stage_state = {
            "module_name": f"{self.modality}_prefill_stage",
            "stage_type": "text_prefill_last" if is_last_stage else "text",
            "start_idx": spec.start_idx,
            "end_idx": spec.end_idx,
            "save_dtype": _save_dtype_name(self.compute_dtype),
            "original_input_dtype": str(stage_input.dtype),
            "original_input_device": str(stage_input.device),
            "attention_mask_2d": _runtime_tensor(self.prefill_attention_mask_2d_raw, device=self.stage_state_device),
            "stage_input": _runtime_tensor(stage_input, device=self.stage_state_device, compute_dtype=self.compute_dtype),
            "layer_input": _runtime_tensor(stage_input, device=self.stage_state_device, compute_dtype=self.compute_dtype),
            "attention_mask": _runtime_tensor(prefill_runtime_state.attention_mask, device=self.stage_state_device),
            "cos": _runtime_tensor(
                prefill_runtime_state.cos,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            ),
            "sin": _runtime_tensor(
                prefill_runtime_state.sin,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            ),
            "visual_pos_masks": visual_pos_masks,
            "deepstack_by_layer": deepstack_by_layer,
            "deepstack_layer_indices": sorted(deepstack_by_layer),
            "layers": layer_bundles,
        }

        if self.modality == "text":
            stage_state["prompt"] = self.extra["prompt"]
            stage_state["input_ids"] = _runtime_tensor(prefill_runtime_state.input_ids, device=self.stage_state_device)
            stage_state["tp_weight_sharded"] = (
                False if stage_static_weights is None else stage_static_weights.tp_weight_sharded
            )
            stage_state["tp_shard_rank"] = None if stage_static_weights is None else stage_static_weights.tp_shard_rank
            stage_state["tp_shard_world_size"] = (
                None if stage_static_weights is None else stage_static_weights.tp_shard_world_size
            )
            if spec.start_idx == 0 and stage_static_weights is not None:
                stage_state["embed_tokens_weight"] = stage_static_weights.embed_tokens_weight
        else:
            stage_state["num_frames"] = self.extra["num_frames"]
            stage_state["frame_paths"] = self.extra["frame_paths"]
            stage_state["input_ids"] = _runtime_tensor(prefill_runtime_state.input_ids, device=self.stage_state_device)

        if is_last_stage:
            if not hasattr(self, "model"):
                local_prefill = self._run_local_last_stage_prefill_reference(
                    hidden_stage_output=hidden_stage_output,
                    stage_input=stage_input,
                    prefill_runtime_state=prefill_runtime_state,
                    layer_bundles=layer_bundles,
                )
                hidden_stage_output = local_prefill["hidden_stage_output"]
                norm_output = local_prefill["norm_output"]
                logits = local_prefill["logits"]
            else:
                text_model = self.model.model.language_model
                norm_output = text_model.norm(hidden_stage_output).detach().clone()
                logits = _compute_logits(self.model, norm_output).detach().clone()
            stage_state.update(
                {
                    "original_output_dtype": str(logits.dtype),
                    "original_output_device": str(logits.device),
                    "stage_output": _runtime_tensor(logits, device=self.stage_state_device, compute_dtype=self.compute_dtype),
                    "layer_output": _runtime_tensor(logits, device=self.stage_state_device, compute_dtype=self.compute_dtype),
                    "hidden_stage_output": _runtime_tensor(
                        hidden_stage_output,
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "norm_output": _runtime_tensor(norm_output, device=self.stage_state_device, compute_dtype=self.compute_dtype),
                    "logits": _runtime_tensor(logits, device=self.stage_state_device, compute_dtype=self.compute_dtype),
                }
            )
            if self.include_text_weights and (stage_static_weights is not None or self.modality != "text"):
                stage_state.update(self._get_stage_state_final_output_weights(stage_static_weights))
        else:
            stage_state.update(
                {
                    "original_output_dtype": str(hidden_stage_output.dtype),
                    "original_output_device": str(hidden_stage_output.device),
                    "stage_output": _runtime_tensor(
                        hidden_stage_output,
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "layer_output": _runtime_tensor(
                        hidden_stage_output,
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    ),
                }
            )

        return stage_state

    def _build_decode_stage_state(self, spec: StageSpec) -> StageState:
        state = self._ensure_decode_state()
        stage_input = self._lookup_stage_boundary_tensor(state, spec, key="stage_input")
        if stage_input is None:
            raise RuntimeError(
                f"decode state 缺少 stage_idx={spec.stage_idx} 的 stage_input handoff。"
            )
        is_last_stage = spec.end_idx == self.num_layers - 1
        layer_bundles = self._build_layers_for_stage(spec, cache_by_layer=state["cache_by_layer"])
        decode_runtime_state = None
        if self.modality == "multimodal":
            decode_runtime_state = self._build_mm_stage_state(
                spec,
                mm_state_from_decode_state(state),
                stage_input=stage_input,
            )
        else:
            decode_runtime_state = None
        visual_pos_masks = _runtime_tensor(
            None if decode_runtime_state is None else decode_runtime_state.visual_pos_masks,
            device=self.stage_state_device,
        )
        deepstack_by_layer = (
            {}
            if decode_runtime_state is None
            else {
                int(layer_idx): _runtime_tensor(
                    deepstack,
                    device=self.stage_state_device,
                    compute_dtype=self.compute_dtype,
                )
                for layer_idx, deepstack in decode_runtime_state.deepstack_by_layer.items()
            }
        )
        stage_static_weights = self._get_stage_state_static_weights(spec)

        stage_state = {
            "module_name": f"{self.modality}_decode_stage",
            "stage_type": "text_decode_last" if is_last_stage else "text_decode",
            "start_idx": spec.start_idx,
            "end_idx": spec.end_idx,
            "save_dtype": _save_dtype_name(self.compute_dtype),
            "original_input_dtype": str(stage_input.dtype),
            "original_input_device": str(stage_input.device),
            "decode_source": state["decode_source"],
            "decode_token_id": state["decode_token_id"],
            "prefill_seq_len": int(self.prefill_input_ids.shape[-1]),
            "total_seq_len": int(state["attention_mask_2d"].shape[-1]),
            "prefill_input_ids": _runtime_tensor(self.prefill_input_ids, device=self.stage_state_device),
            "decode_input_ids": _runtime_tensor(state["decode_input_ids"], device=self.stage_state_device),
            "prefill_attention_mask_2d": _runtime_tensor(self.prefill_attention_mask_2d, device=self.stage_state_device),
            "attention_mask_2d": _runtime_tensor(state["attention_mask_2d"], device=self.stage_state_device),
            "stage_input": _runtime_tensor(stage_input, device=self.stage_state_device, compute_dtype=self.compute_dtype),
            "layer_input": _runtime_tensor(stage_input, device=self.stage_state_device, compute_dtype=self.compute_dtype),
            "attention_mask": _runtime_tensor(state["attention_mask"], device=self.stage_state_device),
            "cos": _runtime_tensor(state["cos"], device=self.stage_state_device, compute_dtype=self.compute_dtype),
            "sin": _runtime_tensor(state["sin"], device=self.stage_state_device, compute_dtype=self.compute_dtype),
            "visual_pos_masks": visual_pos_masks,
            "deepstack_by_layer": deepstack_by_layer,
            "deepstack_layer_indices": sorted(deepstack_by_layer),
            "layers": layer_bundles,
        }
        if self.modality == "text":
            stage_state["cache_by_layer"] = state["cache_by_layer"]

        if self.modality == "text":
            stage_state["prompt"] = self.extra["prompt"]
            stage_state["tp_weight_sharded"] = (
                False if stage_static_weights is None else stage_static_weights.tp_weight_sharded
            )
            stage_state["tp_shard_rank"] = None if stage_static_weights is None else stage_static_weights.tp_shard_rank
            stage_state["tp_shard_world_size"] = (
                None if stage_static_weights is None else stage_static_weights.tp_shard_world_size
            )
        else:
            stage_state["num_frames"] = self.extra["num_frames"]
            stage_state["frame_paths"] = self.extra["frame_paths"]
            stage_state["position_ids"] = _runtime_tensor(state["position_ids"], device=self.stage_state_device)

        if spec.start_idx == 0:
            if self.include_text_weights and (stage_static_weights is not None or self.modality != "text"):
                stage_state["embed_tokens_weight"] = self._get_stage_state_embed_tokens_weight(stage_static_weights)

        if is_last_stage:
            hidden_stage_output = state["hidden_stage_output"]
            if hidden_stage_output is None:
                raise RuntimeError("decode last stage 没有拿到 final norm 前的 hidden_stage_output。")
            logits = state["logits"]
            norm_output = state["norm_output"]
            stage_state.update(
                {
                    "original_output_dtype": str(logits.dtype),
                    "original_output_device": str(logits.device),
                    "stage_output": _runtime_tensor(logits, device=self.stage_state_device, compute_dtype=self.compute_dtype),
                    "layer_output": _runtime_tensor(logits, device=self.stage_state_device, compute_dtype=self.compute_dtype),
                    "hidden_stage_output": _runtime_tensor(
                        hidden_stage_output,
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "norm_output": _runtime_tensor(norm_output, device=self.stage_state_device, compute_dtype=self.compute_dtype),
                    "logits": _runtime_tensor(logits, device=self.stage_state_device, compute_dtype=self.compute_dtype),
                }
            )
            if stage_static_weights is not None or self.modality != "text":
                stage_state.update(self._get_stage_state_final_output_weights(stage_static_weights))
        else:
            stage_output = self._lookup_stage_boundary_tensor(state, spec, key="stage_output")
            if stage_output is None:
                raise RuntimeError(
                    f"decode state 缺少 stage_idx={spec.stage_idx} 的 stage_output handoff。"
                )
            stage_state.update(
                {
                    "original_output_dtype": str(stage_output.dtype),
                    "original_output_device": str(stage_output.device),
                    "stage_output": _runtime_tensor(
                        stage_output,
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "layer_output": _runtime_tensor(
                        stage_output,
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    ),
                }
            )

        return stage_state

    def _build_generate_stage_state(self, spec: StageSpec) -> StageState:
        state = self._ensure_generate_state()
        is_last_stage = spec.end_idx == self.num_layers - 1
        prefill_stage_input = self._prefill_stage_inputs_by_stage[spec.stage_idx]
        prefill_hidden_stage_output = self._prefill_stage_outputs_by_stage[spec.stage_idx]
        layer_bundles = self._build_layers_for_stage(spec, cache_by_layer=state["cache_by_layer"])
        if self.modality == "multimodal":
            prefill_runtime_state = self._build_mm_stage_state(
                spec,
                self._mm_prefill_shared if self._mm_prefill_shared is not None else self._prefill_mm_inputs(),
                stage_input=prefill_stage_input,
            )
        else:
            prefill_runtime_state = self._prefill_mm_inputs()
        visual_pos_masks = _runtime_tensor(
            getattr(prefill_runtime_state, "visual_pos_masks", None),
            device=self.stage_state_device,
        )
        deepstack_by_layer = {
            int(layer_idx): _runtime_tensor(
                deepstack,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            )
            for layer_idx, deepstack in getattr(prefill_runtime_state, "deepstack_by_layer", {}).items()
        }
        stage_static_weights = self._get_stage_state_static_weights(spec)

        prefill_payload = {
            "attention_mask_2d": _runtime_tensor(self.prefill_attention_mask_2d, device=self.stage_state_device),
            "stage_input": _runtime_tensor(
                prefill_stage_input,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            ),
            "attention_mask": _runtime_tensor(prefill_runtime_state.attention_mask, device=self.stage_state_device),
            "cos": _runtime_tensor(
                prefill_runtime_state.cos,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            ),
            "sin": _runtime_tensor(
                prefill_runtime_state.sin,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            ),
        }
        if is_last_stage:
            prefill_payload.update(
                {
                    "stage_output": _runtime_tensor(
                        state["prefill_logits"],
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "hidden_stage_output": _runtime_tensor(
                        prefill_hidden_stage_output,
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "norm_output": _runtime_tensor(
                        state["prefill_norm_output"],
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "logits": _runtime_tensor(
                        state["prefill_logits"],
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "output_token_id": state["generated_token_ids"][0],
                }
            )
        else:
            prefill_payload["stage_output"] = _runtime_tensor(
                prefill_hidden_stage_output,
                device=self.stage_state_device,
                compute_dtype=self.compute_dtype,
            )

        decode_steps = []
        for step_result in state["step_results"]:
            step_stage_input = self._lookup_stage_boundary_tensor(step_result, spec, key="stage_input")
            if step_stage_input is None:
                raise RuntimeError(
                    f"generate step 缺少 stage_idx={spec.stage_idx} 的 stage_input handoff。"
                )
            decode_runtime_state = None
            if self.modality == "multimodal":
                decode_runtime_state = self._build_mm_stage_state(
                    spec,
                    mm_state_from_decode_state(step_result),
                    stage_input=step_stage_input,
                )
            step_payload = {
                "step_idx": step_result["step_idx"],
                "decode_input_ids": _runtime_tensor(step_result["decode_input_ids"], device=self.stage_state_device),
                "attention_mask_2d": _runtime_tensor(step_result["attention_mask_2d"], device=self.stage_state_device),
                "total_seq_len": int(step_result["attention_mask_2d"].shape[-1]),
                "stage_input": _runtime_tensor(
                    step_stage_input,
                    device=self.stage_state_device,
                    compute_dtype=self.compute_dtype,
                ),
                "attention_mask": _runtime_tensor(step_result["attention_mask"], device=self.stage_state_device),
                "cos": _runtime_tensor(
                    step_result["cos"],
                    device=self.stage_state_device,
                    compute_dtype=self.compute_dtype,
                ),
                "sin": _runtime_tensor(
                    step_result["sin"],
                    device=self.stage_state_device,
                    compute_dtype=self.compute_dtype,
                ),
                "visual_pos_masks": None,
                "deepstack_by_layer": {},
                "deepstack_layer_indices": [],
            }
            if self.modality == "multimodal":
                step_payload["position_ids"] = _runtime_tensor(step_result["position_ids"], device=self.stage_state_device)
                step_payload["visual_pos_masks"] = _runtime_tensor(
                    decode_runtime_state.visual_pos_masks,
                    device=self.stage_state_device,
                )
                step_payload["deepstack_by_layer"] = {
                    int(layer_idx): _runtime_tensor(
                        deepstack,
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    )
                    for layer_idx, deepstack in decode_runtime_state.deepstack_by_layer.items()
                }
                step_payload["deepstack_layer_indices"] = sorted(step_payload["deepstack_by_layer"])

            if is_last_stage:
                hidden_stage_output = step_result["hidden_stage_output"]
                if hidden_stage_output is None:
                    raise RuntimeError("generate decode last stage 没有拿到 final norm 前的 hidden_stage_output。")
                step_payload.update(
                    {
                        "stage_output": _runtime_tensor(
                            step_result["logits"],
                            device=self.stage_state_device,
                            compute_dtype=self.compute_dtype,
                        ),
                        "hidden_stage_output": _runtime_tensor(
                            hidden_stage_output,
                            device=self.stage_state_device,
                            compute_dtype=self.compute_dtype,
                        ),
                        "norm_output": _runtime_tensor(
                            step_result["norm_output"],
                            device=self.stage_state_device,
                            compute_dtype=self.compute_dtype,
                        ),
                        "logits": _runtime_tensor(
                            step_result["logits"],
                            device=self.stage_state_device,
                            compute_dtype=self.compute_dtype,
                        ),
                        "output_token_id": step_result["output_token_id"],
                    }
                )
            else:
                step_stage_output = self._lookup_stage_boundary_tensor(
                    step_result,
                    spec,
                    key="stage_output",
                )
                if step_stage_output is None:
                    raise RuntimeError(
                        f"generate step 缺少 stage_idx={spec.stage_idx} 的 stage_output handoff。"
                    )
                step_payload["stage_output"] = _runtime_tensor(
                    step_stage_output,
                    device=self.stage_state_device,
                    compute_dtype=self.compute_dtype,
                )
            decode_steps.append(step_payload)

        stage_state = {
            "module_name": f"{self.modality}_generate_stage",
            "stage_type": f"{self.modality}_generate_last" if is_last_stage else f"{self.modality}_generate",
            "start_idx": spec.start_idx,
            "end_idx": spec.end_idx,
            "save_dtype": _save_dtype_name(self.compute_dtype),
            "original_input_dtype": str(prefill_stage_input.dtype),
            "original_input_device": str(prefill_stage_input.device),
            "max_new_tokens": state["max_new_tokens"],
            "prefill_seq_len": int(self.prefill_input_ids.shape[-1]),
            "prefill_input_ids": _runtime_tensor(self.prefill_input_ids, device=self.stage_state_device),
            "prefill_attention_mask_2d": _runtime_tensor(
                self.prefill_attention_mask_2d,
                device=self.stage_state_device,
            ),
            "generated_token_ids": torch.tensor(
                [state["generated_token_ids"]],
                device=self.stage_state_device,
                dtype=self.prefill_input_ids.dtype,
            ),
            "prefill": prefill_payload,
            "visual_pos_masks": visual_pos_masks,
            "deepstack_by_layer": deepstack_by_layer,
            "deepstack_layer_indices": sorted(deepstack_by_layer),
            "layers": layer_bundles,
            "decode_steps": decode_steps,
        }
        if self.modality == "text":
            stage_state["cache_by_layer"] = state["cache_by_layer"]

        if self.modality == "text":
            stage_state["prompt"] = self.extra["prompt"]
            stage_state["tp_weight_sharded"] = (
                False if stage_static_weights is None else stage_static_weights.tp_weight_sharded
            )
            stage_state["tp_shard_rank"] = None if stage_static_weights is None else stage_static_weights.tp_shard_rank
            stage_state["tp_shard_world_size"] = (
                None if stage_static_weights is None else stage_static_weights.tp_shard_world_size
            )
            if spec.start_idx == 0:
                stage_state["input_ids"] = _runtime_tensor(self.prefill_input_ids, device=self.stage_state_device)
        else:
            stage_state["num_frames"] = self.extra["num_frames"]
            stage_state["frame_paths"] = self.extra["frame_paths"]

        if spec.start_idx == 0:
            if self.include_text_weights and (stage_static_weights is not None or self.modality != "text"):
                stage_state["embed_tokens_weight"] = self._get_stage_state_embed_tokens_weight(stage_static_weights)

        if is_last_stage:
            if self.include_text_weights and (stage_static_weights is not None or self.modality != "text"):
                stage_state.update(self._get_stage_state_final_output_weights(stage_static_weights))

        return stage_state

    def _build_generate_stage_state_runtime_only(self, spec: StageSpec) -> StageState:
        if self.modality not in {"text", "multimodal"}:
            raise RuntimeError(f"runtime-only generate StageState 不支持 modality={self.modality!r}。")

        text_stage_weights = (
            self._get_text_stage_static_weights(spec)
            if self.include_text_weights
            else None
        )
        layer_bundles = self._build_layers_for_stage(spec)
        stage_state = build_text_stage_state(
            spec=spec,
            stage_state_device=self.stage_state_device,
            compute_dtype=self.compute_dtype,
            prefill_attention_mask_2d=self.prefill_attention_mask_2d,
            prefill_seq_len=int(self.prefill_input_ids.shape[-1]),
            batch_size=int(self.prefill_input_ids.shape[0]),
            token_id_dtype=self.prefill_input_ids.dtype,
            hidden_size=self._text_model_config.hidden_size if self._text_model_config is not None else 0,
            layers=layer_bundles,
            text_stage_weights=text_stage_weights,
        )
        stage_state["module_name"] = f"{self.modality}_generate_stage"
        stage_state["stage_type"] = f"{self.modality}_generate_runtime_only"
        stage_state["modality"] = self.modality
        stage_state["max_new_tokens"] = int(self.runtime_config.get("max_new_tokens", 4))

        if self.modality == "multimodal":
            stage_input = self._prefill_stage_inputs_by_stage.get(spec.stage_idx)
            if stage_input is None:
                raise RuntimeError(
                    f"multimodal runtime-only generate 缺少 stage_idx={spec.stage_idx} 的 startup handoff。"
                )
            prefill_runtime_state = self._build_mm_stage_state(
                spec,
                self._mm_prefill_shared if self._mm_prefill_shared is not None else self._prefill_mm_inputs(),
                stage_input=stage_input,
            )
            visual_pos_masks = _runtime_tensor(
                getattr(prefill_runtime_state, "visual_pos_masks", None),
                device=self.stage_state_device,
            )
            deepstack_by_layer = {
                int(layer_idx): _runtime_tensor(
                    deepstack,
                    device=self.stage_state_device,
                    compute_dtype=self.compute_dtype,
                )
                for layer_idx, deepstack in getattr(prefill_runtime_state, "deepstack_by_layer", {}).items()
            }
            stage_state.update(
                {
                    "num_frames": self.extra["num_frames"],
                    "frame_paths": self.extra["frame_paths"],
                    "stage_input": _runtime_tensor(
                        stage_input,
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "layer_input": _runtime_tensor(
                        stage_input,
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "prefill_attention_mask": _runtime_tensor(
                        prefill_runtime_state.attention_mask,
                        device=self.stage_state_device,
                    ),
                    "prefill_position_ids": _runtime_tensor(
                        prefill_runtime_state.position_ids,
                        device=self.stage_state_device,
                    ),
                    "prefill_cos": _runtime_tensor(
                        prefill_runtime_state.cos,
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "prefill_sin": _runtime_tensor(
                        prefill_runtime_state.sin,
                        device=self.stage_state_device,
                        compute_dtype=self.compute_dtype,
                    ),
                    "rope_deltas": _runtime_tensor(
                        getattr(prefill_runtime_state, "rope_deltas", None),
                        device=self.stage_state_device,
                    ),
                    "visual_pos_masks": visual_pos_masks,
                    "deepstack_by_layer": deepstack_by_layer,
                    "deepstack_layer_indices": sorted(deepstack_by_layer),
                }
            )

        if self.modality == "text" and spec.start_idx == 0:
            stage_state["input_ids"] = _runtime_tensor(self.prefill_input_ids, device=self.stage_state_device)
            if text_stage_weights is not None and text_stage_weights.embed_tokens_weight is not None:
                stage_state["embed_tokens_weight"] = text_stage_weights.embed_tokens_weight
        elif self.modality == "multimodal" and spec.start_idx == 0 and text_stage_weights is not None:
            if text_stage_weights.embed_tokens_weight is not None:
                stage_state["embed_tokens_weight"] = text_stage_weights.embed_tokens_weight

        if spec.end_idx == self.num_layers - 1 and text_stage_weights is not None:
            if text_stage_weights.final_norm_weight is not None:
                stage_state["final_norm_weight"] = text_stage_weights.final_norm_weight
            if text_stage_weights.final_norm_eps is not None:
                stage_state["final_norm_eps"] = text_stage_weights.final_norm_eps
            if text_stage_weights.lm_head_weight is not None:
                stage_state["lm_head_weight"] = text_stage_weights.lm_head_weight
            stage_state["lm_head_bias"] = text_stage_weights.lm_head_bias

        return stage_state

    def build_stage_state(self, stage_idx: int) -> StageState:
        spec = self.stage_specs_by_idx[stage_idx]
        with startup_timer(
            self.log_component,
            f"materialize stage_idx={stage_idx} range={spec.start_idx}:{spec.end_idx}",
        ):
            if self.mode == "prefill":
                stage_state = self._build_prefill_stage_state(spec)
            elif self.mode == "decode":
                stage_state = self._build_decode_stage_state(spec)
            elif self.mode == "generate":
                if not self.include_runtime_reference:
                    stage_state = self._build_generate_stage_state_runtime_only(spec)
                else:
                    stage_state = self._build_generate_stage_state(spec)
            else:
                raise ValueError(
                    f"不支持的 direct stage 构造组合: modality={self.modality!r} mode={self.mode!r} stage_idx={stage_idx}"
                )

            if (
                self.mode == "generate"
                and not self.include_text_weights
            ):
                if self.include_runtime_reference:
                    # Historical helper name is text_scaffold, but the scaffold
                    # compaction/rebuild path is also valid for multimodal
                    # generate states when we want local TP ranks to load
                    # weights themselves instead of broadcasting them.
                    return compact_text_scaffold(stage_state)
                if stage_state.get("runtime_only_generate") and self.modality == "text":
                    return compact_text_stage_state(stage_state)
            assert_text_weight_scope(stage_state)
            assert_text_tp_shard_shapes(stage_state)
            return stage_state

    def build_stage_bundle(self, stage_idx: int) -> dict[str, Any]:
        return self.build_stage_state(stage_idx)


DirectStageBundleBuilder = DirectStageStateBuilder


def build_direct_stage_state(
    *,
    stage_idx: int,
    start_idx: int,
    end_idx: int,
    runtime_config: dict[str, Any],
    tp_shard_rank: int | None = None,
    tp_shard_world_size: int | None = None,
    include_text_weights: bool = True,
    mm_activate_frontend: bool | None = None,
) -> StageState:
    with DirectStageStateBuilder(
        stage_specs=[
            StageSpec(
                stage_idx=stage_idx,
                start_idx=start_idx,
                end_idx=end_idx,
                num_layers=end_idx - start_idx + 1,
                save_dtype=runtime_config.get("save_dtype", "auto"),
                bundle_path=None,
            )
        ],
        runtime_config=runtime_config,
        tp_shard_rank=tp_shard_rank,
        tp_shard_world_size=tp_shard_world_size,
        include_text_weights=include_text_weights,
        mm_activate_frontend=mm_activate_frontend,
    ) as builder:
        return builder.build_stage_state(stage_idx)


def build_direct_stage_bundle(
    *,
    stage_idx: int,
    start_idx: int,
    end_idx: int,
    runtime_config: dict[str, Any],
    tp_shard_rank: int | None = None,
    tp_shard_world_size: int | None = None,
    include_text_weights: bool = True,
    mm_activate_frontend: bool | None = None,
) -> dict:
    return build_direct_stage_state(
        stage_idx=stage_idx,
        start_idx=start_idx,
        end_idx=end_idx,
        runtime_config=runtime_config,
        tp_shard_rank=tp_shard_rank,
        tp_shard_world_size=tp_shard_world_size,
        include_text_weights=include_text_weights,
        mm_activate_frontend=mm_activate_frontend,
    )


def prepare_mm_startup_contract(
    *,
    stage_specs: list[StageSpec],
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    if str(runtime_config.get("modality", "")) != "multimodal":
        raise ValueError("prepare_mm_startup_contract 只支持 multimodal runtime_config。")
    with DirectStageStateBuilder(
        stage_specs=stage_specs,
        runtime_config=dict(runtime_config),
        include_text_weights=False,
        mm_activate_frontend=True,
    ) as builder:
        return builder.export_mm_startup_contract()


def select_mm_startup_contract(
    payload: dict[str, Any],
    *,
    local_stage_indices: list[int] | None = None,
) -> dict[str, Any]:
    _assert_thin_mm_startup_payload(payload, context="select")
    normalized = _normalize_mm_startup_contract(payload)
    stage_handoffs = normalized["stage_handoffs"]
    if local_stage_indices is None:
        selected_stage_indices = sorted(stage_handoffs)
    else:
        selected_stage_indices = [int(stage_idx) for stage_idx in local_stage_indices]
    missing_stage_indices = [
        stage_idx
        for stage_idx in selected_stage_indices
        if stage_idx not in stage_handoffs
    ]
    if missing_stage_indices:
        raise RuntimeError(
            f"multimodal startup contract 缺少 local stage handoff: {missing_stage_indices}"
        )

    local_payload: dict[str, Any] = {
        "shared": _clone_mm_shared_to_cpu(normalized["shared"]),
        "stage_handoffs": _clone_mm_stage_handoffs_to_cpu(
            {
                stage_idx: stage_handoffs[stage_idx]
                for stage_idx in selected_stage_indices
            }
        ),
        "num_frames": int(normalized["num_frames"]),
        "frame_paths": list(normalized["frame_paths"]),
    }

    stage_visuals = normalized["stage_visuals"]
    selected_stage_visuals = {
        stage_idx: stage_visuals[stage_idx]
        for stage_idx in selected_stage_indices
        if stage_idx in stage_visuals
    }
    if selected_stage_visuals:
        local_payload["stage_visuals"] = _clone_mm_stage_visuals_to_cpu(
            selected_stage_visuals,
        )
    else:
        if normalized["visual_pos_masks"] is not None:
            local_payload["visual_pos_masks"] = _clone_tensor_to_cpu(
                normalized["visual_pos_masks"],
            )
        if normalized["deepstack_by_layer"]:
            local_payload["deepstack_by_layer"] = _clone_mm_deepstack_to_cpu(
                normalized["deepstack_by_layer"],
            )

    return local_payload


def seed_mm_startup_runtime_config(
    runtime_config: dict[str, Any],
    payload: dict[str, Any],
    *,
    local_stage_indices: list[int] | None = None,
) -> None:
    selected_payload = select_mm_startup_contract(
        payload,
        local_stage_indices=local_stage_indices,
    )

    runtime_config["_mm_startup_shared"] = selected_payload["shared"]
    runtime_config["_mm_startup_stage_handoffs"] = selected_payload["stage_handoffs"]
    runtime_config.pop("_mm_startup_root_input", None)
    selected_stage_visuals = selected_payload.get("stage_visuals")
    if selected_stage_visuals:
        runtime_config["_mm_startup_stage_visuals"] = selected_stage_visuals
        runtime_config.pop("_mm_startup_visual_pos_masks", None)
        runtime_config.pop("_mm_startup_deepstack_by_layer", None)
    else:
        runtime_config.pop("_mm_startup_stage_visuals", None)
        runtime_config["_mm_startup_visual_pos_masks"] = selected_payload.get("visual_pos_masks")
        runtime_config["_mm_startup_deepstack_by_layer"] = selected_payload.get("deepstack_by_layer", {})
    runtime_config["_mm_num_frames"] = int(selected_payload["num_frames"] or runtime_config.get("num_frames", 0))
    runtime_config["_mm_frame_paths"] = list(
        selected_payload["frame_paths"] or runtime_config.get("_mm_frame_paths") or []
    )
    runtime_config["_mm_startup_contract_ready"] = True
    runtime_config["_mm_frontend_state_ready"] = True
    runtime_config.pop("_mm_startup_boundaries", None)
    runtime_config.pop("_mm_frontend_seed", None)
    runtime_config.pop("_mm_frontend_meta", None)
    runtime_config.pop("_mm_frontend_plan", None)
    runtime_config.pop("_mm_frontend_state", None)


class StageStateLoader:
    """Loads a rank-local StageState from a lightweight direct-runtime scaffold."""

    def __init__(
        self,
        *,
        runtime_config: dict[str, Any],
        compute_dtype: torch.dtype,
        tp_shard_rank: int | None = None,
        tp_shard_world_size: int | None = None,
    ) -> None:
        self.runtime_config = runtime_config
        self.compute_dtype = compute_dtype
        self.tp_shard_rank = tp_shard_rank
        self.tp_shard_world_size = tp_shard_world_size

    def load_from_scaffold(self, stage_state_scaffold: StageState) -> StageState:
        return _materialize_text_stage_state(
            stage_state_scaffold=stage_state_scaffold,
            runtime_config=self.runtime_config,
            compute_dtype=self.compute_dtype,
            tp_shard_rank=self.tp_shard_rank,
            tp_shard_world_size=self.tp_shard_world_size,
        )


def materialize_text_stage_state(
    *,
    stage_state_scaffold: StageState,
    runtime_config: dict[str, Any],
    compute_dtype: torch.dtype,
    tp_shard_rank: int | None = None,
    tp_shard_world_size: int | None = None,
) -> StageState:
    return StageStateLoader(
        runtime_config=runtime_config,
        compute_dtype=compute_dtype,
        tp_shard_rank=tp_shard_rank,
        tp_shard_world_size=tp_shard_world_size,
    ).load_from_scaffold(stage_state_scaffold)


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
    include_runtime_reference: bool | None = None,
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
        include_runtime_reference=include_runtime_reference,
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
        bundle_dir=None,
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
    include_runtime_reference: bool | None = None,
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
        include_runtime_reference=include_runtime_reference,
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


def build_direct_tp_manifest(
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
    include_runtime_reference: bool | None = None,
) -> TensorParallelManifest:
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
        include_runtime_reference=include_runtime_reference,
    )
    parsed_tp_degrees = parse_tp_degrees(tp_degrees)
    if pipeline_manifest.num_stages != 1:
        raise ValueError(
            f"backend=tp 是单 stage TP，当前 stage 数是 {pipeline_manifest.num_stages}。"
        )
    if len(parsed_tp_degrees) != 1:
        raise ValueError(f"backend=tp 要求恰好一个 TP degree，当前拿到 {parsed_tp_degrees!r}。")
    return TensorParallelManifest.from_pipeline_manifest(
        pipeline_manifest,
        tp_degree=parsed_tp_degrees[0],
        runtime=_runtime_name(modality, mode, "tp"),
    )

__all__ = [
    "DirectStageStateBuilder",
    "StageStateLoader",
    "build_direct_stage_state",
    "pack_mm_startup_transport",
    "prepare_mm_startup_contract",
    "restore_mm_startup_transport",
    "seed_mm_startup_runtime_config",
    "materialize_text_stage_state",
    "pack_text_scaffold_transport",
    "build_direct_pipeline_manifest",
    "build_direct_tp_manifest",
    "build_direct_hybrid_manifest",
    "compact_text_prompt_meta",
    "prepare_text_prompt_meta",
    "restore_text_scaffold_transport",
    "restore_text_prompt_meta",
]
