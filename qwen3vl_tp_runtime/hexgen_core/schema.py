"""Structured runtime schemas for manifests, rank context, and stage handoff payloads."""

from dataclasses import asdict, dataclass, field
from typing import Any, TypeAlias

import torch

StageState: TypeAlias = dict[str, Any]


HYBRID_RUNTIME_INPUT_PROTOCOL = "hybrid_runtime_inputs_v1"


class HybridRuntimeInputSchema:
    """Formal schema for HYBRID runtime-only stage-group input broadcast."""

    PROTOCOL = HYBRID_RUNTIME_INPUT_PROTOCOL
    COMMON_REQUIRED_KEYS = frozenset(
        (
            "protocol",
            "modality",
            "mode",
            "runtime_only_generate",
        )
    )
    COMMON_ALLOWED_KEYS = COMMON_REQUIRED_KEYS
    TEXT_REQUIRED_KEYS = COMMON_REQUIRED_KEYS | frozenset(
        (
            "input_ids",
            "runtime_only_prompt_local_rebuild",
        )
    )
    TEXT_ALLOWED_KEYS = TEXT_REQUIRED_KEYS | frozenset(("attention_mask_2d",))
    MULTIMODAL_REQUIRED_KEYS = COMMON_REQUIRED_KEYS | frozenset(
        (
            "shared",
            "stage_handoff",
        )
    )
    MULTIMODAL_ALLOWED_KEYS = MULTIMODAL_REQUIRED_KEYS | frozenset(("stage_visuals",))
    MULTIMODAL_SHARED_REQUIRED_KEYS = frozenset(
        (
            "input_ids",
            "rope_deltas",
        )
    )
    MULTIMODAL_SHARED_ALLOWED_KEYS = MULTIMODAL_SHARED_REQUIRED_KEYS | frozenset(
        (
            "attention_mask_2d",
            "position_ids",
            "mm_token_type_ids",
            "image_grid_thw",
            "video_grid_thw",
        )
    )
    MULTIMODAL_STAGE_HANDOFF_REQUIRED_KEYS = frozenset(("stage_input",))
    MULTIMODAL_STAGE_HANDOFF_ALLOWED_KEYS = MULTIMODAL_STAGE_HANDOFF_REQUIRED_KEYS
    MULTIMODAL_STAGE_VISUAL_ALLOWED_KEYS = frozenset(
        (
            "visual_pos_masks",
            "deepstack_by_layer",
        )
    )

    COMMON_LOCAL_REBUILD_FIELDS = frozenset(
        (
            "module_name",
            "stage_type",
            "stage_idx",
            "start_idx",
            "end_idx",
            "save_dtype",
            "max_new_tokens",
            "layers",
        )
    )
    TEXT_LOCAL_REBUILD_FIELDS = COMMON_LOCAL_REBUILD_FIELDS | frozenset(
        (
            "prefill_attention_mask_2d",
            "prefill_seq_len",
            "batch_size",
            "token_id_dtype",
        )
    )
    MULTIMODAL_LOCAL_REBUILD_FIELDS = COMMON_LOCAL_REBUILD_FIELDS | frozenset(
        (
            "prefill_attention_mask_2d",
            "prefill_attention_mask",
            "prefill_position_ids",
            "prefill_cos",
            "prefill_sin",
            "num_frames",
            "frame_paths",
        )
    )

    FORBIDDEN_BROADCAST_KEYS = frozenset(
        (
            "attention_mask",
            "batch_size",
            "boundaries",
            "bundle",
            "cache_by_layer",
            "cos",
            "end_idx",
            "frame_dir",
            "frame_paths",
            "frontend_paths",
            "hidden_size",
            "hidden_states",
            "layer_input",
            "layers",
            "module_name",
            "num_frames",
            "prefill_attention_mask",
            "prefill_attention_mask_2d",
            "prefill_cos",
            "prefill_position_ids",
            "prefill_sin",
            "replay_bundle",
            "replay_bundle_path",
            "root_input",
            "save_dtype",
            "sin",
            "stage_bundle",
            "stage_idx",
            "stage_output",
            "stage_type",
            "start_idx",
            "token_id_dtype",
        )
    )
    FORBIDDEN_BROADCAST_KEY_SUFFIXES = ("_weight", "_bias")

    @classmethod
    def allowed_top_level_keys(cls, modality: str) -> frozenset[str]:
        if modality == "text":
            return cls.TEXT_ALLOWED_KEYS
        if modality == "multimodal":
            return cls.MULTIMODAL_ALLOWED_KEYS
        raise RuntimeError(f"HYBRID runtime input schema 不支持 modality={modality!r}。")

    @classmethod
    def local_rebuild_fields(cls, modality: str) -> frozenset[str]:
        if modality == "text":
            return cls.TEXT_LOCAL_REBUILD_FIELDS
        if modality == "multimodal":
            return cls.MULTIMODAL_LOCAL_REBUILD_FIELDS
        raise RuntimeError(f"HYBRID runtime input schema 不支持 modality={modality!r}。")

    @classmethod
    def validate(cls, payload: dict[str, Any], *, context: str = "runtime_inputs") -> None:
        if not isinstance(payload, dict):
            raise RuntimeError(f"HYBRID runtime input schema 需要 dict，context={context}。")

        cls._assert_no_forbidden_keys(payload, context=context)
        cls._assert_required_keys(
            payload,
            required=cls.COMMON_REQUIRED_KEYS,
            context=context,
            scope="runtime_inputs",
        )
        protocol = payload.get("protocol")
        if protocol != cls.PROTOCOL:
            raise RuntimeError(
                "HYBRID runtime input protocol 不匹配，"
                f"context={context} expected={cls.PROTOCOL!r} actual={protocol!r}"
            )
        mode = payload.get("mode")
        if mode != "generate":
            raise RuntimeError(
                "HYBRID runtime input 目前只支持 generate mode，"
                f"context={context} mode={mode!r}"
            )
        if payload.get("runtime_only_generate") is not True:
            raise RuntimeError(
                "HYBRID runtime input 必须显式 runtime_only_generate=True，"
                f"context={context}"
            )

        modality = payload.get("modality")
        if modality == "text":
            cls._validate_text(payload, context=context)
            return
        if modality == "multimodal":
            cls._validate_multimodal(payload, context=context)
            return
        raise RuntimeError(
            "HYBRID runtime input modality 不在协议内，"
            f"context={context} modality={modality!r}"
        )

    @classmethod
    def _validate_text(cls, payload: dict[str, Any], *, context: str) -> None:
        cls._assert_allowed_keys(
            payload,
            allowed=cls.TEXT_ALLOWED_KEYS,
            context=context,
            scope="text runtime_inputs",
        )
        cls._assert_required_keys(
            payload,
            required=cls.TEXT_REQUIRED_KEYS,
            context=context,
            scope="text runtime_inputs",
        )
        if payload.get("runtime_only_prompt_local_rebuild") is not True:
            raise RuntimeError(
                "text runtime input 必须声明 runtime_only_prompt_local_rebuild=True，"
                f"context={context}"
            )
        if not torch.is_tensor(payload.get("input_ids")):
            raise RuntimeError(f"text runtime input 必须携带 tensor input_ids，context={context}。")
        attention_mask = payload.get("attention_mask_2d")
        if attention_mask is not None and not torch.is_tensor(attention_mask):
            raise RuntimeError(
                "text runtime input attention_mask_2d 必须是 tensor 或省略，"
                f"context={context}"
            )

    @classmethod
    def _validate_multimodal(cls, payload: dict[str, Any], *, context: str) -> None:
        cls._assert_allowed_keys(
            payload,
            allowed=cls.MULTIMODAL_ALLOWED_KEYS,
            context=context,
            scope="multimodal runtime_inputs",
        )
        cls._assert_required_keys(
            payload,
            required=cls.MULTIMODAL_REQUIRED_KEYS,
            context=context,
            scope="multimodal runtime_inputs",
        )

        shared = payload.get("shared")
        if not isinstance(shared, dict):
            raise RuntimeError(f"multimodal runtime input shared 必须是 dict，context={context}。")
        cls._assert_allowed_keys(
            shared,
            allowed=cls.MULTIMODAL_SHARED_ALLOWED_KEYS,
            context=context,
            scope="multimodal runtime_inputs.shared",
        )
        cls._assert_required_keys(
            shared,
            required=cls.MULTIMODAL_SHARED_REQUIRED_KEYS,
            context=context,
            scope="multimodal runtime_inputs.shared",
        )
        for tensor_key in cls.MULTIMODAL_SHARED_REQUIRED_KEYS:
            if not torch.is_tensor(shared.get(tensor_key)):
                raise RuntimeError(
                    "multimodal runtime input shared 必须携带必要 tensor，"
                    f"context={context} key={tensor_key}"
                )
        for tensor_key in cls.MULTIMODAL_SHARED_ALLOWED_KEYS:
            value = shared.get(tensor_key)
            if value is not None and not torch.is_tensor(value):
                raise RuntimeError(
                    "multimodal runtime input shared 字段必须是 tensor 或省略，"
                    f"context={context} key={tensor_key}"
                )

        stage_handoff = payload.get("stage_handoff")
        if not isinstance(stage_handoff, dict):
            raise RuntimeError(f"multimodal runtime input stage_handoff 必须是 dict，context={context}。")
        cls._assert_allowed_keys(
            stage_handoff,
            allowed=cls.MULTIMODAL_STAGE_HANDOFF_ALLOWED_KEYS,
            context=context,
            scope="multimodal runtime_inputs.stage_handoff",
        )
        cls._assert_required_keys(
            stage_handoff,
            required=cls.MULTIMODAL_STAGE_HANDOFF_REQUIRED_KEYS,
            context=context,
            scope="multimodal runtime_inputs.stage_handoff",
        )
        if not torch.is_tensor(stage_handoff.get("stage_input")):
            raise RuntimeError(
                "multimodal runtime input stage_handoff 必须携带 tensor stage_input，"
                f"context={context}"
            )

        stage_visuals = payload.get("stage_visuals")
        if stage_visuals is None:
            return
        if not isinstance(stage_visuals, dict):
            raise RuntimeError(f"multimodal runtime input stage_visuals 必须是 dict，context={context}。")
        cls._assert_allowed_keys(
            stage_visuals,
            allowed=cls.MULTIMODAL_STAGE_VISUAL_ALLOWED_KEYS,
            context=context,
            scope="multimodal runtime_inputs.stage_visuals",
        )
        visual_pos_masks = stage_visuals.get("visual_pos_masks")
        if visual_pos_masks is not None and not torch.is_tensor(visual_pos_masks):
            raise RuntimeError(
                "multimodal runtime input stage_visuals.visual_pos_masks 必须是 tensor 或省略，"
                f"context={context}"
            )
        deepstack_by_layer = stage_visuals.get("deepstack_by_layer")
        if deepstack_by_layer is None:
            return
        if not isinstance(deepstack_by_layer, dict):
            raise RuntimeError(
                "multimodal runtime input stage_visuals.deepstack_by_layer 必须是 dict，"
                f"context={context}"
            )
        for layer_idx, deepstack in deepstack_by_layer.items():
            if not isinstance(layer_idx, int):
                raise RuntimeError(
                    "multimodal runtime input deepstack layer key 必须是 int，"
                    f"context={context} key={layer_idx!r}"
                )
            if deepstack is not None and not torch.is_tensor(deepstack):
                raise RuntimeError(
                    "multimodal runtime input deepstack value 必须是 tensor 或 None，"
                    f"context={context} key={layer_idx!r}"
                )

    @classmethod
    def _assert_allowed_keys(
        cls,
        payload: dict[Any, Any],
        *,
        allowed: frozenset[str],
        context: str,
        scope: str,
    ) -> None:
        invalid = [str(key) for key in payload if str(key) not in allowed]
        if invalid:
            raise RuntimeError(
                "HYBRID runtime input 字段不在协议内，"
                f"context={context} scope={scope} invalid_keys={invalid}"
            )

    @classmethod
    def _assert_required_keys(
        cls,
        payload: dict[str, Any],
        *,
        required: frozenset[str],
        context: str,
        scope: str,
    ) -> None:
        missing = [key for key in sorted(required) if key not in payload]
        if missing:
            raise RuntimeError(
                "HYBRID runtime input 缺少协议必需字段，"
                f"context={context} scope={scope} missing_keys={missing}"
            )

    @classmethod
    def _assert_no_forbidden_keys(
        cls,
        value: Any,
        *,
        context: str,
        path: tuple[str, ...] = (),
    ) -> None:
        if isinstance(value, dict):
            for raw_key, item in value.items():
                key = str(raw_key)
                current_path = (*path, key)
                if cls._is_forbidden_broadcast_key(key):
                    joined = ".".join(current_path)
                    raise RuntimeError(
                        "HYBRID runtime input 禁止广播 stage-local、weight、frontend path "
                        "或 derived attention 字段，"
                        f"context={context} key={joined}"
                    )
                cls._assert_no_forbidden_keys(item, context=context, path=current_path)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                cls._assert_no_forbidden_keys(item, context=context, path=(*path, str(index)))

    @classmethod
    def _is_forbidden_broadcast_key(cls, key: str) -> bool:
        return key in cls.FORBIDDEN_BROADCAST_KEYS or any(
            key.endswith(suffix)
            for suffix in cls.FORBIDDEN_BROADCAST_KEY_SUFFIXES
        )


@dataclass(slots=True)
class StageReplaySpec:
    """Replay-only metadata for a captured pipeline stage."""

    bundle_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: "StageReplaySpec | dict[str, Any] | str | None") -> "StageReplaySpec | None":
        if data is None:
            return None
        if isinstance(data, StageReplaySpec):
            return data
        if isinstance(data, str):
            return cls(bundle_path=data)
        return cls(bundle_path=data["bundle_path"])


@dataclass(slots=True)
class ManifestReplaySpec:
    """Replay-only metadata shared by captured pipeline manifests."""

    bundle_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: "ManifestReplaySpec | dict[str, Any] | str | None",
    ) -> "ManifestReplaySpec | None":
        if data is None:
            return None
        if isinstance(data, ManifestReplaySpec):
            return data
        if isinstance(data, str):
            return cls(bundle_dir=data)
        return cls(bundle_dir=data.get("bundle_dir"))


@dataclass(slots=True, init=False)
class StageSpec:
    """Direct runtime metadata describing one pipeline stage."""

    stage_idx: int
    start_idx: int
    end_idx: int
    num_layers: int
    save_dtype: str
    replay: StageReplaySpec | None = None

    def __init__(
        self,
        stage_idx: int,
        start_idx: int,
        end_idx: int,
        num_layers: int,
        save_dtype: str,
        replay: StageReplaySpec | dict[str, Any] | str | None = None,
        bundle_path: str | None = None,
    ) -> None:
        self.stage_idx = stage_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_layers = num_layers
        self.save_dtype = save_dtype
        self.replay = StageReplaySpec.from_dict(replay)
        if bundle_path is None:
            return
        if self.replay is not None and self.replay.bundle_path != bundle_path:
            raise ValueError("StageSpec replay.bundle_path 和 legacy bundle_path 不一致。")
        self.replay = StageReplaySpec(bundle_path=bundle_path)

    @property
    def is_direct(self) -> bool:
        return self.replay_bundle_path is None

    @property
    def replay_bundle_path(self) -> str | None:
        return None if self.replay is None else self.replay.bundle_path

    @property
    def bundle_path(self) -> str | None:
        """Compatibility shim for legacy replay callers."""

        return self.replay_bundle_path

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "stage_idx": self.stage_idx,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "num_layers": self.num_layers,
            "save_dtype": self.save_dtype,
        }
        if self.replay is not None:
            payload["replay"] = self.replay.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageSpec":
        replay = StageReplaySpec.from_dict(data.get("replay"))
        return cls(
            stage_idx=data["stage_idx"],
            start_idx=data["start_idx"],
            end_idx=data["end_idx"],
            num_layers=data["num_layers"],
            save_dtype=data["save_dtype"],
            replay=replay,
            bundle_path=data.get("bundle_path"),
        )


@dataclass(slots=True)
class BoundaryStats:
    """Numeric diff stats for one inter-stage boundary."""

    src_stage_idx: int
    dst_stage_idx: int
    max_diff: float
    mean_diff: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoundaryStats":
        return cls(**data)


@dataclass(slots=True, init=False)
class TextPipelineManifest:
    """Serializable manifest for a multi-stage direct runtime."""

    pipeline_type: str
    num_stages: int
    stage_ranges: list[tuple[int, int]]
    stages: list[StageSpec]
    boundaries: list[BoundaryStats]
    num_frames: int
    save_dtype: str
    runtime_config: dict[str, Any] = field(default_factory=dict)
    replay: ManifestReplaySpec | None = None

    def __init__(
        self,
        pipeline_type: str,
        num_stages: int,
        stage_ranges: list[tuple[int, int]],
        bundle_dir: str | None = None,
        *,
        stages: list[StageSpec],
        boundaries: list[BoundaryStats],
        num_frames: int,
        save_dtype: str,
        runtime_config: dict[str, Any] | None = None,
        replay: ManifestReplaySpec | dict[str, Any] | str | None = None,
    ) -> None:
        self.pipeline_type = pipeline_type
        self.num_stages = num_stages
        self.stage_ranges = stage_ranges
        self.stages = stages
        self.boundaries = boundaries
        self.num_frames = num_frames
        self.save_dtype = save_dtype
        self.runtime_config = {} if runtime_config is None else runtime_config
        self.replay = ManifestReplaySpec.from_dict(replay)
        if bundle_dir is None:
            return
        if self.replay is not None and self.replay.bundle_dir != bundle_dir:
            raise ValueError("manifest replay.bundle_dir 和 legacy bundle_dir 不一致。")
        self.replay = ManifestReplaySpec(bundle_dir=bundle_dir)

    @property
    def is_direct(self) -> bool:
        return self.replay is None and all(stage.is_direct for stage in self.stages)

    @property
    def replay_bundle_dir(self) -> str | None:
        return None if self.replay is None else self.replay.bundle_dir

    @property
    def bundle_dir(self) -> str | None:
        """Compatibility shim for legacy replay callers."""

        return self.replay_bundle_dir

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "pipeline_type": self.pipeline_type,
            "num_stages": self.num_stages,
            "stage_ranges": self.stage_ranges,
            "stages": [stage.to_dict() for stage in self.stages],
            "boundaries": [asdict(boundary) for boundary in self.boundaries],
            "num_frames": self.num_frames,
            "save_dtype": self.save_dtype,
            "runtime_config": self.runtime_config,
        }
        if self.replay is not None:
            payload["replay"] = self.replay.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextPipelineManifest":
        replay = ManifestReplaySpec.from_dict(data.get("replay"))
        return cls(
            pipeline_type=data["pipeline_type"],
            num_stages=data["num_stages"],
            stage_ranges=[tuple(item) for item in data["stage_ranges"]],
            stages=[StageSpec.from_dict(item) for item in data["stages"]],
            boundaries=[BoundaryStats.from_dict(item) for item in data["boundaries"]],
            num_frames=data["num_frames"],
            save_dtype=data["save_dtype"],
            runtime_config=data.get("runtime_config", {}),
            replay=replay,
            bundle_dir=data.get("bundle_dir"),
        )


@dataclass(slots=True, init=False)
class TensorParallelManifest:
    """Serializable manifest for a standalone TP direct runtime."""

    runtime: str
    tp_degree: int
    tp_degrees: list[int]
    stage_rank_groups: list[list[int]]
    pp_rank_groups: list[list[int]]
    world_size: int
    num_stages: int
    send_list: list[list[int]]
    recv_list: list[list[int]]
    send_empty_list: list[list[bool]]
    recv_empty_list: list[list[bool]]
    stage_ranges: list[tuple[int, int]]
    stages: list[StageSpec]
    boundaries: list[BoundaryStats]
    num_frames: int
    save_dtype: str
    pipeline_type: str = "text"
    runtime_config: dict[str, Any] = field(default_factory=dict)
    replay: ManifestReplaySpec | None = None

    def __init__(
        self,
        runtime: str,
        tp_degree: int | None = None,
        stage_ranges: list[tuple[int, int]] | None = None,
        bundle_dir: str | None = None,
        *,
        stages: list[StageSpec],
        boundaries: list[BoundaryStats],
        num_frames: int,
        save_dtype: str,
        pipeline_type: str = "text",
        runtime_config: dict[str, Any] | None = None,
        replay: ManifestReplaySpec | dict[str, Any] | str | None = None,
        tp_degrees: list[int] | None = None,
        stage_rank_groups: list[list[int]] | None = None,
        pp_rank_groups: list[list[int]] | None = None,
        world_size: int | None = None,
        num_stages: int | None = None,
        send_list: list[list[int]] | None = None,
        recv_list: list[list[int]] | None = None,
        send_empty_list: list[list[bool]] | None = None,
        recv_empty_list: list[list[bool]] | None = None,
    ) -> None:
        if tp_degree is None:
            if tp_degrees is None or len(tp_degrees) != 1:
                raise ValueError("TensorParallelManifest 需要恰好一个 TP degree。")
            tp_degree = int(tp_degrees[0])
        if tp_degree <= 1:
            raise ValueError(f"backend=tp 要求 TP degree > 1，当前拿到 {tp_degree}。")
        if stage_ranges is None:
            raise ValueError("TensorParallelManifest 需要 stage_ranges。")
        if len(stage_ranges) != 1 or len(stages) != 1:
            raise ValueError("TensorParallelManifest 是单 stage TP manifest。")
        if num_stages is not None and num_stages != 1:
            raise ValueError(f"TensorParallelManifest num_stages 必须是 1，当前拿到 {num_stages}。")
        if world_size is not None and world_size != tp_degree:
            raise ValueError(
                "TensorParallelManifest world_size 必须等于 TP degree，"
                f"world_size={world_size} tp_degree={tp_degree}。"
            )

        self.runtime = runtime
        self.tp_degree = int(tp_degree)
        self.tp_degrees = [self.tp_degree]
        self.stage_rank_groups = [list(range(self.tp_degree))]
        self.pp_rank_groups = [[rank] for rank in range(self.tp_degree)]
        self.world_size = self.tp_degree
        self.num_stages = 1
        self.send_list = [[] for _ in range(self.tp_degree)]
        self.recv_list = [[] for _ in range(self.tp_degree)]
        self.send_empty_list = [[] for _ in range(self.tp_degree)]
        self.recv_empty_list = [[] for _ in range(self.tp_degree)]
        self.stage_ranges = stage_ranges
        self.stages = stages
        self.boundaries = boundaries
        self.num_frames = num_frames
        self.save_dtype = save_dtype
        self.pipeline_type = pipeline_type
        self.runtime_config = {} if runtime_config is None else runtime_config
        self.replay = ManifestReplaySpec.from_dict(replay)
        if bundle_dir is not None:
            if self.replay is not None and self.replay.bundle_dir != bundle_dir:
                raise ValueError("tp manifest replay.bundle_dir 和 legacy bundle_dir 不一致。")
            self.replay = ManifestReplaySpec(bundle_dir=bundle_dir)

        expected_layout = {
            "tp_degrees": self.tp_degrees,
            "stage_rank_groups": self.stage_rank_groups,
            "pp_rank_groups": self.pp_rank_groups,
            "send_list": self.send_list,
            "recv_list": self.recv_list,
            "send_empty_list": self.send_empty_list,
            "recv_empty_list": self.recv_empty_list,
        }
        provided_layout = {
            "tp_degrees": tp_degrees,
            "stage_rank_groups": stage_rank_groups,
            "pp_rank_groups": pp_rank_groups,
            "send_list": send_list,
            "recv_list": recv_list,
            "send_empty_list": send_empty_list,
            "recv_empty_list": recv_empty_list,
        }
        for key, value in provided_layout.items():
            if value is not None and value != expected_layout[key]:
                raise ValueError(f"TensorParallelManifest 的 {key} 和单 stage TP layout 不一致。")

    @property
    def is_direct(self) -> bool:
        return self.replay is None and all(stage.is_direct for stage in self.stages)

    @property
    def replay_bundle_dir(self) -> str | None:
        return None if self.replay is None else self.replay.bundle_dir

    @property
    def bundle_dir(self) -> str | None:
        """Compatibility shim for legacy replay callers."""

        return self.replay_bundle_dir

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "runtime": self.runtime,
            "tp_degree": self.tp_degree,
            "tp_degrees": self.tp_degrees,
            "stage_rank_groups": self.stage_rank_groups,
            "pp_rank_groups": self.pp_rank_groups,
            "world_size": self.world_size,
            "num_stages": self.num_stages,
            "send_list": self.send_list,
            "recv_list": self.recv_list,
            "send_empty_list": self.send_empty_list,
            "recv_empty_list": self.recv_empty_list,
            "pipeline_type": self.pipeline_type,
            "stage_ranges": self.stage_ranges,
            "stages": [stage.to_dict() for stage in self.stages],
            "boundaries": [asdict(boundary) for boundary in self.boundaries],
            "num_frames": self.num_frames,
            "save_dtype": self.save_dtype,
            "runtime_config": self.runtime_config,
        }
        if self.replay is not None:
            payload["replay"] = self.replay.to_dict()
        return payload

    @classmethod
    def from_pipeline_manifest(
        cls,
        manifest: TextPipelineManifest,
        *,
        tp_degree: int,
        runtime: str = "text_tp",
    ) -> "TensorParallelManifest":
        return cls(
            runtime=runtime,
            tp_degree=tp_degree,
            pipeline_type=manifest.pipeline_type,
            stage_ranges=manifest.stage_ranges,
            stages=manifest.stages,
            boundaries=manifest.boundaries,
            num_frames=manifest.num_frames,
            save_dtype=manifest.save_dtype,
            runtime_config=dict(manifest.runtime_config),
            replay=manifest.replay,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorParallelManifest":
        replay = ManifestReplaySpec.from_dict(data.get("replay"))
        tp_degrees = data.get("tp_degrees")
        tp_degree = data.get("tp_degree")
        if tp_degree is None and tp_degrees is not None and len(tp_degrees) == 1:
            tp_degree = tp_degrees[0]
        return cls(
            runtime=data.get("runtime", "text_tp"),
            tp_degree=tp_degree,
            tp_degrees=tp_degrees,
            stage_rank_groups=data.get("stage_rank_groups"),
            pp_rank_groups=data.get("pp_rank_groups"),
            world_size=data.get("world_size"),
            num_stages=data.get("num_stages"),
            send_list=data.get("send_list"),
            recv_list=data.get("recv_list"),
            send_empty_list=data.get("send_empty_list"),
            recv_empty_list=data.get("recv_empty_list"),
            pipeline_type=data.get("pipeline_type", "text"),
            stage_ranges=[tuple(item) for item in data["stage_ranges"]],
            stages=[StageSpec.from_dict(item) for item in data["stages"]],
            boundaries=[BoundaryStats.from_dict(item) for item in data["boundaries"]],
            num_frames=data["num_frames"],
            save_dtype=data["save_dtype"],
            runtime_config=data.get("runtime_config", {}),
            replay=replay,
            bundle_dir=data.get("bundle_dir"),
        )


@dataclass(slots=True)
class HybridLayout:
    """Serializable communication layout for hybrid PP+TP execution."""

    tp_degrees: list[int]
    stage_rank_groups: list[list[int]]
    pp_rank_groups: list[list[int]]
    world_size: int
    num_stages: int
    send_list: list[list[int]]
    recv_list: list[list[int]]
    send_empty_list: list[list[bool]]
    recv_empty_list: list[list[bool]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HybridLayout":
        return cls(**data)


@dataclass(slots=True, init=False)
class TextHybridManifest:
    """Serializable manifest for a hybrid PP+TP direct runtime."""

    runtime: str
    tp_degrees: list[int]
    stage_rank_groups: list[list[int]]
    pp_rank_groups: list[list[int]]
    world_size: int
    num_stages: int
    send_list: list[list[int]]
    recv_list: list[list[int]]
    send_empty_list: list[list[bool]]
    recv_empty_list: list[list[bool]]
    stage_ranges: list[tuple[int, int]]
    stages: list[StageSpec]
    boundaries: list[BoundaryStats]
    num_frames: int
    save_dtype: str
    pipeline_type: str = "text"
    runtime_config: dict[str, Any] = field(default_factory=dict)
    replay: ManifestReplaySpec | None = None

    def __init__(
        self,
        runtime: str,
        tp_degrees: list[int],
        stage_rank_groups: list[list[int]],
        pp_rank_groups: list[list[int]],
        world_size: int,
        num_stages: int,
        send_list: list[list[int]],
        recv_list: list[list[int]],
        send_empty_list: list[list[bool]],
        recv_empty_list: list[list[bool]],
        stage_ranges: list[tuple[int, int]],
        bundle_dir: str | None = None,
        *,
        stages: list[StageSpec],
        boundaries: list[BoundaryStats],
        num_frames: int,
        save_dtype: str,
        pipeline_type: str = "text",
        runtime_config: dict[str, Any] | None = None,
        replay: ManifestReplaySpec | dict[str, Any] | str | None = None,
    ) -> None:
        self.runtime = runtime
        self.tp_degrees = tp_degrees
        self.stage_rank_groups = stage_rank_groups
        self.pp_rank_groups = pp_rank_groups
        self.world_size = world_size
        self.num_stages = num_stages
        self.send_list = send_list
        self.recv_list = recv_list
        self.send_empty_list = send_empty_list
        self.recv_empty_list = recv_empty_list
        self.stage_ranges = stage_ranges
        self.stages = stages
        self.boundaries = boundaries
        self.num_frames = num_frames
        self.save_dtype = save_dtype
        self.pipeline_type = pipeline_type
        self.runtime_config = {} if runtime_config is None else runtime_config
        self.replay = ManifestReplaySpec.from_dict(replay)
        if bundle_dir is None:
            return
        if self.replay is not None and self.replay.bundle_dir != bundle_dir:
            raise ValueError("hybrid manifest replay.bundle_dir 和 legacy bundle_dir 不一致。")
        self.replay = ManifestReplaySpec(bundle_dir=bundle_dir)

    @property
    def is_direct(self) -> bool:
        return self.replay is None and all(stage.is_direct for stage in self.stages)

    @property
    def replay_bundle_dir(self) -> str | None:
        return None if self.replay is None else self.replay.bundle_dir

    @property
    def bundle_dir(self) -> str | None:
        """Compatibility shim for legacy replay callers."""

        return self.replay_bundle_dir

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "runtime": self.runtime,
            "tp_degrees": self.tp_degrees,
            "stage_rank_groups": self.stage_rank_groups,
            "pp_rank_groups": self.pp_rank_groups,
            "world_size": self.world_size,
            "num_stages": self.num_stages,
            "send_list": self.send_list,
            "recv_list": self.recv_list,
            "send_empty_list": self.send_empty_list,
            "recv_empty_list": self.recv_empty_list,
            "pipeline_type": self.pipeline_type,
            "stage_ranges": self.stage_ranges,
            "stages": [stage.to_dict() for stage in self.stages],
            "boundaries": [asdict(boundary) for boundary in self.boundaries],
            "num_frames": self.num_frames,
            "save_dtype": self.save_dtype,
            "runtime_config": self.runtime_config,
        }
        if self.replay is not None:
            payload["replay"] = self.replay.to_dict()
        return payload

    @classmethod
    def from_pipeline_manifest(
        cls,
        manifest: TextPipelineManifest,
        layout: HybridLayout,
        runtime: str = "text_hybrid",
    ) -> "TextHybridManifest":
        return cls(
            runtime=runtime,
            tp_degrees=layout.tp_degrees,
            stage_rank_groups=layout.stage_rank_groups,
            pp_rank_groups=layout.pp_rank_groups,
            world_size=layout.world_size,
            num_stages=layout.num_stages,
            send_list=layout.send_list,
            recv_list=layout.recv_list,
            send_empty_list=layout.send_empty_list,
            recv_empty_list=layout.recv_empty_list,
            pipeline_type=manifest.pipeline_type,
            stage_ranges=manifest.stage_ranges,
            stages=manifest.stages,
            boundaries=manifest.boundaries,
            num_frames=manifest.num_frames,
            save_dtype=manifest.save_dtype,
            runtime_config=dict(manifest.runtime_config),
            replay=manifest.replay,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextHybridManifest":
        replay = ManifestReplaySpec.from_dict(data.get("replay"))
        return cls(
            runtime=data.get("runtime", "text_hybrid"),
            tp_degrees=data["tp_degrees"],
            stage_rank_groups=data["stage_rank_groups"],
            pp_rank_groups=data["pp_rank_groups"],
            world_size=data["world_size"],
            num_stages=data["num_stages"],
            send_list=data["send_list"],
            recv_list=data["recv_list"],
            send_empty_list=data["send_empty_list"],
            recv_empty_list=data["recv_empty_list"],
            pipeline_type=data.get("pipeline_type", "text"),
            stage_ranges=[tuple(item) for item in data["stage_ranges"]],
            stages=[StageSpec.from_dict(item) for item in data["stages"]],
            boundaries=[BoundaryStats.from_dict(item) for item in data["boundaries"]],
            num_frames=data["num_frames"],
            save_dtype=data["save_dtype"],
            runtime_config=data.get("runtime_config", {}),
            replay=replay,
            bundle_dir=data.get("bundle_dir"),
        )


@dataclass(slots=True)
class HybridRankContext:
    """Resolved runtime context for one rank within a hybrid layout."""

    stage_idx: int
    stage_ranks: list[int]
    tp_degree: int
    local_rank: int
    leader_rank: int
    prev_leader_rank: int | None
    next_leader_rank: int | None
    stage_group: Any
    pp_group_idx: int
    current_pp_group: list[int]
    send_list: list[int]
    recv_list: list[int]
    send_empty_list: list[bool]
    recv_empty_list: list[bool]


@dataclass(slots=True)
class PayloadSummary:
    """Metadata describing one transport send call."""

    is_empty: bool
    num_tensors: int
    payload_keys: list[str]
    tensor_shapes: dict[str, tuple[int, ...] | None]
    tensor_dtypes: dict[str, str | None] = field(default_factory=dict)
    tensor_numels: dict[str, int] = field(default_factory=dict)
    tensor_bytes: dict[str, int] = field(default_factory=dict)
    total_tensor_bytes: int = 0

    @classmethod
    def empty(cls) -> "PayloadSummary":
        return cls(
            is_empty=True,
            num_tensors=0,
            payload_keys=[],
            tensor_shapes={},
            tensor_dtypes={},
            tensor_numels={},
            tensor_bytes={},
            total_tensor_bytes=0,
        )

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, torch.Tensor | None] | None,
    ) -> "PayloadSummary":
        if payload is None:
            return cls.empty()
        tensor_shapes = {
            name: (None if tensor is None else tuple(tensor.shape))
            for name, tensor in payload.items()
        }
        tensor_dtypes = {
            name: (None if tensor is None else str(tensor.dtype))
            for name, tensor in payload.items()
        }
        tensor_numels = {
            name: (0 if tensor is None else int(tensor.numel()))
            for name, tensor in payload.items()
        }
        tensor_bytes = {
            name: (0 if tensor is None else int(tensor.numel() * tensor.element_size()))
            for name, tensor in payload.items()
        }
        return cls(
            is_empty=False,
            num_tensors=len(payload),
            payload_keys=list(payload.keys()),
            tensor_shapes=tensor_shapes,
            tensor_dtypes=tensor_dtypes,
            tensor_numels=tensor_numels,
            tensor_bytes=tensor_bytes,
            total_tensor_bytes=sum(tensor_bytes.values()),
        )


@dataclass(slots=True)
class StageHandoffPayload:
    """Canonical multimodal payload exchanged across stage boundaries."""

    hidden_states: torch.Tensor | None = None
    visual_pos_masks: torch.Tensor | None = None
    deepstack_feature_pack: dict[int, torch.Tensor | None] = field(default_factory=dict)
    multimodal_meta: dict[str, torch.Tensor | None] = field(default_factory=dict)

    HIDDEN_STATES_KEY = "hidden_states"
    VISUAL_POS_MASKS_KEY = "visual_pos_masks"
    DEEPSTACK_PREFIX = "deepstack_feature_pack__layer_"
    MULTIMODAL_META_PREFIX = "multimodal_meta__"

    @classmethod
    def deepstack_key(cls, layer_idx: int) -> str:
        return f"{cls.DEEPSTACK_PREFIX}{layer_idx}"

    @classmethod
    def multimodal_meta_key(cls, name: str) -> str:
        return f"{cls.MULTIMODAL_META_PREFIX}{name}"

    def to_transport_payload(self) -> dict[str, torch.Tensor | None]:
        payload: dict[str, torch.Tensor | None] = {}

        if self.hidden_states is not None:
            payload[self.HIDDEN_STATES_KEY] = self.hidden_states
        if self.visual_pos_masks is not None:
            payload[self.VISUAL_POS_MASKS_KEY] = self.visual_pos_masks

        for layer_idx in sorted(self.deepstack_feature_pack):
            payload[self.deepstack_key(layer_idx)] = self.deepstack_feature_pack[layer_idx]

        for name in sorted(self.multimodal_meta):
            payload[self.multimodal_meta_key(name)] = self.multimodal_meta[name]

        return payload

    @classmethod
    def from_transport_payload(
        cls,
        payload: dict[str, torch.Tensor | None] | None,
    ) -> "StageHandoffPayload | None":
        if payload is None:
            return None

        hidden_states = payload.get(cls.HIDDEN_STATES_KEY)
        visual_pos_masks = payload.get(cls.VISUAL_POS_MASKS_KEY)
        deepstack_feature_pack: dict[int, torch.Tensor | None] = {}
        multimodal_meta: dict[str, torch.Tensor | None] = {}

        for name, tensor in payload.items():
            if name in (cls.HIDDEN_STATES_KEY, cls.VISUAL_POS_MASKS_KEY):
                continue
            if name.startswith(cls.DEEPSTACK_PREFIX):
                layer_idx = int(name.removeprefix(cls.DEEPSTACK_PREFIX))
                deepstack_feature_pack[layer_idx] = tensor
                continue
            if name.startswith(cls.MULTIMODAL_META_PREFIX):
                meta_name = name.removeprefix(cls.MULTIMODAL_META_PREFIX)
                multimodal_meta[meta_name] = tensor
                continue
            multimodal_meta[name] = tensor

        return cls(
            hidden_states=hidden_states,
            visual_pos_masks=visual_pos_masks,
            deepstack_feature_pack=deepstack_feature_pack,
            multimodal_meta=multimodal_meta,
        )
