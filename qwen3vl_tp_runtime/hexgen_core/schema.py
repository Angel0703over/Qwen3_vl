"""Structured runtime schemas for manifests, rank context, and stage handoff payloads."""

from dataclasses import asdict, dataclass, field
from typing import Any, TypeAlias

import torch

StageState: TypeAlias = dict[str, Any]


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
