"""Structured runtime schemas for manifests, rank context, and stage handoff payloads."""

from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class StageSpec:
    """Metadata describing one pipeline stage bundle on disk."""

    stage_idx: int
    start_idx: int
    end_idx: int
    bundle_path: str
    num_layers: int
    save_dtype: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageSpec":
        return cls(**data)


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


@dataclass(slots=True)
class TextPipelineManifest:
    """Serializable manifest for a multi-stage text pipeline capture."""

    pipeline_type: str
    num_stages: int
    stage_ranges: list[tuple[int, int]]
    bundle_dir: str
    stages: list[StageSpec]
    boundaries: list[BoundaryStats]
    num_frames: int
    save_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_type": self.pipeline_type,
            "num_stages": self.num_stages,
            "stage_ranges": self.stage_ranges,
            "bundle_dir": self.bundle_dir,
            "stages": [asdict(stage) for stage in self.stages],
            "boundaries": [asdict(boundary) for boundary in self.boundaries],
            "num_frames": self.num_frames,
            "save_dtype": self.save_dtype,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextPipelineManifest":
        return cls(
            pipeline_type=data["pipeline_type"],
            num_stages=data["num_stages"],
            stage_ranges=[tuple(item) for item in data["stage_ranges"]],
            bundle_dir=data["bundle_dir"],
            stages=[StageSpec.from_dict(item) for item in data["stages"]],
            boundaries=[BoundaryStats.from_dict(item) for item in data["boundaries"]],
            num_frames=data["num_frames"],
            save_dtype=data["save_dtype"],
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


@dataclass(slots=True)
class TextHybridManifest:
    """Serializable manifest for a hybrid PP+TP text runtime."""

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
    bundle_dir: str
    stages: list[StageSpec]
    boundaries: list[BoundaryStats]
    num_frames: int
    save_dtype: str
    pipeline_type: str = "text"

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "bundle_dir": self.bundle_dir,
            "stages": [asdict(stage) for stage in self.stages],
            "boundaries": [asdict(boundary) for boundary in self.boundaries],
            "num_frames": self.num_frames,
            "save_dtype": self.save_dtype,
        }

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
            bundle_dir=manifest.bundle_dir,
            stages=manifest.stages,
            boundaries=manifest.boundaries,
            num_frames=manifest.num_frames,
            save_dtype=manifest.save_dtype,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextHybridManifest":
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
            bundle_dir=data["bundle_dir"],
            stages=[StageSpec.from_dict(item) for item in data["stages"]],
            boundaries=[BoundaryStats.from_dict(item) for item in data["boundaries"]],
            num_frames=data["num_frames"],
            save_dtype=data["save_dtype"],
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

    @classmethod
    def empty(cls) -> "PayloadSummary":
        return cls(
            is_empty=True,
            num_tensors=0,
            payload_keys=[],
            tensor_shapes={},
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
        return cls(
            is_empty=False,
            num_tensors=len(payload),
            payload_keys=list(payload.keys()),
            tensor_shapes=tensor_shapes,
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
