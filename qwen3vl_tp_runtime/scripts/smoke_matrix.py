#!/usr/bin/env python3
"""Frozen smoke matrix metadata for runtime baseline checks."""

from __future__ import annotations

from dataclasses import dataclass


TEXT_GENERATE_IDS = [104455, 9909, 9286, 16488]
TEXT_GENERATE_TEXT = "人工智能（Artificial"

FRAME_MM_GENERATE_IDS = [87140, 15946, 3837, 101177]
FRAME_MM_GENERATE_TEXT = "视频中，一名"

FRAME_MM_LONG_GENERATE_IDS = [
    87140,
    15946,
    3837,
    101177,
    105611,
    99194,
    38035,
    113727,
    33108,
    104362,
    38035,
    113233,
    9370,
    104253,
    104224,
    46944,
]
FRAME_MM_LONG_GENERATE_TEXT = "视频中，一名穿着深色衬衫和浅色裤子的男子站在一个"

FULL_VIDEO_GENERATE_IDS = [87140, 108869, 100369, 102122]
FULL_VIDEO_GENERATE_TEXT = "视频展示了两个场景"


@dataclass(frozen=True)
class SmokeCase:
    case_id: str
    description: str
    expected_ids: list[int]
    expected_text: str
    backend: str | None = None
    expected_rank_count: int | None = None
    modality: str = "text"
    expected_video_source: str | None = None
    require_transport_metrics: bool = False
    require_consume_only: bool = False
    require_tp_sharded: bool = False
    optional: bool = False


STEP22_SMOKE_MATRIX: tuple[SmokeCase, ...] = (
    SmokeCase(
        case_id="hf-text-generate",
        description="HF text generate reference",
        expected_ids=TEXT_GENERATE_IDS,
        expected_text=TEXT_GENERATE_TEXT,
        backend="hf",
        expected_rank_count=1,
    ),
    SmokeCase(
        case_id="hf-mm-generate",
        description="HF frame-dir multimodal reference",
        expected_ids=TEXT_GENERATE_IDS,
        expected_text=TEXT_GENERATE_TEXT,
        backend="hf",
        expected_rank_count=1,
        modality="multimodal",
        expected_video_source="frame_paths",
    ),
    SmokeCase(
        case_id="pp-mm-generate",
        description="PP frame-dir multimodal generate",
        expected_ids=FRAME_MM_GENERATE_IDS,
        expected_text=FRAME_MM_GENERATE_TEXT,
        backend="pp",
        expected_rank_count=2,
        modality="multimodal",
        expected_video_source="frame_paths",
        require_consume_only=True,
    ),
    SmokeCase(
        case_id="tp-mm-generate",
        description="TP frame-dir multimodal generate",
        expected_ids=FRAME_MM_GENERATE_IDS,
        expected_text=FRAME_MM_GENERATE_TEXT,
        backend="tp",
        expected_rank_count=2,
        modality="multimodal",
        expected_video_source="frame_paths",
        require_consume_only=True,
        require_tp_sharded=True,
    ),
    SmokeCase(
        case_id="hybrid-mm-generate",
        description="HYBRID frame-dir multimodal generate",
        expected_ids=FRAME_MM_GENERATE_IDS,
        expected_text=FRAME_MM_GENERATE_TEXT,
        backend="hybrid",
        expected_rank_count=3,
        modality="multimodal",
        expected_video_source="frame_paths",
        require_consume_only=True,
    ),
    SmokeCase(
        case_id="tp-mm-generate-long",
        description="TP frame-dir multimodal long decode guard",
        expected_ids=FRAME_MM_LONG_GENERATE_IDS,
        expected_text=FRAME_MM_LONG_GENERATE_TEXT,
        backend="tp",
        expected_rank_count=2,
        modality="multimodal",
        expected_video_source="frame_paths",
        require_consume_only=True,
        require_tp_sharded=True,
    ),
    SmokeCase(
        case_id="tp-mm-generate-frame-regression",
        description="TP frame-dir regression after full-video support",
        expected_ids=FRAME_MM_GENERATE_IDS,
        expected_text=FRAME_MM_GENERATE_TEXT,
        backend="tp",
        expected_rank_count=2,
        modality="multimodal",
        expected_video_source="frame_paths",
        require_transport_metrics=True,
        require_consume_only=True,
        require_tp_sharded=True,
    ),
    SmokeCase(
        case_id="hf-mm-generate-video-builder-prompt",
        description="HF full-video reference with direct-builder prompt",
        expected_ids=FULL_VIDEO_GENERATE_IDS,
        expected_text=FULL_VIDEO_GENERATE_TEXT,
        backend="hf",
        expected_rank_count=1,
        modality="multimodal",
        expected_video_source="video_path",
        require_transport_metrics=True,
        optional=True,
    ),
    SmokeCase(
        case_id="pp-mm-generate-video",
        description="PP full-video generate",
        expected_ids=FULL_VIDEO_GENERATE_IDS,
        expected_text=FULL_VIDEO_GENERATE_TEXT,
        backend="pp",
        expected_rank_count=2,
        modality="multimodal",
        expected_video_source="video_path",
        require_transport_metrics=True,
        require_consume_only=True,
        optional=True,
    ),
    SmokeCase(
        case_id="tp-mm-generate-video",
        description="TP full-video generate",
        expected_ids=FULL_VIDEO_GENERATE_IDS,
        expected_text=FULL_VIDEO_GENERATE_TEXT,
        backend="tp",
        expected_rank_count=2,
        modality="multimodal",
        expected_video_source="video_path",
        require_transport_metrics=True,
        require_consume_only=True,
        require_tp_sharded=True,
        optional=True,
    ),
    SmokeCase(
        case_id="hybrid-mm-generate-video-pp2tp1",
        description="HYBRID 2-node full-video generate",
        expected_ids=FULL_VIDEO_GENERATE_IDS,
        expected_text=FULL_VIDEO_GENERATE_TEXT,
        backend="hybrid",
        expected_rank_count=2,
        modality="multimodal",
        expected_video_source="video_path",
        require_transport_metrics=True,
        require_consume_only=True,
        optional=True,
    ),
)


def iter_smoke_cases(*, include_optional: bool = False) -> list[SmokeCase]:
    return [case for case in STEP22_SMOKE_MATRIX if include_optional or not case.optional]


def get_smoke_case(case_id: str) -> SmokeCase | None:
    by_id = {case.case_id: case for case in STEP22_SMOKE_MATRIX}
    if case_id in by_id:
        return by_id[case_id]

    if case_id.startswith("tp-mm-generate-long"):
        return by_id["tp-mm-generate-long"]
    if case_id.startswith("tp-mm-generate-frame-regression"):
        return by_id["tp-mm-generate-frame-regression"]

    if "video" in case_id:
        if case_id.startswith("hf-mm-generate"):
            return by_id["hf-mm-generate-video-builder-prompt"]
        if case_id.startswith("pp-mm-generate"):
            return by_id["pp-mm-generate-video"]
        if case_id.startswith("tp-mm-generate"):
            return by_id["tp-mm-generate-video"]
        if case_id.startswith("hybrid-mm-generate"):
            return by_id["hybrid-mm-generate-video-pp2tp1"]

    if case_id.startswith("pp-mm-generate"):
        return by_id["pp-mm-generate"]
    if case_id.startswith("tp-mm-generate"):
        return by_id["tp-mm-generate"]
    if case_id.startswith("hybrid-mm-generate"):
        return by_id["hybrid-mm-generate"]
    if case_id.startswith("hf-mm-generate"):
        return by_id["hf-mm-generate"]
    if case_id.startswith("hf-text-generate"):
        return by_id["hf-text-generate"]
    return None

