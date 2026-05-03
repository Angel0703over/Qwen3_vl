"""Video window metadata index for observing local KV ownership."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

import torch


VIDEO_WINDOW_CACHE_SCHEMA = "video_window_cache_index_v1"
VIDEO_TOKEN_TYPE_ID = 2


@dataclass(frozen=True)
class VideoWindowId:
    batch_index: int
    window_index: int


@dataclass(frozen=True)
class KVLocation:
    owner_rank: int | None
    stage_idx: int
    layer_start: int
    layer_end: int
    tp_rank: int | None
    tp_degree: int
    token_start: int
    token_end: int
    kv_offset_start: int
    kv_offset_end: int
    cache_max_seq_len: int | None = None


@dataclass(frozen=True)
class VideoWindowMetadata:
    window_id: VideoWindowId
    token_start: int
    token_end: int
    frame_start: int | None
    frame_end: int | None
    time_start_s: float | None
    time_end_s: float | None
    grid_thw: tuple[int, int, int] | None
    kv_location: KVLocation

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start

    @property
    def frame_count(self) -> int | None:
        if self.frame_start is None or self.frame_end is None:
            return None
        return self.frame_end - self.frame_start

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_count"] = self.token_count
        payload["frame_count"] = self.frame_count
        if self.grid_thw is not None:
            payload["grid_thw"] = list(self.grid_thw)
        return payload


class VideoWindowCacheIndex:
    """Local metadata copy mapping video windows to this rank's KV offsets."""

    def __init__(self, windows: list[VideoWindowMetadata]) -> None:
        self.windows = list(windows)

    @property
    def window_count(self) -> int:
        return len(self.windows)

    @property
    def total_video_tokens(self) -> int:
        return sum(window.token_count for window in self.windows)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": VIDEO_WINDOW_CACHE_SCHEMA,
            "observation_only": True,
            "compression_enabled": False,
            "eviction_enabled": False,
            "remote_fetch_enabled": False,
            "window_count": self.window_count,
            "total_video_tokens": self.total_video_tokens,
            "windows": [window.to_dict() for window in self.windows],
        }
        payload["metadata_bytes"] = _json_size_bytes(payload)
        return payload


def attach_video_window_cache_index(
    stage_state: dict[str, Any],
    *,
    owner_rank: int | None,
    stage_idx: int,
    layer_start: int,
    layer_end: int,
    tp_rank: int | None,
    tp_degree: int,
    cache_max_seq_len: int | None,
    sample_fps: float | None = None,
) -> dict[str, Any] | None:
    """Attach a local video-window KV index to a runtime-only multimodal StageState."""

    if not stage_state.get("runtime_only_generate"):
        return None
    if stage_state.get("modality") != "multimodal":
        return None

    index = build_video_window_cache_index(
        mm_token_type_ids=stage_state.get("mm_token_type_ids"),
        video_grid_thw=stage_state.get("video_grid_thw"),
        num_frames=stage_state.get("num_frames"),
        owner_rank=owner_rank,
        stage_idx=stage_idx,
        layer_start=layer_start,
        layer_end=layer_end,
        tp_rank=tp_rank,
        tp_degree=tp_degree,
        cache_max_seq_len=cache_max_seq_len,
        sample_fps=sample_fps,
    )
    if index is None:
        stage_state.pop("video_window_cache", None)
        return None
    summary = index.to_dict()
    stage_state["video_window_cache"] = summary
    return summary


def build_video_window_cache_index(
    *,
    mm_token_type_ids: Any,
    video_grid_thw: Any,
    num_frames: Any,
    owner_rank: int | None,
    stage_idx: int,
    layer_start: int,
    layer_end: int,
    tp_rank: int | None,
    tp_degree: int,
    cache_max_seq_len: int | None,
    sample_fps: float | None = None,
) -> VideoWindowCacheIndex | None:
    token_rows = _token_type_rows(mm_token_type_ids)
    if not token_rows:
        return None

    grid_rows = _grid_rows(video_grid_thw)
    total_frames = _safe_int(num_frames)
    fps = float(sample_fps) if sample_fps else 1.0
    windows: list[VideoWindowMetadata] = []

    for batch_index, token_types in enumerate(token_rows):
        ranges = _contiguous_ranges(token_types, VIDEO_TOKEN_TYPE_ID)
        for window_index, (token_start, token_end) in enumerate(ranges):
            frame_start, frame_end = _infer_frame_range(
                window_index=window_index,
                window_count=len(ranges),
                total_frames=total_frames,
                grid_rows=grid_rows,
            )
            time_start_s, time_end_s = _infer_time_range(frame_start, frame_end, fps)
            grid_thw = _window_grid_thw(
                window_index=window_index,
                window_count=len(ranges),
                token_count=token_end - token_start,
                grid_rows=grid_rows,
            )
            location = KVLocation(
                owner_rank=None if owner_rank is None else int(owner_rank),
                stage_idx=int(stage_idx),
                layer_start=int(layer_start),
                layer_end=int(layer_end),
                tp_rank=None if tp_rank is None else int(tp_rank),
                tp_degree=int(tp_degree),
                token_start=int(token_start),
                token_end=int(token_end),
                kv_offset_start=int(token_start),
                kv_offset_end=int(token_end),
                cache_max_seq_len=None if cache_max_seq_len is None else int(cache_max_seq_len),
            )
            windows.append(
                VideoWindowMetadata(
                    window_id=VideoWindowId(
                        batch_index=int(batch_index),
                        window_index=int(window_index),
                    ),
                    token_start=int(token_start),
                    token_end=int(token_end),
                    frame_start=frame_start,
                    frame_end=frame_end,
                    time_start_s=time_start_s,
                    time_end_s=time_end_s,
                    grid_thw=grid_thw,
                    kv_location=location,
                )
            )

    if not windows:
        return None
    return VideoWindowCacheIndex(windows)


def _token_type_rows(mm_token_type_ids: Any) -> list[list[int]]:
    if not torch.is_tensor(mm_token_type_ids):
        return []
    token_types = mm_token_type_ids.detach().to(device="cpu", dtype=torch.long)
    if token_types.ndim == 1:
        return [[int(value) for value in token_types.tolist()]]
    if token_types.ndim == 2:
        return [[int(value) for value in row.tolist()] for row in token_types]
    return []


def _grid_rows(video_grid_thw: Any) -> list[tuple[int, int, int]]:
    if not torch.is_tensor(video_grid_thw):
        return []
    grid = video_grid_thw.detach().to(device="cpu", dtype=torch.long)
    if grid.ndim == 1 and grid.numel() == 3:
        grid = grid.view(1, 3)
    if grid.ndim != 2 or grid.shape[-1] != 3:
        return []
    return [tuple(int(value) for value in row.tolist()) for row in grid]


def _contiguous_ranges(values: list[int], target: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value == target and start is None:
            start = index
        elif value != target and start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, len(values)))
    return ranges


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_frame_range(
    *,
    window_index: int,
    window_count: int,
    total_frames: int | None,
    grid_rows: list[tuple[int, int, int]],
) -> tuple[int | None, int | None]:
    if window_count <= 0:
        return None, None
    if total_frames is None:
        if len(grid_rows) == 1:
            total_frames = int(grid_rows[0][0])
        elif grid_rows:
            total_frames = sum(int(row[0]) for row in grid_rows)
    if total_frames is None or total_frames <= 0:
        return None, None

    frame_start = (int(window_index) * total_frames) // int(window_count)
    frame_end = ((int(window_index) + 1) * total_frames) // int(window_count)
    if frame_end <= frame_start:
        frame_end = min(total_frames, frame_start + 1)
    return frame_start, frame_end


def _infer_time_range(
    frame_start: int | None,
    frame_end: int | None,
    sample_fps: float,
) -> tuple[float | None, float | None]:
    if frame_start is None or frame_end is None or sample_fps <= 0:
        return None, None
    return frame_start / sample_fps, frame_end / sample_fps


def _window_grid_thw(
    *,
    window_index: int,
    window_count: int,
    token_count: int,
    grid_rows: list[tuple[int, int, int]],
) -> tuple[int, int, int] | None:
    if not grid_rows:
        return None
    if len(grid_rows) == window_count:
        return grid_rows[window_index]
    if len(grid_rows) == 1:
        total_t, h, w = grid_rows[0]
        if window_count == int(total_t) and window_count > 0:
            return (1, int(h), int(w))
        return (int(total_t), int(h), int(w))
    if 0 <= window_index < len(grid_rows):
        return grid_rows[window_index]
    return None


def _json_size_bytes(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))


__all__ = [
    "KVLocation",
    "VIDEO_WINDOW_CACHE_SCHEMA",
    "VideoWindowCacheIndex",
    "VideoWindowId",
    "VideoWindowMetadata",
    "attach_video_window_cache_index",
    "build_video_window_cache_index",
]
