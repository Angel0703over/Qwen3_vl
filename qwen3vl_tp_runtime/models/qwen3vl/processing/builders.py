"""Input builders for text-only and multimodal Qwen3-VL experiments."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from qwen_vl_utils import process_vision_info

from qwen3vl_tp_runtime.hexgen_core.config import FRAME_DIR


VIDEO_INPUT_METADATA_SCHEMA = "qwen3vl_video_input_v1"


@dataclass(slots=True)
class VideoInputSpec:
    """User-facing video input, close to vLLM media IO naming but Qwen-backed."""

    frame_paths: list[str] | None = None
    video_path: str | None = None
    video_url: str | None = None
    sample_fps: int | float | None = 1
    video_fps: float | None = None
    video_nframes: int | None = None
    video_start: float | None = None
    video_end: float | None = None
    video_min_frames: int | None = None
    video_max_frames: int | None = None

    @property
    def source_kind(self) -> str:
        if self.frame_paths is not None:
            return "frame_paths"
        if self.video_path is not None:
            return "video_path"
        if self.video_url is not None:
            return "video_url"
        return "missing"

    def validate(self) -> None:
        source_count = sum(
            1
            for value in (self.frame_paths, self.video_path, self.video_url)
            if value is not None
        )
        if source_count != 1:
            raise ValueError("video input 必须且只能指定 frame_paths / video_path / video_url 之一。")
        if self.frame_paths is not None and not self.frame_paths:
            raise ValueError("frame_paths 不能为空。")
        if self.video_fps is not None and self.video_nframes is not None:
            raise ValueError("video_fps 和 video_nframes 不能同时使用；Qwen3-VL 采样规则要求二选一。")
        if self.video_nframes is not None and int(self.video_nframes) <= 0:
            raise ValueError(f"video_nframes 必须大于 0，当前拿到 {self.video_nframes}。")
        if self.video_min_frames is not None and int(self.video_min_frames) <= 0:
            raise ValueError(f"video_min_frames 必须大于 0，当前拿到 {self.video_min_frames}。")
        if self.video_max_frames is not None and int(self.video_max_frames) <= 0:
            raise ValueError(f"video_max_frames 必须大于 0，当前拿到 {self.video_max_frames}。")
        if (
            self.video_min_frames is not None
            and self.video_max_frames is not None
            and int(self.video_min_frames) > int(self.video_max_frames)
        ):
            raise ValueError("video_min_frames 不能大于 video_max_frames。")


def list_frames(num_frames: int, frame_dir: str = FRAME_DIR) -> list[str]:
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frame_dir}")
    return frame_paths[:num_frames]


def _file_uri(path: str) -> str:
    return path if "://" in path else f"file://{path}"


def _set_if_present(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


def build_video_messages(
    spec: VideoInputSpec,
    *,
    prompt: str = "请用中文简要描述这个视频的主要内容。",
):
    spec.validate()
    if spec.frame_paths is not None:
        video_payload: dict[str, Any] = {
            "type": "video",
            "video": [_file_uri(path) for path in spec.frame_paths],
        }
        _set_if_present(video_payload, "sample_fps", spec.sample_fps)
    else:
        video_source = spec.video_path if spec.video_path is not None else spec.video_url
        video_payload = {
            "type": "video",
            "video": video_source,
        }
        _set_if_present(video_payload, "fps", spec.video_fps)
        _set_if_present(video_payload, "nframes", spec.video_nframes)
        _set_if_present(video_payload, "video_start", spec.video_start)
        _set_if_present(video_payload, "video_end", spec.video_end)
        _set_if_present(video_payload, "min_frames", spec.video_min_frames)
        _set_if_present(video_payload, "max_frames", spec.video_max_frames)

    return [
        {
            "role": "user",
            "content": [
                video_payload,
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "shape"):
        try:
            return {
                "type": type(value).__name__,
                "shape": [int(dim) for dim in value.shape],
            }
        except Exception:
            pass
    return repr(value)


def _metadata_frame_count(metadata: Any) -> int | None:
    if not isinstance(metadata, dict):
        return None
    frames_indices = metadata.get("frames_indices")
    if frames_indices is not None:
        try:
            return len(frames_indices)
        except TypeError:
            pass
    nframes = metadata.get("nframes")
    if nframes is not None:
        try:
            return int(nframes)
        except (TypeError, ValueError):
            return None
    return None


def _probe_video_duration_s(video_path: str, *, video_start: float | None = None, video_end: float | None = None) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        duration = float(result.stdout.strip())
    except Exception:
        return None
    if video_start is not None:
        duration = max(duration - float(video_start), 0.0)
    if video_end is not None:
        end_duration = float(video_end) - (0.0 if video_start is None else float(video_start))
        duration = min(duration, max(end_duration, 0.0))
    return duration if duration > 0 else None


def _extract_video_frames_ffmpeg(spec: VideoInputSpec, output_dir: str) -> list[str]:
    if spec.video_path is None:
        raise RuntimeError("ffmpeg fallback 只支持本地 video_path。")
    if not Path(spec.video_path).exists():
        raise FileNotFoundError(spec.video_path)

    frame_budget = spec.video_nframes
    if frame_budget is None and spec.video_fps is None:
        frame_budget = spec.video_min_frames or 4
    if frame_budget is not None and spec.video_max_frames is not None:
        frame_budget = min(int(frame_budget), int(spec.video_max_frames))
    if frame_budget is not None and spec.video_min_frames is not None:
        frame_budget = max(int(frame_budget), int(spec.video_min_frames))

    resolved_fps = spec.video_fps
    if resolved_fps is None and frame_budget is not None:
        duration = _probe_video_duration_s(
            spec.video_path,
            video_start=spec.video_start,
            video_end=spec.video_end,
        )
        if duration is not None:
            resolved_fps = max(float(frame_budget) / duration, 0.001)

    output_pattern = os.path.join(output_dir, "frame_%06d.jpg")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if spec.video_start is not None:
        cmd.extend(["-ss", str(spec.video_start)])
    cmd.extend(["-i", spec.video_path])
    if spec.video_end is not None:
        clip_start = 0.0 if spec.video_start is None else float(spec.video_start)
        clip_duration = max(float(spec.video_end) - clip_start, 0.001)
        cmd.extend(["-t", str(clip_duration)])
    if resolved_fps is not None:
        cmd.extend(["-vf", f"fps={resolved_fps}"])
    if frame_budget is not None:
        cmd.extend(["-frames:v", str(int(frame_budget))])
    cmd.append(output_pattern)
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    frame_paths = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))
    if not frame_paths:
        raise RuntimeError(f"ffmpeg fallback 没有从视频中抽出帧: {spec.video_path}")
    return frame_paths


def _extract_video_frames_pyav(spec: VideoInputSpec, output_dir: str) -> list[str]:
    try:
        import av
        import numpy as np
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("PyAV/Pillow fallback 不可用，无法从完整视频抽帧。") from exc

    if spec.video_path is None:
        raise RuntimeError("PyAV fallback 只支持本地 video_path。")
    if not Path(spec.video_path).exists():
        raise FileNotFoundError(spec.video_path)

    container = av.open(spec.video_path)
    stream = container.streams.video[0]
    total_frames = int(stream.frames or 0)
    fps = float(stream.average_rate) if stream.average_rate else None
    if total_frames <= 0:
        total_frames = sum(1 for _frame in container.decode(stream))
        container.close()
        container = av.open(spec.video_path)
        stream = container.streams.video[0]
    if total_frames <= 0:
        raise RuntimeError(f"PyAV 没有从视频中读到帧: {spec.video_path}")

    start_frame = 0
    end_frame = total_frames - 1
    if fps is not None and spec.video_start is not None:
        start_frame = max(0, min(end_frame, int(float(spec.video_start) * fps)))
    if fps is not None and spec.video_end is not None:
        end_frame = max(start_frame, min(end_frame, int(float(spec.video_end) * fps)))

    frame_count = spec.video_nframes
    if frame_count is None and spec.video_fps is not None and fps is not None:
        duration = max((end_frame - start_frame + 1) / fps, 0.0)
        frame_count = max(1, int(round(duration * float(spec.video_fps))))
    if frame_count is None:
        frame_count = spec.video_min_frames or 4
    if spec.video_max_frames is not None:
        frame_count = min(int(frame_count), int(spec.video_max_frames))
    if spec.video_min_frames is not None:
        frame_count = max(int(frame_count), int(spec.video_min_frames))
    frame_count = max(1, min(int(frame_count), end_frame - start_frame + 1))

    selected_indices = set(
        int(value)
        for value in np.linspace(start_frame, end_frame, frame_count).round().astype(int).tolist()
    )
    frame_paths: list[str] = []
    for frame_idx, frame in enumerate(container.decode(stream)):
        if frame_idx not in selected_indices:
            continue
        rgb = _pyav_frame_to_rgb_array(frame)
        output_path = os.path.join(output_dir, f"frame_{len(frame_paths):06d}.jpg")
        Image.fromarray(rgb).save(output_path, quality=95)
        frame_paths.append(output_path)
        if len(frame_paths) >= len(selected_indices):
            break
    container.close()
    if not frame_paths:
        raise RuntimeError(f"PyAV fallback 没有从视频中抽出帧: {spec.video_path}")
    return frame_paths


def _pyav_frame_to_rgb_array(frame):
    import numpy as np

    fmt = frame.format.name
    if fmt in {"yuv420p", "yuvj420p"}:
        y_plane, u_plane, v_plane = frame.planes
        y = _plane_to_array(y_plane, frame.height, frame.width).astype(np.float32)
        u = _plane_to_array(u_plane, (frame.height + 1) // 2, (frame.width + 1) // 2).astype(np.float32)
        v = _plane_to_array(v_plane, (frame.height + 1) // 2, (frame.width + 1) // 2).astype(np.float32)
        u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)[: frame.height, : frame.width]
        v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)[: frame.height, : frame.width]
        if fmt == "yuvj420p":
            c = y
            d = u - 128.0
            e = v - 128.0
            r = c + 1.402 * e
            g = c - 0.344136 * d - 0.714136 * e
            b = c + 1.772 * d
        else:
            c = y - 16.0
            d = u - 128.0
            e = v - 128.0
            r = 1.164383 * c + 1.596027 * e
            g = 1.164383 * c - 0.391762 * d - 0.812968 * e
            b = 1.164383 * c + 2.017232 * d
        return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)
    return frame.to_ndarray(format="rgb24")


def _plane_to_array(plane, height: int, width: int):
    import numpy as np

    raw = np.frombuffer(plane, dtype=np.uint8)
    rows = raw.reshape(plane.height, plane.line_size)
    return rows[:height, :width]


def _process_video_messages(
    processor,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> tuple[dict[str, Any], list[Any] | None, dict[str, Any]]:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs.pop("token_type_ids", None)
    return inputs, video_metadatas, video_kwargs


def summarize_video_input(
    spec: VideoInputSpec,
    *,
    video_metadatas: list[Any] | None,
    video_kwargs: dict[str, Any] | None,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec.validate()
    sanitized_metadatas = [_json_safe(metadata) for metadata in (video_metadatas or [])]
    first_metadata = video_metadatas[0] if video_metadatas else None
    sampled_frame_count = _metadata_frame_count(first_metadata)
    if sampled_frame_count is None and spec.frame_paths is not None:
        sampled_frame_count = len(spec.frame_paths)

    summary: dict[str, Any] = {
        "schema": VIDEO_INPUT_METADATA_SCHEMA,
        "source": spec.source_kind,
        "frame_count": sampled_frame_count,
        "video_kwargs": _json_safe(video_kwargs or {}),
        "video_metadata": sanitized_metadatas,
    }
    if spec.frame_paths is not None:
        summary["frame_count"] = len(spec.frame_paths)
        summary["frame_path_count"] = len(spec.frame_paths)
        summary["sample_fps"] = spec.sample_fps
    if spec.video_path is not None:
        summary["video_path_basename"] = os.path.basename(spec.video_path)
    if spec.video_url is not None:
        summary["video_url"] = spec.video_url
    sampling = {
        "fps": spec.video_fps,
        "nframes": spec.video_nframes,
        "video_start": spec.video_start,
        "video_end": spec.video_end,
        "min_frames": spec.video_min_frames,
        "max_frames": spec.video_max_frames,
    }
    summary["sampling"] = {key: value for key, value in sampling.items() if value is not None}
    if inputs is not None and inputs.get("video_grid_thw") is not None:
        summary["video_grid_thw"] = _json_safe(inputs.get("video_grid_thw"))
    return summary


def build_inputs_with_metadata(
    processor,
    frame_paths: list[str] | None = None,
    *,
    video_path: str | None = None,
    video_url: str | None = None,
    prompt: str = "请用中文简要描述这个视频的主要内容。",
    sample_fps: int = 1,
    video_fps: float | None = None,
    video_nframes: int | None = None,
    video_start: float | None = None,
    video_end: float | None = None,
    video_min_frames: int | None = None,
    video_max_frames: int | None = None,
    add_generation_prompt: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = VideoInputSpec(
        frame_paths=frame_paths,
        video_path=video_path,
        video_url=video_url,
        sample_fps=sample_fps,
        video_fps=video_fps,
        video_nframes=video_nframes,
        video_start=video_start,
        video_end=video_end,
        video_min_frames=video_min_frames,
        video_max_frames=video_max_frames,
    )
    messages = build_video_messages(spec, prompt=prompt)
    reader = "qwen_vl_utils"

    if spec.video_path is not None:
        with tempfile.TemporaryDirectory(prefix="qwen3vl_video_frames_") as tmp_dir:
            if shutil.which("ffmpeg") is not None:
                fallback_frame_paths = _extract_video_frames_ffmpeg(spec, tmp_dir)
                reader = "ffmpeg_frame_adapter"
            else:
                fallback_frame_paths = _extract_video_frames_pyav(spec, tmp_dir)
                reader = "pyav_frame_adapter"
            fallback_sample_fps = spec.video_fps if spec.video_fps is not None else spec.sample_fps
            fallback_messages = build_video_messages(
                VideoInputSpec(
                    frame_paths=fallback_frame_paths,
                    sample_fps=fallback_sample_fps,
                ),
                prompt=prompt,
            )
            inputs, video_metadatas, video_kwargs = _process_video_messages(
                processor,
                fallback_messages,
                add_generation_prompt=add_generation_prompt,
            )
    else:
        inputs, video_metadatas, video_kwargs = _process_video_messages(
            processor,
            messages,
            add_generation_prompt=add_generation_prompt,
        )

    video_input_metadata = summarize_video_input(
        spec,
        video_metadatas=video_metadatas,
        video_kwargs=video_kwargs,
        inputs=inputs,
    )
    video_input_metadata["reader"] = reader
    return inputs, video_input_metadata


def build_inputs(
    processor,
    frame_paths: list[str] | None = None,
    *,
    video_path: str | None = None,
    video_url: str | None = None,
    prompt: str = "请用中文简要描述这个视频的主要内容。",
    sample_fps: int = 1,
    video_fps: float | None = None,
    video_nframes: int | None = None,
    video_start: float | None = None,
    video_end: float | None = None,
    video_min_frames: int | None = None,
    video_max_frames: int | None = None,
    add_generation_prompt: bool = True,
):
    inputs, _metadata = build_inputs_with_metadata(
        processor,
        frame_paths=frame_paths,
        video_path=video_path,
        video_url=video_url,
        prompt=prompt,
        sample_fps=sample_fps,
        video_fps=video_fps,
        video_nframes=video_nframes,
        video_start=video_start,
        video_end=video_end,
        video_min_frames=video_min_frames,
        video_max_frames=video_max_frames,
        add_generation_prompt=add_generation_prompt,
    )
    return inputs


def build_text_inputs(
    processor,
    prompt: str,
    *,
    add_generation_prompt: bool = True,
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    inputs = processor(
        text=text,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return inputs


__all__ = [
    "VIDEO_INPUT_METADATA_SCHEMA",
    "VideoInputSpec",
    "build_video_messages",
    "list_frames",
    "build_inputs",
    "build_inputs_with_metadata",
    "summarize_video_input",
    "build_text_inputs",
]
