"""Input builders for text-only and multimodal Qwen3-VL experiments."""

from __future__ import annotations

import glob
import os

from qwen_vl_utils import process_vision_info

from qwen3vl_tp_runtime.hexgen_core.config import FRAME_DIR


def list_frames(num_frames: int, frame_dir: str = FRAME_DIR) -> list[str]:
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frame_dir}")
    return frame_paths[:num_frames]


def build_inputs(processor, frame_paths: list[str]):
    # 这里保持和现有实验脚本一致，默认使用视频帧 + 中文描述提示。
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [f"file://{p}" for p in frame_paths],
                    "sample_fps": 1,
                },
                {
                    "type": "text",
                    "text": "请用中文简要描述这个视频的主要内容。",
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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
    "list_frames",
    "build_inputs",
    "build_text_inputs",
]
