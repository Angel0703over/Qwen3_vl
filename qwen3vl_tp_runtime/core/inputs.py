import glob
import os

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

from qwen3vl_tp_runtime.core.config import FRAME_DIR, MODEL_PATH


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


def load_model(
    model_path: str = MODEL_PATH,
    *,
    attn_implementation: str = "eager",
):
    return Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=attn_implementation,
        local_files_only=True,
    ).eval()


def load_processor(model_path: str = MODEL_PATH):
    return AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )
