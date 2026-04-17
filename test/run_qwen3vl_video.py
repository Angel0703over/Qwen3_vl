import os
import glob
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct"
FRAME_DIR = "/mnt/ssd/code/Qwen3_vl/frames"

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

frame_paths = sorted(glob.glob(os.path.join(FRAME_DIR, "*.jpg")))
assert frame_paths, f"No frames found in {FRAME_DIR}"

# 先只取前 8 帧，降低 Jetson 压力
frame_paths = frame_paths[:8]

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

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True,
).eval()

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

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

inputs = inputs.to(model.device)

print("Generating...")
with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

generated_ids = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
]

output_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

print("\n===== OUTPUT =====")
print(output_text[0])