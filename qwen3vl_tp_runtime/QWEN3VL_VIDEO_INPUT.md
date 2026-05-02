# Qwen3-VL Video Input Flow

这份文档只说明 Qwen3-VL 如何把完整视频变成模型可消费的多模态输入，以及当前 runtime 应该怎么接入。

## 结论

当前项目已经支持两条路径：

```text
frame_dir/*.jpg
  -> qwen_vl_utils.process_vision_info
  -> processor(...)
  -> pixel_values_videos / video_grid_thw / mm_token_type_ids
  -> frontend_state

video_path
  -> ffmpeg_frame_adapter 或 pyav_frame_adapter 抽临时帧
  -> qwen_vl_utils.process_vision_info
  -> processor(...)
  -> pixel_values_videos / video_grid_thw / mm_token_type_ids
  -> frontend_state
```

真实 Jetson baseline 见 `baseline_runs/20260502-step21-video-input/`：`test/demo.mp4 --video-nframes 4` 已通过 HF/PP/TP/HYBRID smoke，frame-dir 旧路径回归不变。

如果要支持真实完整视频，Qwen3-VL 的推荐入口不是先手写抽帧脚本，而是把 message 里的 `video` 改成视频路径或 URL：

```python
{
    "type": "video",
    "video": "/path/to/video.mp4",
    "fps": 2.0,
    "min_frames": 4,
    "max_frames": 768,
}
```

然后仍然调用 `qwen_vl_utils.process_vision_info(...)` 和 Qwen3-VL `processor(...)`。

## 上游流程

### 1. video path / url 到采样帧

`qwen_vl_utils.fetch_video()` 支持：

- 本地路径。
- `file://...`
- `http://...` / `https://...`
- 已经抽好的帧列表。

当 `video` 是字符串路径或 URL 时，它会选择视频 reader：

```text
torchcodec -> decord -> torchvision
```

也可以用环境变量指定：

```bash
FORCE_QWENVL_VIDEO_READER=torchvision
FORCE_QWENVL_VIDEO_READER=decord
FORCE_QWENVL_VIDEO_READER=torchcodec
```

当前本地环境里：

- `qwen_vl_utils` 已安装。
- `torchvision` 已安装。
- `av` 已安装。
- `decord` / `torchcodec` 未安装。

所以当前会优先落到 `torchvision` reader。

### 2. 采样规则

`qwen_vl_utils.smart_nframes()` 负责决定抽多少帧。

支持两种互斥配置：

- `nframes`：直接指定抽帧数。
- `fps`：按目标 fps 抽帧。

默认规则：

- `fps = 2.0`
- `min_frames = 4`
- `max_frames = 768`
- 帧数会对齐到 `FRAME_FACTOR = 2`
- 支持 `video_start` / `video_end` 截取时间段

抽帧索引用 `linspace(start_frame, end_frame, nframes)` 均匀采样。

返回的 metadata 包括：

- `fps`
- `frames_indices`
- `total_num_frames`
- `video_backend`

这些 metadata 对 Qwen3-VL 很重要，因为 Qwen3-VL 会把时间戳写进 prompt。

### 3. 帧 tensor 到 processor 输入

`qwen_vl_utils.process_vision_info(..., return_video_metadata=True, return_video_kwargs=True)` 返回：

```text
images
videos
video_kwargs
```

对视频来说，`videos` 里是：

```text
(video_tensor, video_metadata)
```

当前项目的 [build_inputs](/mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/models/qwen3vl/processing/builders.py:20) 已经这样处理：

```python
images, videos, video_kwargs = process_vision_info(
    messages,
    image_patch_size=16,
    return_video_kwargs=True,
    return_video_metadata=True,
)

inputs = processor(
    text=text,
    images=images,
    videos=videos,
    video_metadata=video_metadatas,
    do_resize=False,
    return_tensors="pt",
    **video_kwargs,
)
```

这里 `video_kwargs` 会包含：

```python
{"do_sample_frames": False}
```

意思是：`qwen_vl_utils` 已经完成视频读取、采样和 resize，processor 不再二次采样。

### 4. processor 做什么

Qwen3-VL `Qwen3VLVideoProcessor` 把视频 tensor 处理成：

- `pixel_values_videos`
- `video_grid_thw`

默认参数：

- `patch_size = 16`
- `temporal_patch_size = 2`
- `merge_size = 2`
- `fps = 2`
- `min_frames = 4`
- `max_frames = 768`

`Qwen3VLProcessor` 还会用 `video_metadata.frames_indices` 和 `video_metadata.fps` 计算时间戳，把一个视频占位符展开成类似：

```text
<0.5 seconds><|vision_start|>...<|vision_end|>
<1.0 seconds><|vision_start|>...<|vision_end|>
```

并返回 `mm_token_type_ids`，其中：

- text = `0`
- image = `1`
- video = `2`

### 5. 模型 frontend 做什么

Qwen3-VL 模型 forward 里：

1. 用 `input_ids` 得到 text `inputs_embeds`。
2. 如果有 `pixel_values_videos`，调用 `get_video_features()`。
3. `get_video_features()` 复用 image feature path，跑视觉 tower。
4. 用 video placeholder mask 把 video features 写回 `inputs_embeds`。
5. 构造：
   - `visual_pos_masks`
   - `deepstack_visual_embeds`
6. 用 `input_ids + mm_token_type_ids + video_grid_thw` 计算 3D position ids / `rope_deltas`。
7. 调用 language model decoder。

在我们项目里，这一步的结果就是 `frontend_state`：

```text
input_ids
inputs_embeds
attention_mask
position_ids
cos / sin
rope_deltas
visual_pos_masks
deepstack_by_layer
image_grid_thw / video_grid_thw
```

## 当前项目怎么接完整视频

当前项目只需要在 input builder 层扩展，不应该改变 PP/TP/HYBRID 主路径。

当前代码已完成第一版接入：

- `models/qwen3vl/processing/builders.py`：`VideoInputSpec`、`build_video_messages`、`build_inputs_with_metadata`。
- `scripts/runtime.py`：支持 `--video-path` / `--video-url` 和 Qwen 采样参数。
- `scripts/helpers/run-*-mm-generate.sh`：支持 `VIDEO_PATH` / `VIDEO_URL` 环境变量。
- 本地 `video_path` 第一版使用 `ffmpeg` media adapter 抽临时帧，再交给 Qwen processor；这是为了绕开当前 Jetson 环境里 `torchvision/av` reader 卡住的问题。
- 分布式路径仍只让 stage0 / input-owner 跑 video decode 和视觉 frontend。

已接入 runtime config / CLI 字段：

- `video_path`
- `video_url`
- `video_fps`
- `video_nframes`
- `video_start`
- `video_end`
- `video_min_frames`
- `video_max_frames`

然后让 [build_inputs](/mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/models/qwen3vl/processing/builders.py:20) 支持两种入口：

```text
frame_paths != None:
  video = ["file://frame0.jpg", "file://frame1.jpg", ...]

video_path != None:
  ffmpeg 临时抽帧 -> video = ["file://frame0.jpg", ...]

video_url != None:
  video = "https://..."
```

其他逻辑保持一致：

```text
messages
  -> processor.apply_chat_template(...)
  -> process_vision_info(... return_video_metadata=True)
  -> processor(... video_metadata=..., do_resize=False, **video_kwargs)
```

## 和 vLLM 的完整视频处理对比

### 一句话对比

vLLM 把完整视频处理抽象成通用 media IO：

```text
video_url / video bytes
  -> VideoLoader.load_bytes(...)
  -> frames_array + metadata
  -> model-specific processor
  -> model runner
```

我们项目当前更贴近 Qwen 官方 `qwen_vl_utils` 路径：

```text
video path / frame list
  -> qwen_vl_utils.process_vision_info(...)
  -> Qwen3VLProcessor
  -> frontend_state
  -> startup contract
  -> PP/TP/HYBRID StageState
```

因此建议是：**media IO 层对齐 vLLM，model processor/frontend 层继续对齐 Qwen3-VL，分布式边界继续保持当前 startup contract 设计。**

### 关键差异

| 维度 | vLLM | 当前项目 / Qwen3-VL 路径 | 建议 |
| --- | --- | --- | --- |
| 在线 API | OpenAI-compatible `video_url` | 当前 helper 只从本地 frame list 构造 message | 增加 `video_path/video_url` CLI/runtime config |
| 离线输入 | `multi_modal_data["video"]` 可直接放 ndarray / tensor list | `qwen_vl_utils.process_vision_info` 返回 video tensor | 保留 frame-list 旧入口 |
| 完整视频解码 | `VideoLoader.load_bytes(data, num_frames, fps, max_duration, frame_recovery)` | `qwen_vl_utils.fetch_video` 选择 `torchcodec/decord/torchvision` | 短期用 `qwen_vl_utils`，后续可封装成 `VideoLoader` |
| 采样规则 | `num_frames` 和 `fps` 都可限制，取更小采样数；dynamic backend 默认 `fps=2` | `nframes` 或 `fps` 二选一，默认 `fps=2.0`，`min_frames=4`，`max_frames=768` | CLI 暴露 `fps/nframes/min/max/start/end` |
| metadata | 返回 HF-compatible metadata：`fps/duration/frames_indices/total_num_frames/video_backend/do_sample_frames` | 返回 `fps/frames_indices/total_num_frames/video_backend`，Qwen3-VL 用它算时间戳 | 必须记录并传给 processor |
| 预抽帧 | 支持 `video/jpeg` base64，并用 `media_io_kwargs` 补 metadata | 当前就是预抽帧列表，metadata 用 fake/raw fps 构造 | 旧路径等价于 vLLM pre-extracted frames |
| 坏帧恢复 | 支持 `frame_recovery` | 当前没有显式坏帧恢复 | 后续可加 opt-in |
| processor | 通用 media IO 后交给模型自己的 processor | 直接走 Qwen3VLProcessor | 不要绕开 Qwen3VLProcessor |
| 分布式 | vLLM engine 内部调度，用户不感知 PP/TP handoff | 我们必须显式保证只有 input-owner 解码/frontend | 保持 non-owner consume-only |

### vLLM 值得照搬的部分

1. 输入参数命名：

```text
video_url
video_path
fps
num_frames
max_duration
frame_recovery
media_io_kwargs
```

2. media IO 返回结构：

```text
frames
metadata = {
    "fps": ...,
    "duration": ...,
    "frames_indices": ...,
    "total_num_frames": ...,
    "video_backend": ...,
    "do_sample_frames": ...,
}
```

3. 对预抽帧保留原视频 metadata。

vLLM 文档明确说明：客户端已经抽帧时，应通过 `media_io_kwargs` 传 `fps`、`frames_indices`、`total_num_frames`、`duration`，否则服务端会丢失原始时间信息。

4. 坏帧恢复做成 opt-in。

vLLM 的 `frame_recovery` 是为损坏或截断视频准备的，不应该默认改变当前 correctness baseline。

### 不建议照搬的部分

1. 不直接照搬 vLLM 的 OpenCV-only backend。

Jetson 当前环境里 `torchvision` 和 `av` 可用，`decord/torchcodec` 不可用。Qwen 官方 `qwen_vl_utils` 已经有 backend fallback，更适合先落地。

2. 不让每个 rank 都执行 media IO。

vLLM 对用户屏蔽了 engine 内部并行；我们当前显式管理 PP/TP/HYBRID，所以必须保持：

```text
input-owner decode/sample/frontend
non-owner consume startup contract
```

3. 不跳过 Qwen3-VL 自己的 processor。

Qwen3-VL 的 processor 会基于 `video_metadata.frames_indices` 和 `fps` 展开带时间戳的视频 token，并生成 `mm_token_type_ids`。这部分和 Qwen3-VL M-RoPE 强相关。

### 对我们项目的落地形态

建议新增一个很薄的 `VideoInputSpec`，语义接近 vLLM media IO，但实现先用 Qwen 官方工具：

```python
{
    "video": video_path_or_url_or_frame_list,
    "fps": video_fps,
    "nframes": video_nframes,
    "video_start": video_start,
    "video_end": video_end,
    "min_frames": video_min_frames,
    "max_frames": video_max_frames,
}
```

然后统一进入：

```text
process_vision_info(..., return_video_metadata=True, return_video_kwargs=True)
processor(..., video_metadata=..., do_resize=False, **video_kwargs)
```

这样旧 frame-list 路径、新 full-video 路径和未来 vLLM-style serving 路径可以共用同一层。

## 分布式边界

完整视频解码和视觉 frontend 仍然只能发生在：

- PP stage0
- pure TP rank0/input-owner
- HYBRID 的 input-owner / stage0 leader

其他 rank 必须继续 consume-only：

- 不读完整视频。
- 不抽帧。
- 不跑视觉 frontend。
- 不接收 raw video path / frame paths。
- 只消费 startup contract / model input / stage handoff。

也就是说，完整视频支持应该放在 startup input owner 之前：

```text
video_path/video_url
  -> input-owner 解码 + 采样
  -> processor
  -> frontend_state
  -> startup contract
  -> non-owner consume-only
```

## 验收建议

实现完整视频入口后，先跑小视频 smoke：

- `tp-mm-generate`
- `pp-mm-generate`
- `hybrid-mm-generate --pp 2 --tp-degrees 2 1`

每个 case 记录：

- video reader backend。
- raw fps。
- sampled frame count。
- `frames_indices` 前后几个值。
- `pixel_values_videos.shape`。
- `video_grid_thw`。
- startup contract keys / tensor count / bytes。
- generated ids/text。

正确性要求：

- frame-list 旧路径不变。
- full-video 新路径 generated ids/text 可解释。
- non-stage0 / non-input-owner 日志里不能出现 video decode / processor / frontend active。

## 参考源码

- 本项目当前 frame-list 入口：[processing/builders.py](/mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/models/qwen3vl/processing/builders.py:20)
- 本项目 frontend state 构建：[runtime_mm.py](/mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/models/qwen3vl/runtime_mm.py:147)
- Qwen3-VL video processor：[qwen3_vl/video_processing_qwen3_vl.py](/mnt/ssd/code/Qwen3_vl/qwen3_vl/video_processing_qwen3_vl.py:86)
- Qwen3-VL processor timestamp expansion：[qwen3_vl/modular_qwen3_vl.py](/mnt/ssd/code/Qwen3_vl/qwen3_vl/modular_qwen3_vl.py:1205)
- Qwen3-VL model video frontend：[qwen3_vl/modeling_qwen3_vl.py](/mnt/ssd/code/Qwen3_vl/qwen3_vl/modeling_qwen3_vl.py:1126)
- qwen-vl-utils video decode/sample：[vision_process.py](/mnt/ssd/miniconda3/envs/vlm/lib/python3.10/site-packages/qwen_vl_utils/vision_process.py:133)
- 官方 qwen-vl-utils 源码：https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
- Hugging Face Qwen3-VL processor 文档：https://huggingface.co/docs/transformers/model_doc/qwen3_vl
- vLLM multimodal video input 文档：https://docs.vllm.ai/en/stable/features/multimodal_inputs/
- vLLM video loader API：https://docs.vllm.ai/en/stable/api/vllm/multimodal/video/
