# Step 21 Full Video Input Baseline

验证目标：完整视频文件 `test/demo.mp4` 走 `--video-path -> 抽帧 -> processor -> runtime input`，并确认分布式路径只由 input-owner/stage0 跑视频读取和视觉 frontend。

## 输入

| item | value |
| --- | --- |
| video | `/mnt/ssd/code/Qwen3_vl/test/demo.mp4` |
| sampling | `--video-nframes 4` |
| reader on Jetson | `pyav_frame_adapter` |
| video grid | `[[2, 40, 74]]` |
| video tokens | `1480` |
| prompt used for frozen match | `请用中文简要描述这个视频的主要内容。` |

## 结果

| case | rank | total s | generated ids | generated text | startup | handoff | TP collective | CUDA peak |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `hf-mm-generate-video-builder-prompt` | - | `37.12` | `[87140, 108869, 100369, 102122]` | `视频展示了两个场景` | `0 B` | `0 B` | `0 B` | `9.76 GiB` |
| `pp-mm-generate-video` | 0 | `74.54` | `[87140, 108869, 100369, 102122]` | `视频展示了两个场景` | `7.42 MiB` | `7.41 MiB` | `0 B` | `7.48 GiB` |
| `pp-mm-generate-video` | 1 | `76.42` | `[87140, 108869, 100369, 102122]` | `视频展示了两个场景` | `7.42 MiB` | `7.41 MiB` | `0 B` | `7.44 GiB` |
| `tp-mm-generate-video` | 0 | `106.15` | `[87140, 108869, 100369, 102122]` | `视频展示了两个场景` | `29.10 MiB` | `0 B` | `55.69s / 533.67 MiB` | `8.86 GiB` |
| `tp-mm-generate-video` | 1 | `106.20` | `[87140, 108869, 100369, 102122]` | `视频展示了两个场景` | `29.10 MiB` | `0 B` | `55.12s / 533.67 MiB` | `8.85 GiB` |
| `hybrid-mm-generate-video-pp2tp1` | 0 | `58.76` | `[87140, 108869, 100369, 102122]` | `视频展示了两个场景` | `7.42 MiB` | `7.41 MiB` | `0 B` | `7.50 GiB` |
| `hybrid-mm-generate-video-pp2tp1` | 1 | `58.79` | `[87140, 108869, 100369, 102122]` | `视频展示了两个场景` | `7.42 MiB` | `7.41 MiB` | `0 B` | `7.45 GiB` |

frame-dir 回归：

| case | rank | generated ids | generated text | startup | TP collective |
| --- | ---: | --- | --- | ---: | ---: |
| `tp-mm-generate-frame-regression` | 0 | `[87140, 15946, 3837, 101177]` | `视频中，一名` | `11.51 MiB` | `24.63s / 221.48 MiB` |
| `tp-mm-generate-frame-regression` | 1 | `[87140, 15946, 3837, 101177]` | `视频中，一名` | `11.51 MiB` | `23.84s / 221.48 MiB` |

## Payload

PP/HYBRID startup contract:

- tensor count: `5`
- bytes: `7,781,072`
- keys: `shared.input_ids`, `shared.rope_deltas`, `shared.mm_token_type_ids`, `shared.video_grid_thw`, `stage_handoffs.1.stage_input`

TP startup contract:

- tensor count: `9`
- bytes: `30,515,387`
- keys: `shared.input_ids`, `shared.rope_deltas`, `shared.mm_token_type_ids`, `shared.video_grid_thw`, `stage_handoffs.0.stage_input`, `stage_visuals.0.visual_pos_masks`, `stage_visuals.0.deepstack_by_layer.0/1/2`

没有传原始视频、frame bytes、root/full/replay payload。non-owner rank 日志显示 `multimodal_frontend_mode=consume-only`。

## 备注

- `live-mm-generate-video.log` 15 分钟后仍停在权重加载后的 live trace 阶段，没有 JSON；先不作为 Step 21 冻结 baseline。
- `hf-mm-generate-video.log` 是自定义 prompt `请用中文简要描述这个视频。`，输出不同是预期的 prompt 差异。
- 当前 distributed multimodal direct builder 仍使用 builder 默认视频 prompt；后续代码/API 清理时再把 CLI prompt 贯通到 direct manifest。
