# qwen3vl_tp_runtime Roadmap

这份文档只保留当前状态、下一阶段规划和固定规则。详细数字看 `BASELINE.md`，新对话接手看 `SESSION_HANDOFF.md`，每轮原始日志看 `baseline_runs/*/README.md`。

## 当前结论

KV cache 管理先冻结到 20C-4。20D 历史窗口检索回取暂不推进。

主路径状态：

| 方向 | 状态 |
| --- | --- |
| direct runtime | `PP / TP / HYBRID` 已稳定 |
| 后端边界 | `PP` 和 `TP` 是基础后端，`HYBRID` 是组合层 |
| TP 权重 | decoder/MLP projection 已 rank-local materialize |
| multimodal startup | 不传 root/full/replay payload，不传 dense derived tensors |
| runtime input | pure TP 避免 dense `stage_input` broadcast；HYBRID schema 已固化 |
| comm dtype | 默认 `bfloat16` |
| payload 减量 | Step 15 已结束，owner/rebuild 规则冻结 |
| buffer/pinned | Step 16 已结束，`--transport-pin-memory` 保持 opt-in |
| KV cache | Step 20A/20B/20C 已完成到 `infinipot-v` opt-in compaction |
| 完整视频输入 | Step 21 `--video-path` 已通过 HF/PP/TP/HYBRID smoke，frame-dir 回归不变 |

关键效果：

| 阶段 | 修改前 | 修改后 | 效果 |
| --- | --- | --- | --- |
| TP 后端 | TP 借用 HYBRID 路径 | 独立 `TensorParallelRunner` | 后端边界清楚 |
| startup contract | 带 reference `stage_output` 和 dense derived tensors | 只传必要 input/metadata | startup payload 下降 |
| HYBRID `tp_degree=1` | stage1 记录伪 TP collective | single-rank bypass | rank2 TP collective 归零 |
| TP comm dtype | 默认 `float32` | 默认 `bfloat16` | TP collective bytes 约减半 |
| pure TP runtime input | 广播 dense `stage_input` | 本地 embedding / local stage input | broadcast events 归零 |
| Step 20A KV cache | decode 用 `torch.cat([past,current])` | `StageKVCache` append/view | correctness 不变 |
| Step 20B video window | 无 window -> KV location 索引 | `VideoWindowCacheIndex` | 为窗口压缩/检索打基础 |
| Step 20C compaction | 完整 visual KV | opt-in `uniform/swa/infinipot-v` compact 本地 KV | active KV bytes 约减半 |

## 冻结 Baseline

| 阶段 | baseline | 结论 |
| --- | --- | --- |
| correctness | `baseline_runs/20260428/` | 固定 text/mm generated ids/text |
| current perf | `baseline_runs/20260430-bfloat16-default/` | 当前性能快照 |
| Step 15 | `baseline_runs/20260430-step15-derived-rebuild/` | payload derived tensor 本地重建 |
| Step 16 | `baseline_runs/20260501-step16-pinned-ab/` | pinned memory opt-in，默认关闭 |
| Step 20A | `baseline_runs/20260501-step20a-kv-cache-smoke/` | `StageKVCache` smoke 通过 |
| Step 20A long | `baseline_runs/20260501-step20a-kv-cache-long-decode/` | `MAX_NEW_TOKENS=16` 输出不变 |
| Step 20B | `baseline_runs/20260501-step20b-video-window-cache/` | 每 rank `4` windows / `576` video tokens |
| Step 20C-3 | `baseline_runs/20260502-step20c3-compaction/` | `uniform` active KV bytes 约减半 |
| Step 20C-4 | `baseline_runs/20260502-step20c4-infinipot-selector/` | `infinipot-v` 本地 K/V scoring，输出不变 |
| Step 21 | `baseline_runs/20260502-step21-video-input/` | 完整视频 `--video-path` 跑通，frame-dir 旧路径不变 |
| Step 22 | `baseline_runs/20260502-step22-2node-smoke/` | 2-node smoke matrix 子集通过，checker/perf table 产物完整 |

固定 multimodal 输出：

- ids：`[87140, 15946, 3837, 101177]`
- text：`视频中，一名`

## 下一阶段规划

当前不继续做 KV retrieval。接下来优先把 runtime 做得更可用、更稳、更容易汇报。

| 优先级 | 阶段 | 目标 | 验收 |
| ---: | --- | --- | --- |
| 1 | 22. Runtime smoke 和 baseline 自动化收口 | 固化一键跑 PP/TP/HYBRID/text/mm 的最小矩阵 | 新 baseline 目录完整；checker 自动验证 generated ids/text 和关键 bytes |
| 2 | 23. 代码/API 清理 | 清理主路径命名、prompt 贯通、过时 helper 和文档引用 | 主路径只暴露 `StageState / model_input / StageKVCache` 等清晰对象；测试通过 |
| 3 | 24. 汇报材料整理 | 把架构、before/after、性能表、KV cache 路线整理成可讲版本 | README/BASELINE/图表可直接用于汇报 |
| 4 | 25. 后续性能候选 | 只做规划，不马上改语义 | 明确 TP collective、PP handoff overlap、stage partition 的优先级 |

## 21. 完整视频输入路径

目标：现在多模态 smoke 主要依赖准备好的 frame 目录；下一步补齐“完整视频文件 -> 抽帧 -> processor -> runtime input”的主路径。

当前状态：21A 已接入并已用真实 Jetson baseline 冻结。完整日志在 `baseline_runs/20260502-step21-video-input/`。

已完成：

- `VideoInputSpec` / `build_video_messages` / `build_inputs_with_metadata`。
- CLI 增加 `--video-path`、`--video-url` 和 Qwen 采样参数。
- 本地 `video_path` 先用 `ffmpeg` 抽帧，再进 Qwen processor；避免当前 `torchvision/av` reader 在 Jetson 环境里卡住。
- `hf` / `live` / `PP` / `TP` / `HYBRID` direct runtime 都能从 runtime config 选择 frame-dir 或完整视频。
- helper scripts 支持 `VIDEO_PATH` / `VIDEO_URL` 环境变量。
- startup contract 只携带 `video_input_metadata`，不传原始视频、frame bytes、root/full/replay payload。

结论：可以做成 vLLM-style 的输入体验，但不照搬 vLLM serving 层。

| 层次 | vLLM 做法 | 我们这里的规划 |
| --- | --- | --- |
| 用户输入 | offline 支持 `multi_modal_data["video"]`；online 支持 OpenAI-style `video_url` | CLI/runtime 支持 `--video-path` / `--video-url`，继续保留 `--frame-dir` |
| media IO | 统一把 URL/文件/bytes 变成 frames + metadata | 本地视频用轻量 `ffmpeg` adapter，URL 可继续交给 Qwen 官方 `qwen_vl_utils` |
| processor | model-specific processor 处理视频 tensor | 继续用 Qwen3-VL `processor + process_vision_info` |
| 分布式边界 | serving engine 内部处理调度和缓存 | 不引入 serving；仍由 stage0/input-owner 读取视频、跑 frontend、发 startup contract |
| payload | 不把原始视频当作模型主输入传播 | 不跨 rank 传视频文件/frames/root payload，只传 compact runtime tensors/metadata |

建议顺序：

1. 输入 schema（已完成第一版）：
   - `frame_dir + num_frames`：保留当前 smoke 路径。
   - `video_path`：本地完整视频文件。
   - `video_url`：CLI/schema 已保留；真实网络路径后续再单独验证。
   - 采样参数：`video_fps`、`video_nframes`、`video_start`、`video_end`、`video_min_frames`、`video_max_frames`。
2. 处理边界（已完成第一版）：
   - 新增轻量 `VideoInputSpec` / `build_video_messages` helper。
   - `frame_dir` 转成 `{"type": "video", "video": ["file://frame0.jpg", ...]}`。
   - `video_path` 先抽成临时帧列表，再转成 `{"type": "video", "video": ["file://frame0.jpg", ...]}`。
   - 统一走 `process_vision_info(... return_video_metadata=True, return_video_kwargs=True)`。
3. runtime/CLI（已完成第一版）：
   - `scripts/runtime.py` 增加 `--video-path` 和采样参数。
   - helper scripts 支持 `VIDEO_PATH` 环境变量；有 `VIDEO_PATH` 时不要求 `FRAME_DIR`。
   - `video_path` 和 `video_url` 互斥；frame-dir 作为默认 fallback。
4. 分布式约束（已完成第一版）：
   - 只有 stage0 / input-owner 读取视频、抽帧、跑 Qwen3-VL frontend。
   - non-input-owner 只 consume startup contract，不读视频、不抽帧、不跑 frontend。
   - startup/runtime payload 只允许现有 compact tensors/metadata，不传原始视频、frame bytes、root payload。
5. metadata 观测（已完成第一版）：
   - rank log 记录 `video_input.source`、`video_path` basename、`video_backend`、`fps`、`frame_count`、`frames_indices`、`total_num_frames`、`video_grid_thw`。
   - 记录 processor 是否设置 `do_sample_frames=False`，避免重复采样。
6. 验证结果（已完成）：
   - `hf-mm-generate` 完整视频对照通过：`[87140, 108869, 100369, 102122]` / `视频展示了两个场景`。
   - `pp-mm-generate`、`tp-mm-generate`、`hybrid-mm-generate-pp2tp1` 完整视频输出一致。
   - frame-dir 回归通过：`[87140, 15946, 3837, 101177]` / `视频中，一名`。
   - non-owner rank 保持 `consume-only`，startup/runtime payload 不传原始视频、frame bytes、root/full/replay payload。

遗留点：

- `live-mm-generate` 完整视频 trace 15 分钟未产出 JSON，暂不作为冻结 baseline。
- 当前 distributed multimodal direct builder 还没有贯通 CLI `--prompt`，使用 builder 默认视频 prompt；后续在 23 代码/API 清理里修掉。
- 当前 HYBRID 完整视频只跑了 2-node `--pp 2 --tp-degrees 1 1` 路径；3-rank `2 1` 需要第三个 CUDA 可用节点。

验收：

- frame-dir 旧路径 generated ids/text 不变。
- 完整视频路径能生成 rank log。
- non-input-owner 不读视频、不抽帧、不跑 frontend。
- startup/runtime payload 不引入 root/full/replay payload。
- `video_window_cache` 仍能从 `mm_token_type_ids == 2` 识别 video token range。
- 如果完整视频和 frame-dir 使用同一批采样帧，generated ids/text 应一致；如果采样策略不同，必须记录 frame indices 和输出差异。

## 22. Runtime Smoke 和 Baseline 自动化收口

目标：把当前常用 smoke 固化成一键矩阵，减少手动跑漏。

当前状态：22A/22B/22C/22D/22E 第一版已完成。真实 baseline 已冻结在 `baseline_runs/20260502-step22-2node-smoke/`。

已固定矩阵定义：

- 代码：`qwen3vl_tp_runtime/scripts/smoke_matrix.py`
- checker：`qwen3vl_tp_runtime/scripts/check_baseline_logs.py`

固定矩阵：

| case | 目的 |
| --- | --- |
| `hf-text-generate` | 原生 transformers text 对照 |
| `hf-mm-generate` | 原生 transformers frame-dir multimodal 对照 |
| `pp-mm-generate` | PP correctness |
| `tp-mm-generate` | TP correctness + collective |
| `hybrid-mm-generate` | HYBRID correctness + stage/TP 组合 |
| `tp-mm-generate-long` | 长 decode guard |
| `tp-mm-generate-frame-regression` | 完整视频接入后确认 frame-dir 旧路径不变 |
| optional full-video smoke | `hf/pp/tp/hybrid` 完整视频 `--video-path` |

checker 已检查：

- `generated_token_ids` / `generated_text` 是否等于固定输出。
- rank 数量是否符合矩阵定义。
- `runtime_metrics.transport` 里的 startup/scaffold/handoff/TP collective bytes 是否存在且非负。
- startup/runtime payload keys 不含 root/full/replay/stage_output 等禁用 payload。
- multimodal non-owner 是否 `consume-only`。
- TP rank 的 `tp_weight_sharded` / shard rank / shard world size。
- full-video 和 frame-dir 的 `video_input.source` 是否正确。

perf table 已统一输出：

- 脚本：`qwen3vl_tp_runtime/scripts/collect_runtime_perf.py`
- 输出文件：`runtime-perf-records.json` / `runtime-perf-table.md`
- Markdown 字段：`total s`、`startup bytes`、`scaffold bytes`、`handoff bytes`、`TP coll s/bytes`、`CUDA peak`、`loaded weights`、`stage KV bytes`
- `stage KV bytes` 有 active bytes 时显示为 `active / allocated`，否则显示 allocated bytes。

一键 helper：

- 脚本：`qwen3vl_tp_runtime/scripts/helpers/run-step22-smoke-matrix.sh`
- 负责创建 baseline 目录、跑 fixed smoke matrix、调用 checker、生成 perf records/table、写 README。
- 分布式 case 通过 `TP_HOSTS` / `PP_HOSTS` / `HYBRID_HOSTS` 指定 rank host；`local` 表示当前机器，其他 host 通过 ssh 启动。
- required matrix 里的 `hybrid-mm-generate` 需要 3 个 rank；只有两台 Jetson 时要补第三个 CUDA host，或先单独跑可用 case。

22E 结果：

- 已跑 2-node Jetson 子集：`hf-text-generate`、`hf-mm-generate`、`pp-mm-generate`、`tp-mm-generate`、`tp-mm-generate-long`、`tp-mm-generate-frame-regression`。
- `check-smoke-matrix.txt` 全部 PASS。
- `runtime-perf-records.json` / `runtime-perf-table.md` 已生成。
- 暂缺 3-rank `hybrid-mm-generate`：这轮 Codex tool shell 看不到 jetson1 CUDA device nodes，只用 jetson2/jetson3 跑了 2-node 子集；物理 jetson1 普通终端 CUDA 可用，可后续通过普通登录/SSH 作为第三 rank 补跑完整矩阵。

用法：

```bash
# 单个 case
PYTHONPATH=. python qwen3vl_tp_runtime/scripts/check_baseline_logs.py \
  --case-id tp-mm-generate-frame-regression \
  baseline_runs/20260502-step21-video-input/tp-mm-generate-frame-regression-rank0.log \
  baseline_runs/20260502-step21-video-input/tp-mm-generate-frame-regression-rank1.log

# 完整 Step 22 矩阵目录，跑新 baseline 时使用
PYTHONPATH=. python qwen3vl_tp_runtime/scripts/check_baseline_logs.py \
  --matrix step22 \
  --baseline-dir baseline_runs/<new-step22-dir> \
  --require-transport-metrics

# 包含完整视频 optional cases
PYTHONPATH=. python qwen3vl_tp_runtime/scripts/check_baseline_logs.py \
  --matrix step22 \
  --baseline-dir baseline_runs/<new-step22-dir> \
  --include-optional \
  --require-transport-metrics

# 生成 perf records/table
PYTHONPATH=. python qwen3vl_tp_runtime/scripts/collect_runtime_perf.py \
  --baseline-dir baseline_runs/<new-step22-dir> \
  --matrix step22 \
  --output-json baseline_runs/<new-step22-dir>/runtime-perf-records.json \
  --output-md baseline_runs/<new-step22-dir>/runtime-perf-table.md

# 一键跑 Step 22 required matrix
TP_HOSTS="local 10.126.126.4" \
PP_HOSTS="local 10.126.126.4" \
HYBRID_HOSTS="local 10.126.126.4 10.126.126.5" \
bash qwen3vl_tp_runtime/scripts/helpers/run-step22-smoke-matrix.sh

# 包含 full-video optional cases
VIDEO_PATH=/mnt/ssd/code/Qwen3_vl/test/demo.mp4 \
bash qwen3vl_tp_runtime/scripts/helpers/run-step22-smoke-matrix.sh --include-optional
```

验收：

- 每个 case 有 stdout/stderr 或 rank log。
- checker 验证 generated ids/text。
- perf table 自动汇总 total seconds、transport bytes、TP collective、CUDA peak、loaded weights、stage KV cache bytes。
- baseline 目录命名统一，旧目录不再堆积。

下一步：

- 通过普通登录/SSH 使用 jetson1 作为第三 CUDA rank 后，补跑完整 Step 22 matrix，尤其是 required `hybrid-mm-generate`。
- 进入 Step 23：代码/API 清理，优先检查主路径命名、prompt 贯通和 helper 参数一致性。

## 23. 代码/API 清理

目标：在继续加功能前，把主路径 API 收窄，降低后续维护成本。

候选清理：

- 检查 `runtime_input` / `model_input` 命名残留。
- 检查 `bundle` 是否只存在于 replay/debug/capture。
- 检查 `StageState` 是否仍是主路径唯一运行对象。
- 检查 video KV compression 是否只在 opt-in 下 mutate KV。
- 检查 helper script 参数是否和 README 一致。
- 删除不再引用的 debug/replay helper 或迁到 debug 目录。

验收：

- `rg "bundle|runtime_input|manifest-path|allow-debug-paths"` 的结果符合预期。
- 本地单测和最小 smoke wrapper 通过。
- 不改变 generated ids/text 和 baseline payload 语义。

## 24. 汇报材料整理

目标：把项目讲清楚，而不是只堆日志。

建议输出：

- 一张架构图：`PP / TP / HYBRID`、stage、rank、input-owner、startup contract、handoff。
- 一张 before/after 表：payload、TP collective、KV active bytes。
- 一张 KV cache 路线图：`StageKVCache -> VideoWindowCacheIndex -> opt-in compression`。
- 一个 demo 流程：完整视频或 frame-dir 输入，跑到 generated text。

验收：

- `README.md` 能说明项目是什么、怎么跑、当前效果。
- `BASELINE.md` 能支撑汇报数字。
- `ROADMAP.md` 能说明下一步为什么这么排。

## 暂不推进

| 方向 | 原因 |
| --- | --- |
| 20D 历史窗口检索回取 | KV cache 管理先冻结到 20C-4 |
| 默认启用 video KV compression | 还缺更长视频和更多问题质量评估 |
| vLLM-style serving engine | 当前目标不是 serving 系统 |
| BlockPool / prefix cache / scheduler | 暂不引入 serving 复杂度 |
| 远端 dense KV 回取 | 风险高，容易破坏当前 correctness guard |
| PP handoff overlap / stage partition 搜索 | 等完整视频输入和 smoke 自动化稳定后再评估 |

## 固定规则

- 主路径对象叫 `StageState`。
- `bundle` 只保留给 replay/debug/capture。
- `hexgen_core/modules/` 只放 `pipeline_parallel.py`、`tensor_parallel.py`、`hybrid_parallel.py`。
- HYBRID 可以调用 PP/TP helper；TP 不能反向依赖 HYBRID。
- payload/transport 改动必须记录 before/after keys、tensor count、bytes。
- 性能改动必须保留 before/after runtime records。
- 改 runtime 主路径后，至少验证 `generated_token_ids`、`generated_text`、CUDA peak、transport bytes、weight shard scope。

## 常用同步

```bash
bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed
bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed
```
