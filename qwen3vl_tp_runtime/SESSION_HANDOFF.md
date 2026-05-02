# qwen3vl_tp_runtime Handoff

这份文件只给新对话快速接手。详细数字看 `BASELINE.md`，路线看 `ROADMAP.md`，具体某次 smoke 看 `baseline_runs/*/README.md`。

## 一句话上下文

这是 Qwen3-VL 的 correctness-first 分布式推理 runtime 原型。主路径已从 replay bundle 迁移到启动时直接从 `model_path` 构建 `StageState`，支持 `PP / TP / HYBRID`。

## 环境

| 项目 | 值 |
| --- | --- |
| 工作目录 | `/mnt/ssd/code/Qwen3_vl` |
| runtime | `qwen3vl_tp_runtime` |
| Python | `/mnt/ssd/miniconda3/envs/vlm/bin/python` |
| Torchrun | `/mnt/ssd/miniconda3/envs/vlm/bin/torchrun` |
| 模型 | `/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct` |
| 帧目录 | `/mnt/ssd/code/Qwen3_vl/frames` |
| jetson2 | `10.126.126.3` |
| jetson3 | `10.126.126.4` |

同步：

```bash
bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed
bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed
```

## 架构不变量

- `pipeline_parallel.py` 是纯 PP 基础后端。
- `tensor_parallel.py` 是纯 TP 基础后端。
- `hybrid_parallel.py` 是 PP+TP 组合后端。
- HYBRID 可以调用 PP/TP helper；TP 不能反向依赖 HYBRID。
- `hexgen_core/modules/` 只放三种并行后端。
- 主路径对象叫 `StageState`。
- `bundle` 只保留给 replay/debug/capture。
- 内部 helper 优先用 vLLM-style `model_input`；wire key/protocol 仍保留 `runtime_inputs`。
- `--manifest-path`、capture、trace、dump 都属于 debug/replay 路径。

## 当前状态

| 项目 | 当前状态 |
| --- | --- |
| PP | direct StageState 主路径已通过 |
| TP | 独立 runner，不依赖 HYBRID |
| HYBRID | 组合层，stage-group runtime input schema 已固化 |
| TP 权重 | decoder/MLP projection 已 rank-local materialize |
| multimodal TP | rank0/input-owner 准备 startup contract，其他 TP rank consume-only |
| startup contract | 不传 root/full/replay payload，不传 dense derived tensors |
| comm dtype | 默认 `bfloat16` |
| Step 15 | payload owner/rebuild 语义已冻结 |
| Step 16 | decode 小 tensor 复用完成；`--transport-pin-memory` 默认关闭 |
| Step 20A | `StageKVCache` 已通过真实 Jetson smoke |
| Step 20B | `VideoWindowCacheIndex` 已通过真实 Jetson smoke |
| Step 20C | `uniform/swa/infinipot-v` opt-in video KV compaction 已通过到 20C-4；KV cache 管理阶段先到这里 |

## 当前 Baseline

| 用途 | 目录 |
| --- | --- |
| correctness baseline | `baseline_runs/20260428/` |
| 当前性能 baseline | `baseline_runs/20260430-bfloat16-default/` |
| Step 15 payload baseline | `baseline_runs/20260430-step15-derived-rebuild/` |
| Step 16 pinned A/B | `baseline_runs/20260501-step16-pinned-ab/` |
| Step 20A KV cache smoke | `baseline_runs/20260501-step20a-kv-cache-smoke/` |
| Step 20A long decode | `baseline_runs/20260501-step20a-kv-cache-long-decode/` |
| Step 20B video window cache | `baseline_runs/20260501-step20b-video-window-cache/` |
| Step 20C-3 compaction | `baseline_runs/20260502-step20c3-compaction/` |
| Step 20C-4 InfiniPot-V selector | `baseline_runs/20260502-step20c4-infinipot-selector/` |
| Step 21 full video input | `baseline_runs/20260502-step21-video-input/` |
| Step 22 smoke automation | `baseline_runs/20260502-step22-2node-smoke/` |

固定输出：

- text：`[104455, 9909, 9286, 16488]`，`人工智能（Artificial`
- multimodal：`[87140, 15946, 3837, 101177]`，`视频中，一名`

## 最近结论

| 阶段 | 结论 |
| --- | --- |
| Step 15 | `None` slot 跳过；`attention_mask_2d/position_ids` 可重建时不传 |
| Step 16 | decode mask/token 小 buffer 复用；pinned memory 收益小，保持 opt-in |
| Step 20A | `StageKVCache` append/view 替代 runtime-only decode `torch.cat` 路径 |
| Step 20B | 每 rank 记录 `4` windows / `576` video tokens，不压缩、不回取 |
| Step 20C-3 | `uniform keep_ratio=0.5` active KV bytes 约减半，输出不变 |
| Step 20C-4 | `infinipot-v` 用本地 K/V value-norm + TaR 打分，输出不变 |
| Step 21 | 完整视频 `--video-path` 跑通，HF/PP/TP/HYBRID 输出一致，frame-dir 旧路径不变 |
| Step 22 | 2-node smoke matrix 子集通过，checker/perf table 产物完整；3-rank HYBRID 待用普通登录/SSH 方式启用 jetson1 rank |

Step 20C 固定规则：

- 默认 `--video-kv-compression none` 不修改 KV。
- opt-in 才 compact 本 rank/stage 的 `StageKVCache`。
- compact 后 attention mask key length 跟 physical KV length 对齐。
- decode position 仍使用 logical uncompressed position。
- 不重跑视觉 frontend，不广播 dense KV。

## 当前下一步

KV cache 管理先冻结在 20C-4，暂不开始 20D 历史窗口检索回取。新的阶段规划见 `ROADMAP.md`。

Step 21A 完整视频输入已接入并验证：

- `--video-path` / `--video-url` 和采样参数已进 CLI/runtime config。
- 本地 `video_path` 有 `ffmpeg` 时使用 `ffmpeg_frame_adapter`；Jetson2/3 当前无 `ffmpeg`，使用 `pyav_frame_adapter`。
- helper scripts 支持 `VIDEO_PATH` / `VIDEO_URL`。
- 真实 Jetson baseline 已用 `test/demo.mp4 --video-nframes 4` 跑通。
- 完整视频 frozen output：`[87140, 108869, 100369, 102122]`，`视频展示了两个场景`。
- frame-dir 回归 output 仍是 `[87140, 15946, 3837, 101177]`，`视频中，一名`。
- `live-mm-generate` 完整视频 trace 15 分钟未产出 JSON，暂不作为冻结 baseline。

如果继续推进，优先顺序是：

1. Step 23：代码/API 清理，尤其是 distributed multimodal direct builder 贯通 CLI `--prompt`。
2. 通过普通登录/SSH 使用 jetson1 作为第三 CUDA rank，补跑完整 Step 22 matrix，特别是 required `hybrid-mm-generate`。
3. `live-mm-generate` 完整视频 trace 卡顿定位。
4. 汇报材料整理。

Step 22A/22B/22C/22D/22E 已完成：

- `scripts/smoke_matrix.py` 固定 smoke case 和 expected ids/text。
- `scripts/check_baseline_logs.py` 支持 `--matrix step22`、transport bytes、consume-only、TP shard、full-video/frame-dir source 检查。
- `scripts/collect_runtime_perf.py` 支持 `--matrix step22`，统一生成 `runtime-perf-records.json` / `runtime-perf-table.md`，字段包含 total、startup/scaffold/handoff、TP collective、CUDA peak、loaded weights、stage KV bytes。
- `scripts/helpers/run-step22-smoke-matrix.sh` 会创建 baseline 目录、启动 required/optional smoke、运行 checker/perf collector、写 README。
- 真实 Jetson baseline：`baseline_runs/20260502-step22-2node-smoke/`。已跑 `hf-text`、`hf-mm`、`pp-mm`、`tp-mm`、`tp-mm-long`、`tp-mm-frame-regression`，checker 全部 PASS，perf table 已生成。

20D 原计划先做 metadata / retrieval contract，当前只作为后续备选：

- 定义 query range、candidate windows、score、KV location、fetch plan。
- 先 dry-run，不删除 KV、不回取 KV、不改默认生成语义。
- 验收看 top-k windows、预计回取 bytes、generated ids/text 是否不变。

下一阶段如果没有额外指定，按 Roadmap 继续 Step 23：代码/API 清理。

## 常用命令

TP multimodal：

```bash
NODE_RANK=0 MASTER_ADDR=10.126.126.3 bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh
NODE_RANK=1 MASTER_ADDR=10.126.126.3 bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh
```

HYBRID multimodal：

```bash
NODE_RANK=0 MASTER_ADDR=10.126.126.3 bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh
NODE_RANK=1 MASTER_ADDR=10.126.126.3 bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh
NODE_RANK=2 MASTER_ADDR=10.126.126.3 bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh
```

本地最小回归：

```bash
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh
```

## 工作习惯

- 默认中文回答。
- 修改 runtime/架构前先读 `SESSION_HANDOFF.md`、`README.md`、`ROADMAP.md`、`BASELINE.md`。
- 搜索用 `rg`，文件列表用 `rg --files`。
- 手动编辑用 `apply_patch`。
- 不做顺手重构，不回滚用户改动。
- 改 runtime 主路径后，按 `BASELINE.md` 的验收字段跑对应 smoke。
