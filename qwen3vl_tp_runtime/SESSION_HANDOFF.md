# qwen3vl_tp_runtime Handoff

给新对话快速接手用。路线看 `ROADMAP.md`，数字看 `BASELINE.md`，原始日志看 `baseline_runs/*/README.md`。

## 一句话上下文

这是 Qwen3-VL 的 correctness-first 分布式推理 runtime 原型。主路径已从 replay bundle 迁移到启动时直接从 `model_path` 构建 `StageState`，支持 `PP / TP / HYBRID`。

## 环境

| 项目 | 值 |
| --- | --- |
| repo | `/mnt/ssd/code/Qwen3_vl` |
| runtime | `qwen3vl_tp_runtime` |
| Python | `/mnt/ssd/miniconda3/envs/vlm/bin/python` |
| Torchrun | `/mnt/ssd/miniconda3/envs/vlm/bin/torchrun` |
| model | `/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct` |
| frame dir | `/mnt/ssd/code/Qwen3_vl/frames` |
| jetson2 | `10.126.126.3` |
| jetson3 | `10.126.126.4` |

同步：

```bash
bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed
bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed
```

## 架构规则

- `pipeline_parallel.py` 是纯 PP 基础后端。
- `tensor_parallel.py` 是纯 TP 基础后端。
- `hybrid_parallel.py` 是 PP+TP 组合后端。
- HYBRID 可以调用 PP/TP helper；TP 不能反向依赖 HYBRID。
- 主路径对象叫 `StageState`。
- `bundle` 只保留给 replay/debug/capture/legacy compat。
- KV cache 实现集中在 `models/qwen3vl/kv_cache/`；不再保留 `models/qwen3vl/execution/kv_cache.py` 这类旧子模块 shim。
- 内部 helper 可以用 `model_input`；wire key/protocol 继续保留 `runtime_inputs`。
- `--manifest-path`、capture、trace、dump 都属于 debug/replay 路径。

## 当前状态

| 方向 | 状态 |
| --- | --- |
| PP | direct StageState 主路径已通过 |
| TP | 独立 `TensorParallelRunner`，不依赖 HYBRID |
| HYBRID | 组合层，stage-group runtime input schema 已固化 |
| TP 权重 | decoder/MLP projection 已 rank-local materialize |
| multimodal TP | rank0/input-owner 准备 startup contract，其他 TP rank consume-only |
| startup contract | 不传 root/full/replay payload，不传 dense derived tensors |
| comm dtype | 默认 `bfloat16` |
| KV cache | 20A/20B/20C 到 `infinipot-v` opt-in compaction 已跑通；实现位于 `models/qwen3vl/kv_cache/`；20D 暂不做 |
| 完整视频 | `--video-path` 已通过 HF/PP/TP/HYBRID smoke |
| Step 22 | smoke matrix / checker / perf table 已自动化 |
| Step 23 | API 命名、prompt 贯通、legacy lazy export 第一轮完成 |
| Step 24 | 24A-24K 已完成；PP/TP/HYBRID 三个 backend 已统一 facade；KV cache 已独立子包；旧 KV 子模块 shim 已删除；真实 Jetson `pp-mm/tp-mm/hybrid-mm` 子集通过 |

固定输出：

- text：`[104455, 9909, 9286, 16488]`，`人工智能（Artificial`
- frame-dir multimodal with CLI prompt：`[104455, 9909, 9286, 16488]`，`人工智能（Artificial`
- full-video default video prompt：`[87140, 108869, 100369, 102122]`，`视频展示了两个场景`

## 重要 Baseline

| 用途 | 目录 |
| --- | --- |
| current perf | `baseline_runs/20260430-bfloat16-default/` |
| Step 20C-4 InfiniPot-V selector | `baseline_runs/20260502-step20c4-infinipot-selector/` |
| Step 21 full video input | `baseline_runs/20260502-step21-video-input/` |
| Step 22 smoke automation | `baseline_runs/20260502-step22-full-smoke/` |
| Step 23C prompt smoke | `baseline_runs/20260502-step23c-prompt-smoke/` |
| Step 24H code cleanup verify | `baseline_runs/20260503-step24h-verify/` |

## 最近结论

| 阶段 | 结论 |
| --- | --- |
| Step 15 | derived tensor 本地重建，payload owner/rebuild 规则冻结 |
| Step 16 | decode 小 buffer 复用完成；pinned memory 保持 opt-in |
| Step 20A | `StageKVCache` 替代 runtime-only decode `torch.cat` 路径 |
| Step 20B | 每 rank 记录 video window -> KV location metadata |
| Step 20C | `uniform/swa/infinipot-v` opt-in compaction 可把 active KV bytes 约减半 |
| Step 21 | 完整视频输入跑通，不跨 rank 传原始视频/frame bytes |
| Step 22 | required smoke matrix 完整通过，含 PP=3 和 3-rank HYBRID |
| Step 23 | frame-dir multimodal 已贯通 CLI `--prompt`；主路径 execution 不再用 `bundle` 命名 |

## 当前下一步

1. 代码脚手架整理按小步推进；Step 25 性能候选暂不做。
2. 当前已完成 KV cache 独立子包拆分，后续可继续审计 `runtime_builder.py` 是否适合按 startup/manifest 拆分。
3. `live-mm-generate` 完整视频 trace 卡顿可以单独定位，不阻塞主 runtime。

## 常用命令

```bash
# 本地最小回归
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh

# Step 22 baseline checker
PYTHONPATH=. python qwen3vl_tp_runtime/scripts/check_baseline_logs.py \
  --matrix step22 \
  --baseline-dir baseline_runs/<new-step22-dir> \
  --require-transport-metrics

# TP multimodal
NODE_RANK=0 MASTER_ADDR=10.126.126.3 bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh
NODE_RANK=1 MASTER_ADDR=10.126.126.3 bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh

# HYBRID multimodal
NODE_RANK=0 MASTER_ADDR=10.126.126.3 bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh
NODE_RANK=1 MASTER_ADDR=10.126.126.3 bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh
NODE_RANK=2 MASTER_ADDR=10.126.126.3 bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh
```
