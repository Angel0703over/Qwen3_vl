# qwen3vl_tp_runtime

Qwen3-VL 分布式推理 runtime 原型。主路径支持 `pp`、`tp`、`hybrid`，启动时直接从 `model_path` 构建每个 stage/rank 的 `StageState`。

## 快速使用

主入口：

```bash
/mnt/ssd/miniconda3/envs/vlm/bin/python qwen3vl_tp_runtime/scripts/runtime.py
```

常用并行参数：

```bash
# 纯 PP：按层平均切成 2 个 stage
--backend pp --pp 2

# 纯 TP：单 stage，全模型 TP=2
--backend tp --tp 2

# 均匀 HYBRID：2 个 PP stage，每个 stage TP=2
--backend hybrid --pp 2 --tp 2

# 异构 HYBRID：stage0 TP=2，stage1 TP=1
--backend hybrid --pp 2 --tp-degrees 2 1
```

高级覆盖：

- `--stage-ranges 0:17 18:35`
- `--tp-degrees 2 1`

## 当前状态

| 方向 | 当前结果 |
| --- | --- |
| direct runtime | `PP / TP / HYBRID` 都从 `model_path` 直接构建 `StageState` |
| TP 后端 | 已独立为 `TensorParallelRunner`，不依赖 HYBRID |
| TP 权重 | decoder/MLP projection 已 rank-local materialize |
| multimodal TP | rank0/input-owner 准备 startup contract，其他 TP rank consume-only |
| HYBRID | PP+TP 组合层，runtime input 已收口到 `hybrid_runtime_inputs_v1` |
| transport payload | 不传 root/full/replay payload，不传 dense derived attention/RoPE tensor |
| comm dtype | 默认 `bfloat16` |

## 修改效果

| 修改 | 修改前 | 修改后 |
| --- | ---: | ---: |
| startup contract 移除 `stage_output` | `7,563,328` bytes | `4,353,088` bytes |
| startup contract 移除 dense derived tensor | `4,353,088` bytes | `3,245,806` bytes |
| HYBRID stage1 `tp_degree=1` collective | `648.46 MiB` | `0 B` |
| pure TP comm dtype | `449.12 MiB` collective | `221.48 MiB` collective |
| pure TP runtime input broadcast | `4` events / rank | `0` events |
| Step 15 derived shared payload | `12,093,371` bytes | `12,068,291` bytes |

## 固定术语

- 主路径执行对象叫 `StageState`。
- `bundle` 只保留给 replay、capture、debug 路径。
- `PP` 和 `TP` 是基础后端；`HYBRID` 是组合后端。
- HYBRID 可以调用 PP/TP helper；TP 不能反向依赖 HYBRID。

## 目录职责

- `hexgen_core/modules/pipeline_parallel.py`：纯 PP 后端。
- `hexgen_core/modules/tensor_parallel.py`：纯 TP 后端。
- `hexgen_core/modules/hybrid_parallel.py`：PP+TP HYBRID 后端。
- `hexgen_core/schema.py`：manifest、rank context、runtime input schema。
- `models/qwen3vl/runtime_builder.py`：从 `model_path` 构建 stage/rank `StageState`。
- `models/qwen3vl/runtime_mm_stage.py`：multimodal shared/runtime tensor rebuild。
- `models/qwen3vl/runtime_text_stage.py`：text runtime input rebuild 和 stage materialization。
- `models/qwen3vl/weights/`：权重 index、load plan、TP shard slicing。
- `scripts/runtime.py`：统一 CLI 入口。
- `scripts/helpers/`：稳定 smoke wrapper。

## 调试路径

下面路径只用于 replay/capture/debug，需要显式传 `--allow-debug-paths`：

- `--manifest-path`
- `--compare-direct`
- `--trace-layers`
- `--dump-layer`

## 文档

- `ROADMAP.md`：当前任务和后续队列。
- `BASELINE.md`：当前 baseline、before/after 效果、验收字段。
- `SESSION_HANDOFF.md`：新对话接手用的简明上下文。
- `baseline_runs/*/README.md`：具体某轮真实 Jetson profile。
