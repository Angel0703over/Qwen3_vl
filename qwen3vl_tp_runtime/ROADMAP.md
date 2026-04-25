# qwen3vl_tp_runtime Roadmap

这份文档用于固定我们已经对齐的推进顺序。

默认规则：

- 后续工作按这里的顺序推进。
- 如果没有新的明确讨论，不随意换顺序。
- 当前先不做 KV cache manager。

## 当前快照

已基本完成：

- `text` 的 `pp/tp/hybrid` 主运行路径已经 direct-first，不再默认依赖磁盘 bundle。
- `text` 权重索引层已经有了：`weights/index.py`、`weights/loader.py`、`weights/planner.py`。
- `text` 的 `PP stage-only load` 已经基本成型。
- `text` 的 `TP local-shard load + local-shard kernel` 已经基本成型。
- `--manifest-path replay` 已经降级为 debug-only，用法上不再是主路径。

部分完成：

- `schema` 和 loader 里仍然保留 `bundle_path / bundle_dir` 兼容字段和兼容分支。
- `multimodal` 已经开始从 `runtime_builder.py` 里拆边界，新增了 `models/qwen3vl/runtime_mm.py`。
- `hybrid multimodal` 已经不会误走 text prompt metadata 路径，但还没有自己的 frontend metadata/runtime contract。
- `legacy/debug` 兼容层仍然大量保留在 `capture/`、`tensor_parallel.py`、`prepare_*`、`__init__` 导出面里。

尚未完成：

- `multimodal direct/stage-only load`
- `hybrid multimodal` 的本地 shard/stage 构建
- 主 schema 的最终瘦身
- legacy/debug 最后一波下沉
- 基线验收矩阵
- 系统化性能验收

## 约定顺序

### 1. 冻结最小基线验收

目标：

- 固定一组 `hf/live/pp/tp/hybrid` 的 `text/multimodal generate` 回归命令。
- 每次回归统一比较 `generated_token_ids`、`generated_text`、启动时间。

主要文件：

- `scripts/runtime.py`
- `scripts/runtime_summary.py`
- `README.md`

说明：

- 峰值显存统计也要纳入，但可以在最后补齐；当前先把命令矩阵固定下来。

### 2. 立起 multimodal 的显式状态边界

目标：

- 把 `multimodal` 的 `frontend state`、`decode state`、`stage state` 从当前的大一统对象里拆出来。
- 新增明确边界模块，例如 `models/qwen3vl/runtime_mm_stage.py`。

主要文件：

- `models/qwen3vl/runtime_mm.py`
- `models/qwen3vl/live/common.py`
- `models/qwen3vl/live/inputs.py`

说明：

- 这是当前最优先的一步。
- 这一步先做边界，不要求一次把 vision shard-only 全部做完。

### 3. 让 multimodal decode 只依赖显式状态

目标：

- `decode` 不再从 `model.model.rope_deltas` 读取隐式状态。
- `rope_deltas`、`position_ids`、`visual_pos_masks`、`deepstack_by_layer` 全部来自显式 runtime state。

主要文件：

- `models/qwen3vl/live/inputs.py`
- `models/qwen3vl/runtime_mm.py`
- `models/qwen3vl/runtime_mm_stage.py`

### 4. 把 runtime_builder 接到新的 multimodal state 边界

目标：

- 收掉 `runtime_builder.py` 里 multimodal 相关的裸 `dict` 状态和临时拼装逻辑。
- 让 builder 明确消费新的 `frontend state / stage state`。

主要文件：

- `models/qwen3vl/runtime_builder.py`

### 5. 给 pp/hybrid 接上 multimodal direct loader 协议

目标：

- 给 multimodal 定义自己的 frontend metadata / runtime metadata / scaffold 约定。
- 不再只是“跳过 text prompt metadata”，而是真正有自己的 direct contract。

主要文件：

- `hexgen_core/modules/pipeline_parallel.py`
- `hexgen_core/modules/hybrid_parallel.py`

### 6. 做 multimodal 的 stage-only load

目标：

- `stage0` 负责 vision/frontend。
- 非 `stage0` 只负责 decoder stage。
- 主 direct runtime 分支里不再整模 `load_model()`。

主要文件：

- `models/qwen3vl/runtime_mm.py`
- `models/qwen3vl/runtime_mm_stage.py`
- `models/qwen3vl/vision/encoder.py`
- `models/qwen3vl/vision/deepstack.py`
- 可能新增 `models/qwen3vl/weights/vision.py`

### 7. 收 hybrid multimodal

目标：

- `hybrid` 的每个 rank 按 `(stage_idx, tp_rank)` 本地构建自己的 multimodal runtime。
- stage leader 只广播轻量元数据，不再广播完整 multimodal stage bundle。

主要文件：

- `hexgen_core/modules/hybrid_parallel.py`

### 8. 收 schema 和主 CLI

目标：

- manifest 只保留 direct runtime 所需最小信息。
- 主路径不再显式暴露 bundle/replay 语义。
- `bundle_path is None` 这类主路径分叉继续减少。

主要文件：

- `hexgen_core/schema.py`
- `scripts/runtime.py`
- `scripts/runtime_cli.py`

### 9. 清理 legacy/debug 兼容层

目标：

- `capture/*`
- `prepare_*`
- `tensor_parallel.py`
- compatibility `__init__` 导出

这些都进一步下沉为 debug-only，不再污染主运行面。

主要文件：

- `hexgen_core/modules/tensor_parallel.py`
- `hexgen_core/modules/__init__.py`
- `models/qwen3vl/__init__.py`
- `models/qwen3vl/capture/*`
- `hexgen_core/modules/pipeline_parallel.py`
- `hexgen_core/modules/hybrid_parallel.py`

### 10. 做最终性能验收

目标：

- 验证启动时间下降。
- 验证峰值显存下降。
- 验证 `PP` rank 不再加载无关层。
- 验证 `TP` rank 不再持有完整层权重。
- 验证 `PP/TP/hybrid` 输出一致性。

## 当前默认下一步

当前默认下一步是：

`步骤 2：立起 multimodal 的显式状态边界`

只有当这一步稳定后，后面的 `multimodal direct loader`、`stage-only load`、`hybrid multimodal` 才会顺下来。
