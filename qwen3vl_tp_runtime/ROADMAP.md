# qwen3vl_tp_runtime Roadmap

这份文件只回答三个问题：

- 现在项目处在哪。
- 下一步优先做什么。
- 做完一项怎么验收。

详细历史、命令和 payload 数字放在：

- `README.md`：当前架构和目录职责。
- `BASELINE.md`：固定 baseline、真实 profile、payload/perf 数字。
- `SESSION_HANDOFF.md`：新对话接手用完整上下文。

## 当前状态

已完成的主线里程碑：

- `PP / TP / hybrid` 主路径都能从 `model_path` 直接构建 `StageState`。
- `StageState` 是主路径术语；`bundle` 只保留给 replay/debug/capture/file-backed reference。
- `PP` 和 `TP` 是基础后端，`HYBRID` 是组合后端。
- `backend=tp` 已经是独立后端，不依赖 HYBRID。
- TP decoder/MLP projection 已经 rank-local materialize。
- pure TP multimodal 已改成 rank0/input-owner startup contract，其他 TP rank consume-only。
- HYBRID runtime-only broadcast 已收口到正式 `hybrid_runtime_inputs_v1` schema。
- 2026-04-29 已冻结 step 13 长期目标真实 profile：
  - `tp-mm-generate`
  - `hybrid-text-generate`
  - `hybrid-mm-generate --pp 2 --tp-degrees 2 1`

当前推荐对照：

- correctness baseline：`baseline_runs/20260428/`
- payload/perf baseline：`baseline_runs/20260429-longterm-profile/`
- 本地最小回归：

```bash
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh
```

## 当前下一步

### 14. TP collective profiling 和低风险调整

为什么先做这个：

- `tp-mm-generate` 每 rank TP collective 约 `449.12 MiB`，耗时约 `44-45s`。
- `hybrid-mm-generate` stage0 TP collective 约 `227.64 MiB`，stage1 约 `648.46 MiB`。
- 这已经是当前真实 profile 里最明显的性能观察点。

先只补观测，不急着改语义：

- 记录每个 prefill/decode step 的 collective 次数。
- 记录 collective tensor shape / dtype / bytes / elapsed seconds。
- 记录 rank 间耗时是否不一致。

低风险优化候选：

- 减少不必要 dtype cast。
- 合并小 collective。
- 避免重复 all-gather。

验收：

- `tp-text-generate` / `tp-mm-generate` generated ids/text 不变。
- `hybrid-mm-generate` generated ids/text 不变。
- `weight_load.tp_weight_sharded=true` 不回退。
- summary 或 perf records 能看到更细的 collective timing。
- 优化后 TP collective 总耗时下降，或至少不增加。

## 后续任务队列

### 15. Multimodal payload 减量

目标：

- 单独检查 `stage_input` / `deepstack_by_layer` 的大 tensor。
- 判断哪些是必须跨 rank/stage 传输，哪些可以本地重建或延迟构造。

验收：

- payload keys/count/bytes 有 before/after。
- generated ids/text 不变。
- 不重新引入 root/full/replay payload。
- non-stage0 / non-input-owner 仍然 consume-only。

### 16. Buffer reuse / pinned memory 实验

目标：

- 减少 hidden / handoff / decode step tensor 的重复分配。
- 评估 pinned memory 对 Jetson CPU transport / CPU->GPU copy 是否有收益。

执行顺序：

1. 先 profiling，不改语义。
2. 再做 stage-local buffer reuse。
3. 最后单独实验 pinned memory。

验收：

- correctness 不变。
- CUDA peak allocated/reserved 不上升，最好下降。
- runtime_total_seconds 或 transport seconds 有可解释变化。

### 17. PP handoff overlap

目标：

- 尝试让 PP stage 间 hidden handoff 与本地计算重叠。

前置条件：

- PP handoff payload bytes 和 send/recv timing 已经可观测。
- 已确认阻塞点，不盲目改通信结构。

验收：

- `pp-text-generate` / `pp-mm-generate` correctness 不变。
- PP handoff wait 时间下降。
- 没有引入 rank 间死锁风险。

### 18. Stage partition 搜索

目标：

- 根据每个 stage 的耗时和显存搜索更合适的 stage range。
- 支持不同 Jetson 数量和异构性能。

候选方向：

- 先手工比较 `0:17 / 18:35` 附近切分。
- 再写脚本自动枚举 stage ranges。
- HYBRID 同时考虑 stage range 和 `tp_degrees`。

验收：

- 每组 partition 都有 correctness + perf 表。
- 找到比当前 baseline 更好的 total time / peak memory 组合。
- 文档记录最终推荐 partition。

## 长期功能

### 20. Embedding / lm_head vocab parallelism

当前状态：

- decoder projection / MLP projection 已按 TP shard-local 加载。
- `embed_tokens_weight` 和 `lm_head_weight` 仍按当前执行语义复制。

目标：

- 研究并实现 vocab parallel embedding / lm_head。

风险：

- 会影响 token embedding、logits gather、top-k、sampling。
- 会影响 PP stage0 和 last stage 的权重 scope。

### 21. KV cache session 化

当前状态：

- KV cache 主要以 correctness-first 的 `cache_by_layer` 字典存在。

目标：

- 设计 stage-local cache manager。
- prefill 后 cache 持久化。
- decode step 复用 cache。
- 后续支持多请求。

### 22. 更接近 serving engine

长期目标：

- request scheduler
- paged KV cache
- block manager
- continuous batching
- streaming output
- prefill/decode overlap

这些不是当前优先项，等 correctness、baseline 和性能记录稳定后再推进。

## 架构不变量

后续所有改动都必须遵守：

- `pipeline_parallel.py` 是纯 PP 基础后端。
- `tensor_parallel.py` 是纯 TP 基础后端。
- `hybrid_parallel.py` 是 PP+TP 组合后端。
- `hexgen_core/modules/` 只放三种并行后端文件和 `__init__.py`。
- HYBRID 可以调用 PP/TP helper。
- TP 不能反向依赖 HYBRID。
- PP 不应为了少量复用反向依赖 TP；需要复用时优先提公共模块。
- 主路径 public API 不再使用 `bundle` 表达 runtime 对象。
- `--manifest-path`、capture、replay、trace、dump 都属于 debug/replay 路径。

## 性能任务规则

每个性能优化都要保留：

- before/after `runtime-perf-records.json`
- before/after `runtime-perf-table.md`
- generated ids/text
- payload keys/count/bytes
- loaded weight bytes
- CUDA peak allocated/reserved

默认先跑：

```bash
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh
```

如果改到权重加载，再跑：

```bash
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh --include-weight-loader
```

如果改到真实 distributed generate 路径，至少重跑对应 Jetson smoke：

- `tp-text-generate`
- `tp-mm-generate`
- `pp-mm-generate`
- `hybrid-mm-generate`

## 已完成索引

这些任务已经完成，不要重复展开到 Roadmap 里：

- 1-4：TP worker 空壳删除、debug/replay 移出、HYBRID 调 TP helper 规范化、StageState/bundle 命名收紧。
- 5-8：baseline checker、runtime smoke wrapper、完整 correctness baseline、runtime core regression wrapper。
- 9-11：启动耗时、显存、transport/payload profiling。
- 12：vLLM-style startup contract tensor 减量。
- 13：vLLM-style scaffold broadcast 减量、runtime input schema、pure TP multimodal input-owner、长期 profile 冻结。

需要查细节时看：

- `BASELINE.md`
- `SESSION_HANDOFF.md`
- `baseline_runs/20260429-longterm-profile/README.md`

## Definition Of Done

一个 Roadmap 任务完成时，至少满足：

- 代码改动符合架构不变量。
- 主路径术语仍是 `StageState`。
- debug/replay 路径没有被误认为主路径。
- 相关 unit tests 通过。
- 如果影响分布式 runtime，至少跑一个真实 Jetson smoke。
- 如果影响主路径文档或命令，同步更新 `README.md` / `BASELINE.md` / `SESSION_HANDOFF.md`。
- 如需继续在 Jetson 上测试，完成同步：

```bash
bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed
bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed
```
