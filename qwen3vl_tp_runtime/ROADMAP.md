# qwen3vl_tp_runtime Roadmap

Roadmap 只保留当前方向、下一步和验收口径。详细历史和数字看：

- `README.md`：架构和目录职责。
- `BASELINE.md`：baseline、真实 profile、payload/perf 记录。
- `SESSION_HANDOFF.md`：完整接手上下文。

## 当前快照

- `PP / TP / HYBRID` 都能从 `model_path` 直接构建 `StageState`。
- `PP` 和 `TP` 是基础后端；`HYBRID` 是组合后端。
- `StageState` 是主路径术语；`bundle` 只保留给 replay/debug/capture。
- `backend=tp` 已独立，不依赖 HYBRID。
- TP decoder/MLP projection 已经 rank-local materialize。
- pure TP multimodal 已是 rank0/input-owner startup contract，其他 TP rank consume-only。
- HYBRID runtime-only broadcast 已收口到 `hybrid_runtime_inputs_v1`。
- `--comm-dtype` 默认值已落到 `bfloat16`，collective 语义未改。

推荐对照：

- correctness baseline：`baseline_runs/20260428/`
- payload/perf baseline：`baseline_runs/20260429-longterm-profile/`
- step 14 profile：`baseline_runs/20260430-bfloat16-default/`
- 本地最小回归：`bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh`

## 当前任务

### 15. Multimodal Payload 减量

目标：

- 检查 `stage_input` / `deepstack_by_layer` 等大 tensor。
- 区分必须跨 rank/stage 传输的 payload 和可本地重建的 payload。
- 保持 non-stage0 / non-input-owner consume-only。
- 不重新引入 root/full/replay payload。

建议顺序：

1. 先对照 vLLM multimodal input / embeddings 边界。
2. 统计当前 `tp-mm-generate` 和 `hybrid-mm-generate` payload keys/count/bytes。
3. 只规划一类可移除字段，先不急着改多处语义。
4. 优先处理 rank-local 或 stage-local 可重建字段。
5. 重跑对应 Jetson smoke，冻结 before/after。

验收：

- payload keys/count/bytes 有 before/after。
- generated ids/text 不变。
- `weight_load.tp_weight_sharded=true` 不回退。
- `BASELINE.md` / `SESSION_HANDOFF.md` 记录实际结果。

## 后续队列

| step | 任务 | 目标 | 验收重点 |
| --- | --- | --- | --- |
| 16 | Buffer reuse / pinned memory | 减少 hidden/handoff/decode tensor 重复分配，评估 pinned memory | correctness 不变，CUDA peak 不上升，transport/time 有解释 |
| 17 | PP handoff overlap | 让 PP handoff 与本地计算尝试重叠 | PP correctness 不变，wait 时间下降，无死锁 |
| 18 | Stage partition 搜索 | 搜索更好的 stage range，支持异构 Jetson | 每组有 correctness + perf，记录推荐 partition |
| 20 | Embedding / lm_head vocab parallelism | 研究 vocab parallel embedding / lm_head | token/logits/top-k 语义稳定，权重 scope 清楚 |
| 21 | KV cache session 化 | stage-local cache manager，prefill 后 decode 复用 | cache 行为可测，decode correctness 不变 |
| 22 | Serving engine 方向 | scheduler、paged KV、block manager、batching、streaming | correctness/baseline 稳定后再推进 |

## 做每一步前

1. 查 vLLM 对应设计。
2. 写清楚映射到本项目后哪些照搬、哪些不照搬。
3. 本轮只改一个主目标。
4. 先补观测，再做低风险改动。
5. 完成后更新 `BASELINE.md` / `SESSION_HANDOFF.md`。

## 架构不变量

- `pipeline_parallel.py` 是纯 PP 基础后端。
- `tensor_parallel.py` 是纯 TP 基础后端。
- `hybrid_parallel.py` 是 PP+TP 组合后端。
- `hexgen_core/modules/` 只放三种并行后端文件和 `__init__.py`。
- HYBRID 可以调用 PP/TP helper；TP 不能反向依赖 HYBRID。
- PP 不应为了少量复用反向依赖 TP；需要复用时优先提公共模块。
- 主路径 public API 不再用 `bundle` 表达 runtime 对象。
- `--manifest-path`、capture、replay、trace、dump 都属于 debug/replay 路径。

## 已完成索引

- 1-4：TP worker 空壳删除、debug/replay 移出、HYBRID 调 TP helper 规范化、StageState/bundle 命名收紧。
- 5-8：baseline checker、runtime smoke wrapper、完整 correctness baseline、runtime core regression wrapper。
- 9-11：启动耗时、显存、transport/payload profiling。
- 12：vLLM-style startup contract tensor 减量。
- 13：vLLM-style scaffold broadcast 减量、runtime input schema、pure TP multimodal input-owner、长期 profile 冻结。
- 14：TP collective profiling、`tp_degree=1` bypass、substage profiling、pure TP runtime input broadcast 减量、`bfloat16` 默认通信 dtype 落地。

## Definition Of Done

- 符合架构不变量。
- 主路径术语仍是 `StageState`。
- 相关 unit tests 通过。
- 分布式 runtime 改动至少跑一个真实 Jetson smoke。
- payload/transport 改动记录 before/after keys/count/bytes。
- 性能改动保留 `runtime-perf-records.json` 和 `runtime-perf-table.md`。
- 主路径文档或命令变化时，同步更新 `README.md` / `BASELINE.md` / `SESSION_HANDOFF.md`。

Jetson 同步：

```bash
bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed
bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed
```
