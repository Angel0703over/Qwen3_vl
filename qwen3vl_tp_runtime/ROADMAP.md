# qwen3vl_tp_runtime Roadmap

这份 Roadmap 只保留当前方向、已完成效果和下一步。详细数字看 `BASELINE.md`，完整交接看 `SESSION_HANDOFF.md`。

## 当前状态

- `PP / TP / HYBRID` 都已从 `model_path` 直接构建 `StageState`。
- `PP` 和 `TP` 是基础后端；`HYBRID` 是组合后端。
- `backend=tp` 已独立，不依赖 HYBRID。
- TP decoder/MLP projection 已 rank-local materialize。
- pure TP multimodal 已是 rank0/input-owner startup contract。
- HYBRID runtime input 已收口到 `hybrid_runtime_inputs_v1`。
- `--comm-dtype` 默认值已改为 `bfloat16`。
- Step 15 已结束：空 slot 清理、derived shared tensor 本地重建、`stage_input/deepstack_by_layer` owner 语义已冻结。

## 已完成效果

| 阶段 | 修改前 | 修改后 | 效果 |
| --- | --- | --- | --- |
| TP 独立后端 | TP 借用 HYBRID manifest/runner | 独立 `TensorParallelManifest / TensorParallelRunner` | PP/TP/HYBRID 边界清楚 |
| startup contract `stage_output` | contract 带 reference `stage_output` | 只传后续 stage 必需的 `stage_input` | `7,563,328 -> 4,353,088` bytes |
| startup derived tensor | 传 dense `attention_mask/cos/sin` | non-stage0 本地重建 | `4,353,088 -> 3,245,806` bytes |
| HYBRID `tp_degree=1` | stage1 仍记录伪 TP collective | single-rank collective bypass | rank2 TP collective `648.46 MiB -> 0 B` |
| TP comm dtype | 默认 `float32` 通信 | 默认 `bfloat16` 通信 | `tp-mm` collective `449.12 -> 221.48 MiB` |
| pure TP runtime input | generate 时广播 dense `stage_input` | 本地 embedding / local stage input | runtime input events `4 -> 0` |
| Step 15 空 slot | payload 带 `None` tensor slot | 跳过 `None` slot | key count 下降，bytes 不变 |
| Step 15 derived shared | 传 `attention_mask_2d/position_ids` | 可重建时本地恢复 | affected payload 少 `25,080` bytes |
| Step 15 大 payload | `stage_input/deepstack` 语义未冻结 | owner/rebuild 规则写清楚 | 后续删大 tensor 有安全边界 |
| 代码冗余清理 | PP/HYBRID 有 worker 薄封装和 transport 旧别名 | 直接 phase impl + `StageCommunicator` | 主路径 API 更窄 |
| vLLM-style 命名 | HYBRID helper 叫 `runtime_input` | 内部 helper 改为 `model_input` | 代码更直观，wire protocol 不变 |

## 当前任务：16. Buffer reuse / pinned memory

目标：

- 减少 hidden / handoff / decode tensor 的重复分配和 clone。
- 评估 CPU transport 前后的 pinned memory 是否值得引入。
- 不改变 correctness、payload 协议和 TP shard scope。

执行顺序：

1. 统计 generate 主路径里 hidden / handoff / decode tensor 的分配和 clone 位置。
2. 区分必须持久保存的 tensor、可复用 buffer、只为 transport 临时创建的 tensor。
3. 先做低风险 buffer reuse，不改变 stage/handoff 语义。
4. pinned memory 只做 opt-in 实验，有数据再决定是否默认启用。

验收：

- generated ids/text 不变。
- `weight_load.tp_weight_sharded=true` 不回退。
- CUDA peak allocated/reserved 不上升，或上升原因明确。
- transport/time 有 before/after 记录。
- `BASELINE.md` / `SESSION_HANDOFF.md` 更新实际结果。

## 后续队列

| step | 任务 | 目标 | 验收重点 |
| --- | --- | --- | --- |
| 17 | PP handoff overlap | 尝试让 PP handoff 与本地计算重叠 | correctness 不变，wait 时间下降，无死锁 |
| 18 | Stage partition 搜索 | 搜索更好的 stage range，支持异构 Jetson | 每组有 correctness + perf |
| 20 | Embedding / lm_head vocab parallelism | 研究 vocab parallel embedding / lm_head | token/logits/top-k 语义稳定 |
| 21 | KV cache session 化 | stage-local cache manager，prefill 后 decode 复用 | cache 行为可测 |
| 22 | Serving engine 方向 | scheduler、paged KV、batching、streaming | correctness/baseline 稳定后再推进 |

## 固定规则

- 做每一步前先对照 vLLM，写清楚哪些能照搬、哪些不能照搬。
- 本轮只改一个主目标。
- payload/transport 改动必须记录 before/after keys、tensor count、bytes。
- 性能改动必须保留 before/after runtime records。
- `StageState` 是主路径术语；`bundle` 只保留给 replay/debug/capture。
- 内部函数名优先用 vLLM-style `model_input`；已有 wire key/protocol 不为命名重构单独改动。
- `hexgen_core/modules/` 只放 `pipeline_parallel.py`、`tensor_parallel.py`、`hybrid_parallel.py`。
- HYBRID 可以调用 PP/TP helper；TP 不能反向依赖 HYBRID。

## 常用同步

```bash
bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed
bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed
```
