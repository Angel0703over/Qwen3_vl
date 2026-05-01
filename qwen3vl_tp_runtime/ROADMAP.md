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
- Step 16 已结束：decode 小 tensor 复用已落地，`--transport-pin-memory` A/B 已完成，默认关闭。

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
| Step 16 decode 小 tensor | 每 step `cat(mask, ones)` / 新建 token tensor | 预分配 mask buffer / 复用 token buffer | 减少 decode loop 小分配，payload bytes 不变 |
| Step 16 pinned memory | 无实验开关 | `--transport-pin-memory` best-effort pinned CPU staging | TP 小幅变快，CUDA peak 不变，收益不大所以默认关闭 |
| 代码冗余清理 | PP/HYBRID 有 worker 薄封装和 transport 旧别名 | 直接 phase impl + `StageCommunicator` | 主路径 API 更窄 |
| vLLM-style 命名 | HYBRID helper 叫 `runtime_input` | 内部 helper 改为 `model_input` | 代码更直观，wire protocol 不变 |

## 已结束：16. Buffer reuse / pinned memory

目标：

- 减少 hidden / handoff / decode tensor 的重复分配和 clone。
- 评估 CPU transport 前后的 pinned memory 是否值得引入。
- 不改变 correctness、payload 协议和 TP shard scope。

完成内容：

1. 已完成 allocation / clone 盘点：`BUFFER_REUSE_AUDIT.md`。
2. 已完成 decode loop 小 tensor reuse：`PP / TP / HYBRID` runtime-only generate 共享 `generate_buffers.py`。
3. 已完成 pinned memory opt-in：`--transport-pin-memory`，覆盖 TP collective 和 tensor payload CPU staging。
4. KV cache 的 `full_key/full_value` clone 和 `torch.cat([past, current])` 留到 KV cache manager 阶段。

真实 A/B：

- 目录：`baseline_runs/20260501-step16-pinned-ab/`。
- `tp-mm-generate` generated ids/text 不变，payload bytes 不变。
- pinned total time 从 `53.47 / 53.21s` 到 `53.01 / 52.97s`。
- pinned TP collective time 从 `24.34 / 23.76s` 到 `23.91 / 23.51s`。
- CUDA peak allocated 不上升；rank0 reserved 约多 `2 MiB`。
- HYBRID 功能性 A/B 通过，但 rank0/rank1 共用 jetson2，只作为 correctness 验证。
- 结论：保留 opt-in，不改默认值。

已验证：

- `py_compile` 已覆盖 `generate_buffers.py` 和三个后端。
- `run-runtime-core-regression.sh --skip-baseline-checks` 已通过，包含 distributed transport 单测。
- 真实 Jetson A/B 已完成。

验收：

- generated ids/text 不变。
- `weight_load.tp_weight_sharded=true` 不回退。
- CUDA peak allocated 不上升；reserved 小幅波动已解释。
- transport/time 有 before/after 记录。
- `BASELINE.md` / `SESSION_HANDOFF.md` 已更新实际结果。

## 当前下一步：20. KV cache 管理部分

当前只围绕三条路线推进：Jupiter-style 连续 KV buffer、InfiniPot-V-style 视觉 token 压缩、ReKV-style 历史窗口检索。vLLM serving 体系暂不考虑。

### 20A. Jupiter-style 连续 KV buffer

先做这个。

目标：

- 提前在 GPU 上为每个本地 stage/rank 分配连续 KV buffer。
- prefill/decode 直接把 K/V 写入 buffer。
- 用 `current_length` 记录有效长度。
- 避免 decode 每步 `torch.cat([past, current])`。
- 避免每层每步 `full_key/full_value.detach().clone()`。

Jupiter 对照：

- `Jupiter/tasks/medusa_llama/kv_cache.py` 的 `KVCache.cat()` 是蓝本。
- 它不是重新 `torch.cat`，而是 `narrow -> copy_ -> current_length += append_len`。
- 我们不直接复制模块，而是做适配 Qwen3-VL `PP / TP / HYBRID` 的 stage-local / rank-local 版本。

实现顺序：

1. 新增轻量 KV cache 对象：
   - 建议文件：`models/qwen3vl/execution/kv_cache.py`。
   - `LayerKVCache`：管理单层 key/value buffer。
   - `StageKVCache`：管理本 stage 的多个 layer。
2. 第一版只支持 runtime-only generate：
   - `batch_size=1` 或当前 smoke 覆盖的 batch。
   - `max_seq_len = prefill_seq_len + max_new_tokens`。
   - 不碰 replay/capture/debug。
3. 在 prefill 后创建或填充 `StageKVCache`：
   - PP：每个 PP stage 只保存本 stage layer range。
   - TP：每个 TP rank 只保存本地 KV head shard。
   - HYBRID：每个 stage 内每个 TP rank 都本地保存自己的 shard。
4. attention 路径加 opt-in cache 参数：
   - 有 `StageKVCache` 时走 `append/get_view`。
   - 没有 cache 时保留当前 `cache_by_layer` + `torch.cat` 旧路径。
5. stage trace 更新：
   - 不再把 `full_key/full_value` clone 回 `cache_by_layer`。
   - 返回同一个 cache handle 或轻量 cache metadata。

验收：

- `tp-text-generate` / `tp-mm-generate` generated ids/text 不变。
- `hybrid-mm-generate` generated ids/text 不变。
- `weight_load.tp_weight_sharded=true` 不回退。
- decode 阶段 `torch.cat([past, current])` 次数下降到 0。
- `full_key/full_value.detach().clone()` 次数下降到 0。
- 长 decode 下 allocation 或 elapsed 有解释；即使 CUDA peak 因预分配上升，也要能说明原因。

### 20B. InfiniPot-V-style 视觉 token KV 压缩

第二阶段再做。

目标：

- 关注视频输入里的时间冗余和空间语义重要性。
- 对视觉 token 的 KV 做压缩或筛选。
- 在 cache budget 下优先保留重要 token。

边界：

- 先做 opt-in，不影响默认 correctness baseline。
- 不让 non-input-owner 重新跑视觉 frontend。
- 保留默认完整 KV 路径作为 correctness guard。

### 20C. ReKV-style 历史窗口检索回取

第三阶段再做。

目标：

- 长视频场景下不让 GPU KV 无限增长。
- query 到来时检索相关历史窗口。
- 只把必要 KV 回取到模型上下文。

边界：

- 先规划全局检索 + 本地/远端回取接口。
- 再决定 RAM/磁盘/远端存储策略。
- 不影响短视频默认路径。

当前不考虑：

- vLLM-style BlockPool / prefix cache / serving scheduler。
- paged/block cache 只有在 20A/20B/20C 稳定后再重新评估。

## 后续队列

| step | 任务 | 目标 | 验收重点 |
| --- | --- | --- | --- |
| 17（这条先不需要做） | PP handoff overlap | 尝试让 PP handoff 与本地计算重叠 | correctness 不变，wait 时间下降，无死锁 |
| 18（这条先不需要做） | Stage partition 搜索 | 搜索更好的 stage range，支持异构 Jetson | 每组有 correctness + perf |
| 19（这条先不需要做） | Embedding / lm_head vocab parallelism | 研究 vocab parallel embedding / lm_head | token/logits/top-k 语义稳定 |
| 20 | KV cache 管理部分 | Jupiter-style 连续 KV、InfiniPot-V 压缩、ReKV 检索 | generated ids/text 不变，decode cat/clone 下降，长视频 KV 增长可控 |
| 21 | Serving engine 方向 | scheduler、paged KV、batching、streaming | correctness/baseline 稳定后再推进 |

## 固定规则

- KV cache 当前阶段先不对照 vLLM serving 体系，重点放在 Jupiter / InfiniPot-V / ReKV 三条路线。
- 其他步骤如果需要对照 vLLM，写清楚哪些能照搬、哪些不能照搬。
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
