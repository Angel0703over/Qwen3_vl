# qwen3vl_tp_runtime Roadmap

这份 Roadmap 只保留当前方向、已完成效果和下一步。详细数字看 `BASELINE.md`，完整交接看 `SESSION_HANDOFF.md`。

## 当前方向

- `PP / TP / HYBRID` 是主路径：启动时直接从 `model_path` 构建 `StageState`。
- `PP` 和 `TP` 是基础后端；`HYBRID` 是 PP+TP 组合层。
- 主路径只围绕 correctness-first runtime 继续推进。
- 当前阶段重点：KV cache 管理，不考虑 vLLM serving 体系。

## 当前状态

| 方向 | 状态 |
| --- | --- |
| direct runtime | `PP / TP / HYBRID` 已稳定 |
| TP 后端 | 已独立，不依赖 HYBRID |
| TP 权重 | decoder/MLP projection 已 rank-local materialize |
| multimodal startup | 不传 root/full/replay payload，不传 dense derived tensors |
| runtime input | pure TP 已避免 dense `stage_input` broadcast；HYBRID schema 已固化 |
| comm dtype | 默认 `bfloat16` |
| Step 15 payload | 已结束，owner/rebuild 语义已冻结 |
| Step 16 buffer/pinned | 已结束，`--transport-pin-memory` 保持 opt-in |
| Step 20A KV cache | 已结束，`StageKVCache` 真实 Jetson smoke 和 16-token long decode 通过 |
| Step 20B video window cache | 已结束，只记录窗口 metadata，不压缩/删除/回取 KV |
| Step 20C-0 video KV planner | 已结束，只记录 compression plan，不压缩/删除/回取 KV |
| Step 20C-1 video KV selector | 已结束，`uniform/swa` opt-in 只记录 selected token stats |

## 已完成效果

| 阶段 | 修改前 | 修改后 | 效果 |
| --- | --- | --- | --- |
| TP 独立后端 | TP 借用 HYBRID 路径 | 独立 `TensorParallelRunner` | 后端边界清楚 |
| startup contract | 带 reference `stage_output` 和 dense derived tensors | 只传必要 input/metadata | startup payload 下降 |
| HYBRID `tp_degree=1` | stage1 记录伪 TP collective | single-rank bypass | rank2 TP collective 归零 |
| TP comm dtype | 默认 `float32` | 默认 `bfloat16` | TP collective bytes 约减半 |
| pure TP runtime input | 广播 dense `stage_input` | 本地 embedding / local stage input | broadcast events 归零 |
| Step 15 payload | 空 slot 和部分 derived shared tensor 仍传输 | `None` slot 跳过，`attention_mask_2d/position_ids` 可重建时本地恢复 | payload 更薄 |
| Step 16 decode buffer | 每 step 新建小 tensor | 复用 decode mask/token buffer | 减少小分配 |
| Step 16 pinned memory | 无开关 | `--transport-pin-memory` opt-in | 小幅收益，默认关闭 |
| Step 20A KV cache | decode 用 `torch.cat([past,current])` 并 clone cache | `StageKVCache` append/view，旧路径保留 | correctness 不变，CUDA peak 基本持平 |
| Step 20B video window cache | 没有 window -> KV location 索引 | prefill log 记录 video window metadata | correctness 不变，后续可做窗口压缩/检索 |

## KV Cache 路线

当前按这四层推进：

| 模块 | 参考对象 | KV 管理位置 | 状态 | 核心作用 |
| --- | --- | --- | --- | --- |
| `StageKVCacheManager` | Jupiter | 每个 Jetson / stage / rank 本地都有 | 已完成：当前代码名是 `LayerKVCache / StageKVCache` | 管理本 stage/rank 的 K/V，替代零散 `cache_by_layer` |
| `VideoWindowCacheManager` | 自己设计 + ReKV 思想 | 每个 Jetson / stage / rank 本地元数据副本 | 已完成：当前代码名是 `VideoWindowCacheIndex` | 按时间窗口组织视频流 KV，建立 `window -> KV location` 映射 |
| 窗口内视觉 KV 压缩 | InfiniPot-V | 产生该窗口 KV 的 Jetson 本地执行 | 下一步 | 在单个窗口内筛选/压缩 visual token KV，控制窗口 KV 规模 |
| 历史窗口检索回取 | ReKV | 全局检索 + 本地/远端回取 | 后续 | query 来时找相关历史窗口，只回取必要 KV，避免 GPU cache 无限增长 |

## 20A. StageKVCacheManager

状态：已结束。

代码位置：

- `models/qwen3vl/execution/kv_cache.py`
- `LayerKVCache`
- `StageKVCache`

完成内容：

- runtime-only generate 创建 stage-local/rank-local KV buffer。
- prefill/decode 直接 append K/V，再通过 view 读取有效长度。
- 有 `StageKVCache` 时不再走 decode 每步 `torch.cat([past,current])`。
- 有 `StageKVCache` 时不再 clone `full_key/full_value` 回 `cache_by_layer`。
- 非 runtime-only 路径保留旧 `cache_by_layer` guard。

验证结果：

| case | 结果 |
| --- | --- |
| `tp-text-generate` | generated ids/text 不变 |
| `tp-mm-generate` | generated ids/text 不变 |
| `hybrid-mm-generate` | generated ids/text 不变 |
| `tp-mm-generate MAX_NEW_TOKENS=16` | generated ids/text 不变 |

Profile：

- 短 smoke：`baseline_runs/20260501-step20a-kv-cache-smoke/`
- 16-token long decode：`baseline_runs/20260501-step20a-kv-cache-long-decode/`
- 长 decode `stage_kv_cache.tensor_bytes=47,407,104` / rank。
- CUDA peak 与 `baseline_runs/20260430-bfloat16-default/` 基本持平。

## 20B. VideoWindowCacheManager

状态：已结束。

代码位置：

- `models/qwen3vl/execution/video_window_cache.py`
- `VideoWindowId`
- `VideoWindowMetadata`
- `KVLocation`
- `VideoWindowCacheIndex`

完成内容：

- 从 `mm_token_type_ids == 2` 的连续区间识别 video windows。
- 记录 token range、frame range、time range、grid metadata。
- 记录本 rank 的 KV location：owner rank、stage、layer range、TP rank、local KV offset。
- `PP / TP / HYBRID` runtime-only multimodal prefill stats 输出 `video_window_cache`。
- 只做观测：不删除 KV、不压缩 KV、不跨机回取 KV。

验证结果：

| case | 结果 | window metadata |
| --- | --- | --- |
| `tp-mm-generate` | generated ids/text 不变 | 每 rank `4` windows / `576` video tokens |
| `hybrid-mm-generate` | generated ids/text 不变 | 每 rank `4` windows / `576` video tokens |

Profile：

- `baseline_runs/20260501-step20b-video-window-cache/`
- `tp-mm` elapsed `53.22-53.28s`，CUDA peak `6.52-6.53 GiB`。
- `hybrid-mm` elapsed `33.00-33.15s`，CUDA peak 与 Step 20A 基本持平。
- metadata bytes：TP/HYBRID stage0 `2027` bytes，HYBRID stage1 `2031` bytes。

## 20C. 窗口内视觉 KV 压缩

状态：20C-0 / 20C-1 已冻结，下一步是 20C-2 compression contract。

目标：

- 在单个 video window 内筛选或压缩 visual token KV。
- 优先保留时间/空间上更重要的 token。
- 先做 opt-in，默认完整 KV 路径保持 correctness guard。

对照 InfiniPot-V：

- 源码参考：`InfiniPot-V/kvcache_utils.py::process_kv_cache` 和 `InfiniPot-V/qwen_inference_ovu.py::_block_wise_prefill`。
- InfiniPot-V 是 block-wise video prefill：每个视频块跑 vision frontend + LLM forward，然后压缩非最后块的视觉 KV。
- 它的策略是 `uniform`、`swa` 和 `infinipot-v`，其中 `infinipot-v` 结合 TaR recent-query similarity 和 value-norm。
- 我们第一阶段不改成 block-wise frontend；直接复用 20A `StageKVCache` 和 20B `VideoWindowCacheIndex`。
- 压缩只在产生该 KV 的本地 stage/rank/layer shard 上执行，不广播 dense KV，non-input-owner 不重新跑视觉 frontend。

建议接口：

- 新模块：`models/qwen3vl/execution/video_kv_compression.py`。
- CLI opt-in：当前支持 `--video-kv-compression none|uniform|swa`，默认 `none`；`infinipot-v` 放到 20C-4。
- budget 参数：先用 `--video-kv-keep-ratio` 或 `--video-kv-keep-tokens-per-window`，二选一后固定。

实现顺序：

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| 20C-0 planner/stats | 已新增窗口压缩 plan，只计算每个 window 的 keep budget、候选 token、预计 bytes，不改 KV | 真实 Jetson `tp-mm` / `hybrid-mm` 通过 |
| 20C-1 opt-in selector | 已实现 `uniform` 和 `swa` token 选择；只依赖 window token range 和本地 KV shape | `uniform` 真实 Jetson `tp-mm` / `hybrid-mm` 通过；`swa` 有单测 |
| 20C-2 compression contract | 物理压缩前先解决 attention mask 和 key length 对齐；明确压缩后 `StageKVCache.current_length`、past length、position 语义 | 有单测覆盖 mask/key 长度匹配；默认路径仍不变 |
| 20C-3 opt-in compaction | 在本地 `StageKVCache` 中压缩 visual token KV，保留 text/system/instruction KV | opt-in 有压缩率、CUDA peak、elapsed、输出差异记录 |
| 20C-4 InfiniPot-V selector | 加入 TaR + value-norm 选择器；score 来自本地 K/V，不重跑 frontend | 与 `uniform/swa` 对比质量和资源收益 |

20C-0 / 20C-1 只做统计，暂不物理删除 KV。
20C-2 是 20C-3 的前置协议闸门：当前 decode mask 仍按完整 prefill 长度构造，不能直接删除 visual KV。

验收：

- 默认路径 generated ids/text 不变。
- opt-in 压缩路径有明确压缩率、CUDA peak、输出差异记录。
- non-input-owner 不重新跑视觉 frontend。
- TP/HYBRID 下每个 rank 只压缩自己的 local KV shard；不新增跨 rank KV 传输。
- rank log 记录 before/after token count、KV bytes、selector、window id、layer range。

## 后续：20D. 历史窗口检索回取

状态：待规划。

目标：

- 长视频场景下，GPU 不无限保存所有历史窗口 KV。
- query 来时检索相关历史窗口。
- 只把必要 KV 回取到当前上下文。

验收：

- 默认短视频路径不受影响。
- 检索命中窗口、回取 bytes、回取耗时可观测。
- 支持本地回取，再考虑远端回取。

## 暂不推进

| step | 任务 | 原因 |
| --- | --- | --- |
| 17 | PP handoff overlap | 当前先做 KV 管理 |
| 18 | Stage partition 搜索 | KV 路线稳定后再做 |
| 19 | Embedding / lm_head vocab parallelism | 与当前 KV 管理无直接关系 |
| 21 | Serving engine | 暂不考虑 vLLM-style BlockPool / prefix cache / scheduler |

## 固定规则

- 主路径对象叫 `StageState`。
- `bundle` 只保留给 replay/debug/capture。
- `hexgen_core/modules/` 只放 `pipeline_parallel.py`、`tensor_parallel.py`、`hybrid_parallel.py`。
- HYBRID 可以调用 PP/TP helper；TP 不能反向依赖 HYBRID。
- payload/transport 改动必须记录 before/after keys、tensor count、bytes。
- 性能改动必须保留 before/after runtime records。
- 当前 KV 阶段优先 Jupiter / InfiniPot-V / ReKV 路线，不照搬 vLLM serving 体系。

## 常用同步

```bash
bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed
bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed
```
