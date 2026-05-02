# Step 20C-4 InfiniPot-V Selector

本轮加入 opt-in `--video-kv-compression infinipot-v`。

核心变化：

- selector 从本地 `StageKVCache` 的 K/V 计算分数，不重跑视觉 frontend。
- value-norm 负责全局重要性，TaR recent-query similarity 给近期 query 相关 token 加权。
- 当前 runtime compaction contract 要求 stage/rank 内共享一份 token list，所以这里把 local layer/head score 聚合成 rank-local shared selection。
- 默认 `--video-kv-compression none` 不变。

## 验证

| case | rank | stage | total s | CUDA peak | TP coll | startup/scaffold/handoff | active KV before -> after | video tokens orig/keep/drop | score layers | generated |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | ---: | --- |
| `tp-mm-generate-step20c4-infinipot-j23` | 0 | 0 | `53.52` | `6.54 GiB` | `24.48s / 221.48 MiB` | `11.51 MiB / 0 B / 0 B` | `46,227,456 -> 24,993,792` | `576 / 288 / 288` | `36` | pass |
| `tp-mm-generate-step20c4-infinipot-j23` | 1 | 0 | `53.52` | `6.53 GiB` | `23.68s / 221.48 MiB` | `11.51 MiB / 0 B / 0 B` | `46,227,456 -> 24,993,792` | `576 / 288 / 288` | `36` | pass |
| `hybrid-mm-generate-step20c4-infinipot-j23shared` | 0 | 0 | `32.81` | `3.73 GiB` | `2.11s / 113.82 MiB` | `3.07 MiB / 11.51 MiB / 3.08 MiB` | `23,113,728 -> 12,496,896` | `576 / 288 / 288` | `18` | pass |
| `hybrid-mm-generate-step20c4-infinipot-j23shared` | 1 | 0 | `32.90` | `3.23 GiB` | `1.67s / 113.82 MiB` | `0 B / 11.51 MiB / 0 B` | `23,113,728 -> 12,496,896` | `576 / 288 / 288` | `18` | pass |
| `hybrid-mm-generate-step20c4-infinipot-j23shared` | 2 | 1 | `32.90` | `5.49 GiB` | `0.00s / 0 B` | `3.07 MiB / 0 B / 3.08 MiB` | `46,227,456 -> 24,993,792` | `576 / 288 / 288` | `18` | pass |

固定输出：

- generated ids：`[87140, 15946, 3837, 101177]`
- generated text：`视频中，一名`

## 对比 uniform

| 项目 | `uniform` | `infinipot-v` |
| --- | --- | --- |
| token budget | `576 -> 288` | `576 -> 288` |
| active KV bytes | 约减半 | 约减半 |
| allocated KV bytes | 不变 | 不变 |
| token selection | 均匀采样，例如 `[10,11], [12,13]...` | 本地 K/V 打分，例如 rank0 第一窗口 `[13,15], [24,25], [30,31]...` |
| frontend | 不重跑 | 不重跑 |
| dense KV broadcast | 无新增 | 无新增 |
| short smoke 输出 | 不变 | 不变 |

结论：

- 20C-4 已具备 InfiniPot-V-style opt-in selector 和物理 compaction。
- 当前只证明短 smoke correctness 和资源统计；质量收益需要后续更长视频、更长 decode 和更丰富问题集评估。
- allocated KV 暂时不下降，因为 20A 第一版仍预分配整块 buffer，本轮只收紧 `current_length` 和 active KV。
