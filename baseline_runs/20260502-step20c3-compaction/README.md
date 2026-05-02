# Step 20C-3 Opt-in Video KV Compaction

本轮验证 `--video-kv-compression uniform --video-kv-keep-ratio 0.5`。默认 `none` 路径不修改 KV；本轮只冻结 opt-in 路径的真实 Jetson smoke。

## 结果

| case | rank | stage | total s | CUDA peak | TP coll | startup/scaffold/handoff | active KV before -> after | allocated KV | prefill len | video tokens orig/keep/drop | generated |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | --- | --- |
| `tp-mm-generate-step20c3-uniform-j23` | 0 | 0 | `53.69` | `6.53 GiB` | `24.72s / 221.48 MiB` | `11.51 / 0.00 / 0.00 MiB` | `46,227,456 -> 24,993,792` | `46,522,368` | `627 -> 339` | `576 / 288 / 288` | `视频中，一名` |
| `tp-mm-generate-step20c3-uniform-j23` | 1 | 0 | `53.69` | `6.52 GiB` | `24.10s / 221.48 MiB` | `11.51 / 0.00 / 0.00 MiB` | `46,227,456 -> 24,993,792` | `46,522,368` | `627 -> 339` | `576 / 288 / 288` | `视频中，一名` |
| `hybrid-mm-generate-step20c3-uniform-j23shared` | 0 | 0 | `32.37` | `3.73 GiB` | `2.05s / 113.82 MiB` | `3.07 / 11.51 / 3.08 MiB` | `23,113,728 -> 12,496,896` | `23,261,184` | `627 -> 339` | `576 / 288 / 288` | `视频中，一名` |
| `hybrid-mm-generate-step20c3-uniform-j23shared` | 1 | 0 | `32.44` | `3.23 GiB` | `1.64s / 113.82 MiB` | `0.00 / 11.51 / 0.00 MiB` | `23,113,728 -> 12,496,896` | `23,261,184` | `627 -> 339` | `576 / 288 / 288` | `视频中，一名` |
| `hybrid-mm-generate-step20c3-uniform-j23shared` | 2 | 1 | `32.49` | `5.47 GiB` | `0.00s / 0.00 MiB` | `3.07 / 0.00 / 3.08 MiB` | `46,227,456 -> 24,993,792` | `46,522,368` | `627 -> 339` | `576 / 288 / 288` | `视频中，一名` |

Generated ids 均为 `[87140, 15946, 3837, 101177]`。

## 结论

- `video_kv_compaction.schema=video_kv_compaction_v1` 已在 TP/HYBRID rank log 中冻结。
- `uniform` opt-in 会物理 compact 本地 `StageKVCache`，video tokens 从 `576` 保留 `288`。
- active KV bytes 约减半；allocated KV bytes 不变，因为第一版仍保留预分配 buffer。
- compact 后 prefill physical length 是 `339`，logical uncompressed length 仍按 `627` 推 decode position。
- TP/HYBRID 每个 rank 只压缩自己的 local KV shard；没有新增 dense KV broadcast，non-input-owner 不重新跑视觉 frontend。
- 本次短 smoke 输出与固定 multimodal baseline 一致。opt-in 压缩未来在更长输出上允许有质量差异，需要单独记录。
