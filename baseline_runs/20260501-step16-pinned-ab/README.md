# Step 16 Pinned Memory A/B

日期：2026-05-01

目的：验证 `--transport-pin-memory` 只作为 opt-in 实验时，correctness、CUDA peak 和 transport time 是否可接受。

## 环境说明

- jetson1 当前 `torch.cuda.is_available()` 为 `False`，报 `NvRmMemInitNvmap failed`。
- 纯 TP A/B 使用 jetson2 + jetson3，各 1 个 rank，是真实 2-GPU TP。
- HYBRID A/B 使用 jetson2 承载 rank0/rank1、jetson3 承载 rank2，属于功能性验证，不作为 3-GPU 性能 baseline。
- jetson1 失败试跑未纳入本目录结果；有效 case 都带 `j23` 或 `j23shared` 后缀。

## 结果摘要

| case | ranks | total s | TP coll s | startup/scaffold/handoff | CUDA peak | pin used | correctness |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| `tp-mm-generate-step16-default-j23` | 0/1 | `53.47 / 53.21` | `24.34 / 23.76` | startup `11.51 MiB` | `6.53 / 6.52 GiB` | `0 / 0` | pass |
| `tp-mm-generate-step16-pinned-j23` | 0/1 | `53.01 / 52.97` | `23.91 / 23.51` | startup `11.51 MiB` | `6.53 / 6.52 GiB` | `289 / 289 events` | pass |
| `hybrid-mm-generate-step16-default-j23shared` | 0/1/2 | `32.69 / 32.74 / 32.85` | `2.25 / 1.58 / 0.00` | payload bytes unchanged | `3.73 / 3.22 / 5.46 GiB` | `0 / 0 / 0` | pass |
| `hybrid-mm-generate-step16-pinned-j23shared` | 0/1/2 | `32.80 / 32.85 / 32.96` | `2.27 / 1.49 / 0.00` | payload bytes unchanged | `3.73 / 3.22 / 5.46 GiB` | `154 / 149 / 1 events` | pass |

固定输出：

- generated ids：`[87140, 15946, 3837, 101177]`
- generated text：`视频中，一名`

## 结论

- `--transport-pin-memory` 不改变 payload bytes、generated ids/text、weight shard scope。
- 纯 TP 上 pinned run 的 total time 小幅下降约 `0.24-0.46s`，TP collective time 小幅下降约 `0.25-0.43s`。
- HYBRID 功能性验证通过，但因为 rank0/rank1 共用 jetson2 GPU，不作为性能结论。
- CUDA peak allocated 不上升；TP rank0 reserved 增加约 `2 MiB`，属于 pinned staging 相关小波动。
- 当前收益不大，保持默认关闭。

详细表：`runtime-perf-table.md`
详细 records：`runtime-perf-records.json`
