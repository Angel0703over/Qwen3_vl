# Step 22 Smoke Matrix Baseline

| item | value |
| --- | --- |
| created_utc | 2026-05-02 07:05:36 |
| include_optional | 0 |
| selected_case_ids | `hf-text-generate hf-mm-generate pp-mm-generate tp-mm-generate tp-mm-generate-long tp-mm-generate-frame-regression` |
| model | `/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct` |
| frame_dir | `/mnt/ssd/code/Qwen3_vl/frames` |
| tp_hosts | `10.126.126.3 10.126.126.4` |
| pp_hosts | `10.126.126.3 10.126.126.4` |
| hybrid_hosts | not used in this 2-node subset |

## Notes

- 这是 Step 22 的 2-node Jetson smoke baseline 子集，实际 CUDA ranks 使用 `10.126.126.3` / `10.126.126.4`。
- 覆盖了 `hf-text-generate`、`hf-mm-generate`、`pp-mm-generate`、`tp-mm-generate`、`tp-mm-generate-long`、`tp-mm-generate-frame-regression`。
- `check-smoke-matrix.txt` 已验证 generated ids/text、rank count、transport metrics、consume-only 和 TP shard 关键字段。
- 当前未跑 required matrix 里的 3-rank `hybrid-mm-generate`：这次由 Codex tool shell 作为 controller 启动，而该 shell 看不到 jetson1 的 CUDA device nodes；物理 jetson1 普通终端 CUDA 可用。后续可用普通登录/SSH 方式把 jetson1 作为第三 CUDA rank 补跑完整矩阵。

## Perf Table

| case | rank | total s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | CUDA peak | loaded weights | stage KV bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hf-text-generate | - | 8.76 | 0 B | 0 B | 0 B | 0.00 | 0 B | 8.28 GiB | - | - |
| hf-mm-generate | - | 10.92 | 0 B | 0 B | 0 B | 0.00 | 0 B | 8.55 GiB | - | - |
| pp-mm-generate | 0 | 30.25 | 3.07 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.39 GiB | 4.11 GiB | 44.09 MiB / 44.37 MiB |
| pp-mm-generate | 1 | 30.29 | 3.07 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.47 GiB | 4.11 GiB | 44.09 MiB / 44.37 MiB |
| tp-mm-generate | 0 | 52.95 | 11.51 MiB | 0 B | 0 B | 24.42 | 221.48 MiB | 6.53 GiB | 4.83 GiB | 44.09 MiB / 44.37 MiB |
| tp-mm-generate | 1 | 52.99 | 11.51 MiB | 0 B | 0 B | 23.83 | 221.48 MiB | 6.52 GiB | 4.83 GiB | 44.09 MiB / 44.37 MiB |
| tp-mm-generate-long | 0 | 62.03 | 11.51 MiB | 0 B | 0 B | 28.58 | 225.70 MiB | 6.53 GiB | 4.83 GiB | 44.09 MiB / 45.21 MiB |
| tp-mm-generate-long | 1 | 62.05 | 11.51 MiB | 0 B | 0 B | 27.89 | 225.70 MiB | 6.52 GiB | 4.83 GiB | 44.09 MiB / 45.21 MiB |
| tp-mm-generate-frame-regression | 0 | 52.96 | 11.51 MiB | 0 B | 0 B | 24.28 | 221.48 MiB | 6.53 GiB | 4.83 GiB | 44.09 MiB / 44.37 MiB |
| tp-mm-generate-frame-regression | 1 | 52.97 | 11.51 MiB | 0 B | 0 B | 23.58 | 221.48 MiB | 6.52 GiB | 4.83 GiB | 44.09 MiB / 44.37 MiB |

## Files

- `check-smoke-matrix.txt`
- `collect-runtime-perf.txt`
- `runtime-perf-records.json`
- `runtime-perf-table.md`
