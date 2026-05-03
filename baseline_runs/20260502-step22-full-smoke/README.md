# Step 22 Smoke Matrix Baseline

| item | value |
| --- | --- |
| created_utc | 2026-05-02 08:08:56 |
| include_optional | 0 |
| model | `/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct` |
| frame_dir | `/mnt/ssd/code/Qwen3_vl/frames` |
| tp_hosts | `local 10.126.126.3` |
| pp_hosts | `local 10.126.126.3` |
| pp3_hosts | `local 10.126.126.3 10.126.126.4` |
| hybrid_hosts | `local 10.126.126.3 10.126.126.4` |

## Notes

- 这是 Step 22 required smoke matrix 的完整 baseline。
- 覆盖了 `hf-text-generate`、`hf-mm-generate`、`pp-mm-generate`、`pp3-mm-generate`、`tp-mm-generate`、`hybrid-mm-generate`、`tp-mm-generate-long`、`tp-mm-generate-frame-regression`。
- `pp3-mm-generate` 已使用 3 ranks：jetson1 rank0、jetson2 rank1、jetson3 rank2。
- `hybrid-mm-generate` 已使用 3 ranks：jetson1 rank0、jetson2 rank1、jetson3 rank2。
- `check-smoke-matrix.txt` 已验证 generated ids/text、rank count、transport metrics、consume-only 和 TP shard 关键字段。

## Perf Table

| case | rank | total s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | CUDA peak | loaded weights | stage KV bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hf-text-generate | - | 28.69 | 0 B | 0 B | 0 B | 0.00 | 0 B | 8.28 GiB | - | - |
| hf-mm-generate | - | 11.60 | 0 B | 0 B | 0 B | 0.00 | 0 B | 8.55 GiB | - | - |
| pp-mm-generate | 0 | 30.60 | 3.07 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.39 GiB | 4.11 GiB | 44.09 MiB / 44.37 MiB |
| pp-mm-generate | 1 | 30.80 | 3.07 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.47 GiB | 4.11 GiB | 44.09 MiB / 44.37 MiB |
| pp3-mm-generate | 0 | 31.57 | 6.14 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 3.86 GiB | 2.98 GiB | 29.39 MiB / 29.58 MiB |
| pp3-mm-generate | 1 | 31.67 | 3.07 MiB | 0 B | 6.15 MiB | 0.00 | 0 B | 3.13 GiB | 2.26 GiB | 29.39 MiB / 29.58 MiB |
| pp3-mm-generate | 2 | 31.67 | 3.07 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 3.95 GiB | 2.98 GiB | 29.39 MiB / 29.58 MiB |
| tp-mm-generate | 0 | 52.95 | 11.51 MiB | 0 B | 0 B | 24.28 | 221.48 MiB | 6.53 GiB | 4.83 GiB | 44.09 MiB / 44.37 MiB |
| tp-mm-generate | 1 | 53.08 | 11.51 MiB | 0 B | 0 B | 23.73 | 221.48 MiB | 6.52 GiB | 4.83 GiB | 44.09 MiB / 44.37 MiB |
| hybrid-mm-generate | 0 | 45.03 | 3.07 MiB | 11.51 MiB | 3.08 MiB | 13.26 | 113.82 MiB | 3.73 GiB | 2.42 GiB | 22.04 MiB / 22.18 MiB |
| hybrid-mm-generate | 1 | 45.24 | 0 B | 11.51 MiB | 0 B | 12.98 | 113.82 MiB | 3.23 GiB | 2.42 GiB | 22.04 MiB / 22.18 MiB |
| hybrid-mm-generate | 2 | 45.28 | 3.07 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.47 GiB | 4.11 GiB | 44.09 MiB / 44.37 MiB |
| tp-mm-generate-long | 0 | 62.30 | 11.51 MiB | 0 B | 0 B | 28.98 | 225.70 MiB | 6.53 GiB | 4.83 GiB | 44.09 MiB / 45.21 MiB |
| tp-mm-generate-long | 1 | 62.36 | 11.51 MiB | 0 B | 0 B | 28.33 | 225.70 MiB | 6.52 GiB | 4.83 GiB | 44.09 MiB / 45.21 MiB |
| tp-mm-generate-frame-regression | 0 | 53.05 | 11.51 MiB | 0 B | 0 B | 24.58 | 221.48 MiB | 6.53 GiB | 4.83 GiB | 44.09 MiB / 44.37 MiB |
| tp-mm-generate-frame-regression | 1 | 53.23 | 11.51 MiB | 0 B | 0 B | 23.83 | 221.48 MiB | 6.52 GiB | 4.83 GiB | 44.09 MiB / 44.37 MiB |

## Files

- `check-smoke-matrix.txt`
- `collect-runtime-perf.txt`
- `runtime-perf-records.json`
- `runtime-perf-table.md`
