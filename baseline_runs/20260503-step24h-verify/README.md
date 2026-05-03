# Step 22 Smoke Matrix Baseline

| item | value |
| --- | --- |
| created_utc | 2026-05-03 04:15:58 |
| include_optional | 0 |
| selected_case_ids | `pp-mm-generate tp-mm-generate hybrid-mm-generate` |
| model | `/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct` |
| frame_dir | `/mnt/ssd/code/Qwen3_vl/frames` |
| tp_hosts | `local 10.126.126.3` |
| pp_hosts | `local 10.126.126.3` |
| pp3_hosts | `local 10.126.126.3 10.126.126.4` |
| hybrid_hosts | `local 10.126.126.3 10.126.126.4` |

## Perf Table

| case | rank | total s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | CUDA peak | loaded weights | stage KV bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pp-mm-generate | 0 | 30.41 | 3.06 MiB | 0 B | 3.06 MiB | 0.00 | 0 B | 5.37 GiB | 4.11 GiB | 43.88 MiB / 44.16 MiB |
| pp-mm-generate | 1 | 30.51 | 3.06 MiB | 0 B | 3.06 MiB | 0.00 | 0 B | 5.46 GiB | 4.11 GiB | 43.88 MiB / 44.16 MiB |
| tp-mm-generate | 0 | 53.06 | 11.50 MiB | 0 B | 0 B | 24.36 | 220.43 MiB | 6.52 GiB | 4.83 GiB | 43.88 MiB / 44.16 MiB |
| tp-mm-generate | 1 | 53.18 | 11.50 MiB | 0 B | 0 B | 23.77 | 220.43 MiB | 6.52 GiB | 4.83 GiB | 43.88 MiB / 44.16 MiB |
| hybrid-mm-generate | 0 | 44.73 | 3.06 MiB | 11.50 MiB | 3.06 MiB | 13.23 | 113.28 MiB | 3.73 GiB | 2.42 GiB | 21.94 MiB / 22.08 MiB |
| hybrid-mm-generate | 1 | 44.90 | 0 B | 11.50 MiB | 0 B | 12.93 | 113.28 MiB | 3.23 GiB | 2.42 GiB | 21.94 MiB / 22.08 MiB |
| hybrid-mm-generate | 2 | 44.81 | 3.06 MiB | 0 B | 3.06 MiB | 0.00 | 0 B | 5.46 GiB | 4.11 GiB | 43.88 MiB / 44.16 MiB |

## Files

- `check-smoke-matrix.txt`
- `collect-runtime-perf.txt`
- `runtime-perf-records.json`
- `runtime-perf-table.md`
