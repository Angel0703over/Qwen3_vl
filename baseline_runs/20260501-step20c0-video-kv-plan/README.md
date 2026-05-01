# Step 20C-0 Video KV Compression Planner

Date: 2026-05-01

Purpose:

- Freeze planner-only `video_kv_compression_plan` logs on real Jetsons.
- Confirm default multimodal generate output is unchanged.
- Confirm no KV compression, KV deletion, or remote KV fetch is introduced.

Run layout:

- `tp-mm-generate-step20c0-j23`: rank0 on jetson2, rank1 on jetson3.
- `hybrid-mm-generate-step20c0-j23shared`: rank0/rank1 on jetson2, rank2 on jetson3.

Correctness:

| case | ranks | generated ids | generated text |
| --- | ---: | --- | --- |
| `tp-mm-generate-step20c0-j23` | 2 | `[87140, 15946, 3837, 101177]` | `视频中，一名` |
| `hybrid-mm-generate-step20c0-j23shared` | 3 | `[87140, 15946, 3837, 101177]` | `视频中，一名` |

Planner summary:

| case | rank | method | windows | original tokens | keep tokens | drop tokens | estimated original KV bytes | estimated keep KV bytes | plan metadata bytes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tp-mm-generate-step20c0-j23` | 0 | `none` | 4 | 576 | 576 | 0 | 42,467,328 | 42,467,328 | 3,463 |
| `tp-mm-generate-step20c0-j23` | 1 | `none` | 4 | 576 | 576 | 0 | 42,467,328 | 42,467,328 | 3,463 |
| `hybrid-mm-generate-step20c0-j23shared` | 0 | `none` | 4 | 576 | 576 | 0 | 21,233,664 | 21,233,664 | 3,455 |
| `hybrid-mm-generate-step20c0-j23shared` | 1 | `none` | 4 | 576 | 576 | 0 | 21,233,664 | 21,233,664 | 3,455 |
| `hybrid-mm-generate-step20c0-j23shared` | 2 | `none` | 4 | 576 | 576 | 0 | 42,467,328 | 42,467,328 | 3,467 |

Notes:

- `video_kv_compression_plan.schema=video_kv_compression_plan_v1`.
- `planner_only=true`, `mutates_kv=false`, `compression_enabled=false`.
- Default method is `none`, so this run only records keep budget and estimated bytes.
- `video_window_cache` remains 4 windows / 576 video tokens on every rank.

Performance:

- See `runtime-perf-table.md`.
- Machine-readable records: `runtime-perf-records.json`.

Validation commands:

```bash
PYTHONPATH=/mnt/ssd/code/Qwen3_vl /mnt/ssd/miniconda3/envs/vlm/bin/python \
  qwen3vl_tp_runtime/scripts/check_baseline_logs.py \
  --case-id tp-mm-generate-step20c0-j23 \
  baseline_runs/20260501-step20c0-video-kv-plan/tp-mm-generate-step20c0-j23-rank0.log \
  baseline_runs/20260501-step20c0-video-kv-plan/tp-mm-generate-step20c0-j23-rank1.log

PYTHONPATH=/mnt/ssd/code/Qwen3_vl /mnt/ssd/miniconda3/envs/vlm/bin/python \
  qwen3vl_tp_runtime/scripts/check_baseline_logs.py \
  --case-id hybrid-mm-generate-step20c0-j23shared \
  baseline_runs/20260501-step20c0-video-kv-plan/hybrid-mm-generate-step20c0-j23shared-rank0.log \
  baseline_runs/20260501-step20c0-video-kv-plan/hybrid-mm-generate-step20c0-j23shared-rank1.log \
  baseline_runs/20260501-step20c0-video-kv-plan/hybrid-mm-generate-step20c0-j23shared-rank2.log
```
