# 20260501 Step 20A KV cache long decode

Purpose:

- Add a longer decode profile for Jupiter-style `StageKVCache`.
- Run the same `tp-mm-generate` 16-token case as the frozen long baseline.
- Record generated ids/text, elapsed time, CUDA peak, TP collective, and `stage_kv_cache.tensor_bytes`.

Setup:

- jetson2 rank0: `10.126.126.3`
- jetson3 rank1: `10.126.126.4`
- model: `/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct`
- frames: `/mnt/ssd/code/Qwen3_vl/frames`
- `MAX_NEW_TOKENS=16`
- `MASTER_PORT=29633`

Command shape:

```bash
CASE_ID=tp-mm-generate-long-step20a NODE_RANK=0 NNODES=2 MASTER_ADDR=10.126.126.3 MASTER_PORT=29633 MAX_NEW_TOKENS=16 OUT=... bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh
CASE_ID=tp-mm-generate-long-step20a NODE_RANK=1 NNODES=2 MASTER_ADDR=10.126.126.3 MASTER_PORT=29633 MAX_NEW_TOKENS=16 OUT=... bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh
```

Correctness:

| case | generated ids | generated text |
| --- | --- | --- |
| `tp-mm-generate-long-step20a` | `[87140, 15946, 3837, 101177, 105611, 99194, 38035, 113727, 33108, 104362, 38035, 113233, 9370, 104253, 104224, 46944]` | `视频中，一名穿着深色衬衫和浅色裤子的男子站在一个` |

KV cache summary:

| rank | prefill seq | max new | max seq | final length | layers | append count | `stage_kv_cache.tensor_bytes` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `627` | `16` | `643` | `642` | `36` | `576` | `47,407,104` |
| 1 | `627` | `16` | `643` | `642` | `36` | `576` | `47,407,104` |

Performance:

| rank | total s | CUDA peak alloc/reserved | TP coll s | TP coll bytes | loaded weights | TP shard |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `62.98` | `6.53 / 6.60 GiB` | `29.17` | `225.70 MiB` | `4.83 GiB` | `true` |
| 1 | `62.94` | `6.52 / 6.62 GiB` | `28.51` | `225.70 MiB` | `4.83 GiB` | `true` |

Comparison with `baseline_runs/20260430-bfloat16-default/`:

| case | rank0 total | rank1 total | rank0 TP coll | rank1 TP coll | CUDA peak | TP coll bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old `tp-mm-generate-long-default-bfloat16` | `62.66` | `62.56` | `28.84` | `28.07` | `6.53 / 6.52 GiB` | `225.70 MiB` |
| Step 20A long decode | `62.98` | `62.94` | `29.17` | `28.51` | `6.53 / 6.52 GiB` | `225.70 MiB` |

Conclusion:

- Generated ids/text are unchanged.
- TP collective bytes are unchanged.
- CUDA peak is effectively unchanged.
- `stage_kv_cache.tensor_bytes` grows with `max_seq_len`: short decode `46,522,368` bytes at `max_seq=631`, long decode `47,407,104` bytes at `max_seq=643`.
- The small elapsed delta is within current Jetson/Gloo variability; no regression signal from this profile.
- Final `current_length=642` is expected: prefill predicts the first generated token, then decode appends the remaining 15 tokens.

Artifacts:

- `runtime-perf-records.json`
- `runtime-perf-table.md`
- `tp-mm-generate-long-step20a-rank*.log`
- `*.ssh.stdout`
- `*.ssh.stderr` (empty)
