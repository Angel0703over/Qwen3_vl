# 20260501 Step 20A KV cache smoke

Purpose:

- Verify Jupiter-style `StageKVCache` on real Jetsons.
- Confirm generated ids/text stay unchanged.
- Record elapsed time, CUDA peak, and `stage_kv_cache.tensor_bytes`.

Setup:

- jetson2: `10.126.126.3`
- jetson3: `10.126.126.4`
- model: `/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct`
- frames: `/mnt/ssd/code/Qwen3_vl/frames`
- `MAX_NEW_TOKENS=4`

Cases:

| case | ranks | ports | result |
| --- | ---: | --- | --- |
| `tp-text-generate-step20a` | 2 | `29630` | PASS |
| `tp-mm-generate-step20a` | 2 | `29631` | PASS |
| `hybrid-mm-generate-step20a` | 3 | `29632` | PASS |

Correctness:

| case | generated ids | generated text |
| --- | --- | --- |
| `tp-text-generate-step20a` | `[104455, 9909, 9286, 16488]` | `人工智能（Artificial` |
| `tp-mm-generate-step20a` | `[87140, 15946, 3837, 101177]` | `视频中，一名` |
| `hybrid-mm-generate-step20a` | `[87140, 15946, 3837, 101177]` | `视频中，一名` |

KV cache summary:

| case | rank | max seq | final length | layers | append count | `stage_kv_cache.tensor_bytes` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tp-text-generate-step20a` | 0 | `20` | `19` | `36` | `144` | `1,474,560` |
| `tp-text-generate-step20a` | 1 | `20` | `19` | `36` | `144` | `1,474,560` |
| `tp-mm-generate-step20a` | 0 | `631` | `630` | `36` | `144` | `46,522,368` |
| `tp-mm-generate-step20a` | 1 | `631` | `630` | `36` | `144` | `46,522,368` |
| `hybrid-mm-generate-step20a` | 0 | `631` | `630` | `18` | `72` | `23,261,184` |
| `hybrid-mm-generate-step20a` | 1 | `631` | `630` | `18` | `72` | `23,261,184` |
| `hybrid-mm-generate-step20a` | 2 | `631` | `630` | `18` | `72` | `46,522,368` |

Performance snapshot:

| case | rank | total s | CUDA peak | TP coll s | TP coll bytes | loaded weights | TP shard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tp-text-generate-step20a` | 0 | `9.07` | `4.91 GiB` | `2.14` | `6.68 MiB` | `4.83 GiB` | `true` |
| `tp-text-generate-step20a` | 1 | `8.96` | `4.91 GiB` | `2.08` | `6.68 MiB` | `4.83 GiB` | `true` |
| `tp-mm-generate-step20a` | 0 | `53.44` | `6.53 GiB` | `24.42` | `221.48 MiB` | `4.83 GiB` | `true` |
| `tp-mm-generate-step20a` | 1 | `53.36` | `6.52 GiB` | `23.77` | `221.48 MiB` | `4.83 GiB` | `true` |
| `hybrid-mm-generate-step20a` | 0 | `32.51` | `3.73 GiB` | `2.30` | `113.82 MiB` | `2.42 GiB` | `true` |
| `hybrid-mm-generate-step20a` | 1 | `32.55` | `3.23 GiB` | `1.60` | `113.82 MiB` | `2.42 GiB` | `true` |
| `hybrid-mm-generate-step20a` | 2 | `32.62` | `5.47 GiB` | `0.00` | `0 B` | `4.11 GiB` | `false` |

Notes:

- Text output and multimodal output match the frozen baseline.
- `stage_kv_cache.current_lengths` ends at `prefill_seq_len + 3` because prefill predicts the first generated token and decode runs the remaining three steps for `max_new_tokens=4`.
- CUDA peak is effectively unchanged versus the `bfloat16` default baseline. The preallocated KV bytes are visible in `stage_kv_cache.tensor_bytes`, but peak is still dominated by model weights and existing activation/runtime buffers.
- `hybrid-mm` rank2 has `tp_degree=1`; `tp_weight_sharded=false` and `TP coll bytes=0 B` are expected.

Artifacts:

- `runtime-perf-records.json`
- `runtime-perf-table.md`
- `*-rank*.log`
- `*.ssh.stdout`
- `*.ssh.stderr` (empty)
