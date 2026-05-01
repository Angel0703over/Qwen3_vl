# 20260501 Step 20B Video Window Cache

Purpose:

- Verify `VideoWindowCacheIndex` on real Jetsons.
- Record video window metadata without changing generation semantics.
- Confirm no KV compression, deletion, or cross-node KV fetch is introduced.

Setup:

- jetson2 rank0/rank1 for HYBRID stage0 TP, jetson3 rank2 for HYBRID stage1.
- jetson2 rank0 and jetson3 rank1 for pure TP.
- model: `/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct`
- frames: `/mnt/ssd/code/Qwen3_vl/frames`
- `MAX_NEW_TOKENS=4`
- ports: TP `29640`, HYBRID `29641`

Correctness:

| case | generated ids | generated text |
| --- | --- | --- |
| `tp-mm-generate-step20b` | `[87140, 15946, 3837, 101177]` | `视频中，一名` |
| `hybrid-mm-generate-step20b` | `[87140, 15946, 3837, 101177]` | `视频中，一名` |

Window metadata:

| case | rank | stage | TP rank/degree | windows | video tokens | metadata bytes | token ranges | KV owner/layers |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `tp-mm-generate-step20b` | 0 | 0 | `0/2` | 4 | 576 | 2027 | `10:154 ... 466:610` | rank0 / `0:35` |
| `tp-mm-generate-step20b` | 1 | 0 | `1/2` | 4 | 576 | 2027 | `10:154 ... 466:610` | rank1 / `0:35` |
| `hybrid-mm-generate-step20b` | 0 | 0 | `0/2` | 4 | 576 | 2027 | `10:154 ... 466:610` | rank0 / `0:17` |
| `hybrid-mm-generate-step20b` | 1 | 0 | `1/2` | 4 | 576 | 2027 | `10:154 ... 466:610` | rank1 / `0:17` |
| `hybrid-mm-generate-step20b` | 2 | 1 | `0/1` | 4 | 576 | 2031 | `10:154 ... 466:610` | rank2 / `18:35` |

Performance:

| case | rank | total s | CUDA peak | TP coll s | TP coll bytes | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tp-mm-generate-step20b` | 0 | `53.28` | `6.53 GiB` | `24.51` | `221.48 MiB` | `4.83 GiB` |
| `tp-mm-generate-step20b` | 1 | `53.22` | `6.52 GiB` | `23.80` | `221.48 MiB` | `4.83 GiB` |
| `hybrid-mm-generate-step20b` | 0 | `33.00` | `3.73 GiB` | `2.34` | `113.82 MiB` | `2.42 GiB` |
| `hybrid-mm-generate-step20b` | 1 | `33.15` | `3.23 GiB` | `1.64` | `113.82 MiB` | `2.42 GiB` |
| `hybrid-mm-generate-step20b` | 2 | `33.13` | `5.47 GiB` | `0.00` | `0 B` | `4.11 GiB` |

Conclusion:

- `video_window_cache` appears in prefill summary on every rank.
- It records window count, token range, frame/time range, and local KV location.
- `StageKVCache` behavior is unchanged.
- CUDA peak and TP collective bytes match Step 20A-level behavior.

Artifacts:

- `runtime-perf-records.json`
- `runtime-perf-table.md`
- `tp-mm-generate-step20b-rank*.log`
- `hybrid-mm-generate-step20b-rank*.log`
- `*.ssh.stdout`
- `*.ssh.stderr` (empty)
