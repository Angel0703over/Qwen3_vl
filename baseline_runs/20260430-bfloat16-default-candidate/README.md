# 2026-04-30 bfloat16 default-candidate regression

Purpose:

- Broaden `--comm-dtype bfloat16` coverage before considering it as the default.
- Keep the default unchanged in this round.
- Compare generated ids/text, TP collective bytes, and TP collective seconds.

Cases:

| case | max new tokens | dtype | ranks |
| --- | ---: | --- | ---: |
| `tp-text-generate-bfloat16` | 4 | `bfloat16` | 2 |
| `tp-mm-generate-bfloat16-wide` | 4 | `bfloat16` | 2 |
| `hybrid-mm-generate-bfloat16-wide` | 4 | `bfloat16` | 3 |
| `tp-mm-generate-long-default` | 16 | default `float32` | 2 |
| `tp-mm-generate-long-bfloat16` | 16 | `bfloat16` | 2 |

Topology:

- Pure TP: rank0 on `10.126.126.3`, rank1 on `10.126.126.4`.
- HYBRID: rank0/rank1 on `10.126.126.3`, rank2 on `10.126.126.4`, `--pp 2 --tp-degrees 2 1`.
- `MASTER_ADDR=10.126.126.3`.

Validation:

- `check-tp-text-generate-bfloat16.txt`: PASS.
- `check-tp-mm-generate-bfloat16-wide.txt`: PASS.
- `check-hybrid-mm-generate-bfloat16-wide.txt`: PASS.
- `check-tp-mm-generate-long-default.txt`: PASS.
- `check-tp-mm-generate-long-bfloat16.txt`: PASS.
- No non-empty `.ssh.stderr` files.

Short smoke comparison:

| case | rank | generated ids/text | TP collective bytes | TP collective seconds |
| --- | ---: | --- | ---: | ---: |
| default `tp-text-generate` | 0 | `[104455, 9909, 9286, 16488]` / `人工智能（Artificial` | `13.54 MiB` | `2.631429s` |
| default `tp-text-generate` | 1 | `[104455, 9909, 9286, 16488]` / `人工智能（Artificial` | `13.54 MiB` | `2.563248s` |
| `tp-text-generate-bfloat16` | 0 | same | `6.68 MiB` | `2.206259s` |
| `tp-text-generate-bfloat16` | 1 | same | `6.68 MiB` | `2.028520s` |
| default `tp-mm-generate` current | 0 | `[87140, 15946, 3837, 101177]` / `视频中，一名` | `442.97 MiB` | `45.082559s` |
| default `tp-mm-generate` current | 1 | `[87140, 15946, 3837, 101177]` / `视频中，一名` | `442.97 MiB` | `44.431829s` |
| `tp-mm-generate-bfloat16-wide` | 0 | same | `221.48 MiB` | `24.365094s` |
| `tp-mm-generate-bfloat16-wide` | 1 | same | `221.48 MiB` | `23.736783s` |
| default `hybrid-mm-generate` | 0 | `[87140, 15946, 3837, 101177]` / `视频中，一名` | `227.64 MiB` | `2.234700s` |
| default `hybrid-mm-generate` | 1 | `[87140, 15946, 3837, 101177]` / `视频中，一名` | `227.64 MiB` | `1.734920s` |
| default `hybrid-mm-generate` | 2 | `[87140, 15946, 3837, 101177]` / `视频中，一名` | `0 B` | `0s` |
| `hybrid-mm-generate-bfloat16-wide` | 0 | same | `113.82 MiB` | `1.948550s` |
| `hybrid-mm-generate-bfloat16-wide` | 1 | same | `113.82 MiB` | `1.481880s` |
| `hybrid-mm-generate-bfloat16-wide` | 2 | same | `0 B` | `0s` |

Long decode comparison:

| case | rank | generated ids/text | total seconds | TP collective bytes | TP collective seconds |
| --- | ---: | --- | ---: | ---: | ---: |
| `tp-mm-generate-long-default` | 0 | `[87140, 15946, 3837, 101177, 105611, 99194, 38035, 113727, 33108, 104362, 38035, 113233, 9370, 104253, 104224, 46944]` / `视频中，一名穿着深色衬衫和浅色裤子的男子站在一个` | `84.236407s` | `451.41 MiB` | `50.078444s` |
| `tp-mm-generate-long-default` | 1 | same | `84.145172s` | `451.41 MiB` | `49.534976s` |
| `tp-mm-generate-long-bfloat16` | 0 | same as default | `62.997012s` | `225.70 MiB` | `29.048884s` |
| `tp-mm-generate-long-bfloat16` | 1 | same as default | `62.879746s` | `225.70 MiB` | `28.288443s` |

Conclusion:

- `bfloat16` preserves generated ids/text across this broader matrix.
- Pure TP text and multimodal collective bytes are about halved.
- Long decode default vs bfloat16 generated ids/text match exactly.
- This is enough to treat `bfloat16` as a strong default candidate, but the runtime default was not changed in this round.
