# 20260430 bfloat16 default landing

Purpose:

- Land `--comm-dtype bfloat16` as the CLI default.
- Do not change collective semantics.
- Re-run the short smoke matrix and `tp-mm-generate` long decode without passing `--comm-dtype`.

Code change:

- `qwen3vl_tp_runtime/scripts/runtime.py`: `--comm-dtype` default changed from `float32` to `bfloat16`.
- `test/test_runtime_cli_modes.py`: parser test asserts the new default.

Cases:

| case | ranks | checker | generated ids/text |
| --- | ---: | --- | --- |
| `tp-text-generate-default` | 2 | PASS | `[104455, 9909, 9286, 16488]` / `人工智能（Artificial` |
| `tp-mm-generate-default` | 2 | PASS | `[87140, 15946, 3837, 101177]` / `视频中，一名` |
| `hybrid-mm-generate-default` | 3 | PASS | `[87140, 15946, 3837, 101177]` / `视频中，一名` |
| `tp-mm-generate-long-default-bfloat16` | 2 | PASS | `[87140, 15946, 3837, 101177, 105611, 99194, 38035, 113727, 33108, 104362, 38035, 113233, 9370, 104253, 104224, 46944]` / `视频中，一名穿着深色衬衫和浅色裤子的男子站在一个` |

Perf summary:

| case | rank | TP collective bytes | TP collective seconds | comm dtype evidence |
| --- | ---: | ---: | ---: | --- |
| `tp-text-generate-default` | 0 | `6.68 MiB` | `2.159819s` | `torch.bfloat16` |
| `tp-text-generate-default` | 1 | `6.68 MiB` | `2.031675s` | `torch.bfloat16` |
| `tp-mm-generate-default` | 0 | `221.48 MiB` | `24.513640s` | `torch.bfloat16` |
| `tp-mm-generate-default` | 1 | `221.48 MiB` | `23.768131s` | `torch.bfloat16` |
| `hybrid-mm-generate-default` | 0 | `113.82 MiB` | `2.081890s` | `torch.bfloat16` |
| `hybrid-mm-generate-default` | 1 | `113.82 MiB` | `1.633925s` | `torch.bfloat16` |
| `hybrid-mm-generate-default` | 2 | `0 B` | `0s` | `tp_degree=1` bypass |
| `tp-mm-generate-long-default-bfloat16` | 0 | `225.70 MiB` | `28.840353s` | `torch.bfloat16` |
| `tp-mm-generate-long-default-bfloat16` | 1 | `225.70 MiB` | `28.074485s` | `torch.bfloat16` |

Artifacts:

- `check-*.txt`: checker output, all PASS.
- `runtime-perf-records.json`: machine-readable perf records.
- `runtime-perf-table.md`: perf table.
- `*.ssh.stderr`: empty for all ranks.
