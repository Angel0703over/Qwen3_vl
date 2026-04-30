# 2026-04-30 comm dtype opt-in profile

Purpose:

- Evaluate `--comm-dtype bfloat16` and `--comm-dtype float16` as opt-in TP collective experiments.
- Do not change the default runtime behavior.
- Compare generated ids/text, TP collective bytes, and TP collective seconds against the float32 comm baseline.

Topology:

- Case: `tp-mm-generate`
- Backend: pure TP
- Nodes/ranks: 2 Jetsons, rank0 on `10.126.126.3`, rank1 on `10.126.126.4`
- Model: `/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct`
- Max new tokens: `4`

Commands:

```bash
NODE_RANK=0 MASTER_ADDR=10.126.126.3 OUT=baseline_runs/20260430-comm-dtype-profile/tp-mm-generate-comm-bfloat16-rank0.log \
  bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh --comm-dtype bfloat16
NODE_RANK=1 MASTER_ADDR=10.126.126.3 OUT=baseline_runs/20260430-comm-dtype-profile/tp-mm-generate-comm-bfloat16-rank1.log \
  bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh --comm-dtype bfloat16

NODE_RANK=0 MASTER_ADDR=10.126.126.3 OUT=baseline_runs/20260430-comm-dtype-profile/tp-mm-generate-comm-float16-rank0.log \
  bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh --comm-dtype float16
NODE_RANK=1 MASTER_ADDR=10.126.126.3 OUT=baseline_runs/20260430-comm-dtype-profile/tp-mm-generate-comm-float16-rank1.log \
  bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh --comm-dtype float16
```

Checker:

- `check-tp-mm-generate-comm-bfloat16.txt`: PASS
- `check-tp-mm-generate-comm-float16.txt`: PASS

Baseline comparison:

| variant | rank | generated ids | generated text | runtime total | TP collective seconds | TP collective bytes | gloo seconds |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| default float32 baseline | 0 | `[87140, 15946, 3837, 101177]` | `视频中，一名` | `74.781979s` | `45.646768s` | `449.12 MiB` | `43.010443s` |
| default float32 baseline | 1 | `[87140, 15946, 3837, 101177]` | `视频中，一名` | `74.715201s` | `44.913286s` | `449.12 MiB` | `42.215497s` |
| `--comm-dtype bfloat16` | 0 | `[87140, 15946, 3837, 101177]` | `视频中，一名` | `53.682284s` | `24.773918s` | `224.56 MiB` | `22.552867s` |
| `--comm-dtype bfloat16` | 1 | `[87140, 15946, 3837, 101177]` | `视频中，一名` | `53.624671s` | `24.117012s` | `224.56 MiB` | `21.820210s` |
| `--comm-dtype float16` | 0 | `[87140, 15946, 3837, 101177]` | `视频中，一名` | `54.026301s` | `24.959947s` | `224.56 MiB` | `22.540502s` |
| `--comm-dtype float16` | 1 | `[87140, 15946, 3837, 101177]` | `视频中，一名` | `53.985323s` | `24.327708s` | `224.56 MiB` | `21.718270s` |

Conclusion:

- Both opt-in dtypes preserve this smoke case's generated ids/text.
- Both halve TP collective bytes: `449.12 MiB -> 224.56 MiB`.
- TP collective seconds drop from about `44-45s` to about `24-25s`.
- End-to-end runtime drops from about `74.7s` to about `53.6-54.0s`.
- `bfloat16` was slightly faster than `float16` in this run.
- Keep the default unchanged for now; document `--comm-dtype bfloat16` as the preferred Jetson opt-in candidate pending broader regression.
