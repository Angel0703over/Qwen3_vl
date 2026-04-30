# 20260429 step 14 TP degree=1 bypass

This directory freezes the real Jetson rerun after adding the single-rank TP
collective bypass.

Topology:

- `hybrid-mm-generate`: Jetson2 rank0/rank1, Jetson3 rank2.
- Runtime args: `--backend hybrid --pp 2 --tp-degrees 2 1`.
- `MASTER_ADDR=10.126.126.3`.

Validation:

- `check-hybrid-mm-baseline.txt`: PASS.
- `runtime-perf-records.json`: structured timing / payload / memory records.
- `runtime-perf-table.md`: summarized timing / payload / memory table.

Key result:

- rank0/rank1 are the real TP group and remain at `227.64 MiB` TP collective
  bytes.
- rank2 is `tp_degree=1`; TP collective bytes drop from the previous
  `648.46 MiB` to `0 B`.
- Generated ids/text are unchanged:
  `[87140, 15946, 3837, 101177]`, `视频中，一名`.

Before/after reference:

- before: `baseline_runs/20260429-step14-profile/`
- after: `baseline_runs/20260429-step14-tp1-bypass/`

Outputs:

- `hybrid-mm-generate-rank*.log`: local copies of rank stdout including JSON
  summaries.
- `hybrid-mm-generate-rank*.ssh.stdout`: SSH command stdout captured from the
  local launcher.
- `hybrid-mm-generate-rank*.ssh.stderr`: SSH stderr; empty for this successful
  run.
