# 20260429 step 14 TP collective profiling

This directory freezes the first real Jetson rerun after adding semantic TP
collective profiling labels.

Topology:

- `tp-text-generate`: Jetson2 rank0, Jetson3 rank1, `--backend tp --tp 2`.
- `tp-mm-generate`: Jetson2 rank0, Jetson3 rank1, `--backend tp --tp 2`.
- `hybrid-mm-generate`: Jetson2 rank0/rank1, Jetson3 rank2,
  `--backend hybrid --pp 2 --tp-degrees 2 1`.
- `MASTER_ADDR=10.126.126.3`.

Validation:

- `check-step14-baseline.txt`: all three cases PASS.
- `runtime-perf-records.json`: structured timing / payload / memory records,
  including `payload.tp_collective_breakdown`.
- `runtime-perf-table.md`: summarized timing / payload / memory table.

Key collective observations:

- `tp-mm-generate`: prefill attention and MLP row-parallel all-reduce dominate.
  Each rank records 36 attention all-reduce events and 36 MLP all-reduce events,
  about 231.14 MiB each side, around 21 seconds per module.
- `tp-text-generate`: same event counts as TP multimodal, but much smaller prefill
  tensors: about 5.63 MiB per attention/MLP prefill side.
- `hybrid-mm-generate` stage0 ranks: 18 attention and 18 MLP prefill
  row-parallel all-reduce events, about 110.21 MiB per module.
- `hybrid-mm-generate` stage1 rank: TP degree is 1, so the current generic path
  still records fallback MLP all-gather and leader broadcast events; this is the
  main low-risk optimization candidate to inspect next.

Outputs:

- `*-rank*.log`: local copies of rank stdout including JSON summaries.
- `*.ssh.stderr`: SSH stderr; all files are empty for successful runs.
- `check-step14-baseline.txt`: baseline checker output.
- `runtime-perf-records.json`: structured records.
- `runtime-perf-table.md`: markdown table.
