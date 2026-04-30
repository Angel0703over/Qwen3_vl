# 20260430 step 14 TP collective substage profiling

This directory freezes the real Jetson rerun after adding TP collective substage
profiling fields.

Topology:

- `tp-mm-generate`: Jetson2 rank0, Jetson3 rank1, `--backend tp --tp 2`.
- `hybrid-mm-generate`: Jetson2 rank0/rank1, Jetson3 rank2,
  `--backend hybrid --pp 2 --tp-degrees 2 1`.
- `MASTER_ADDR=10.126.126.3`.

Validation:

- `check-tp-mm-baseline.txt`: PASS.
- `check-hybrid-mm-baseline.txt`: PASS.
- `runtime-perf-records.json`: structured timing / payload / memory records,
  including TP collective substage totals.
- `runtime-perf-table.md`: summarized timing / payload / memory table.

Key result:

- `tp-mm-generate` is dominated by gloo collective time, not device/CPU copy.
  - rank0: TP collective `45.646768s`; gloo `43.010443s` (`94.2%`).
  - rank1: TP collective `44.913286s`; gloo `42.215497s` (`94.0%`).
- `hybrid-mm-generate` stage0 ranks are same-host TP and much less dominated by
  gloo:
  - rank0: TP collective `2.234700s`; gloo `1.020526s` (`45.7%`).
  - rank1: TP collective `1.734920s`; gloo `0.497019s` (`28.6%`).
- `hybrid-mm-generate` rank2 remains `tp_degree=1`; TP collective bytes/time stay
  `0`.

Representative `tp-mm-generate` prefill row-reduce:

- shape: `(1, 627, 2560)`
- source dtype / target dtype: `torch.bfloat16`
- comm dtype: `torch.float32`
- per event bytes counted at comm dtype: `6.12 MiB`

Outputs:

- `*-rank*.log`: local copies of rank stdout including JSON summaries.
- `*.ssh.stdout` / `*.ssh.stderr`: SSH launcher stdout/stderr.
- `runtime-perf-records.json`: machine-readable perf records.
- `runtime-perf-table.md`: compact perf table.
