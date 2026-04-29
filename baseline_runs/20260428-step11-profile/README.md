# 20260428 step11 transport/profile rerun

This directory contains the first rerun after adding `runtime_metrics.transport`
and payload byte profiling.

Run topology:

- `pp-text-generate`: Jetson2 rank0 + Jetson3 rank1, `MASTER_ADDR=10.126.126.3`
- `pp-mm-generate`: Jetson2 rank0 + Jetson3 rank1, `MASTER_ADDR=10.126.126.3`
- `tp-text-generate`: Jetson2 rank0 + Jetson3 rank1, `MASTER_ADDR=10.126.126.3`
- `tp-mm-generate`: Jetson2 rank0 + Jetson3 rank1, `MASTER_ADDR=10.126.126.3`

Validation:

- `check-baseline-logs.txt`: all four distributed cases PASS.
- `runtime-perf-records.json`: machine-readable timing, memory, and payload metrics.
- `runtime-perf-table.md`: summarized timing, memory, and payload table.

Notes:

- The original `baseline_runs/20260428/` directory remains the frozen
  correctness baseline.
- Hybrid was intentionally left out of this first PP/TP profiling directory.
  The later `baseline_runs/20260429-longterm-profile/` directory freezes the
  real 3-rank hybrid profile.
