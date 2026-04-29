# 20260429 pure PP=3 smoke

This directory contains an exploratory pure pipeline-parallel run with
`--backend pp --pp 3`.

Topology:

- `WORLD_SIZE=3`
- Jetson2 runs rank0 and rank1 on its local CUDA device.
- Jetson3 runs rank2 on its local CUDA device.
- `MASTER_ADDR=10.126.126.3`
- The ranks were launched manually with `RANK/WORLD_SIZE/LOCAL_RANK` instead of
  `torchrun`, because the Codex sandbox cannot access Jetson1 CUDA.

Cases:

- `pp3-text-generate`: PASS
- `pp3-mm-generate`: PASS

Outputs:

- `pp3-text-generate-rank*.log`
- `pp3-mm-generate-rank*.log`
- `check-pp3-logs.txt`
- `runtime-perf-records.json`
- `runtime-perf-table.md`
