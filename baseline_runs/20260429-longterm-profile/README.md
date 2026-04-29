# 20260429 step 13 long-term baseline

This directory freezes the real Jetson profile for ROADMAP step 13 long-term
targets after the runtime input schema and pure TP multimodal input-owner work.

Topology:

- `tp-mm-generate`: Jetson2 rank0, Jetson3 rank1, `--backend tp --tp 2`.
- `hybrid-text-generate`: Jetson2 rank0/rank1, Jetson3 rank2,
  `--backend hybrid --pp 2 --tp-degrees 2 1`.
- `hybrid-mm-generate`: Jetson2 rank0/rank1, Jetson3 rank2,
  `--backend hybrid --pp 2 --tp-degrees 2 1`.
- `MASTER_ADDR=10.126.126.3`.

Cases:

- `tp-mm-generate`: PASS, generated ids `[87140, 15946, 3837, 101177]`,
  text `视频中，一名`.
- `hybrid-text-generate`: PASS, generated ids `[104455, 9909, 9286, 16488]`,
  text `人工智能（Artificial`.
- `hybrid-mm-generate`: PASS, generated ids `[87140, 15946, 3837, 101177]`,
  text `视频中，一名`.

Key payload records:

- `tp-mm-generate`: input-owner startup contract is 422 object bytes plus
  12 tensors / 12,093,371 tensor bytes on both ranks. No scaffold broadcast.
- `hybrid-text-generate`: stage0 runtime input broadcast is 249 object bytes
  plus 1 tensor / 128 tensor bytes, key `runtime_inputs.input_ids`.
- `hybrid-mm-generate`: stage1 startup contract is 422 object bytes plus
  9 tensors / 3,245,384 tensor bytes; stage0 runtime input broadcast is
  938 object bytes plus 11 tensors / 12,093,371 tensor bytes.

Outputs:

- `*-rank*.log`: local copies of rank stdout including JSON summaries.
- `*.ssh.stderr`: SSH stderr; all files are empty for successful runs.
- `check-longterm-baseline.txt`: baseline checker output.
- `runtime-perf-records.json`: structured timing / payload / memory records.
- `runtime-perf-table.md`: markdown timing / payload / memory table.
