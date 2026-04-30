# 2026-04-30 pure TP runtime input local profile

Purpose:

- Evaluate pure TP runtime input broadcast reduction.
- Avoid rank0 broadcasting dense `stage_input` when every TP rank can locally provide the runtime input.
- Keep RowParallel attention/MLP all-reduce unchanged.

vLLM mapping:

- vLLM uses compact model input fields such as `input_ids`, `positions`, `inputs_embeds`, and `intermediate_tensors` at the Qwen3-VL forward boundary.
- vLLM also centralizes tensor dict communication in `broadcast_tensor_dict`.
- For this runtime, the low-risk equivalent is: keep the current startup contract, but avoid the extra pure TP generate-time `stage_input_broadcast` when local `stage_input` or local token embeddings are already available.

Implementation:

- Prefill:
  - use local `input_ids + embed_tokens_weight` when available;
  - otherwise use local `stage_input` / `layer_input` when available;
  - only fall back to rank0 `broadcast_cpu` when neither local path exists.
- Decode:
  - each rank already receives the generated token id through the existing token-id broadcast;
  - each rank then builds the one-token embedding locally from its replicated `embed_tokens_weight`.

Profile:

- Output directory: `baseline_runs/20260430-runtime-input-local-profile/`
- Case: `tp-mm-generate-runtime-input-local`
- Topology: pure TP, 2 Jetsons
- Checker: `check-tp-mm-generate-runtime-input-local.txt`
- Perf records: `runtime-perf-records.json`
- Perf table: `runtime-perf-table.md`

Before / after:

| metric | before default `tp-mm-generate` | after runtime-input-local |
| --- | ---: | ---: |
| generated ids | `[87140, 15946, 3837, 101177]` | `[87140, 15946, 3837, 101177]` |
| generated text | `视频中，一名` | `视频中，一名` |
| runtime input TP events / rank | `4` | `0` |
| runtime input TP bytes / rank | `6,451,200` | `0` |
| rank0 runtime input seconds | `1.035093s` | `0s` |
| rank1 runtime input seconds | `0.563669s` | `0s` |
| rank0 total TP collective bytes | `449.12 MiB` | `442.97 MiB` |
| rank1 total TP collective bytes | `449.12 MiB` | `442.97 MiB` |
| rank0 total TP collective seconds | `45.646768s` | `45.082559s` |
| rank1 total TP collective seconds | `44.913286s` | `44.431829s` |
| startup contract bytes / rank | `11.53 MiB` | `11.53 MiB` |

Summary evidence:

- Prefill: `runtime_input_source=local_stage_input`, `runtime_input_broadcast_skipped=true`.
- Decode steps: `runtime_input_source=local_embeddings`, `runtime_input_broadcast_skipped=true`.
- No non-empty `.ssh.stderr` files.

Conclusion:

- This removes the duplicate generate-time dense `stage_input` TP broadcast.
- It is a small win compared with RowParallel all-reduce, but it is low risk and directionally matches the vLLM-style compact model input boundary.
