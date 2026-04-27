# qwen3vl_tp_runtime

Qwen3-VL runtime code for direct `pp` / `tp` / `hybrid` execution.

## Main Path

- Main runtime entry: `scripts/runtime.py`
- Preferred mode: `backend=pp|tp|hybrid`
- Main path builds runtime state directly from `model_path`

## Status

This runtime started with two concrete goals:

- `TP` should load only the local shard for each rank instead of loading full weights on every GPU and slicing at execution time.
- `PP / TP / hybrid` main runs should build stage/rank runtime state directly from `model_path` at startup instead of depending on prepared `bundle` or manifest replay artifacts.

Current state:

- The main `pp / tp / hybrid` path is direct-first and builds runtime state from `model_path`.
- The main text `TP` path no longer loads full decoder projection weights on every GPU and slices only during compute. Direct `tp_degree > 1` stages first broadcast a no-weight scaffold, then each rank materializes its local shard from `model_path`.
- `backend=tp` text generate has passed real Jetson smoke with `weight_load.tp_weight_sharded=true` on both ranks, `tp_shard_rank=0/2` and `1/2`, shard-sized projection shapes, and identical `loaded_weight_tensor_bytes`.
- `backend=hybrid` text generate has passed real Jetson smoke with stage0 running rank-local TP shards and stage1 loading only its own PP stage weights.
- Multimodal direct runtime for `pp / hybrid` has passed real Jetson smoke runs on the runtime-only main path. All ranks produced matching `generated_token_ids=[87140, 15946, 3837, 101177]`, and summaries prove stage-local frontend/weight scope plus hybrid TP shard-local materialization.
- Multimodal startup transport is now stage-local and thin: it carries only runtime shared metadata/tensors, local stage handoffs, local stage visuals, and frame metadata; root/full/replay payloads are rejected.
- The current milestone is considered complete for direct-from-`model_path` PP/TP/hybrid startup, rank-local text decoder shards, and `pp / hybrid` multimodal stage-only/shard-only smoke. Runtime summary now includes PP stage weight scope, TP projection shape proof, and same-stage TP weight-byte equality evidence.

Remaining tail work:

- Embedding and `lm_head` are still replicated where the current execution semantics require them. Vocab/embedding parallelism is a later optimization, not part of the completed milestone.
- Startup time and peak-memory baselines are intentionally deferred; current acceptance is based on output tokens and `weight_load` shard evidence.
- Some schema, legacy compatibility, and debug-only transport cleanup may still continue, but replay/capture paths are no longer considered the main runtime surface.

## Debug Path

These paths are kept for replay, capture, and regression work only:

- `--manifest-path` replay runs
- `--compare-direct`
- `--trace-layers`
- `--dump-layer`

They require `--allow-debug-paths`.
`--compare-direct / --trace-layers / --dump-layer` are currently `backend=tp|hybrid` and non-generate only.

## Roadmap

- Ordered next steps are tracked in `ROADMAP.md`.
- Unless we explicitly realign, continue work in that document's order.

## Baseline

- Fixed regression commands are tracked in `BASELINE.md`.
- Use those case ids as the default smoke/regression set before and after runtime changes.

## Directory Layout

- `hexgen_core/`
  Core distributed runtime pieces: process groups, transport, schema, PP/TP/hybrid runners.
- `models/qwen3vl/`
  Model-specific runtime code.
- `models/qwen3vl/execution/`
  Low-level tensor execution for attention, decoder, and stage forward/trace logic.
- `models/qwen3vl/processing/`
  Input building, processor/tokenizer loading, and model-path helpers.
- `models/qwen3vl/weights/`
  Weight index, load plan, shard slicing, and stage bundle materialization from weights.
- `models/qwen3vl/runtime_builder.py`
  Direct runtime builders that turn `model_path` into stage/rank runtime bundles and manifests.
- `models/qwen3vl/runtime_text.py`
  Text-only prompt metadata, runtime-only scaffold restore, and startup session helpers.
- `models/qwen3vl/runtime_text_stage.py`
  Text scaffold compaction, runtime input rebuild, and local stage materialization helpers.
- `scripts/`
  User-facing runtime and helper scripts.
- `scripts/helpers/`
  Stable shell wrappers for common entrypoints such as `run-runtime.sh` and `generate.sh`.
- `scripts/runtime_cli.py`
  Runtime CLI defaults, validation, and debug-path gating helpers.
- `scripts/runtime_summary.py`
  JSON summary and generated-text decoding helpers for runtime outputs.

## Naming Notes

- Prefer short, specific helper names over long workflow-style names.
- Use `text_prompt_meta` instead of repeating `runtime_only_text_generate_prompt_metadata`.
- Keep public names descriptive, but avoid encoding the whole call chain in the function name.
