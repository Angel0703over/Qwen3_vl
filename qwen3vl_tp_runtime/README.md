# qwen3vl_tp_runtime

Qwen3-VL runtime code for direct `pp` / `tp` / `hybrid` execution.

## Main Path

- Main runtime entry: `scripts/runtime.py`
- Preferred mode: `backend=pp|tp|hybrid`
- Main path builds runtime state directly from `model_path`

## Debug Path

These paths are kept for replay, capture, and regression work only:

- `--manifest-path` replay runs

They require `--allow-debug-paths`.

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
