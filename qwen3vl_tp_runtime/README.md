# qwen3vl_tp_runtime

Qwen3-VL correctness-first distributed inference runtime prototype. The main path builds `StageState` directly from `model_path` at startup and supports `PP`、`TP`、`HYBRID`.

## Overview

This repo is not a serving engine. It is a reproducible runtime prototype for studying Qwen3-VL distributed inference on Jetson-class nodes.

Current scope:

| area | status |
| --- | --- |
| runtime | `PP / TP / HYBRID` direct `StageState` path |
| backend boundary | `PP` and `TP` are base backends; `HYBRID` composes them |
| multimodal startup | sends compact runtime tensors/metadata only; no root/full/replay payload |
| full video input | `--video-path` and frame-dir paths both supported |
| KV cache | `StageKVCache` + `VideoWindowCacheIndex` + opt-in video KV compaction |
| smoke automation | Step 22 matrix, checker, and perf table are scripted |

Fixed outputs used by current checks:

| case | expected ids | expected text |
| --- | --- | --- |
| text generate | `[104455, 9909, 9286, 16488]` | `人工智能（Artificial` |
| frame-dir multimodal with CLI prompt | `[104455, 9909, 9286, 16488]` | `人工智能（Artificial` |
| full-video default video prompt | `[87140, 108869, 100369, 102122]` | `视频展示了两个场景` |

Default local paths:

| item | default |
| --- | --- |
| repo | `/mnt/ssd/code/Qwen3_vl` |
| Python | `/mnt/ssd/miniconda3/envs/vlm/bin/python` |
| Torchrun | `/mnt/ssd/miniconda3/envs/vlm/bin/torchrun` |
| model | `/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct` |
| frame dir | `/mnt/ssd/code/Qwen3_vl/frames` |

## Quickstart

Run from repo root:

```bash
cd /mnt/ssd/code/Qwen3_vl
export PYTHONPATH=.
```

### 1. CLI Sanity Check

```bash
/mnt/ssd/miniconda3/envs/vlm/bin/python qwen3vl_tp_runtime/scripts/runtime.py --help
```

| field | value |
| --- | --- |
| inputs | repo checkout and Python environment |
| outputs | no files; prints CLI help |
| expected | help includes `--backend {hf,live,pp,tp,hybrid}` and video/KV options |
| troubleshoot | if imports fail, check `PYTHONPATH=.` and the `vlm` environment |

### 2. Local HF Text Generate

```bash
/mnt/ssd/miniconda3/envs/vlm/bin/python qwen3vl_tp_runtime/scripts/runtime.py \
  --backend hf \
  --modality text \
  --mode generate \
  --model-path /mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct \
  --prompt "请用中文简要介绍一下人工智能。" \
  --max-new-tokens 4 | tee /tmp/qwen3vl-hf-text.log
```

| field | value |
| --- | --- |
| inputs | model path and text prompt |
| outputs | `/tmp/qwen3vl-hf-text.log` JSON summary |
| expected | `generated_token_ids=[104455, 9909, 9286, 16488]`, `generated_text=人工智能（Artificial` |
| troubleshoot | if CUDA is unavailable, verify `python -c "import torch; print(torch.cuda.is_available())"` in the same shell |

### 3. Local HF Frame-Dir Multimodal Generate

```bash
/mnt/ssd/miniconda3/envs/vlm/bin/python qwen3vl_tp_runtime/scripts/runtime.py \
  --backend hf \
  --modality multimodal \
  --mode generate \
  --model-path /mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct \
  --frame-dir /mnt/ssd/code/Qwen3_vl/frames \
  --num-frames 8 \
  --prompt "请用中文简要介绍一下人工智能。" \
  --max-new-tokens 4 | tee /tmp/qwen3vl-hf-mm-frame.log
```

| field | value |
| --- | --- |
| inputs | model path, frame directory, prompt |
| outputs | `/tmp/qwen3vl-hf-mm-frame.log` JSON summary |
| expected | same text output as HF text generate; `video_input.source=frame_paths` |
| troubleshoot | if frames are missing, check `FRAME_DIR` or run `find frames -maxdepth 1 -type f` |

### 4. Local HF Full-Video Generate

```bash
/mnt/ssd/miniconda3/envs/vlm/bin/python qwen3vl_tp_runtime/scripts/runtime.py \
  --backend hf \
  --modality multimodal \
  --mode generate \
  --model-path /mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct \
  --video-path /mnt/ssd/code/Qwen3_vl/test/demo.mp4 \
  --video-nframes 4 \
  --prompt "请用中文简要描述这个视频的主要内容。" \
  --max-new-tokens 4 | tee /tmp/qwen3vl-hf-mm-video.log
```

| field | value |
| --- | --- |
| inputs | model path, full video file, video prompt |
| outputs | `/tmp/qwen3vl-hf-mm-video.log` JSON summary |
| expected | `generated_token_ids=[87140, 108869, 100369, 102122]`, `generated_text=视频展示了两个场景` |
| troubleshoot | if video decode fails, check the video path and local video backend; frame-dir smoke is the fallback regression path |

## Reproduce

### One-Click Step 22 Smoke Matrix

```bash
TP_HOSTS="local 10.126.126.4" \
PP_HOSTS="local 10.126.126.4" \
PP3_HOSTS="local 10.126.126.4 10.126.126.5" \
HYBRID_HOSTS="local 10.126.126.4 10.126.126.5" \
bash qwen3vl_tp_runtime/scripts/helpers/run-step22-smoke-matrix.sh \
  --out baseline_runs/$(date -u +%Y%m%d-step22-smoke-matrix)
```

| field | value |
| --- | --- |
| inputs | synced repo/model/frame dir on every host; `TP_HOSTS/PP_HOSTS/PP3_HOSTS/HYBRID_HOSTS` |
| outputs | baseline dir with `*.log`, `check-smoke-matrix.txt`, `collect-runtime-perf.txt`, `runtime-perf-records.json`, `runtime-perf-table.md`, `README.md` |
| expected | `check-smoke-matrix.txt` passes all required cases; perf table has total/startup/handoff/TP collective/CUDA/weight/KV fields |
| troubleshoot | run `DRY_RUN=1 ...` to print commands; use `--case-id <id>` for one case; check `*.ssh.stderr` for remote launch failures |

Optional full-video cases:

```bash
VIDEO_PATH=/mnt/ssd/code/Qwen3_vl/test/demo.mp4 \
bash qwen3vl_tp_runtime/scripts/helpers/run-step22-smoke-matrix.sh \
  --include-optional \
  --out baseline_runs/$(date -u +%Y%m%d-step22-video-smoke)
```

| field | value |
| --- | --- |
| inputs | required Step 22 inputs plus `VIDEO_PATH` or `VIDEO_URL` |
| outputs | same baseline files as Step 22; optional `*-video*` rank logs |
| expected | full-video cases generate `视频展示了两个场景` |
| troubleshoot | `--include-optional` requires `VIDEO_PATH` or `VIDEO_URL`; verify the file exists on all hosts |

### Single Distributed Smoke

Pure TP frame-dir multimodal generate:

```bash
# rank 0
OUT=baseline_runs/manual-tp-mm \
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.126.126.3 \
bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh

# rank 1
OUT=baseline_runs/manual-tp-mm \
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.126.126.3 \
bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh
```

| field | value |
| --- | --- |
| inputs | two ranks, shared `MASTER_ADDR`, synced model/frame dir |
| outputs | `baseline_runs/manual-tp-mm/tp-mm-generate-rank0.log` and `rank1.log` |
| expected | rank logs end with JSON containing `generated_text=人工智能（Artificial`; TP collective metrics are present |
| troubleshoot | if ranks hang, check both ranks use the same `MASTER_ADDR/MASTER_PORT/NNODES`; check firewall and SSH session health |

Pure PP frame-dir multimodal generate:

```bash
OUT=baseline_runs/manual-pp-mm NNODES=2 NODE_RANK=0 MASTER_ADDR=10.126.126.3 \
bash qwen3vl_tp_runtime/scripts/helpers/run-pp-mm-generate.sh

OUT=baseline_runs/manual-pp-mm NNODES=2 NODE_RANK=1 MASTER_ADDR=10.126.126.3 \
bash qwen3vl_tp_runtime/scripts/helpers/run-pp-mm-generate.sh
```

| field | value |
| --- | --- |
| inputs | two PP ranks, model/frame dir |
| outputs | `baseline_runs/manual-pp-mm/pp-mm-generate-rank*.log` |
| expected | generated text matches HF frame-dir output; handoff bytes are recorded |
| troubleshoot | PP requires `PP == NNODES` in this wrapper; use `PP=3 NNODES=3` for PP3 |

HYBRID frame-dir multimodal generate:

```bash
OUT=baseline_runs/manual-hybrid-mm NNODES=3 NODE_RANK=0 MASTER_ADDR=10.126.126.3 \
bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh

OUT=baseline_runs/manual-hybrid-mm NNODES=3 NODE_RANK=1 MASTER_ADDR=10.126.126.3 \
bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh

OUT=baseline_runs/manual-hybrid-mm NNODES=3 NODE_RANK=2 MASTER_ADDR=10.126.126.3 \
bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh
```

| field | value |
| --- | --- |
| inputs | three ranks, default `PP=2`, default `TP_DEGREES="2 1"` |
| outputs | `baseline_runs/manual-hybrid-mm/hybrid-mm-generate-rank*.log` |
| expected | stage0 has TP collective metrics; stage1 `tp_degree=1` has `0 B` TP collective |
| troubleshoot | if only two CUDA hosts are available, run Step 22 single `--case-id` subsets or use full-video `TP_DEGREES="1 1"` only for the documented two-node video case |

### Check Existing Logs

```bash
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python \
  qwen3vl_tp_runtime/scripts/check_baseline_logs.py \
  --matrix step22 \
  --baseline-dir baseline_runs/20260502-step23c-prompt-smoke \
  --require-transport-metrics
```

| field | value |
| --- | --- |
| inputs | existing baseline directory |
| outputs | stdout PASS/FAIL report |
| expected | generated ids/text, rank count, transport metrics, consume-only, and TP shard checks pass |
| troubleshoot | use `--case-id <id> rank*.log` to isolate one failed case |

### Generate Perf Table

```bash
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python \
  qwen3vl_tp_runtime/scripts/collect_runtime_perf.py \
  --baseline-dir baseline_runs/20260502-step23c-prompt-smoke \
  --matrix step22 \
  --output-json baseline_runs/20260502-step23c-prompt-smoke/runtime-perf-records.json \
  --output-md baseline_runs/20260502-step23c-prompt-smoke/runtime-perf-table.md
```

| field | value |
| --- | --- |
| inputs | baseline logs |
| outputs | `runtime-perf-records.json`, `runtime-perf-table.md` |
| expected | table columns include total seconds, startup/handoff bytes, TP collective seconds/bytes, CUDA peak, loaded weights, stage KV bytes |
| troubleshoot | missing rows usually mean a rank log does not contain final JSON; inspect that rank log first |

### Local Core Regression

```bash
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh --skip-baseline-checks
```

| field | value |
| --- | --- |
| inputs | local unit tests only |
| outputs | stdout regression log |
| expected | `PASS runtime core minimal matrix` |
| troubleshoot | remove `--skip-baseline-checks` only when `baseline_runs/20260428/` is present |

## Results

Current numbers live in `BASELINE.md`; raw logs live in `baseline_runs/*/README.md`.

High-signal before/after:

| change | before | after |
| --- | ---: | ---: |
| startup contract removes `stage_output` | `7,563,328` bytes | `4,353,088` bytes |
| startup contract removes dense derived tensors | `4,353,088` bytes | `3,245,806` bytes |
| HYBRID stage1 `tp_degree=1` collective | `648.46 MiB` | `0 B` |
| pure TP comm dtype | `449.12 MiB` collective | `221.48 MiB` collective |
| pure TP runtime input broadcast | `4` events / rank | `0` events |
| Step 20C opt-in compaction | full active visual KV | active KV bytes about half |
| Step 23C prompt propagation | distributed mm used default video prompt | distributed mm uses CLI `--prompt`, matching HF-mm |

Current baseline directories:

| purpose | directory |
| --- | --- |
| current perf | `baseline_runs/20260430-bfloat16-default/` |
| Step 20C-4 InfiniPot-V selector | `baseline_runs/20260502-step20c4-infinipot-selector/` |
| Step 21 full video input | `baseline_runs/20260502-step21-video-input/` |
| Step 22 smoke automation | `baseline_runs/20260502-step22-full-smoke/` |
| Step 23C prompt smoke | `baseline_runs/20260502-step23c-prompt-smoke/` |
| Step 24H code cleanup verify | `baseline_runs/20260503-step24h-verify/` |

## Architecture

Core terms:

- `StageState`: main runtime object for PP/TP/HYBRID.
- `PP`: pipeline-parallel base backend.
- `TP`: tensor-parallel base backend.
- `HYBRID`: PP+TP composition backend.
- `model_input`: internal backend helper name.
- `runtime_inputs`: HYBRID wire protocol key, frozen as `hybrid_runtime_inputs_v1`.
- `bundle`: replay/debug/capture or legacy compat only.

Runtime data rules:

- Do not send root/full/replay payloads in startup contracts.
- Rebuild dense derived tensors locally when possible.
- Only stage0/input-owner reads video, samples frames, and runs the vision frontend.
- Non-input-owner ranks consume startup contract and do not rerun the frontend.
- `--video-kv-compression none` is the default correctness path.
- `uniform`、`swa`、`infinipot-v` video KV compaction are opt-in experiments.

Parallel shortcuts:

```bash
# pure PP: split text layers evenly into 2 stages
--backend pp --pp 2

# pure TP: one stage, TP=2
--backend tp --tp 2

# uniform HYBRID: 2 PP stages, each TP=2
--backend hybrid --pp 2 --tp 2

# heterogeneous HYBRID: stage0 TP=2, stage1 TP=1
--backend hybrid --pp 2 --tp-degrees 2 1
```

Input shortcuts:

```bash
# prepared frames
--frame-dir /mnt/ssd/code/Qwen3_vl/frames --num-frames 8

# full local video
--video-path /mnt/ssd/code/Qwen3_vl/test/demo.mp4 --video-nframes 4

# full video URL
--video-url <url> --video-nframes 4
```

## Code Map

- `hexgen_core/modules/pipeline_parallel.py`: pure PP backend runner; direct `StageState` execution; legacy captured-bundle prepare/replay entrypoints.
- `hexgen_core/modules/tensor_parallel.py`: pure TP backend runner; TP manifest load; rank-local `StageState` execution.
- `hexgen_core/modules/hybrid_parallel.py`: PP+TP HYBRID backend; stage-group scaffold/model-input broadcast; PP handoff plus TP stage execution.
- `hexgen_core/distributed.py`: process group, CPU/object/tensor collective, startup/profile logging, transport staging primitive.
- `hexgen_core/transport.py`: `StageCommunicator` and stage payload send/recv/broadcast.
- `hexgen_core/schema.py`: manifest, rank context, `StageState`, HYBRID runtime input schema, payload summary.
- `hexgen_core/stage.py`: `StageStateView`, handoff payload build/apply, stage execution dispatch.
- `hexgen_core/generate_buffers.py`: runtime-only generate decode buffer reuse.
- `models/qwen3vl/execution/`: Qwen text layer forward/trace and TP math wrapper; no distributed I/O.
- `models/qwen3vl/kv_cache/`: `StageKVCache`, video window metadata, and opt-in video KV compression helpers.
- `models/qwen3vl/runtime_builder.py`: builds stage/rank `StageState` from `model_path`, startup contract, transport pack/restore, direct manifest.
- `models/qwen3vl/runtime_mm_stage.py`: multimodal shared/runtime tensor rebuild and stage materialization.
- `models/qwen3vl/runtime_text_stage.py`: text runtime input rebuild and stage materialization.
- `models/qwen3vl/weights/`: safetensors index, load plan, text/vision weight loading, TP shard slicing.
- `models/qwen3vl/processing/`: Qwen processor/input builder, full-video and frame-dir input assembly.
- `models/qwen3vl/vision/`: Qwen3-VL vision frontend runtime, deepstack, bridge/state.
- `models/qwen3vl/capture/`: captured bundle save/load compatibility; not the direct runtime path.
- `models/qwen3vl/live/`: live HF/Qwen single-node helper and old live bundle/input helper.
- `debug/`: replay runner and TP debug config.
- `scripts/runtime.py`: unified CLI entrypoint and backend dispatch.
- `scripts/runtime_cli.py`: CLI validation, debug gate, direct/replay mode selection.
- `scripts/helpers/`: stable smoke wrappers.
- `scripts/helpers/run-step22-smoke-matrix.sh`: one-click Step 22 runner.
- `scripts/smoke_matrix.py`: fixed smoke matrix and expected generated ids/text.
- `scripts/check_baseline_logs.py`: baseline checker.
- `scripts/collect_runtime_perf.py`: perf table collector.

## Debug

Debug/replay paths require explicit opt-in:

```bash
--allow-debug-paths --manifest-path <path>
--allow-debug-paths --compare-direct
--allow-debug-paths --trace-layers
--allow-debug-paths --dump-layer <idx>
```

Failure checklist:

| symptom | first checks |
| --- | --- |
| import error | `PYTHONPATH=.`, correct Python env |
| CUDA unavailable | same shell: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` |
| distributed hang | all ranks use same `MASTER_ADDR`, `MASTER_PORT`, `NNODES`; remote hosts have synced repo/model/input |
| checker generated mismatch | inspect final JSON in every rank log; compare prompt and frame/video input mode |
| missing transport metrics | rerun with `HEXGEN_STARTUP_LOG=1`; verify final JSON was printed |
| video path failure | confirm file exists on every host or use frame-dir fallback |

## Docs

- `ROADMAP.md`: current task queue.
- `BASELINE.md`: current baselines, before/after effects, validation fields.
- `SESSION_HANDOFF.md`: concise handoff for new conversations.
- `CODE_CLEANUP_AUDIT.md`: Step 24 code cleanup audit.
- `BUFFER_REUSE_AUDIT.md`: Step 16 allocation/clone audit and pinned memory A/B.
- `QWEN3VL_VIDEO_INPUT.md`: Qwen3-VL full-video input, frame extraction, processor, frontend flow.
- `baseline_runs/*/README.md`: per-run Jetson profile.
