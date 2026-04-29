# qwen3vl_tp_runtime Baseline

这份文档用于冻结当前阶段的最小回归基线。

目标：

- 固定一组 `hf / live / pp / tp / hybrid` 的 generate 回归命令。
- 后续改动统一拿这组命令做回归。
- 当前先固定 `generated_token_ids`、`generated_text`，以及 `tp / hybrid` 的 `weight_load` shard/stage scope 证据。
- 启动时间和峰值显存先不强制纳入这份基线，留到最终性能验收再统一收。

## 基线规则

- 默认只跑 `generate`。
- 默认关闭采样，不传 `--do-sample`，保证输出尽量稳定。
- text 和 multimodal 都用同一条中文提示词，除非后面专门讨论要拆。
- distributed case 建议保留 `HEXGEN_STARTUP_LOG=1`，方便看启动耗时拆分。
- distributed case 目前要求所有 rank 的 `generated_token_ids` / `generated_text` 一致。
- `tp-text-generate`、`hybrid-text-generate` 和 `hybrid-mm-generate` 额外要求 JSON summary 中的 `weight_load` 能证明 rank-local materialize / stage-local scope。

## 当前支持矩阵

| case_id | backend | modality | 当前是否纳入基线 | 备注 |
| --- | --- | --- | --- | --- |
| `hf-text-generate` | `hf` | `text` | `是` | 本地单进程 |
| `hf-mm-generate` | `hf` | `multimodal` | `是` | 本地单进程 |
| `live-mm-generate` | `live` | `multimodal` | `是` | 本地单进程 |
| `live-text-generate` | `live` | `text` | `否` | CLI 当前不支持 |
| `pp-text-generate` | `pp` | `text` | `是` | 2 stage / 2 rank |
| `pp-mm-generate` | `pp` | `multimodal` | `是` | 2 stage / 2 rank |
| `tp-text-generate` | `tp` | `text` | `是` | 1 stage / TP=2 |
| `tp-mm-generate` | `tp` | `multimodal` | `是` | 当前先作为 smoke 基线 |
| `hybrid-text-generate` | `hybrid` | `text` | `是` | 2 stage, `tp_degrees=2 1` |
| `hybrid-mm-generate` | `hybrid` | `multimodal` | `是` | 2 stage, `tp_degrees=2 1` |

## 统一参数

建议先统一下面这组环境变量：

```bash
export REPO_ROOT=/mnt/ssd/code/Qwen3_vl
export RUNTIME_ROOT="${REPO_ROOT}/qwen3vl_tp_runtime"
export PYTHON_BIN=/mnt/ssd/miniconda3/envs/vlm/bin/python
export TORCHRUN=/mnt/ssd/miniconda3/envs/vlm/bin/torchrun

export MODEL_PATH=/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct
export FRAME_DIR=/mnt/ssd/code/Qwen3_vl/frames

export TEXT_PROMPT="请用中文简要介绍一下人工智能。"
export MM_PROMPT="请用中文简要介绍一下人工智能。"
export MAX_NEW_TOKENS=4
```

如果你不显式传 `--model-path` 和 `--frame-dir`，当前代码默认值也是这两个路径：

- [config.py](/mnt/ssd/code/Qwen3_vl/qwen3vl_tp_runtime/hexgen_core/config.py)

主路径推荐使用短参数：

- `--pp N`：按模型 text 层数平均切成 `N` 个 PP stage。
- `--tp N`：设置统一 TP degree。
- `--stage-ranges`：保留为手工切层覆盖参数。
- `--tp-degrees`：保留为异构 hybrid 覆盖参数，例如当前 3 卡基线用 `--pp 2 --tp-degrees 2 1`。

## 记录方式

当前硬性记录两样东西：

1. `stdout` 里的最终 JSON
2. distributed case 的 `HEXGEN_STARTUP_LOG=1` 启动日志

如果方便，建议额外保存 `stderr` 或终端里的 `/usr/bin/time -p` 结果；这部分当前只作为参考，不作为是否过线的硬门槛。
如果某台 Jetson 没有 `/usr/bin/time`，直接去掉 `/usr/bin/time -p` 即可，不影响当前基线验收。

建议统一输出目录：

```bash
export BASELINE_OUT="${REPO_ROOT}/baseline_runs/$(date -u +%Y%m%d)"
mkdir -p "${BASELINE_OUT}"
```

本地单进程 case 可以直接这样包一层：

```bash
/usr/bin/time -p bash "${RUNTIME_ROOT}/scripts/helpers/run-runtime.sh" ... \
  > "${BASELINE_OUT}/case.stdout" \
  2> "${BASELINE_OUT}/case.stderr"
```

distributed case 因为通常分多个终端/节点执行，建议按 `rank` 分文件保存。

## 分布式 smoke wrapper

常用 multimodal distributed case 已固化成 wrapper。每台机器运行同一个脚本，只改 `NODE_RANK`；默认日志写到 `${REPO_ROOT}/baseline_runs/$(date -u +%Y%m%d)/case-rankN.log`。

PP/TP wrapper 的 2 节点只是默认值，不是上限。更多 Jetson 时设置 `NNODES=N`，每台机器设置对应的 `NODE_RANK=0..N-1`。pure PP 当前要求 `PP == NNODES`，pure TP 当前要求 `TP == NNODES`；两个 wrapper 都默认把 degree 设成 `NNODES`，并会提前拒绝不一致的配置。

```bash
# PP multimodal generate, 2 nodes.
NODE_RANK=0 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-pp-mm-generate.sh"
NODE_RANK=1 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-pp-mm-generate.sh"

# TP multimodal generate, 2 nodes.
NODE_RANK=0 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-tp-mm-generate.sh"
NODE_RANK=1 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-tp-mm-generate.sh"

# HYBRID multimodal generate, 3 nodes.
NODE_RANK=0 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-hybrid-mm-generate.sh"
NODE_RANK=1 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-hybrid-mm-generate.sh"
NODE_RANK=2 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-hybrid-mm-generate.sh"

# PP multimodal generate, 4 nodes.
NNODES=4 NODE_RANK=0 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-pp-mm-generate.sh"
NNODES=4 NODE_RANK=1 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-pp-mm-generate.sh"
NNODES=4 NODE_RANK=2 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-pp-mm-generate.sh"
NNODES=4 NODE_RANK=3 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-pp-mm-generate.sh"

# TP multimodal generate, 4 nodes.
NNODES=4 NODE_RANK=0 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-tp-mm-generate.sh"
NNODES=4 NODE_RANK=1 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-tp-mm-generate.sh"
NNODES=4 NODE_RANK=2 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-tp-mm-generate.sh"
NNODES=4 NODE_RANK=3 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-tp-mm-generate.sh"
```

常用覆盖项：

- `MASTER_PORT=...`
- `OUT=...`
- `MODEL_PATH=...`
- `FRAME_DIR=...`
- `MM_PROMPT=...`
- `MAX_NEW_TOKENS=...`
- `DRY_RUN=1` 只打印最终命令

wrapper 后面追加的参数会原样传给 `runtime.py`，例如：

```bash
NODE_RANK=0 MASTER_ADDR="<rank0-host>" \
  bash "${RUNTIME_ROOT}/scripts/helpers/run-pp-mm-generate.sh" --save-dtype bfloat16
```

## 验收字段

每个 case 都至少看下面三项：

- `generated_token_ids`
- `generated_text`
- 是否成功完成所有 rank

`tp-text-generate` / `tp-mm-generate` 额外看：

- rank0 和 rank1 都是 `weight_load.tp_weight_sharded=true`
- rank0 是 `weight_load.tp_shard_rank=0`，rank1 是 `weight_load.tp_shard_rank=1`
- 两个 rank 的 `weight_load.tp_shard_world_size=2`
- 两个 rank 的 `weight_load.tp_shard_shape_ok=true`，`tp_sharded_projection_examples` 中 q/k/v/o 和 MLP projection 是 shard 后形状
- 两个 rank 的 `weight_load.loaded_weight_tensor_bytes` 完全一致
- 如果 summary 带 `weight_load.tp_stage_loaded_weight_tensor_bytes_equal`，该字段必须为 `true`

`hybrid-text-generate` 额外看：

- stage0 的两个 TP rank 都是 `weight_load.tp_weight_sharded=true`
- stage0 的两个 TP rank 分别是 `tp_shard_rank=0/2` 和 `1/2`
- stage0 的两个 TP rank 都是 `weight_load.tp_shard_shape_ok=true` 且 `tp_stage_loaded_weight_tensor_bytes_equal=true`
- stage1 单卡是 `weight_load.tp_weight_sharded=false`
- stage1 只加载 `18:35 + final_norm/lm_head` 对应权重

`pp / hybrid multimodal` 额外看：

- startup contract transport 只包含本地 stage 的 `stage_handoffs` 和 `stage_visuals`
- runtime config 不能出现 `_mm_startup_root_input` 或 `_mm_startup_boundaries`
- transport payload 不能包含 `root_input / boundaries / hidden_states / replay_bundle / stage_bundle`
- 各 rank 的 `generated_token_ids` / `generated_text` 一致
- `pp-mm-generate` 中 stage0 是 frontend active，stage1 是 `multimodal_frontend_mode=consume-only`
- `hybrid-mm-generate` 中 stage0 的 TP rank 是 `tp_shard_rank=0/2` 和 `1/2`，且 `tp_stage_loaded_weight_tensor_bytes_equal=true`
- `hybrid-mm-generate` 中 stage1 是 `multimodal_frontend_mode=consume-only`，且只加载 `18:35 + final_norm/lm_head`

distributed case 额外建议保留但暂不强制比较：

- `HEXGEN_STARTUP_LOG=1` 中的 prepare / materialize / broadcast / barrier 时间
- JSON summary 中的 `runtime_metrics.transport` 事件和 payload bytes 汇总
- `/usr/bin/time -p` 的端到端 wall-clock 时间

## 20260428 冻结记录

本轮完整 baseline 输出目录：

- `baseline_runs/20260428/`

单进程 case：

| case_id | stdout / stderr | generated_token_ids | generated_text |
| --- | --- | --- | --- |
| `hf-text-generate` | `baseline_runs/20260428/hf-text-generate.stdout`, `baseline_runs/20260428/hf-text-generate.stderr` | `[104455, 9909, 9286, 16488]` | `人工智能（Artificial` |
| `hf-mm-generate` | `baseline_runs/20260428/hf-mm-generate.stdout`, `baseline_runs/20260428/hf-mm-generate.stderr` | `[104455, 9909, 9286, 16488]` | `人工智能（Artificial` |
| `live-mm-generate` | `baseline_runs/20260428/live-mm-generate.stdout`, `baseline_runs/20260428/live-mm-generate.stderr` | `[87140, 15946, 3837, 101177]` | `视频中，一名` |

`live-mm-generate` 同时记录 `reference_generated_token_ids=[87140, 15946, 3837, 101177]`、`reference_generated_text="视频中，一名"`、`token_match=true`。

分布式 case：

| case_id | rank logs | generated_token_ids | generated_text |
| --- | --- | --- | --- |
| `pp-text-generate` | `baseline_runs/20260428/pp-text-generate-rank0.log`, `baseline_runs/20260428/pp-text-generate-rank1.log` | `[104455, 9909, 9286, 16488]` | `人工智能（Artificial` |
| `pp-mm-generate` | `baseline_runs/20260428/pp-mm-generate-rank0.log`, `baseline_runs/20260428/pp-mm-generate-rank1.log` | `[87140, 15946, 3837, 101177]` | `视频中，一名` |
| `tp-text-generate` | `baseline_runs/20260428/tp-text-generate-rank0.log`, `baseline_runs/20260428/tp-text-generate-rank1.log` | `[104455, 9909, 9286, 16488]` | `人工智能（Artificial` |
| `tp-mm-generate` | `baseline_runs/20260428/tp-mm-generate-rank0.log`, `baseline_runs/20260428/tp-mm-generate-rank1.log` | `[87140, 15946, 3837, 101177]` | `视频中，一名` |
| `hybrid-text-generate` | `baseline_runs/20260428/hybrid-text-generate-rank0.log`, `baseline_runs/20260428/hybrid-text-generate-rank1.log`, `baseline_runs/20260428/hybrid-text-generate-rank2.log` | `[104455, 9909, 9286, 16488]` | `人工智能（Artificial` |
| `hybrid-mm-generate` | `baseline_runs/20260428/hybrid-mm-generate-rank0.log`, `baseline_runs/20260428/hybrid-mm-generate-rank1.log`, `baseline_runs/20260428/hybrid-mm-generate-rank2.log` | `[87140, 15946, 3837, 101177]` | `视频中，一名` |

自动检查结果：

- `baseline_runs/20260428/check-baseline-logs.txt`
- `pp-text-generate`: PASS
- `pp-mm-generate`: PASS
- `tp-text-generate`: PASS
- `tp-mm-generate`: PASS
- `hybrid-text-generate`: PASS
- `hybrid-mm-generate`: PASS

分布式 weight/stage scope 证据：

- `pp-text-generate` / `pp-mm-generate`
  - rank0 stage0 range `0:17`, `loaded_top_level_weight_names=["embed_tokens_weight"]`, `loaded_weight_tensor_bytes=4411421696`
  - rank1 stage1 range `18:35`, `loaded_top_level_weight_names=["final_norm_weight", "lm_head_weight"]`, `loaded_weight_tensor_bytes=4411426816`
- `tp-text-generate` / `tp-mm-generate`
  - rank0 `tp_weight_sharded=true`, `tp_shard_rank=0/2`, `loaded_weight_tensor_bytes=5189532672`
  - rank1 `tp_weight_sharded=true`, `tp_shard_rank=1/2`, `loaded_weight_tensor_bytes=5189532672`
  - both ranks `tp_shard_shape_ok=true`
- `hybrid-text-generate` / `hybrid-mm-generate`
  - rank0 stage0 local0 `tp_weight_sharded=true`, `tp_shard_rank=0/2`, `loaded_weight_tensor_bytes=2594763776`
  - rank1 stage0 local1 `tp_weight_sharded=true`, `tp_shard_rank=1/2`, `loaded_weight_tensor_bytes=2594763776`
  - rank2 stage1 local0 `tp_weight_sharded=false`, `loaded_top_level_weight_names=["final_norm_weight", "lm_head_weight"]`, `loaded_weight_tensor_bytes=4411426816`
  - all ranks `stage_weight_scope_ok=true`

## 20260428 性能 / 显存记录

收集入口：

```bash
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python \
  qwen3vl_tp_runtime/scripts/collect_runtime_perf.py \
  --baseline-dir baseline_runs/20260428 \
  --output-json baseline_runs/20260428/runtime-perf-records.json \
  --output-md baseline_runs/20260428/runtime-perf-table.md
```

输出文件：

- `baseline_runs/20260428/runtime-perf-records.json`
- `baseline_runs/20260428/runtime-perf-table.md`

当前 20260428 correctness baseline 是旧日志：有 startup timer 和部分 `/usr/bin/time -p real`，但没有新加的 `runtime_metrics.memory.*` 和 `runtime_metrics.transport.*` 字段；所以 CUDA peak alloc/reserved 与 payload bytes 当前为空。新代码重跑任意 case 后，JSON summary 会直接写：

- `runtime_metrics.timing.runtime_total_seconds`
- `runtime_metrics.startup.events`
- `runtime_metrics.startup.totals_by_kind.*`
- `runtime_metrics.transport.events`
- `runtime_metrics.transport.totals_by_kind.*`
- `runtime_metrics.transport.totals_by_channel.*`
- `runtime_metrics.memory.cpu_max_rss_bytes`
- `runtime_metrics.memory.peak_allocated_bytes`
- `runtime_metrics.memory.peak_reserved_bytes`

`collect_runtime_perf.py` 会把 transport/profile 汇总成以下 payload 指标：

- startup contract bytes / tensor bytes / object bytes
- scaffold bytes / tensor bytes / object bytes
- stage handoff bytes / seconds
- TP collective bytes / seconds
- transport event count

当前从旧 log 解析到的表：

| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hf-mm-generate | - | 20.03 | 0.00 | 0.00 | 0.00 | 0.00 | - | - | - | - | - | - | - | - |
| hf-text-generate | - | 18.89 | 0.00 | 0.00 | 0.00 | 0.00 | - | - | - | - | - | - | - | - |
| hybrid-mm-generate | 0 | - | 19.16 | 0.00 | 0.00 | 0.00 | - | - | - | - | - | - | - | 2.42 GiB |
| hybrid-mm-generate | 1 | - | 0.00 | 0.00 | 0.00 | 0.00 | - | - | - | - | - | - | - | 2.42 GiB |
| hybrid-mm-generate | 2 | - | 0.11 | 0.00 | 0.04 | 0.00 | - | - | - | - | - | - | - | 4.11 GiB |
| hybrid-text-generate | 0 | 21.13 | 0.00 | 0.00 | 0.00 | 0.00 | - | - | - | - | - | - | - | 2.42 GiB |
| hybrid-text-generate | 1 | 20.97 | 0.00 | 0.00 | 0.00 | 0.00 | - | - | - | - | - | - | - | 2.42 GiB |
| hybrid-text-generate | 2 | 21.01 | 0.00 | 0.00 | 0.03 | 0.00 | - | - | - | - | - | - | - | 4.11 GiB |
| live-mm-generate | - | 25.03 | 0.00 | 0.00 | 0.00 | 0.00 | - | - | - | - | - | - | - | - |
| pp-mm-generate | 0 | - | 19.26 | 0.00 | 0.03 | 0.00 | - | - | - | - | - | - | - | 4.11 GiB |
| pp-mm-generate | 1 | - | 0.35 | 0.00 | 0.03 | 0.00 | - | - | - | - | - | - | - | 4.11 GiB |
| pp-text-generate | 0 | 18.24 | 0.00 | 0.00 | 0.03 | 0.00 | - | - | - | - | - | - | - | 4.11 GiB |
| pp-text-generate | 1 | 18.28 | 0.01 | 0.00 | 0.06 | 0.00 | - | - | - | - | - | - | - | 4.11 GiB |
| tp-mm-generate | 0 | - | 19.52 | 0.00 | 0.00 | 0.00 | - | - | - | - | - | - | - | 4.83 GiB |
| tp-mm-generate | 1 | - | 19.12 | 0.00 | 0.00 | 0.00 | - | - | - | - | - | - | - | 4.83 GiB |
| tp-text-generate | 0 | 23.05 | 0.58 | 0.00 | 0.00 | 0.00 | - | - | - | - | - | - | - | 4.83 GiB |
| tp-text-generate | 1 | 23.02 | 0.58 | 0.00 | 0.00 | 0.00 | - | - | - | - | - | - | - | 4.83 GiB |

## 20260428 step11 profiling 重跑记录

新 transport/payload profiling 加入后，已用 Jetson2 + Jetson3 重跑 4 个 2 节点 distributed case，输出目录：

- `baseline_runs/20260428-step11-profile/`

检查结果：

- `baseline_runs/20260428-step11-profile/check-baseline-logs.txt`
- `pp-text-generate`: PASS
- `pp-mm-generate`: PASS
- `tp-text-generate`: PASS
- `tp-mm-generate`: PASS

性能 / payload 汇总：

- `baseline_runs/20260428-step11-profile/runtime-perf-records.json`
- `baseline_runs/20260428-step11-profile/runtime-perf-table.md`

当前表中已能看到：

- `pp-mm-generate` startup contract payload bytes：每 rank 约 `7.21 MiB`
- `pp-mm-generate` stage handoff bytes：rank0 约 `6.15 MiB`，rank1 约 `3.08 MiB`
- `tp-text-generate` TP collective bytes：每 rank 约 `13.54 MiB`
- `tp-mm-generate` TP collective bytes：每 rank 约 `449.12 MiB`

注意：

- 原 `baseline_runs/20260428/` 仍是 frozen correctness baseline。
- 这轮没有重跑 HYBRID，因为当前 Codex 沙箱不能访问 Jetson1 CUDA，且不能免密 SSH 到 `10.126.126.2` 绕过沙箱。

## 固定 case

### 1. `hf-text-generate`

```bash
/usr/bin/time -p bash "${RUNTIME_ROOT}/scripts/helpers/run-runtime.sh" \
  --backend hf \
  --modality text \
  --mode generate \
  --model-path "${MODEL_PATH}" \
  --prompt "${TEXT_PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
```

### 2. `hf-mm-generate`

```bash
/usr/bin/time -p bash "${RUNTIME_ROOT}/scripts/helpers/run-runtime.sh" \
  --backend hf \
  --modality multimodal \
  --mode generate \
  --model-path "${MODEL_PATH}" \
  --frame-dir "${FRAME_DIR}" \
  --num-frames 8 \
  --sample-fps 1 \
  --prompt "${MM_PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
```

### 3. `live-mm-generate`

```bash
/usr/bin/time -p bash "${RUNTIME_ROOT}/scripts/helpers/run-runtime.sh" \
  --backend live \
  --modality multimodal \
  --mode generate \
  --model-path "${MODEL_PATH}" \
  --frame-dir "${FRAME_DIR}" \
  --num-frames 8 \
  --prompt "${MM_PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
```

### 4. `pp-text-generate`

拓扑固定为：

- `world_size=2`
- `--pp 2`，自动平均切成 `0:17 18:35`

每个节点各执行一次，只改 `NODE_RANK`：

```bash
export MASTER_ADDR="<rank0-host>"
export MASTER_PORT=29533
export NNODES=2
export NODE_RANK="<0-or-1>"

HEXGEN_STARTUP_LOG=1 /usr/bin/time -p "${TORCHRUN}" \
  --nnodes "${NNODES}" \
  --nproc-per-node 1 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  "${RUNTIME_ROOT}/scripts/runtime.py" \
  --backend pp \
  --modality text \
  --mode generate \
  --model-path "${MODEL_PATH}" \
  --pp 2 \
  --prompt "${TEXT_PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
```

### 5. `pp-mm-generate`

拓扑固定为：

- `world_size=2`
- `--pp 2`，自动平均切成 `0:17 18:35`

```bash
export MASTER_ADDR="<rank0-host>"
export MASTER_PORT=29534
export NNODES=2
export NODE_RANK="<0-or-1>"

HEXGEN_STARTUP_LOG=1 /usr/bin/time -p "${TORCHRUN}" \
  --nnodes "${NNODES}" \
  --nproc-per-node 1 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  "${RUNTIME_ROOT}/scripts/runtime.py" \
  --backend pp \
  --modality multimodal \
  --mode generate \
  --model-path "${MODEL_PATH}" \
  --frame-dir "${FRAME_DIR}" \
  --num-frames 8 \
  --pp 2 \
  --prompt "${MM_PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
```

### 6. `tp-text-generate`

拓扑固定为：

- `world_size=2`
- `--tp 2`，自动使用单 stage `0:35`

```bash
export MASTER_ADDR="<rank0-host>"
export MASTER_PORT=29535
export NNODES=2
export NODE_RANK="<0-or-1>"

HEXGEN_STARTUP_LOG=1 /usr/bin/time -p "${TORCHRUN}" \
  --nnodes "${NNODES}" \
  --nproc-per-node 1 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  "${RUNTIME_ROOT}/scripts/runtime.py" \
  --backend tp \
  --modality text \
  --mode generate \
  --model-path "${MODEL_PATH}" \
  --tp 2 \
  --prompt "${TEXT_PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
```

### 7. `tp-mm-generate`

拓扑固定为：

- `world_size=2`
- `--tp 2`，自动使用单 stage `0:35`

说明：

- 这组当前先作为 smoke 基线。
- 它现在还不是最终的 multimodal shard-only 形态。

```bash
export MASTER_ADDR="<rank0-host>"
export MASTER_PORT=29536
export NNODES=2
export NODE_RANK="<0-or-1>"

HEXGEN_STARTUP_LOG=1 /usr/bin/time -p "${TORCHRUN}" \
  --nnodes "${NNODES}" \
  --nproc-per-node 1 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  "${RUNTIME_ROOT}/scripts/runtime.py" \
  --backend tp \
  --modality multimodal \
  --mode generate \
  --model-path "${MODEL_PATH}" \
  --frame-dir "${FRAME_DIR}" \
  --num-frames 8 \
  --tp 2 \
  --prompt "${MM_PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
```

### 8. `hybrid-text-generate`

拓扑固定为：

- `world_size=3`
- `--pp 2`，自动平均切成 `0:17 18:35`
- `--tp-degrees 2 1`

```bash
export MASTER_ADDR="<rank0-host>"
export MASTER_PORT=29537
export NNODES=3
export NODE_RANK="<0-or-1-or-2>"

HEXGEN_STARTUP_LOG=1 /usr/bin/time -p "${TORCHRUN}" \
  --nnodes "${NNODES}" \
  --nproc-per-node 1 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  "${RUNTIME_ROOT}/scripts/runtime.py" \
  --backend hybrid \
  --modality text \
  --mode generate \
  --model-path "${MODEL_PATH}" \
  --pp 2 \
  --tp-degrees 2 1 \
  --prompt "${TEXT_PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
```

### 9. `hybrid-mm-generate`

拓扑固定为：

- `world_size=3`
- `--pp 2`，自动平均切成 `0:17 18:35`
- `--tp-degrees 2 1`

```bash
export MASTER_ADDR="<rank0-host>"
export MASTER_PORT=29538
export NNODES=3
export NODE_RANK="<0-or-1-or-2>"

HEXGEN_STARTUP_LOG=1 /usr/bin/time -p "${TORCHRUN}" \
  --nnodes "${NNODES}" \
  --nproc-per-node 1 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  "${RUNTIME_ROOT}/scripts/runtime.py" \
  --backend hybrid \
  --modality multimodal \
  --mode generate \
  --model-path "${MODEL_PATH}" \
  --frame-dir "${FRAME_DIR}" \
  --num-frames 8 \
  --pp 2 \
  --tp-degrees 2 1 \
  --prompt "${MM_PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
```

## 当前比较标准

先按下面这个标准验收：

- 同一个 case，在改动前后，`generated_token_ids` 一致。
- 同一个 case，在改动前后，`generated_text` 一致。
- distributed case 中，不同 rank 打印出的 `generated_token_ids` / `generated_text` 一致。
- `tp / hybrid` case 的 `weight_load` shard shape、stage scope 与 rank-local loaded bytes 证据不回退。
- `pp / hybrid multimodal` case 的 startup contract 不回退到 root/full/replay payload，且 non-stage0 不重新激活 frontend。

runtime core 变更后的最小固定入口：

```bash
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh
```

如果改到权重加载，使用：

```bash
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh --include-weight-loader
```

性能字段目前只记录，不作为 correctness gate 强制比较：

- 更细粒度的 per-rank 常驻显存
- KV cache 占用
- transport/payload bytes 与 elapsed seconds

启动时间、transport/payload profile 和 CUDA peak allocated/reserved 已经进入 `runtime_metrics` 和 `runtime-perf-*` 输出；后续性能优化前后都要保留这张表做对比。

## 下一步

这份文档当前已经记录 20260428 完整 correctness baseline。后续动作是：

- 后续改动统一拿这份结果做对比
- 新代码重跑 baseline 后补齐 CUDA peak alloc/reserved 与 payload bytes
- P3 后续按 ROADMAP 中的具体性能优化顺序推进
