# qwen3vl_tp_runtime Baseline

这份文档用于冻结当前阶段的最小回归基线。

目标：

- 固定一组 `hf / live / pp / tp / hybrid` 的 generate 回归命令。
- 后续改动统一拿这组命令做回归。
- correctness baseline 固定 `generated_token_ids`、`generated_text`，以及 `tp / hybrid` 的 `weight_load` shard/stage scope 证据。
- performance/profile baseline 固定启动耗时、transport payload bytes、TP collective bytes、CUDA peak memory 和 loaded weight bytes。

## 当前推荐 baseline

- 完整 correctness baseline：`baseline_runs/20260428/`。
- step 13 长期目标真实 profile：`baseline_runs/20260429-longterm-profile/`。
- 最小本地回归入口：`bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh`。
- 改动 runtime 主路径后，至少检查对应 distributed case 的 generated ids/text、payload keys/bytes 和 `weight_load`。

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
| `tp-mm-generate` | `tp` | `multimodal` | `是` | 已覆盖 input-owner startup contract |
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

当前硬性记录：

1. `stdout` 里的最终 JSON
2. distributed case 的 `HEXGEN_STARTUP_LOG=1` 启动日志
3. `runtime_metrics.transport` 中的 payload keys/count/bytes
4. `runtime_metrics.memory` 中的 CUDA peak allocated/reserved
5. `weight_load.loaded_weight_tensor_bytes` 和 TP shard/stage scope 证据

如果方便，建议额外保存 `stderr` 或终端里的 `/usr/bin/time -p` 结果；如果某轮没有 `/usr/bin/time`，以 runtime JSON 里的 `runtime_total_seconds` 为准。
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
- 这轮当时只覆盖 PP/TP profiling；后续 3-rank HYBRID profile 已由 `baseline_runs/20260429-longterm-profile/` 补齐并冻结。

## 20260429 startup contract stage_output 减量记录

本轮优化把主路径 multimodal generate startup contract 中的 reference `stage_output` 移出，只保留后续 stage 启动必须的 `stage_input`。

输出目录：

- `baseline_runs/20260429-startup-contract-opt/`

重跑结果：

- `pp-mm-generate-startup-opt`: PASS，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`
- `hybrid-mm-generate-startup-opt`: PASS，2 节点 `--pp 2 --tp-degrees 1 1` 变体，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`

性能 / payload 汇总：

- `baseline_runs/20260429-startup-contract-opt/runtime-perf-records.json`
- `baseline_runs/20260429-startup-contract-opt/runtime-perf-table.md`

对比：

- before `baseline_runs/20260428-step11-profile/pp-mm-generate-*`：startup contract `13` tensors，`7,563,328` bytes，包含 `stage_handoffs.1.stage_output`。
- after `baseline_runs/20260429-startup-contract-opt/*startup-opt*`：startup contract `12` tensors，`4,353,088` bytes，只包含 `stage_handoffs.1.stage_input`，不再包含 `stage_handoffs.1.stage_output`。

注意：

- `hybrid-mm-generate-startup-opt` 是 HYBRID startup contract 路径验证，不替代 frozen 3-rank `hybrid-mm-generate` correctness baseline。
- 完整 3-rank HYBRID `--pp 2 --tp-degrees 2 1` 后续已由 `baseline_runs/20260429-longterm-profile/` 验证并冻结。

## 20260429 startup contract derived tensor 减量记录

本轮继续把主路径 multimodal generate startup contract 中可本地重建的 derived shared tensor 移出：

- 不再传 `shared.attention_mask`
- 不再传 `shared.cos`
- 不再传 `shared.sin`

输出目录：

- `baseline_runs/20260429-startup-contract-derived-opt/`

重跑结果：

- `pp-mm-generate-derived-opt`: PASS，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`
- `hybrid-mm-generate-derived-opt`: PASS，2 节点 `--pp 2 --tp-degrees 1 1` 变体，generated ids `[87140, 15946, 3837, 101177]`，text `视频中，一名`

性能 / payload 汇总：

- `baseline_runs/20260429-startup-contract-derived-opt/runtime-perf-records.json`
- `baseline_runs/20260429-startup-contract-derived-opt/runtime-perf-table.md`

对比：

- before `baseline_runs/20260428-step11-profile/pp-mm-generate-*`：startup contract `13` tensors，`7,563,328` bytes。
- stage-output after `baseline_runs/20260429-startup-contract-opt/*startup-opt*`：startup contract `12` tensors，`4,353,088` bytes。
- derived-tensor after `baseline_runs/20260429-startup-contract-derived-opt/*derived-opt*`：startup contract `9` tensors，`3,245,806` bytes。

最终 startup contract tensor payload keys：

- `shared.input_ids`
- `shared.attention_mask_2d`
- `shared.position_ids`
- `shared.rope_deltas`
- `shared.mm_token_type_ids`
- `shared.image_grid_thw`
- `shared.video_grid_thw`
- `stage_handoffs.1.stage_input`
- `stage_visuals.1.visual_pos_masks`

注意：

- `shared.attention_mask_2d` 和 `shared.position_ids` 是 compact input / position metadata，不是 dense decoder `attention_mask` 或 RoPE `cos/sin`。
- dense `attention_mask`、RoPE `cos`、RoPE `sin` 现在由 non-stage0 在本地 `StageState` materialize 时重建。
- `hybrid-mm-generate-derived-opt` 是 2 节点 HYBRID startup contract 路径验证，不替代 frozen 3-rank `hybrid-mm-generate` correctness baseline。

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

- 这组已纳入当前 correctness baseline。
- 后续 `baseline_runs/20260429-longterm-profile/` 已覆盖 pure TP multimodal input-owner：rank0 准备 compact startup contract，其他 TP rank consume-only，所有 TP rank 本地 materialize 自己的 weight shard。

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

## 2026-04-29 scaffold broadcast alias 去重记录

本轮是 `ROADMAP.md` step 13 的第一段低风险优化，只处理明确重复的 HYBRID scaffold tensor：

- 场景：单元测试覆盖的 multimodal runtime-only generate weightless scaffold。
- 行为：`stage_input` 和 `layer_input` 是同一个 tensor alias 时，scaffold 不再单独携带 `layer_input`；transport packer 也会把重复 alias 折成 tensor ref。
- before tensor count / bytes：3 tensors / 216 bytes。
- before payload keys：`scaffold.stage_input`、`scaffold.layer_input`、`scaffold.prefill_attention_mask_2d`。
- after tensor count / bytes：2 tensors / 120 bytes。
- after payload keys：`scaffold.stage_input`、`scaffold.prefill_attention_mask_2d`。
- correctness：本地 `test_model_weight_loader.py`、`test_hybrid_direct_loader.py`、`test_tensor_parallel_direct.py` 已通过。

本节记录的是单元 slice。完整 3-rank HYBRID `--pp 2 --tp-degrees 2 1` 的真实 Jetson profile 后续已由 `baseline_runs/20260429-longterm-profile/` 冻结，用于对照最终 runtime input / scaffold bytes。

## 2026-04-29 scaffold rank-local 字段减量记录

本轮是 `ROADMAP.md` step 13 的中期第一步，只处理 rank-local 可推导 scalar metadata：

- 场景：HYBRID stage-group scaffold broadcast。
- 行为：leader 发送 scaffold 前移除 `stage_idx`、`start_idx`、`end_idx`、`save_dtype`、`hidden_size`、`batch_size`；本地再从 `StageSpec`、compute dtype、text config 和 runtime tensor shape 恢复。
- before object meta bytes：153 bytes。
- before meta keys：`batch_size`、`end_idx`、`hidden_size`、`layers`、`save_dtype`、`stage_idx`、`start_idx`。
- after object meta bytes：94 bytes。
- after meta keys：`layers`、`rank_local_fields_local_rebuild`。
- tensor count / tensor bytes：0 / 0，本步主要减少 object metadata，tensor payload keys 不因此改变。
- correctness：本地 `test_hybrid_direct_loader.py`、`test_model_weight_loader.py`、`test_tensor_parallel_direct.py` 已通过。

本节记录的是单元 slice。完整 3-rank HYBRID 真实 profile 后续已由 `baseline_runs/20260429-longterm-profile/` 冻结，用于对照最终 `runtime_inputs_meta` / scaffold object bytes。

## 2026-04-29 scaffold prefill runtime tensor 本地重建记录

本轮是 `ROADMAP.md` step 13 的中期第二步，处理 HYBRID multimodal scaffold 中重复的 top-level prefill runtime tensors：

- 场景：HYBRID multimodal runtime-only generate stage-group scaffold broadcast。
- 行为：leader 发送 scaffold 前移除 `prefill_attention_mask_2d`、`prefill_attention_mask`、`prefill_position_ids`、`prefill_cos`、`prefill_sin`；本地用 `_mm_startup_shared` + `stage_input` 通过 `build_mm_stage_state()` 重建。
- before tensor count / bytes：7 tensors / 308 bytes。
- before tensor keys：`scaffold.stage_input`、`scaffold.rope_deltas`、`scaffold.prefill_attention_mask_2d`、`scaffold.prefill_attention_mask`、`scaffold.prefill_position_ids`、`scaffold.prefill_cos`、`scaffold.prefill_sin`。
- before object meta bytes：485 bytes。
- after tensor count / bytes：2 tensors / 56 bytes。
- after tensor keys：`scaffold.stage_input`、`scaffold.rope_deltas`。
- after object meta bytes：251 bytes。
- correctness：本地 `test_hybrid_direct_loader.py`、`test_model_weight_loader.py`、`test_tensor_parallel_direct.py` 已通过。

本节记录的是单元 slice。完整 3-rank HYBRID 真实 profile 后续已由 `baseline_runs/20260429-longterm-profile/` 冻结，用于对照最终 runtime input tensor keys / bytes。

## 2026-04-29 scaffold frontend metadata 收口记录

本轮是 `ROADMAP.md` step 13 的中期收口，处理 HYBRID multimodal scaffold 中只用于 frontend 来源记录的 metadata：

- 场景：HYBRID multimodal runtime-only generate stage-group scaffold broadcast。
- 行为：leader 发送 scaffold 前移除 `num_frames`、`frame_paths`；本地 materialize 后从 `_mm_num_frames` / `_mm_frame_paths` 恢复兼容字段，不重新读媒体、不重新跑视觉 frontend。
- before object meta bytes：347 bytes。
- before meta keys：`frame_paths`、`layers`、`mm_prefill_runtime_tensors_local_rebuild`、`modality`、`num_frames`、`rank_local_fields_local_rebuild`、`rope_deltas`、`runtime_only_generate`、`stage_input`。
- after object meta bytes：324 bytes。
- after meta keys：`layers`、`mm_frontend_metadata_local_rebuild`、`mm_prefill_runtime_tensors_local_rebuild`、`modality`、`rank_local_fields_local_rebuild`、`rope_deltas`、`runtime_only_generate`、`stage_input`。
- tensor count / bytes：2 tensors / 56 bytes，本步不改变 tensor payload。
- tensor keys：`scaffold.stage_input`、`scaffold.rope_deltas`。
- correctness：本地 `test_hybrid_direct_loader.py`、`test_model_weight_loader.py` 已通过。

`text_scaffold` / `stage_scaffold` 当前只保留给 reference/debug 兼容路径和历史 profile label；runtime-only 主路径开始切到 `runtime_inputs` transport，字段差异已收紧到真实 modality runtime input。

## 2026-04-29 runtime input broadcast 第一轮记录

本轮是 `ROADMAP.md` step 13 的长期第一刀：HYBRID `runtime-only generate` 的 stage-group broadcast 不再发送 `scaffold` root，而是发送 `runtime_inputs` root。

- 场景：HYBRID direct runtime-only generate，`include_runtime_reference=false`。
- 新协议：`hybrid_runtime_inputs_v1`。
- 行为：leader 广播 shared request/runtime input dict；follower 用 `StageSpec`、runtime config 和 startup shared metadata 本地恢复最小 StageState scaffold，再 materialize 本 rank 权重 shard。
- reference/debug/file-backed path：暂时保留旧 `text_scaffold` / `stage_scaffold` transport。

Multimodal runtime-only slice：

- before root：`scaffold`。
- before object meta bytes：433 bytes。
- before tensor count / bytes：2 tensors / 56 bytes。
- before tensor keys：`scaffold.stage_input`、`scaffold.rope_deltas`。
- before meta keys：`layers`、`max_new_tokens`、`mm_frontend_metadata_local_rebuild`、`mm_prefill_runtime_tensors_local_rebuild`、`modality`、`module_name`、`rank_local_fields_local_rebuild`、`rope_deltas`、`runtime_only_generate`、`stage_input`、`stage_type`。
- after root：`runtime_inputs`。
- after object meta bytes：270 bytes。
- after tensor count / bytes：2 tensors / 56 bytes。
- after tensor keys：`runtime_inputs.stage_input`、`runtime_inputs.rope_deltas`。
- after meta keys：`modality`、`mode`、`protocol`、`rope_deltas`、`runtime_only_generate`、`stage_input`。

Text runtime-only slice：

- before root：`scaffold`。
- before object meta bytes：271 bytes。
- before tensor count / bytes：0 tensors / 0 bytes。
- after root：`runtime_inputs`。
- after object meta bytes：190 bytes。
- after tensor count / bytes：0 tensors / 0 bytes。
- after meta keys：`modality`、`mode`、`protocol`、`runtime_only_generate`、`runtime_only_prompt_local_rebuild`。

correctness：本地 `test_hybrid_direct_loader.py`、`test_model_weight_loader.py`、`test_runtime_summary.py`、`test_tensor_parallel_direct.py` 已通过；`run-runtime-core-regression.sh` 已通过，baseline checker 中 PP/TP/HYBRID text/mm generate 均为 PASS。

## 2026-04-29 runtime input broadcast 第二轮记录

本轮是 `ROADMAP.md` step 13 的长期第二刀：HYBRID `runtime-only generate` 的 stage leader 不再先构造 weightless scaffold-like `StageState`，再从中抽 `runtime_inputs`。

- 场景：HYBRID direct runtime-only generate，`include_runtime_reference=false`。
- 新行为：
  - text：直接从 `_runtime_only_input_ids` / `_runtime_only_attention_mask` 组 `runtime_inputs`。
  - multimodal：直接从 `_mm_startup_shared`、`_mm_startup_stage_handoffs[stage_idx]` 和可选 `_mm_startup_stage_visuals[stage_idx]` 组 `runtime_inputs`。
  - leader 在 runtime-only stage-group broadcast 分支不再调用 `build_direct_stage_state(..., include_text_weights=False)`。
  - follower 仍会把 `runtime_inputs` 恢复成 materialize 所需的最小 StageState scaffold；权重继续由本 rank 从 `model_path` materialize。
- reference/debug/file-backed path：仍保留旧 `text_scaffold` / `stage_scaffold` transport。

Text runtime-only slice：

- before object meta bytes：190 bytes。
- before tensor count / bytes：0 tensors / 0 bytes。
- before tensor keys：空。
- after object meta bytes：249 bytes。
- after tensor count / bytes：1 tensor / 24 bytes。
- after tensor keys：`runtime_inputs.input_ids`。

Multimodal runtime-only slice：

- before object meta bytes：270 bytes。
- before tensor count / bytes：2 tensors / 56 bytes。
- before tensor keys：`runtime_inputs.stage_input`、`runtime_inputs.rope_deltas`。
- after object meta bytes：441 bytes。
- after tensor count / bytes：4 tensors / 104 bytes。
- after tensor keys：`runtime_inputs.shared.input_ids`、`runtime_inputs.shared.attention_mask_2d`、`runtime_inputs.shared.rope_deltas`、`runtime_inputs.stage_handoff.stage_input`。

说明：本轮 bytes 增加是预期的，因为 payload 从“依赖 scaffold 派生字段 + 本地 runtime_config 隐式补齐”改为“自洽的 request/runtime input dict”。这一步的收益是移除 leader 构造 scaffold-like StageState 的启动工作和语义依赖，而不是单纯压缩 tensor bytes。

correctness：

- `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py`
- `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py`
- `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py`
- `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_distributed_serialization.py`
- `bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh`

## 2026-04-29 runtime input schema 固化记录

本轮是 `ROADMAP.md` step 13 的长期第三刀：把 `hybrid_runtime_inputs_v1` 从内部 helper 约定固化成正式 schema。

- 新增：
  - `HYBRID_RUNTIME_INPUT_PROTOCOL`
  - `HybridRuntimeInputSchema`
- schema 现在明确：
  - text runtime input 只允许 `protocol` / `modality` / `mode` / `runtime_only_generate` / `input_ids` / `attention_mask_2d` / `runtime_only_prompt_local_rebuild`。
  - multimodal runtime input 只允许 `shared` / `stage_handoff` / `stage_visuals` 三类 runtime dict。
  - `stage_handoff` 只允许 `stage_input`，不允许 `stage_output`。
  - `shared` 不允许 dense `attention_mask`、`cos`、`sin` 这类 derived tensor。
  - weights、layers、frontend paths、replay/full payload、rank-local StageState 字段都被拒绝。
- enforcement：
  - leader build runtime input 时校验。
  - broadcast restore 后、resolve compute dtype 前校验。
  - 恢复最小 StageState scaffold 前再次校验。
- payload 对比：
  - 本步不改变 runtime input tensor keys、tensor count 或 bytes。
  - text 仍保持第二刀 after：1 tensor / 24 bytes，key=`runtime_inputs.input_ids`。
  - multimodal 仍保持第二刀 after：4 tensors / 104 bytes，keys=`runtime_inputs.shared.input_ids` / `runtime_inputs.shared.attention_mask_2d` / `runtime_inputs.shared.rope_deltas` / `runtime_inputs.stage_handoff.stage_input`。

correctness：

- `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_runtime_input_schema.py`
- `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py`
- `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_compat_package_exports.py`
- `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py`
- `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py`

## 2026-04-29 pure TP multimodal input-owner 记录

本轮是 `ROADMAP.md` step 13 的长期第四刀：纯 TP multimodal 对齐 vLLM-style input-owner。

- 场景：`backend=tp` direct multimodal generate，`include_runtime_reference=false`。
- before：每个 TP rank 都会在 direct scaffold 构建阶段以 `mm_activate_frontend=True` active multimodal frontend / file-backed prefill reference；没有 input-owner startup broadcast，因此 startup transport 是 0 tensors / 0 bytes。
- after：
  - rank0/input owner 运行一次 frontend，导出 thin multimodal startup contract。
  - rank0 和非 rank0 都 seed 本地 startup contract。
  - 最终 `build_direct_stage_state(... include_text_weights=False, mm_activate_frontend=False)`，所有 TP rank 都是 consume-only。
  - TP weight shard 仍由每个 rank 从 `model_path` 本地 materialize，不进入 broadcast。

Runtime-only generate 代表性 startup contract slice：

- object meta bytes：82 bytes。
- meta keys：`frame_paths`、`num_frames`。
- tensor count / bytes：9 tensors / 239 bytes。
- tensor keys：`shared.input_ids`、`shared.attention_mask_2d`、`shared.position_ids`、`shared.rope_deltas`、`shared.mm_token_type_ids`、`shared.image_grid_thw`、`shared.video_grid_thw`、`stage_handoffs.0.stage_input`、`stage_visuals.0.visual_pos_masks`。
- 不包含：weights、layers、`stage_output`、dense `shared.attention_mask`、`shared.cos`、`shared.sin`、root/full/replay payload。

本地单元覆盖：

- `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py`
- `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py`
- `PYTHONPATH=.:test /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py`
- `bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh`

真实 Jetson profile 已在下一节确认：

- `tp-mm-generate` rank0 log 中出现 `prepare multimodal input-owner startup contract`。
- rank0/rank1 的最终 builder 都是 `multimodal_frontend_mode=consume-only startup_contract_ready=True`。
- rank0/rank1 的 `weight_load.tp_weight_sharded=true`、`tp_shard_rank=0/2` 和 `1/2` 不变。
- `generated_token_ids` / `generated_text` 不变。

## 2026-04-29 step 13 长期目标真实 profile 冻结

输出目录：

- `baseline_runs/20260429-longterm-profile/`

拓扑：

- `tp-mm-generate`: Jetson2 rank0，Jetson3 rank1，`--backend tp --tp 2`。
- `hybrid-text-generate`: Jetson2 rank0/rank1，Jetson3 rank2，`--backend hybrid --pp 2 --tp-degrees 2 1`。
- `hybrid-mm-generate`: Jetson2 rank0/rank1，Jetson3 rank2，`--backend hybrid --pp 2 --tp-degrees 2 1`。
- `MASTER_ADDR=10.126.126.3`。

自动检查：

- `baseline_runs/20260429-longterm-profile/check-longterm-baseline.txt`
- `tp-mm-generate`: PASS
- `hybrid-text-generate`: PASS
- `hybrid-mm-generate`: PASS

Correctness：

| case_id | rank logs | generated_token_ids | generated_text |
| --- | --- | --- | --- |
| `tp-mm-generate` | `baseline_runs/20260429-longterm-profile/tp-mm-generate-rank0.log`, `baseline_runs/20260429-longterm-profile/tp-mm-generate-rank1.log` | `[87140, 15946, 3837, 101177]` | `视频中，一名` |
| `hybrid-text-generate` | `baseline_runs/20260429-longterm-profile/hybrid-text-generate-rank0.log`, `baseline_runs/20260429-longterm-profile/hybrid-text-generate-rank1.log`, `baseline_runs/20260429-longterm-profile/hybrid-text-generate-rank2.log` | `[104455, 9909, 9286, 16488]` | `人工智能（Artificial` |
| `hybrid-mm-generate` | `baseline_runs/20260429-longterm-profile/hybrid-mm-generate-rank0.log`, `baseline_runs/20260429-longterm-profile/hybrid-mm-generate-rank1.log`, `baseline_runs/20260429-longterm-profile/hybrid-mm-generate-rank2.log` | `[87140, 15946, 3837, 101177]` | `视频中，一名` |

关键 payload / weight 证据：

- `tp-mm-generate`
  - rank0 出现 `prepare multimodal input-owner startup contract`。
  - rank0/rank1 最终 builder 都是 `multimodal_frontend_mode=consume-only startup_contract_ready=True`。
  - `tp_multimodal_startup_contract_meta`: 422 object bytes。
  - `tp_multimodal_startup_contract_tensors`: 12 tensors / 12,093,371 tensor bytes。
  - tensor keys：`shared.input_ids` / `shared.attention_mask_2d` / `shared.position_ids` / `shared.rope_deltas` / `shared.mm_token_type_ids` / `shared.image_grid_thw` / `shared.video_grid_thw` / `stage_handoffs.0.stage_input` / `stage_visuals.0.visual_pos_masks` / `stage_visuals.0.deepstack_by_layer.0` / `stage_visuals.0.deepstack_by_layer.1` / `stage_visuals.0.deepstack_by_layer.2`。
  - rank0/rank1 `loaded_weight_tensor_bytes=5,189,532,672`，`tp_weight_sharded=true`，`tp_shard_rank=0/1`。
- `hybrid-text-generate`
  - stage0 `runtime_inputs_meta`: 249 object bytes。
  - stage0 `runtime_inputs_tensors`: 1 tensor / 128 tensor bytes，key=`runtime_inputs.input_ids`。
  - rank0/rank1 `loaded_weight_tensor_bytes=2,594,763,776`，`tp_stage_loaded_weight_tensor_bytes_equal=true`。
  - rank2 `loaded_weight_tensor_bytes=4,411,426,816`。
- `hybrid-mm-generate`
  - stage1 startup contract：422 object bytes，9 tensors / 3,245,384 tensor bytes。
  - stage0 `runtime_inputs_meta`: 938 object bytes。
  - stage0 `runtime_inputs_tensors`: 11 tensors / 12,093,371 tensor bytes。
  - runtime input keys：`runtime_inputs.shared.input_ids` / `runtime_inputs.shared.attention_mask_2d` / `runtime_inputs.shared.position_ids` / `runtime_inputs.shared.rope_deltas` / `runtime_inputs.shared.mm_token_type_ids` / `runtime_inputs.shared.video_grid_thw` / `runtime_inputs.stage_handoff.stage_input` / `runtime_inputs.stage_visuals.visual_pos_masks` / `runtime_inputs.stage_visuals.deepstack_by_layer.0` / `runtime_inputs.stage_visuals.deepstack_by_layer.1` / `runtime_inputs.stage_visuals.deepstack_by_layer.2`。
  - rank0/rank1 `loaded_weight_tensor_bytes=2,594,763,776`，`tp_stage_loaded_weight_tensor_bytes_equal=true`。
  - rank2 `loaded_weight_tensor_bytes=4,411,426,816`。

Perf table：

- `baseline_runs/20260429-longterm-profile/runtime-perf-records.json`
- `baseline_runs/20260429-longterm-profile/runtime-perf-table.md`

| case | rank | total s | startup bytes | scaffold bytes | TP coll bytes | cuda peak alloc | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tp-mm-generate` | 0 | 74.64 | 11.53 MiB | 0 B | 449.12 MiB | 6.53 GiB | 4.83 GiB |
| `tp-mm-generate` | 1 | 74.66 | 11.53 MiB | 0 B | 449.12 MiB | 6.53 GiB | 4.83 GiB |
| `hybrid-text-generate` | 0 | 10.77 | 0 B | 377 B | 6.87 MiB | 2.46 GiB | 2.42 GiB |
| `hybrid-text-generate` | 1 | 7.43 | 0 B | 377 B | 6.87 MiB | 2.46 GiB | 2.42 GiB |
| `hybrid-text-generate` | 2 | 10.56 | 0 B | 0 B | 19.56 MiB | 4.18 GiB | 4.11 GiB |
| `hybrid-mm-generate` | 0 | 35.06 | 3.10 MiB | 11.53 MiB | 227.64 MiB | 3.73 GiB | 2.42 GiB |
| `hybrid-mm-generate` | 1 | 35.36 | 0 B | 11.53 MiB | 227.64 MiB | 3.22 GiB | 2.42 GiB |
| `hybrid-mm-generate` | 2 | 35.14 | 3.10 MiB | 0 B | 648.46 MiB | 5.46 GiB | 4.11 GiB |

## 下一步

这份文档当前已经记录 20260428 完整 correctness baseline 和 20260429 step 13 长期目标真实 profile。后续动作是：

- correctness 改动默认拿 `baseline_runs/20260428/` 做 generated ids/text 和 shard scope 对比。
- startup/runtime input/payload 改动默认拿 `baseline_runs/20260429-longterm-profile/` 做 payload keys/bytes、CUDA peak memory 和 loaded weight bytes 对比。
- P3 后续按 `ROADMAP.md` 的具体性能优化顺序推进。
