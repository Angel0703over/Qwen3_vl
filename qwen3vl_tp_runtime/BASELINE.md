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

## 验收字段

每个 case 都至少看下面三项：

- `generated_token_ids`
- `generated_text`
- 是否成功完成所有 rank

`tp-text-generate` 额外看：

- rank0 和 rank1 都是 `weight_load.tp_weight_sharded=true`
- rank0 是 `weight_load.tp_shard_rank=0`，rank1 是 `weight_load.tp_shard_rank=1`
- 两个 rank 的 `weight_load.tp_shard_world_size=2`
- 两个 rank 的 `weight_load.tp_shard_shape_ok=true`，`tp_sharded_projection_examples` 中 q/k/v/o 和 MLP projection 是 shard 后形状
- 两个 rank 的 `weight_load.loaded_weight_tensor_bytes` 完全一致
- 两个 rank 的 `weight_load.tp_stage_loaded_weight_tensor_bytes_equal=true`

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
- `/usr/bin/time -p` 的端到端 wall-clock 时间

## 已确认的 shard-only smoke

### `tp-text-generate`

- `rank0`: `tp_weight_sharded=true`, `tp_shard_rank=0`, `tp_shard_world_size=2`
- `rank1`: `tp_weight_sharded=true`, `tp_shard_rank=1`, `tp_shard_world_size=2`
- `rank0/rank1 loaded_weight_tensor_bytes`: `5189532672`
- `generated_token_ids`: `[104455, 9909, 9286, 16488]`
- `generated_text`: `人工智能（Artificial`

### `hybrid-text-generate`

- stage0 rank0/rank1: `tp_weight_sharded=true`, `tp_shard_rank=0/2` and `1/2`
- stage0 rank0/rank1 `loaded_weight_tensor_bytes`: `2594763776`
- stage1 rank2: `tp_weight_sharded=false`, `loaded_weight_tensor_bytes=4411426816`
- `generated_token_ids`: `[104455, 9909, 9286, 16488]`
- `generated_text`: `人工智能（Artificial`

## 已确认的 multimodal direct smoke

### `pp-mm-generate`

- baseline logs: `baseline_runs/20260427/pp-mm-generate-rank0.log`, `baseline_runs/20260427/pp-mm-generate-rank1.log`
- rank0 / stage0: `multimodal_frontend_mode=active`, loaded layers `0..17`, `loaded_top_level_weight_names=["embed_tokens_weight"]`, `stage_weight_scope_ok=true`
- rank1 / stage1: `multimodal_frontend_mode=consume-only`, loaded layers `18..35`, `loaded_top_level_weight_names=["final_norm_weight", "lm_head_weight"]`, `stage_weight_scope_ok=true`
- all ranks `generated_token_ids`: `[87140, 15946, 3837, 101177]`
- all ranks `generated_text`: `视频中，一名`

### `hybrid-mm-generate`

- baseline logs: `baseline_runs/20260427/hybrid-mm-generate-rank0.log`, `baseline_runs/20260427/hybrid-mm-generate-rank1.log`, `baseline_runs/20260427/hybrid-mm-generate-rank2.log`
- stage0 rank0: `tp_weight_sharded=true`, `tp_shard_rank=0`, `tp_shard_world_size=2`, `loaded_weight_tensor_bytes=2594763776`
- stage0 rank1: `tp_weight_sharded=true`, `tp_shard_rank=1`, `tp_shard_world_size=2`, `loaded_weight_tensor_bytes=2594763776`
- stage0 TP bytes equality: `tp_stage_loaded_weight_tensor_bytes_equal=true`, `tp_stage_loaded_weight_tensor_bytes=[2594763776, 2594763776]`
- stage0 scope: loaded layers `0..17`, `loaded_top_level_weight_names=["embed_tokens_weight"]`, `stage_weight_scope_ok=true`
- stage1 rank2: `multimodal_frontend_mode=consume-only`, `tp_weight_sharded=false`, loaded layers `18..35`, `loaded_top_level_weight_names=["final_norm_weight", "lm_head_weight"]`, `loaded_weight_tensor_bytes=4411426816`, `stage_weight_scope_ok=true`
- all ranks `generated_token_ids`: `[87140, 15946, 3837, 101177]`
- all ranks `generated_text`: `视频中，一名`
- runtime-only main path uses `include_runtime_reference=false`, so this smoke does not require `reference_generated_token_ids` / `token_match` in summary.

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
- `tp / hybrid` case 的 `weight_load` shard shape、stage scope 与 TP stage bytes equality 证据不回退。
- `pp / hybrid multimodal` case 的 startup contract 不回退到 root/full/replay payload，且 non-stage0 不重新激活 frontend。

当前还不在这份基线里强制比较：

- 启动时间
- 峰值显存
- 更细粒度的 per-rank 常驻显存
- KV cache 占用

这些放到最后一轮性能验收统一做。

## 下一步

这份文档当前已经记录了 `tp-text-generate`、`hybrid-text-generate`、`pp-mm-generate` 和 `hybrid-mm-generate` 的通过结果。后续动作是：

- 后续改动统一拿这份结果做对比
- 启动时间和峰值显存留到性能验收阶段再收
