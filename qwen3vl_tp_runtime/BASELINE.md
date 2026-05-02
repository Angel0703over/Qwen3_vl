# qwen3vl_tp_runtime Baseline

这份文档只保留当前回归要看的 baseline、关键 before/after 和验收字段。每轮完整表格、rank log 和原始 JSON 保留在 `baseline_runs/*/README.md`。

## 推荐对照目录

| 目的 | 目录 | 用途 |
| --- | --- | --- |
| correctness baseline | `baseline_runs/20260428/` | 固定 generated ids/text |
| 当前性能 baseline | `baseline_runs/20260430-bfloat16-default/` | `bfloat16` 默认通信 dtype 后的性能 |
| Step 15 payload baseline | `baseline_runs/20260430-step15-derived-rebuild/` | multimodal payload keys/bytes |
| Step 16 pinned A/B | `baseline_runs/20260501-step16-pinned-ab/` | decode buffer reuse 后的 pinned memory opt-in 对照 |
| Step 20A KV cache smoke | `baseline_runs/20260501-step20a-kv-cache-smoke/` | `StageKVCache` correctness/perf |
| Step 20A long decode | `baseline_runs/20260501-step20a-kv-cache-long-decode/` | `MAX_NEW_TOKENS=16` profile |
| Step 20B video window cache | `baseline_runs/20260501-step20b-video-window-cache/` | `VideoWindowCacheIndex` metadata correctness/perf |
| Step 20C-0 planner | `baseline_runs/20260501-step20c0-video-kv-plan/` | planner-only compression plan |
| Step 20C-1 selector | `baseline_runs/20260501-step20c1-selector/` | `uniform/swa` selected token stats |
| Step 20C-3 compaction | `baseline_runs/20260502-step20c3-compaction/` | `uniform` opt-in physical compaction |
| Step 20C-4 InfiniPot-V selector | `baseline_runs/20260502-step20c4-infinipot-selector/` | `infinipot-v` opt-in local K/V scoring |
| Step 21 full video input | `baseline_runs/20260502-step21-video-input/` | `--video-path` 完整视频输入 |
| Step 22 smoke automation | `baseline_runs/20260502-step22-2node-smoke/` | 2-node smoke matrix 子集，checker/perf table 产物完整 |

## 固定输出

| case | generated_token_ids | generated_text |
| --- | --- | --- |
| text generate | `[104455, 9909, 9286, 16488]` | `人工智能（Artificial` |
| multimodal generate | `[87140, 15946, 3837, 101177]` | `视频中，一名` |
| tp-mm long decode 16 tokens | `[87140, 15946, 3837, 101177, 105611, 99194, 38035, 113727, 33108, 104362, 38035, 113233, 9370, 104253, 104224, 46944]` | `视频中，一名穿着深色衬衫和浅色裤子的男子站在一个` |

## 当前性能快照

来自 `baseline_runs/20260430-bfloat16-default/runtime-perf-table.md`。

| case | rank | total s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | CUDA peak | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pp3-text-generate` | 0 | `5.92` | `0 B` | `0 B` | `190.00 KiB` | `0.00` | `0 B` | `3.03 GiB` | `2.98 GiB` |
| `pp3-text-generate` | 1 | `5.89` | `0 B` | `0 B` | `285.00 KiB` | `0.00` | `0 B` | `2.30 GiB` | `2.26 GiB` |
| `pp3-text-generate` | 2 | `5.95` | `0 B` | `0 B` | `95.00 KiB` | `0.00` | `0 B` | `3.03 GiB` | `2.98 GiB` |
| `pp3-mm-generate` | 0 | `31.91` | `14.43 MiB` | `0 B` | `6.15 MiB` | `0.00` | `0 B` | `3.86 GiB` | `2.98 GiB` |
| `pp3-mm-generate` | 1 | `31.98` | `7.21 MiB` | `0 B` | `9.23 MiB` | `0.00` | `0 B` | `3.12 GiB` | `2.26 GiB` |
| `pp3-mm-generate` | 2 | `32.03` | `7.21 MiB` | `0 B` | `3.08 MiB` | `0.00` | `0 B` | `3.95 GiB` | `2.98 GiB` |
| `tp-text-generate-default` | 0 | `8.96` | `0 B` | `0 B` | `0 B` | `2.16` | `6.68 MiB` | `4.91 GiB` | `4.83 GiB` |
| `tp-text-generate-default` | 1 | `8.89` | `0 B` | `0 B` | `0 B` | `2.03` | `6.68 MiB` | `4.91 GiB` | `4.83 GiB` |
| `tp-mm-generate-default` | 0 | `53.44` | `11.53 MiB` | `0 B` | `0 B` | `24.51` | `221.48 MiB` | `6.53 GiB` | `4.83 GiB` |
| `tp-mm-generate-default` | 1 | `53.43` | `11.53 MiB` | `0 B` | `0 B` | `23.77` | `221.48 MiB` | `6.52 GiB` | `4.83 GiB` |
| `hybrid-mm-generate-default` | 0 | `32.58` | `3.10 MiB` | `11.53 MiB` | `3.08 MiB` | `2.08` | `113.82 MiB` | `3.73 GiB` | `2.42 GiB` |
| `hybrid-mm-generate-default` | 1 | `32.86` | `0 B` | `11.53 MiB` | `0 B` | `1.63` | `113.82 MiB` | `3.22 GiB` | `2.42 GiB` |
| `hybrid-mm-generate-default` | 2 | `32.66` | `3.10 MiB` | `0 B` | `3.08 MiB` | `0.00` | `0 B` | `5.46 GiB` | `4.11 GiB` |
| `tp-mm-generate-long-default-bfloat16` | 0 | `62.66` | `11.53 MiB` | `0 B` | `0 B` | `28.84` | `225.70 MiB` | `6.53 GiB` | `4.83 GiB` |
| `tp-mm-generate-long-default-bfloat16` | 1 | `62.56` | `11.53 MiB` | `0 B` | `0 B` | `28.07` | `225.70 MiB` | `6.52 GiB` | `4.83 GiB` |

## 关键 Before / After

| 修改 | before | after | 结果 |
| --- | ---: | ---: | --- |
| startup contract 移除 `stage_output` | `13` tensors / `7,563,328` bytes | `12` tensors / `4,353,088` bytes | 少传 reference output |
| startup contract 移除 dense derived tensor | `12` tensors / `4,353,088` bytes | `9` tensors / `3,245,806` bytes | `attention_mask/cos/sin` 本地重建 |
| HYBRID stage1 `tp_degree=1` bypass | `648.46 MiB` TP collective | `0 B` | 清掉 single-rank 伪 collective |
| pure TP comm dtype | `449.12 MiB` collective | `221.48 MiB` collective | 默认 `bfloat16` |
| pure TP runtime input broadcast | `4` events / rank | `0` events | 本地 embedding 或 local stage input |
| Step 15 derived shared | `12,093,371` bytes | `12,068,291` bytes | 少传可重建 shared tensors |
| Step 16 decode 小 tensor | 每 step 新建 mask/token tensor | 复用 decode mask/token buffer | 减少小分配 |
| Step 16 pinned memory | 无开关 | `--transport-pin-memory` | 小幅收益，默认关闭 |
| Step 20A KV cache | decode `torch.cat` + clone cache | `StageKVCache` append/view | 输出不变 |
| Step 20B video window cache | 无 window 索引 | 记录 `window -> KV location` metadata | 输出不变 |
| Step 20C-3 `uniform` compaction | selector 只做统计 | opt-in 物理 compact 本地 KV | active KV bytes 约减半 |
| Step 20C-4 `infinipot-v` selector | range-based token selection | 本地 K/V value-norm + TaR scoring | active KV bytes 约减半 |

## Step 20 KV 管理 Baseline

| 阶段 | case | 关键结果 |
| --- | --- | --- |
| 20A | `tp-text-generate` | `stage_kv_cache.tensor_bytes=1,474,560` / rank，输出不变 |
| 20A | `tp-mm-generate` | `stage_kv_cache.tensor_bytes=46,522,368` / rank，输出不变 |
| 20A | `hybrid-mm-generate` | stage0 `23,261,184` bytes，stage1 `46,522,368` bytes，输出不变 |
| 20A long | `tp-mm MAX_NEW_TOKENS=16` | `stage_kv_cache.tensor_bytes=47,407,104` / rank，输出不变 |
| 20B | `tp-mm` / `hybrid-mm` | 每 rank `4` windows / `576` video tokens，只记录 metadata |
| 20C-0 | planner | `method=none`，`576 / 576 / 0` original/keep/drop，不改 KV |
| 20C-1 | selector | `uniform keep_ratio=0.5` 计划 `576 / 288 / 288`，不改 KV |
| 20C-2 | contract | 固化 physical KV length、attention mask key length、logical position 规则 |
| 20C-3 | `uniform` compaction | prefill physical length `627 -> 339`，输出不变 |
| 20C-4 | `infinipot-v` selector | score 来自本地 K/V，输出不变 |

## Step 20C Compaction 对照

| case | method | rank | total s | CUDA peak | TP coll | active KV before -> after | video tokens | generated |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| `tp-mm-generate-step20c3-uniform-j23` | `uniform` | 0/1 | `53.69` | `6.52-6.53 GiB` | `24.10-24.72s / 221.48 MiB` | `46,227,456 -> 24,993,792` | `576 -> 288` | pass |
| `hybrid-mm-generate-step20c3-uniform-j23shared` | `uniform` | 0/1/2 | `32.37-32.49` | `3.23-5.47 GiB` | stage0 `113.82 MiB`，stage1 `0 B` | stage0 `23,113,728 -> 12,496,896`，stage1 `46,227,456 -> 24,993,792` | `576 -> 288` | pass |
| `tp-mm-generate-step20c4-infinipot-j23` | `infinipot-v` | 0/1 | `53.52` | `6.53-6.54 GiB` | `23.68-24.48s / 221.48 MiB` | `46,227,456 -> 24,993,792` | `576 -> 288` | pass |
| `hybrid-mm-generate-step20c4-infinipot-j23shared` | `infinipot-v` | 0/1/2 | `32.81-32.90` | `3.23-5.49 GiB` | stage0 `113.82 MiB`，stage1 `0 B` | stage0 `23,113,728 -> 12,496,896`，stage1 `46,227,456 -> 24,993,792` | `576 -> 288` | pass |

结论：

- `uniform` 和 `infinipot-v` 在当前短 smoke 中都保持固定 multimodal 输出。
- 两者压缩率相同，active KV bytes 约减半。
- allocated KV buffer bytes 不变，因为当前仍预分配整块 buffer，只收紧 active length。
- `infinipot-v` 的 token selection 来自本地 K/V scoring；质量收益需要更长视频和更多问题评估。

## Step 21 完整视频输入 Baseline

来自 `baseline_runs/20260502-step21-video-input/`。输入为 `test/demo.mp4 --video-nframes 4`，Jetson 上使用 `pyav_frame_adapter`，`video_grid_thw=[[2, 40, 74]]`。

| case | rank | total s | generated ids/text | startup | handoff | TP collective | CUDA peak |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `hf-mm-generate-video-builder-prompt` | - | `37.12` | `[87140, 108869, 100369, 102122]` / `视频展示了两个场景` | `0 B` | `0 B` | `0 B` | `9.76 GiB` |
| `pp-mm-generate-video` | 0/1 | `74.54-76.42` | same | `7.42 MiB` | `7.41 MiB` | `0 B` | `7.44-7.48 GiB` |
| `tp-mm-generate-video` | 0/1 | `106.15-106.20` | same | `29.10 MiB` | `0 B` | `55.12-55.69s / 533.67 MiB` | `8.85-8.86 GiB` |
| `hybrid-mm-generate-video-pp2tp1` | 0/1 | `58.76-58.79` | same | `7.42 MiB` | `7.41 MiB` | `0 B` | `7.45-7.50 GiB` |
| `tp-mm-generate-frame-regression` | 0/1 | `53.06-53.08` | `[87140, 15946, 3837, 101177]` / `视频中，一名` | `11.51 MiB` | `0 B` | `23.84-24.63s / 221.48 MiB` | `6.52-6.53 GiB` |

Payload 结论：

- PP/HYBRID startup contract：`5` tensors，`7,781,072` bytes，keys 为 `shared.input_ids`、`shared.rope_deltas`、`shared.mm_token_type_ids`、`shared.video_grid_thw`、`stage_handoffs.1.stage_input`。
- TP startup contract：`9` tensors，`30,515,387` bytes，keys 为 shared tensors、`stage_handoffs.0.stage_input`、`stage_visuals.0.visual_pos_masks` 和 `deepstack_by_layer.0/1/2`。
- 不传原始视频、frame bytes、root/full/replay payload。
- non-owner rank 为 `multimodal_frontend_mode=consume-only`。
- `live-mm-generate` 完整视频 trace 15 分钟未产出 JSON，暂不冻结；后续放到代码/API 清理阶段处理。

## Step 22 Smoke Matrix Baseline

来自 `baseline_runs/20260502-step22-2node-smoke/`。这轮由 Codex tool shell 作为 controller 启动；该 shell 看不到 jetson1 的 CUDA device nodes，所以先冻结 jetson2/jetson3 的 2-node 子集。物理 jetson1 普通终端 CUDA 可用，3-rank `hybrid-mm-generate` 后续可通过普通登录/SSH 把 jetson1 作为第三 rank 补跑。

| case | ranks | total s | generated | key metrics |
| --- | ---: | ---: | --- | --- |
| `hf-text-generate` | 1 | `8.76` | `[104455, 9909, 9286, 16488]` / `人工智能（Artificial` | CUDA peak `8.28 GiB` |
| `hf-mm-generate` | 1 | `10.92` | same text output | CUDA peak `8.55 GiB` |
| `pp-mm-generate` | 2 | `30.25-30.29` | `[87140, 15946, 3837, 101177]` / `视频中，一名` | startup `3.07 MiB`，handoff `3.08 MiB`，stage KV `44.09 / 44.37 MiB` |
| `tp-mm-generate` | 2 | `52.95-52.99` | same multimodal output | startup `11.51 MiB`，TP collective `23.83-24.42s / 221.48 MiB` |
| `tp-mm-generate-long` | 2 | `62.03-62.05` | long decode fixed output | TP collective `27.89-28.58s / 225.70 MiB`，stage KV `44.09 / 45.21 MiB` |
| `tp-mm-generate-frame-regression` | 2 | `52.96-52.97` | same multimodal output | frame-dir 旧路径回归通过 |

产物：

- `check-smoke-matrix.txt`：全部 PASS。
- `runtime-perf-records.json` / `runtime-perf-table.md`：已生成统一 perf 表。
- `collect-runtime-perf.txt`：perf collector 命令日志。

## Payload 规则

- `None` tensor slot 不进入 startup transport。
- `shared.attention_mask_2d` 可重建时不传。
- `shared.position_ids` 可重建时不传。
- `attention_mask/cos/sin` 保持本地重建，不进入 generate startup contract。
- stage0/input-owner 的 `stage_input` 是 processed `inputs_embeds`。
- non-stage0 的 `stage_input` 是上一 PP stage 输出，等价于 `intermediate_tensors`，不能从 prompt 本地重建。
- `deepstack_by_layer` 只发送给实际消费该 layer 的 stage/rank。
- non-input-owner 不能为了重建 deepstack 重新跑视觉 frontend。
- 禁止重新引入 `root_input`、`boundaries`、`hidden_states`、`stage_output`、frontend paths 或 replay/full payload。

## 验收字段

每次 runtime 主路径改动至少检查：

- `generated_token_ids`
- `generated_text`
- distributed ranks 是否全部成功结束
- `runtime_metrics.transport` 的 payload keys/count/bytes
- `runtime_metrics.memory` 的 CUDA peak allocated/reserved
- `weight_load.loaded_weight_tensor_bytes`

Step 22 checker：

```bash
PYTHONPATH=. python qwen3vl_tp_runtime/scripts/check_baseline_logs.py \
  --matrix step22 \
  --baseline-dir baseline_runs/<new-step22-dir> \
  --include-optional \
  --require-transport-metrics
```

矩阵定义在 `qwen3vl_tp_runtime/scripts/smoke_matrix.py`。单个 case 也可以继续用 `--case-id ... rank*.log` 检查。

Step 22 perf table：

```bash
PYTHONPATH=. python qwen3vl_tp_runtime/scripts/collect_runtime_perf.py \
  --baseline-dir baseline_runs/<new-step22-dir> \
  --matrix step22 \
  --output-json baseline_runs/<new-step22-dir>/runtime-perf-records.json \
  --output-md baseline_runs/<new-step22-dir>/runtime-perf-table.md
```

统一表格字段：`total s`、startup/scaffold/handoff bytes、TP collective seconds/bytes、CUDA peak、loaded weights、`stage KV bytes`。`stage KV bytes` 有 active bytes 时显示 `active / allocated`。

Step 22 一键生成 baseline 目录：

```bash
TP_HOSTS="local 10.126.126.4" \
PP_HOSTS="local 10.126.126.4" \
HYBRID_HOSTS="local 10.126.126.4 10.126.126.5" \
bash qwen3vl_tp_runtime/scripts/helpers/run-step22-smoke-matrix.sh
```

包含完整视频 optional cases：

```bash
VIDEO_PATH=/mnt/ssd/code/Qwen3_vl/test/demo.mp4 \
bash qwen3vl_tp_runtime/scripts/helpers/run-step22-smoke-matrix.sh --include-optional
```

脚本会写入 `check-smoke-matrix.txt`、`collect-runtime-perf.txt`、`runtime-perf-records.json`、`runtime-perf-table.md` 和 `README.md`。

TP / HYBRID 额外检查：

- `weight_load.tp_weight_sharded=true` 不回退。
- TP rank 的 `tp_shard_rank` / `tp_shard_world_size` 正确。
- `tp_stage_loaded_weight_tensor_bytes_equal=true`。
- stage-local scope 正确，stage1 只加载自己的 layer range + final output 权重。

multimodal 额外检查：

- non-stage0 / non-input-owner 是 consume-only。
- startup/runtime input 不包含 root/full/replay payload。
- payload 不包含 `stage_output`、`hidden_states`、frontend paths。

## 常用命令

统一环境：

```bash
export REPO_ROOT=/mnt/ssd/code/Qwen3_vl
export RUNTIME_ROOT="${REPO_ROOT}/qwen3vl_tp_runtime"
export PYTHON_BIN=/mnt/ssd/miniconda3/envs/vlm/bin/python
export TORCHRUN=/mnt/ssd/miniconda3/envs/vlm/bin/torchrun
export MODEL_PATH=/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct
export FRAME_DIR=/mnt/ssd/code/Qwen3_vl/frames
export MAX_NEW_TOKENS=4
```

Distributed multimodal wrappers：

```bash
# TP, 2 nodes
NODE_RANK=0 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-tp-mm-generate.sh"
NODE_RANK=1 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-tp-mm-generate.sh"

# PP, N nodes
NNODES=N NODE_RANK=0 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-pp-mm-generate.sh"

# HYBRID, current 3-rank baseline
NODE_RANK=0 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-hybrid-mm-generate.sh"
NODE_RANK=1 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-hybrid-mm-generate.sh"
NODE_RANK=2 MASTER_ADDR="<rank0-host>" bash "${RUNTIME_ROOT}/scripts/helpers/run-hybrid-mm-generate.sh"
```

本地最小回归：

```bash
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh
```
