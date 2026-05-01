# qwen3vl_tp_runtime Baseline

这份文档只记录当前要拿来回归的 baseline、关键 before/after 效果和验收字段。历史细节留在 `baseline_runs/*/README.md` 和 rank logs 里。

## 推荐对照目录

| 目的 | 目录 | 用途 |
| --- | --- | --- |
| correctness baseline | `baseline_runs/20260428/` | 固定 generated ids/text |
| 长期目标 profile | `baseline_runs/20260429-longterm-profile/` | TP/HYBRID direct runtime、weight shard、payload/perf |
| 当前性能 baseline | `baseline_runs/20260430-bfloat16-default/` | `bfloat16` 默认通信 dtype 后的性能 |
| Step 15 payload baseline | `baseline_runs/20260430-step15-derived-rebuild/` | 最新 multimodal payload keys/bytes |
| Step 16 pinned A/B | `baseline_runs/20260501-step16-pinned-ab/` | decode buffer reuse 后的 pinned memory opt-in 对照 |

## 固定输出

| case | generated_token_ids | generated_text |
| --- | --- | --- |
| text generate | `[104455, 9909, 9286, 16488]` | `人工智能（Artificial` |
| multimodal generate | `[87140, 15946, 3837, 101177]` | `视频中，一名` |
| tp-mm long decode 16 tokens | `[87140, 15946, 3837, 101177, 105611, 99194, 38035, 113727, 33108, 104362, 38035, 113233, 9370, 104253, 104224, 46944]` | `视频中，一名穿着深色衬衫和浅色裤子的男子站在一个` |

## 当前性能快照

主要来自 `baseline_runs/20260430-bfloat16-default/runtime-perf-table.md`。

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
| HYBRID stage1 `tp_degree=1` bypass | `648.46 MiB` TP collective / `1.525s` | `0 B` / `0s` | 清掉 single-rank 伪 collective |
| pure TP comm dtype | `449.12 MiB` / `44-45s` collective | `224.56 MiB` / `24-25s` opt-in | generated ids/text 不变 |
| `bfloat16` 默认落地 | 需要显式 `--comm-dtype bfloat16` | 默认 `torch.bfloat16` | `tp-mm` 默认 collective `221.48 MiB` |
| pure TP runtime input broadcast | `4` events / `6,451,200` bytes per rank | `0` events / `0` bytes | 本地 embedding 或 local stage input |
| Step 15 空 tensor slot | `tp-mm`: `12` keys / `12,093,371` bytes | `11` keys / `12,093,371` bytes | key count 下降，bytes 不变 |
| Step 15 derived shared | `tp-mm`: `11` keys / `12,093,371` bytes | `9` keys / `12,068,291` bytes | 少 `25,080` bytes |
| Step 15 hybrid stage1 startup | `7` keys / `3,245,384` bytes | `5` keys / `3,220,304` bytes | 少 `25,080` bytes |
| Step 16 decode 小 tensor | 每 step 追加 attention mask / 新建 token tensor | 预分配 decode mask buffer / 复用 token buffer | payload bytes 不变，减少 decode loop 小分配 |
| Step 16 pinned memory | 无开关 | `--transport-pin-memory` opt-in | TP 小幅变快，CUDA peak allocated 不变，默认关闭 |
| vLLM-style 函数命名 | HYBRID helper 叫 `runtime_input` | 内部 helper 改为 `model_input` | wire protocol 和 bytes 不变 |

## Step 16 Pinned Memory A/B

目录：`baseline_runs/20260501-step16-pinned-ab/`。

| case | rank | total s | TP coll s | startup/scaffold/handoff | TP coll bytes | CUDA peak alloc/reserved | pin used | correctness |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| `tp-mm-generate-step16-default-j23` | 0 | `53.47` | `24.34` | `11.51 MiB / 0 B / 0 B` | `221.48 MiB` | `6.53 / 6.74 GiB` | `0` | pass |
| `tp-mm-generate-step16-default-j23` | 1 | `53.21` | `23.76` | `11.51 MiB / 0 B / 0 B` | `221.48 MiB` | `6.52 / 6.73 GiB` | `0` | pass |
| `tp-mm-generate-step16-pinned-j23` | 0 | `53.01` | `23.91` | `11.51 MiB / 0 B / 0 B` | `221.48 MiB` | `6.53 / 6.74 GiB` | `289` | pass |
| `tp-mm-generate-step16-pinned-j23` | 1 | `52.97` | `23.51` | `11.51 MiB / 0 B / 0 B` | `221.48 MiB` | `6.52 / 6.73 GiB` | `289` | pass |
| `hybrid-mm-generate-step16-default-j23shared` | 0/1/2 | `32.69-32.85` | `2.25 / 1.58 / 0.00` | bytes unchanged | `113.82 MiB / 113.82 MiB / 0 B` | unchanged | `0` | pass |
| `hybrid-mm-generate-step16-pinned-j23shared` | 0/1/2 | `32.80-32.96` | `2.27 / 1.49 / 0.00` | bytes unchanged | `113.82 MiB / 113.82 MiB / 0 B` | unchanged | `154 / 149 / 1` | pass |

结论：

- generated ids/text 固定为 `[87140, 15946, 3837, 101177]` / `视频中，一名`。
- `--transport-pin-memory` 不改变 payload bytes、TP collective bytes 或 weight shard scope。
- 纯 TP total time 小幅下降约 `0.24-0.46s`，TP collective time 小幅下降约 `0.25-0.43s`。
- CUDA peak allocated 不上升；TP rank0 reserved 约多 `2 MiB`。
- HYBRID A/B 因 rank0/rank1 共用 jetson2，只作为功能性验证。
- 当前收益不大，保持默认关闭。

## Step 15 Payload 结论

已完成：

- `None` tensor slot 不再进入 startup transport。
- `shared.attention_mask_2d` 仅在全 1 时省略，并在接收端用 `input_ids` 本地重建。
- `shared.position_ids` 在可由 `input_ids`、`mm_token_type_ids`、grid metadata 和本地 `mm_config` 重建时省略。
- `attention_mask/cos/sin` 继续保持本地重建，不进入 generate startup contract。
- `stage_input/deepstack_by_layer` 不直接删除，只冻结 owner/rebuild 规则。

大 payload owner 规则：

- stage0/input-owner 的 `stage_input` 是 processed `inputs_embeds`；只有每个 TP rank 能本地构建相同 multimodal embeddings 时，才能删除 broadcast。
- non-stage0 的 `stage_input` 是上一 PP stage 输出，等价于 `intermediate_tensors`，不能从 prompt 本地重建。
- `deepstack_by_layer` 只发送给实际消费该 layer 的 stage/rank；当前基线只属于 stage0。
- non-input-owner 不能为了重建 deepstack 重新跑视觉 frontend。
- 禁止重新引入 `root_input`、`boundaries`、`hidden_states`、`stage_output`、frontend paths 或 replay/full payload。

## 验收字段

每次 runtime 主路径改动至少检查：

- `generated_token_ids`
- `generated_text`
- 所有 distributed ranks 是否成功结束
- `runtime_metrics.transport` 的 payload keys/count/bytes
- `runtime_metrics.memory` 的 CUDA peak allocated/reserved
- `weight_load.loaded_weight_tensor_bytes`

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

同步：

```bash
bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed
bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed
```

## 当前下一步

下一阶段是 `ROADMAP.md` step 20：`KV cache 管理部分`。

Step 16 已关闭：

- allocation / clone 盘点已完成：`BUFFER_REUSE_AUDIT.md`。
- decode loop 小 tensor reuse 已完成，本地轻量回归通过。
- pinned memory opt-in 已完成：`--transport-pin-memory`。
- 真实 Jetson A/B 已记录在 `baseline_runs/20260501-step16-pinned-ab/`。
- KV cache clone/cat 留到 step 20 的 Jupiter-style 连续 KV buffer 阶段。
- InfiniPot-V-style 视觉 token 压缩和 ReKV-style 历史窗口检索作为后续重点。
- 暂不考虑 vLLM-style BlockPool / prefix cache / serving scheduler。
