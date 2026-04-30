# 2026-04-30 Step 15 Derived Shared Rebuild

目的：继续 `Multimodal Payload 减量`，把可本地重建的 `shared.attention_mask_2d` / `shared.position_ids` 从 generate 主路径 transport 中移出。

## vLLM 对照

- vLLM Qwen3VL forward 边界是 `input_ids`、`positions`、`intermediate_tensors`、`inputs_embeds` 和 multimodal grid metadata。
- 对我们这里的映射：`positions` 是运行时本地需要的输入，但不一定必须作为 startup/broadcast payload 传输；只要有 `input_ids`、`mm_token_type_ids`、`image_grid_thw` / `video_grid_thw` 和本地 `mm_config`，就能本地重建 mRoPE position。
- 参考：https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/qwen3_vl/

## 改动

- `shared.attention_mask_2d`：仅当它是全 1 tensor 时省略，接收端用 `torch.ones_like(input_ids)` 重建。
- `shared.position_ids`：当可从 `input_ids`、`mm_token_type_ids`、grid metadata 和本地 `mm_config.vision_config.spatial_merge_size` 重建时省略。
- `attention_mask` / `cos` / `sin` 继续保持本地重建，不进入 generate startup contract。
- 若未来出现带 padding 的 `attention_mask_2d`，当前过滤逻辑会保留原 tensor，不会盲目省略。

## Before / After

before 对照目录：`baseline_runs/20260430-step15-payload-profile/`。

| path | before keys / tensor bytes | after keys / tensor bytes | removed |
| --- | ---: | ---: | --- |
| `tp-mm-generate` startup contract rank1 | `11` / `12,093,371` | `9` / `12,068,291` | `shared.attention_mask_2d`, `shared.position_ids` |
| `hybrid-mm-generate` stage0 runtime input broadcast rank1 | `11` / `12,093,371` | `9` / `12,068,291` | `runtime_inputs.shared.attention_mask_2d`, `runtime_inputs.shared.position_ids` |
| `hybrid-mm-generate` stage1 startup contract rank2 | `7` / `3,245,384` | `5` / `3,220,304` | `shared.attention_mask_2d`, `shared.position_ids` |

每个 affected payload 少 `25,080` tensor bytes：

- `attention_mask_2d`: `5,016` bytes
- `position_ids`: `20,064` bytes

## Smoke

| case | checker | generated ids | generated text |
| --- | --- | --- | --- |
| `tp-mm-generate-derived-rebuild` | PASS | `[87140, 15946, 3837, 101177]` | `视频中，一名` |
| `hybrid-mm-generate-derived-rebuild` | PASS | `[87140, 15946, 3837, 101177]` | `视频中，一名` |

输出文件：

- `check-tp-mm-generate-derived-rebuild.txt`
- `check-hybrid-mm-generate-derived-rebuild.txt`
- `runtime-perf-records.json`
- `runtime-perf-table.md`

## 大 Payload Owner / Rebuild 规划

当前剩余大 payload：

| payload | shape / dtype | bytes | owner / 结论 |
| --- | --- | ---: | --- |
| `stage_handoffs.0.stage_input` / `runtime_inputs.stage_handoff.stage_input` | `[1, 627, 2560]` / `bfloat16` | `3,210,240` | stage0/input-owner 或 stage leader 拥有；非 owner 目前必须接收 |
| `stage_visuals.0.deepstack_by_layer.{0,1,2}` | `[576, 2560]` / `bfloat16` each | `8,847,360` total | 归属消费 deepstack 的 stage；当前只属于 stage0 |
| `stage_handoffs.1.stage_input` | `[1, 627, 2560]` / `bfloat16` | `3,210,240` | 上一 PP stage 输出，stage1 不能从 prompt 本地重建 |

规划结论：

- `stage_input` 如果是 stage0/input-owner 的 processed `inputs_embeds`，未来只有在每个 TP rank 能本地构建同样 multimodal embeddings 时，才能减少 broadcast。
- `stage_input` 如果是 non-stage0 的 PP handoff / `intermediate_tensors`，必须跨 stage 传输。
- `deepstack_by_layer` 只发送给实际消费该 layer 的 stage/rank；当前 stage1 没有 deepstack，不应收到 deepstack。
- non-input-owner 不允许为了重建 deepstack 重新跑视觉 frontend。
- 不能重新引入 `root_input`、`boundaries`、`hidden_states`、`stage_output`、frontend paths 或 replay/full payload。

结论：`shared.attention_mask_2d` / `shared.position_ids` 已从 generate 主路径 startup/runtime input payload 中移出并本地重建，correctness 不变。`stage_input` / `deepstack_by_layer` 的 owner/rebuild 语义已冻结；第 15 步到此结束，真正删除这些大 payload 需要另开阶段。
