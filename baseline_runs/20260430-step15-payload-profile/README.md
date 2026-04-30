# 2026-04-30 Step 15 Multimodal Payload First Cut

目的：开始 `ROADMAP.md` step 15，先对照 vLLM 的 multimodal input / embeddings 边界，统计当前 payload，再只移除最低风险字段。

## vLLM 对照

- vLLM 的 multimodal processor 输出是 `MultiModalInputs`：`prompt_token_ids`、`mm_kwargs`、`mm_hashes`、`mm_placeholders`。
- vLLM 支持把 image/video tensor 当作 embeddings 直接进入模型，跳过 HF processing。
- 对我们这里的映射：主路径应传 compact runtime tensor / metadata，不传 frontend root/full/replay payload。
- 参考：
  - https://docs.vllm.ai/en/stable/api/vllm/multimodal/inputs/
  - https://docs.vllm.ai/en/v0.9.0.1/contributing/model/multimodal.html

## 字段分类

当前必须传：

- `stage_handoffs.*.stage_input`：non-owner / non-stage0 需要的 processed hidden input。
- `stage_visuals.*.deepstack_by_layer.*`：对应 stage 的 decoder layer 会消费的 visual deepstack。
- `stage_visuals.*.visual_pos_masks`：当本 stage 有 deepstack 时需要。

后续候选：

- `shared.position_ids`：未来可以尝试从 `input_ids`、grid metadata、token type 和 mm config 本地重建。
- `shared.attention_mask_2d`：若能证明全 1 且只依赖长度，可以本地重建。

本轮最低风险移除：

- `None` tensor slot，例如无图片时的 `shared.image_grid_thw`、stage1 无 deepstack 时的 `stage_visuals.1.visual_pos_masks`。

## Before / After

before 对照目录：`baseline_runs/20260430-bfloat16-default/`。

| path | before keys / bytes | after keys / bytes | removed |
| --- | ---: | ---: | --- |
| `tp-mm-generate` startup contract rank1 | `12` / `12,093,371` | `11` / `12,093,371` | `shared.image_grid_thw` |
| `hybrid-mm-generate` stage0 runtime input broadcast rank1 | `11` / `12,093,371` | `11` / `12,093,371` | none |
| `hybrid-mm-generate` stage1 startup contract rank2 | `9` / `3,245,384` | `7` / `3,245,384` | `shared.image_grid_thw`, `stage_visuals.1.visual_pos_masks` |

bytes 不变是预期结果：本轮删的是 `None`/0-byte 槽位，不是大 tensor。

## Smoke

| case | checker | generated ids | generated text |
| --- | --- | --- | --- |
| `tp-mm-generate-step15` | PASS | `[87140, 15946, 3837, 101177]` | `视频中，一名` |
| `hybrid-mm-generate-step15` | PASS | `[87140, 15946, 3837, 101177]` | `视频中，一名` |

输出文件：

- `check-tp-mm-generate-step15.txt`
- `check-hybrid-mm-generate-step15.txt`
- `runtime-perf-records.json`
- `runtime-perf-table.md`

结论：第一刀已把空 tensor 槽位从 startup transport 中移除，correctness 不变，`weight_load.tp_weight_sharded=true` 未回退。后续 `shared.position_ids` / `shared.attention_mask_2d` 本地重建已在 `baseline_runs/20260430-step15-derived-rebuild/` 完成；`stage_input` / `deepstack_by_layer` 只冻结 owner/rebuild 语义，不直接删除。
