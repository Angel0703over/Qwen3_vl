# qwen3vl_tp_runtime Handoff

这份文件只保留新对话接手必需信息。详细历史看 `BASELINE.md` 和 `baseline_runs/*/README.md`。

## 一句话上下文

这是 Qwen3-VL 的 correctness-first 分布式推理 runtime 原型。主路径已经从 replay bundle 迁移到启动时直接从 `model_path` 构建 `StageState`，并支持 `PP / TP / HYBRID`。

## 环境

- 当前日期：2026-05-01
- 工作目录：`/mnt/ssd/code/Qwen3_vl`
- runtime 目录：`qwen3vl_tp_runtime`
- Python：`/mnt/ssd/miniconda3/envs/vlm/bin/python`
- Torchrun：`/mnt/ssd/miniconda3/envs/vlm/bin/torchrun`
- 模型：`/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct`
- 帧目录：`/mnt/ssd/code/Qwen3_vl/frames`
- Jetson：
  - jetson2：`10.126.126.3`
  - jetson3：`10.126.126.4`

同步：

```bash
bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed
bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed
```

## 架构不变量

- `pipeline_parallel.py` 是纯 PP 基础后端。
- `tensor_parallel.py` 是纯 TP 基础后端。
- `hybrid_parallel.py` 是 PP+TP 组合后端。
- HYBRID 可以调用 PP/TP helper；TP 不能反向依赖 HYBRID。
- `hexgen_core/modules/` 只放三种并行后端。
- 主路径对象叫 `StageState`。
- `bundle` 只保留给 replay/debug/capture。
- 内部 helper 优先用 vLLM-style `model_input`；wire key/protocol 仍保留 `runtime_inputs`。
- `--manifest-path`、capture、trace、dump 都属于 debug/replay 路径。

## 当前状态

| 项目 | 当前状态 |
| --- | --- |
| PP | direct StageState 主路径已通过 |
| TP | 独立 manifest/runner，不依赖 HYBRID |
| HYBRID | 组合层，stage-group runtime input schema 已固化 |
| TP 权重 | decoder/MLP projection 已 rank-local materialize |
| multimodal TP | rank0/input-owner 准备 startup contract，其他 TP rank consume-only |
| startup contract | 不传 root/full/replay payload，不传 dense derived tensors |
| comm dtype | 默认 `bfloat16` |
| Step 15 | 已结束，payload owner/rebuild 语义已冻结 |
| Step 16 | 已结束：decode 小 tensor 复用 + pinned memory A/B，`--transport-pin-memory` 保持默认关闭 |

## 关键 Before / After

| 修改 | before | after | 效果 |
| --- | ---: | ---: | --- |
| startup `stage_output` | `13` tensors / `7,563,328` bytes | `12` tensors / `4,353,088` bytes | 少传 reference output |
| startup dense derived | `12` tensors / `4,353,088` bytes | `9` tensors / `3,245,806` bytes | `attention_mask/cos/sin` 本地重建 |
| HYBRID stage1 `tp_degree=1` | `648.46 MiB` TP collective | `0 B` | 清掉伪 collective |
| comm dtype | `float32`，`tp-mm` 约 `449.12 MiB` | 默认 `bfloat16`，约 `221.48 MiB` | TP collective bytes 约减半 |
| pure TP runtime input | `4` broadcast events / rank | `0` | 本地 embedding 或 local stage input |
| Step 15 derived shared | `11` keys / `12,093,371` bytes | `9` keys / `12,068,291` bytes | affected payload 少 `25,080` bytes |
| Step 16 decode 小 tensor | 每 step `cat(mask, ones)` / 新建 token tensor | 预分配 mask buffer / 复用 token buffer | 减少 decode loop 小分配，payload bytes 不变 |
| Step 16 pinned memory | 无实验开关 | `--transport-pin-memory` | TP 小幅变快，CUDA peak 不变，默认关闭 |
| 代码冗余清理 | PP/HYBRID worker wrapper + `StageHandoffTransport` | 直接 phase impl + `StageCommunicator` | 主路径 API 更窄 |
| vLLM-style 命名 | HYBRID `runtime_input` helper | 内部 `model_input` helper | 代码更直观，wire protocol 不变 |

## Step 15 结论

已完成：

- startup transport 跳过 `None` tensor slot。
- `shared.attention_mask_2d` 可重建时不传。
- `shared.position_ids` 可重建时不传。
- `attention_mask/cos/sin` 保持本地重建。
- `stage_input/deepstack_by_layer` 的 owner/rebuild 语义已冻结。

冻结规则：

- stage0/input-owner 的 `stage_input` 是 processed `inputs_embeds`；只有每个 TP rank 能本地构建相同 multimodal embeddings 时，才能删除 broadcast。
- non-stage0 的 `stage_input` 是上一 PP stage 输出，等价于 `intermediate_tensors`，不能从 prompt 本地重建。
- `deepstack_by_layer` 只发给实际消费该 layer 的 stage/rank。
- non-input-owner 不能为了重建 deepstack 重新跑视觉 frontend。
- 不允许重新引入 `root_input`、`boundaries`、`hidden_states`、`stage_output`、frontend paths 或 replay/full payload。

## 当前 Baseline

| 用途 | 目录 |
| --- | --- |
| correctness baseline | `baseline_runs/20260428/` |
| 长期目标 profile | `baseline_runs/20260429-longterm-profile/` |
| 当前性能 baseline | `baseline_runs/20260430-bfloat16-default/` |
| Step 15 payload baseline | `baseline_runs/20260430-step15-derived-rebuild/` |
| Step 16 pinned A/B | `baseline_runs/20260501-step16-pinned-ab/` |

固定输出：

- text：`[104455, 9909, 9286, 16488]`，`人工智能（Artificial`
- multimodal：`[87140, 15946, 3837, 101177]`，`视频中，一名`

## 常用命令

TP multimodal：

```bash
NODE_RANK=0 MASTER_ADDR=10.126.126.3 bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh
NODE_RANK=1 MASTER_ADDR=10.126.126.3 bash qwen3vl_tp_runtime/scripts/helpers/run-tp-mm-generate.sh
```

HYBRID multimodal：

```bash
NODE_RANK=0 MASTER_ADDR=10.126.126.3 bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh
NODE_RANK=1 MASTER_ADDR=10.126.126.3 bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh
NODE_RANK=2 MASTER_ADDR=10.126.126.3 bash qwen3vl_tp_runtime/scripts/helpers/run-hybrid-mm-generate.sh
```

本地最小回归：

```bash
bash qwen3vl_tp_runtime/scripts/helpers/run-runtime-core-regression.sh
```

## 当前下一步

`ROADMAP.md` step 16：`Buffer reuse / pinned memory` 已结束。

完成内容：

1. allocation / clone 盘点已完成，详见 `BUFFER_REUSE_AUDIT.md`。
2. decode loop 小 tensor reuse 已完成，涉及 `generate_buffers.py` 和 `PP / TP / HYBRID` runtime-only generate。
3. pinned memory opt-in 已完成：运行时加 `--transport-pin-memory`。
4. 真实 Jetson A/B 已记录在 `baseline_runs/20260501-step16-pinned-ab/`。

Step 16 结论：

- `tp-mm-generate` pinned total time 从 `53.47 / 53.21s` 到 `53.01 / 52.97s`。
- TP collective time 从 `24.34 / 23.76s` 到 `23.91 / 23.51s`。
- payload bytes、generated ids/text、weight shard scope 不变。
- CUDA peak allocated 不上升；rank0 reserved 约多 `2 MiB`。
- 收益偏小，`--transport-pin-memory` 保持 opt-in，不改默认。

下一步建议：

- 开始 step 20：`KV cache 管理部分`。
- 当前重点只放前三条路线：
  - Jupiter-style 连续 KV buffer：提前分配 GPU KV，用 `current_length` 避免 decode 每步 `torch.cat`。
  - InfiniPot-V-style 视觉 token KV 压缩：关注视频时间冗余和空间语义重要性。
  - ReKV-style 历史窗口检索回取：query 来时只回取必要 KV，避免 GPU cache 无限增长。
- 暂不考虑 vLLM-style BlockPool / prefix cache / serving scheduler。

## 工作习惯

- 默认中文回答。
- 修改前先读 `SESSION_HANDOFF.md`、`README.md`、`ROADMAP.md`、`BASELINE.md`。
- 搜索用 `rg`，文件列表用 `rg --files`。
- 手动编辑用 `apply_patch`。
- 不做顺手重构。
- 不回滚用户改动。
- 改 runtime 主路径后，按 `BASELINE.md` 的验收字段跑对应 smoke。
