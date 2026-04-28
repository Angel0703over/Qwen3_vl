# qwen3vl_tp_runtime Roadmap

这份 Roadmap 只记录接下来要做的事情。更完整的历史状态、对话迁移信息和已完成细节见：

- `README.md`
- `BASELINE.md`
- `SESSION_HANDOFF.md`

## 当前完成快照

当前主线里程碑已经完成：

- `PP / TP / hybrid` 主路径可以在启动时直接从 `model_path` 构建每个 stage/rank 的 `StageState`。
- 主路径术语已经从旧 `bundle` 收口为 `StageState`；`bundle` 只保留在 replay/debug/capture/file-backed reference 语义里。
- `backend=tp` 已经是独立后端，有自己的 `TensorParallelManifest / build_direct_tp_manifest / TensorParallelRunner / run_tensor_parallel_rank`。
- `backend=tp` 不再依赖 `TextHybridManifest`、`TextHybridRunner` 或 `hybrid_parallel.py`。
- 架构方向已经明确为：`PP` 和 `TP` 是并列基础后端，`HYBRID` 是依赖二者的组合后端。
- direct `tp_degree > 1` stage 已经走 rank-local materialize：每个 TP rank 只加载自己的 text decoder projection shard。
- `pp / hybrid multimodal generate` 已通过 Jetson smoke，stage scope、frontend ownership、startup transport thin contract、TP shard-local materialize 都有 summary/log 证据。
- `--pp N` 和 `--tp N` 已作为推荐 CLI 参数；`--stage-ranges` / `--tp-degrees` 保留为高级覆盖。

当前不要重复做这些大目标，后续主要进入整理、自动化和性能阶段。

## 架构不变量

后续所有改动都必须遵守：

- `pipeline_parallel.py` 是纯 PP 基础后端。
- `tensor_parallel.py` 是纯 TP 基础后端。
- `hybrid_parallel.py` 是 PP+TP 组合后端。
- HYBRID 可以调用 PP/TP 的函数。
- TP 不能反向依赖 HYBRID。
- PP 不应为了复用少量 helper 反向依赖 TP，除非后续明确引入新的公共基础模块。
- 主路径 public API 不再使用 `bundle` 表达 runtime 对象。
- `--manifest-path`、capture、replay、trace、dump 都属于 debug/replay 路径，不是默认主路径。

## P0：当前优先清理

### 1. 已完成：处理 TP 里的空壳 worker 类

背景：

- `tensor_parallel.py` 之前有 `StageRunner / GenerateWorker / DecodeWorker / TensorParallelRunner`。
- `StageRunner` 和 `TensorParallelRunner` 有实际职责。
- `GenerateWorker` / `DecodeWorker` 主要是命名对齐，没有独立职责。

目标：

- 让 TP worker 类名和职责一致。

已采用方案：

- 删除 `GenerateWorker / DecodeWorker` 空壳。
- `TensorParallelRunner` 直接继承 `StageRunner`。
- `tensor_parallel.py.__all__` 不再导出 TP 的 `GenerateWorker / DecodeWorker`。

验收：

- `TensorParallelRunner -> StageRunner` 继承关系清晰。
- TP 主路径不再出现完全空壳 worker 类。
- `test/test_tensor_parallel_direct.py` 通过。
- `test/test_compat_package_exports.py` 已同步导出面预期。

### 2. 移出 TP 文件里的旧 replay/debug TP 路径

背景：

- `tensor_parallel.py` 当前同时包含主路径 TP 和旧 replay/debug TP。
- 旧路径包括：
  - `load_text_stage_bundle`
  - `run_text_tensor_parallel_stage`
  - `TextTensorParallelRunner`
  - `run_text_tensor_parallel_rank`
  - `DEBUG_REPLAY_EXPORTS`

目标：

- 让 `tensor_parallel.py` 更像纯主路径 TP 文件。
- 旧 replay/debug 仍然可用，但移动到更明确的位置。

建议位置：

- `hexgen_core/modules/tp_debug.py`，如果继续视作 debug helper。

验收：

- `tensor_parallel.py.__all__` 只保留 direct TP 主路径。
- 旧 replay/debug import 仍能通过 `DEBUG_REPLAY_EXPORTS` 或明确 compat 路径访问。
- `test/test_compat_package_exports.py` 覆盖新的位置。
- `test/test_tensor_parallel_direct.py` 通过。

### 3. 规范 HYBRID 调用 TP helper 的方式

背景：

- HYBRID 依赖 TP，因此 HYBRID 直接复用 TP helper 是正确方向。
- 当前 `hybrid_parallel.py` 从 `tensor_parallel.py` import 了若干 `_` 开头 helper：
  - `_build_generate_phase_state`
  - `_strip_runtime_layer_cache`
  - `_build_generate_cache_map`
  - `_is_runtime_only_generate_state`
  - `_infer_runtime_tensor_device`
  - `_infer_runtime_tensor_dtype`
  - `_infer_runtime_token_dtype`
  - `_build_runtime_only_stage_input_template`
  - `_token_tensor_to_list`
  - `_broadcast_token_id`

目标：

- 保持依赖方向正确，同时让代码风格更清楚。

可选方案：

- 方案 A：保留私有 helper import，并在 `hybrid_parallel.py` 注释说明 HYBRID 是组合层，允许复用 TP internal helper。
- 方案 B：把这些 helper 改成无下划线的 module-level helper，但不放入 package-level public `__all__`。

建议：

不要新建 `generate_common.py`，除非后续用户重新确认。

验收：

- `tensor_parallel.py` 仍不依赖 HYBRID。
- `hybrid_parallel.py` 不再复制这些 helper。
- `test/test_hybrid_direct_loader.py` 和 `test/test_tensor_parallel_direct.py` 通过。

### 4. 收紧 StageState / bundle 命名残留

背景：

- 主路径已经迁移到 `StageState`。
- 仍有一些 `bundle` 出现在 file-backed reference、capture/replay/debug、layer weight 兼容变量中。

目标：

- 区分“应该保留的 legacy/debug bundle”和“主路径里漏掉的 bundle 命名”。

检查命令：

```bash
rg -n "stage_bundle|bundle" qwen3vl_tp_runtime -g'*.py' -g'*.md'
```

处理规则：

- capture/replay/debug/file-backed reference：可以保留。
- weight loader 内部的 `layer_bundle`：可以暂时保留，因为表示层参数集合，不是 stage runtime object。
- direct 主路径 public API / summary / README 主路径说明：不应该继续叫 bundle。

验收：

- 新增或修改的主路径代码不引入新的 `stage_bundle` 命名。
- `README.md` / `ROADMAP.md` / `BASELINE.md` 术语保持一致。

## P1：自动化回归

### 5. 增加 baseline log 检查脚本

背景：

- 当前 `BASELINE.md` 已记录应检查字段。
- 真实分布式 smoke 仍主要靠人工读 JSON summary 和日志。

目标：

- 增加一个脚本读取 rank log，自动检查关键字段。

建议文件：

- `qwen3vl_tp_runtime/scripts/check_baseline_logs.py`

建议支持：

- 输入一个 case id 和多个 rank log。
- 自动提取最后一个 JSON summary。
- 检查 `generated_token_ids` / `generated_text` 是否一致。
- 对 `tp-text-generate` 检查：
  - `tp_weight_sharded=true`
  - `tp_shard_rank`
  - `tp_shard_world_size`
  - `tp_shard_shape_ok`
  - `tp_stage_loaded_weight_tensor_bytes_equal`
- 对 `pp-mm-generate` 检查：
  - stage0 frontend active
  - stage1 consume-only
  - stage scope
- 对 `hybrid-mm-generate` 检查：
  - stage0 TP ranks shard-local
  - stage1 stage-local
  - all ranks generated ids 一致

验收：

- 能检查 `baseline_runs/20260427/pp-mm-generate-rank*.log`。
- 能检查 `baseline_runs/20260427/hybrid-mm-generate-rank*.log`。
- 检查失败时错误信息指向具体 rank 和字段。

### 6. 固化 runtime smoke wrapper

背景：

- `BASELINE.md` 已有命令，但分布式多 rank 仍需要手动复制。

目标：

- 给常用 case 增加稳定 shell wrapper，减少手动出错。

建议：

- 保留 `scripts/helpers/run-runtime.sh`。
- 新增或扩展：
  - `scripts/helpers/run-pp-mm-generate.sh`
  - `scripts/helpers/run-tp-text-generate.sh`
  - `scripts/helpers/run-hybrid-mm-generate.sh`

要求：

- 支持传 `NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。
- 默认 `OUT=baseline_runs/$(date -u +%Y%m%d)`。
- 默认使用 `--pp` / `--tp` 参数，不再默认写 `--stage-ranges`。

验收：

- 每个 wrapper 打印最终命令。
- 用户可以直接在 rank0/rank1/rank2 机器上运行。

## P2：真实分布式回归补齐

### 7. 重跑并冻结完整 baseline

当前已知 smoke 记录：

- `tp-text-generate` 已通过。
- `pp-mm-generate` 已通过。
- `hybrid-mm-generate` 已通过。
- `hybrid-text-generate` 已通过。

建议补齐或重跑：

- `pp-text-generate`
- `tp-mm-generate`
- `hf-text-generate`
- `hf-mm-generate`
- `live-mm-generate`

验收：

- 每个 case 都有 rank log 或 stdout/stderr 文件。
- `BASELINE.md` 更新实际日期、log 路径、generated ids、generated text。
- distributed case 用新的 baseline checker 自动检查。

### 8. 每次 runtime core 变更后的最小回归矩阵

如果改到这些文件：

- `tensor_parallel.py`
- `pipeline_parallel.py`
- `hybrid_parallel.py`
- `runtime_builder.py`
- `runtime_text_stage.py`
- `weights/`
- `transport.py`
- `schema.py`

至少跑：

```bash
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_tensor_parallel_direct.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_pipeline_direct_loader.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_hybrid_direct_loader.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_runtime_cli_modes.py
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_compat_package_exports.py
```

如果改到权重加载：

```bash
PYTHONPATH=. /mnt/ssd/miniconda3/envs/vlm/bin/python test/test_model_weight_loader.py
```

如果改到真实 generate 路径，额外在 Jetson 跑：

- `tp-text-generate`
- `pp-mm-generate`
- `hybrid-mm-generate`

## P3：性能和显存基线

### 9. 建立启动耗时和显存记录

背景：

- 当前验收重点是 correctness 和 shard/stage scope。
- 启动时间和峰值显存还没有系统记录。

目标：

- 给每个 baseline case 记录：
  - prepare session 时间
  - startup contract broadcast 时间
  - materialize stage 时间
  - post-load barrier 时间
  - generate 总时间
  - 峰值显存

建议：

- 继续使用 `HEXGEN_STARTUP_LOG=1`。
- 在 runner summary 增加可机器读取的 timing 字段，而不仅是 stdout log。
- 如果可行，记录 `torch.cuda.max_memory_allocated()` 和 `torch.cuda.max_memory_reserved()`。

验收：

- `BASELINE.md` 增加性能/显存表。
- 每个 rank 的 timing/memory 可以从 JSON summary 或 log checker 读出。

### 10. 性能优化候选

等 correctness 和 baseline 自动化稳定后，再做：

- startup contract tensor 减量。
- scaffold broadcast 减量。
- buffer reuse。
- pinned memory。
- PP handoff overlap。
- TP all-reduce / all-gather profiling。
- stage partition 搜索。

注意：

- 性能优化不应破坏当前 correctness guard。
- 每个优化都要保留前后 baseline 对比。

## P4：功能增强

### 11. embedding / lm_head vocab parallelism

当前状态：

- decoder projection / MLP projection 已按 TP shard-local 加载。
- `embed_tokens_weight` 和 `lm_head_weight` 仍按当前执行语义复制。

目标：

- 研究并实现 vocab parallel embedding / lm_head。

风险：

- 会影响 token embedding、logits gather、top-k、sampling。
- 会影响 PP stage0 和 last stage 的权重 scope。

验收：

- TP rank 不再完整复制 embedding/lm_head。
- logits 与当前 reference 对齐。
- `weight_load` 能证明 vocab shard shape。

### 12. KV cache session 化

当前状态：

- KV cache 主要以 correctness-first 的 `cache_by_layer` 字典存在。

目标：

- 设计 stage-local cache manager。
- prefill 后 cache 持久化。
- decode step 复用 cache。
- 后续支持多请求。

验收：

- text generate 和 multimodal generate 不依赖 replay/reference cache。
- cache 生命周期清晰。
- 可以区分 session / request / stage / layer。

### 13. 更接近 serving engine

长期目标：

- request scheduler。
- paged KV cache。
- block manager。
- continuous batching。
- streaming output。
- prefill/decode overlap。

这不是当前 P0/P1 任务，只有在 correctness、baseline、性能记录稳定后再推进。

## Definition Of Done

一个 Roadmap 任务完成时，至少满足：

- 代码改动符合依赖方向。
- 主路径术语仍是 `StageState`。
- debug/replay 路径没有被误认为主路径。
- 相关 unit tests 通过。
- 如果影响分布式 runtime，至少跑一个真实 Jetson smoke。
- 如果影响主路径文档或命令，更新 `README.md` / `BASELINE.md` / `SESSION_HANDOFF.md` 中对应内容。
- 如果需要在 Jetson 上继续测试，完成同步：

```bash
bash sync-to-jetson2.sh --host 10.126.126.3 --git-changed
bash sync-to-jetson2.sh --host 10.126.126.4 --git-changed
```
